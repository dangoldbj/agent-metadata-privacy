"""Capture a real A2A task lifecycle to anchor the generative model.

We stand up a real `a2a-sdk` JSON-RPC server with a minimal *streaming* agent, then
drive a full task through the real SDK client (discovery -> message/send ->
streamed updates -> completion) over HTTP on localhost. We record the actual wire
messages -- the agent-card discovery fetch and the JSON-RPC POST via httpx event
hooks (real byte sizes), and each streamed update via the client's event stream
(real cadence). No LLM or API keys: the agent just emits text chunks.

The capture is summarized by `anchor.py` and used to sanity-check that the
generator's message sizes, update cadence, and lifecycle structure sit in realistic
ranges. This is calibration/validation, not a labeled corpus.

Run:  uv run --group anchor python scripts/capture_a2a.py
Output: results/anchor_capture.json
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette

from a2a.client.card_resolver import A2ACardResolver
from a2a.client.client_factory import ClientConfig, ClientFactory, TransportProtocol
from a2a.helpers.proto_helpers import new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Message,
    Part,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    TaskState,
)

HOST, PORT = "127.0.0.1", 8731
RPC_URL = f"http://{HOST}:{PORT}/"
OUT = Path(__file__).resolve().parents[1] / "results" / "anchor_capture.json"

N_TASKS = 12  # number of task lifecycles to capture
UPDATE_DELAY = 0.05  # server delay between streamed updates (s) -> realistic cadence


class StreamingEchoExecutor(AgentExecutor):
    """A minimal agent: submits, streams a few working updates, returns a result."""

    async def execute(self, context, event_queue) -> None:
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)  # Task must precede status updates
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        # a small, slightly variable number of streamed updates with text payloads
        n_updates = 2 + (len(task.id) % 4)
        for i in range(n_updates):
            text = f"processing step {i}: " + ("x" * (200 + 90 * i))
            await updater.update_status(
                TaskState.TASK_STATE_WORKING,
                message=updater.new_agent_message([Part(text=text)]),
            )
            await asyncio.sleep(UPDATE_DELAY)
        await updater.add_artifact(
            [Part(text="result: " + ("y" * 600))], name="result"
        )
        await updater.complete()

    async def cancel(self, context, event_queue) -> None:  # pragma: no cover - unused
        raise NotImplementedError


def build_app() -> Starlette:
    card = AgentCard(
        name="anchor-echo",
        description="minimal streaming agent for wire capture",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(id="echo", name="echo", description="streams text", tags=["demo"])
        ],
        supported_interfaces=[
            AgentInterface(url=RPC_URL, protocol_binding=TransportProtocol.JSONRPC)
        ],
    )
    handler = DefaultRequestHandler(StreamingEchoExecutor(), InMemoryTaskStore(), card)
    routes = create_jsonrpc_routes(handler, "/") + create_agent_card_routes(card)
    return Starlette(routes=routes)


def _serve_in_thread() -> uvicorn.Server:
    config = uvicorn.Config(build_app(), host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # wait for readiness
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            r = httpx.get(RPC_URL + ".well-known/agent-card.json", timeout=0.5)
            if r.status_code == 200:
                return server
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


async def _run_capture() -> list[dict]:
    records: list[dict] = []

    async def on_request(request: httpx.Request) -> None:
        records.append({
            "kind": "http_request", "t": time.perf_counter(), "direction": "c2s",
            "method": request.method, "path": request.url.path,
            "bytes": len(request.content or b""),
        })

    async def on_response(response: httpx.Response) -> None:
        cl = response.headers.get("content-length")
        records.append({
            "kind": "http_response", "t": time.perf_counter(), "direction": "s2c",
            "path": response.request.url.path, "status": response.status_code,
            "bytes": int(cl) if cl is not None else None,
        })

    async with httpx.AsyncClient(
        event_hooks={"request": [on_request], "response": [on_response]}, timeout=30
    ) as http:
        resolver = A2ACardResolver(http, RPC_URL)
        card = await resolver.get_agent_card()  # discovery GET (captured by hooks)
        factory = ClientFactory(ClientConfig(httpx_client=http, streaming=True))
        client = factory.create(card)

        for task_no in range(N_TASKS):
            msg = Message(
                message_id=str(uuid.uuid4()),
                role=Role.ROLE_USER,
                parts=[Part(text=f"task {task_no}: please process")],
            )
            req = SendMessageRequest(
                message=msg,
                configuration=SendMessageConfiguration(accepted_output_modes=["text/plain"]),
            )
            async for event in client.send_message(req):
                records.append({
                    "kind": "stream_event", "t": time.perf_counter(), "direction": "s2c",
                    "task_no": task_no, "bytes": len(event.SerializeToString()),
                    "event_type": type(event).__name__,
                })
        await client.close()
    return records


def main() -> None:
    server = _serve_in_thread()
    try:
        records = asyncio.run(_run_capture())
    finally:
        server.should_exit = True
        time.sleep(0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rpc_url": RPC_URL,
        "n_tasks": N_TASKS,
        "update_delay_s": UPDATE_DELAY,
        "records": records,
    }, indent=2))
    by_kind: dict[str, int] = {}
    for r in records:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    print(f"captured {len(records)} records {by_kind} -> {OUT}")


if __name__ == "__main__":
    main()
