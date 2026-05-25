"""Server-side transport interface and concrete implementations.

A `ServerTransport` accepts incoming requests over some wire (ROS service,
HTTP, gRPC, ...) and dispatches them into a transport-agnostic handler. The
`Server` composes a transport; the handler itself never sees the wire format.

Only `RosServerTransport` imports rclpy, and it does so lazily so that
`messages.py`, `server.py`, and the `ServerTransport` ABC stay pure-python
and unit-testable without a ROS environment.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from .messages import Request, Response

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rclpy.node import Node


# ---- Error hierarchy ---------------------------------------------------------
class TransportError(Exception):
    """Base class for all transport-layer failures."""


# ---- Interface ---------------------------------------------------------------
ReqT = TypeVar("ReqT", bound=Request)
RespT = TypeVar("RespT", bound=Response)

Handler = Callable[[ReqT], RespT]


class ServerTransport(abc.ABC, Generic[ReqT, RespT]):
    """Receive `Request`s, dispatch into a handler, send back `Response`s."""

    @abc.abstractmethod
    def serve(self, handler: Handler[ReqT, RespT]) -> None:
        """Register the handler and begin servicing requests.

        Implementations may be non-blocking (ROS — work happens on the
        executor thread) or blocking (HTTP — runs an event loop). Callers
        should treat this as fire-and-forget; lifecycle is managed via
        `shutdown()` and the surrounding `Server` facade.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Stop servicing and release any underlying resources."""


# ---- ROS implementation ------------------------------------------------------
class RosServerTransport(ServerTransport[ReqT, RespT]):
    """Backs `serve()` with a ROS 2 service server.

    The owning node should spin on a `MultiThreadedExecutor` if the handler
    may take longer than the inter-request interval; otherwise concurrent
    callers will queue behind the single executor thread.

    `decode_request` / `encode_response` keep this class agnostic to the
    specific srv type — the caller is responsible for mapping between the
    generated ROS classes and our pure-python `Request`/`Response`.
    """

    def __init__(
        self,
        node: "Node",
        srv_type: type,
        srv_name: str,
        decode_request: Callable[[object], ReqT],
        encode_response: Callable[[RespT], object],
    ) -> None:
        self._node = node
        self._srv_type = srv_type
        self._srv_name = srv_name
        self._decode = decode_request
        self._encode = encode_response
        self._service = None  # populated in serve()
        self._handler: Handler[ReqT, RespT] | None = None

    def serve(self, handler: Handler[ReqT, RespT]) -> None:
        if self._service is not None:
            raise RuntimeError(f"already serving on '{self._srv_name}'")
        self._handler = handler
        self._service = self._node.create_service(
            self._srv_type, self._srv_name, self._on_request
        )

    def _on_request(self, srv_req, _srv_resp):  # type: ignore[no-untyped-def]
        # ROS pre-allocates the response object and asks us to return one; we
        # ignore the pre-allocation and return a freshly-encoded message.
        assert self._handler is not None
        req = self._decode(srv_req)
        resp = self._handler(req)
        return self._encode(resp)

    def shutdown(self) -> None:
        if self._service is not None:
            self._node.destroy_service(self._service)
            self._service = None
            self._handler = None


# ---- HTTP implementation -----------------------------------------------------
class HttpServerTransport(ServerTransport[ReqT, RespT]):
    """Backs `serve()` with a uvicorn-hosted FastAPI app.

    The single POST /predict route msgpack-decodes the body into a dict, hands
    it to `decode_request`, runs the user-provided handler, then returns the
    msgpack-encoded response. `serve()` runs uvicorn on a daemon thread so the
    caller can keep using the `Server` facade as a non-blocking start/stop.
    """

    def __init__(
        self,
        host: str,
        port: int,
        decode_request: Callable[[dict], ReqT],
        encode_response: Callable[[RespT], dict],
    ) -> None:
        import fastapi  # noqa: F401  (validated at construction time)
        import msgpack  # noqa: F401
        import uvicorn  # noqa: F401

        self._host = host
        self._port = port
        self._decode = decode_request
        self._encode = encode_response
        self._handler: Handler[ReqT, RespT] | None = None
        self._server = None  # uvicorn.Server
        self._thread = None

    def serve(self, handler: Handler[ReqT, RespT]) -> None:
        if self._server is not None:
            raise RuntimeError("HttpServerTransport is already serving")
        self._handler = handler

        import fastapi
        import msgpack
        import uvicorn
        from fastapi import Request, Response

        app = fastapi.FastAPI()

        @app.post("/predict")
        async def predict(request: Request) -> Response:  # noqa: ARG001
            raw = await request.body()
            try:
                payload = msgpack.unpackb(raw, raw=False)
            except Exception as exc:  # noqa: BLE001
                raise fastapi.HTTPException(status_code=400, detail=f"bad msgpack: {exc!r}")
            req_obj = self._decode(payload)
            resp_obj = self._handler(req_obj)
            return Response(
                content=msgpack.packb(self._encode(resp_obj), use_bin_type=True),
                media_type="application/msgpack",
            )

        config = uvicorn.Config(
            app, host=self._host, port=self._port,
            log_level="info", access_log=False, loop="asyncio",
        )
        self._server = uvicorn.Server(config)

        import threading
        self._thread = threading.Thread(
            target=self._server.run, name="omnivla-http", daemon=True
        )
        self._thread.start()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self._server = None
            self._thread = None
            self._handler = None
