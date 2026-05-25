"""Transport-generic server facade.

Mirror of `Client` on the rena-control side: a thin composer that wires one
`ServerTransport` to one handler function. Swap the transport without
touching the handler.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from .messages import Request, Response
from .transport import Handler, ServerTransport

ReqT = TypeVar("ReqT", bound=Request)
RespT = TypeVar("RespT", bound=Response)


class Server(Generic[ReqT, RespT]):
    """Composes any `ServerTransport` with a pure handler."""

    def __init__(
        self,
        transport: ServerTransport[ReqT, RespT],
        handler: Handler[ReqT, RespT],
    ) -> None:
        self._transport = transport
        self._handler = handler

    def start(self) -> None:
        self._transport.serve(self._handler)

    def stop(self) -> None:
        self._transport.shutdown()

    def __enter__(self) -> "Server[ReqT, RespT]":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
