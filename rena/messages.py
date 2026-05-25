"""Transport-agnostic wire types.

Mirror of `rena_navigation/vla/messages.py` in the rena-control repo. Both
sides must agree byte-for-byte on these dataclasses; the wire format itself
(ROS srv, HTTP JSON, gRPC proto, ...) is the responsibility of each
`Transport` / `ServerTransport` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class Message(Protocol):
    """Every message carries a monotonic ns timestamp for staleness checks."""

    stamp_ns: int


@dataclass
class Request:
    """Base request type. Subclass to add task-specific payload fields."""

    stamp_ns: int = 0


@dataclass
class Response:
    """Base response type. Subclass to add task-specific payload fields.

    `stamp_ns` should echo the originating request's stamp so clients can drop
    responses to observations that have since gone stale.
    """

    stamp_ns: int = 0


# ---- NavVla payload ----------------------------------------------------------
# Concrete subtypes for the OmniVLA edge navigation task. These live alongside
# the generic base types for now; they'll move with `messages.py` when we
# extract the shared bits into a small pip package.


@dataclass(kw_only=True)
class NavVlaRequest(Request):
    """Observation + goal sent from the edge nav client to the VLA server."""

    obs_jpegs: list[bytes]              # 6 egocentric frames, 96x96 JPEG, oldest first.
    cur_large_jpeg: bytes               # Current frame at 224x224 (CLIP branch).
    goal_x: float                       # Goal pose in robot frame, x forward (m).
    goal_y: float                       # Goal pose in robot frame, y left (m).
    goal_yaw: float                     # Goal yaw delta (rad).
    modality_id: int = 4                # 4 = pose-only.
    goal_jpeg: bytes | None = None      # Optional egocentric goal image at 96x96.
    language_prompt: str = ""           # Empty => unused.


@dataclass(kw_only=True)
class NavVlaResponse(Response):
    """Robot command + predicted waypoint chunk returned by the VLA server."""

    linear: float
    angular: float
    waypoint_x: list[float] = field(default_factory=list)
    waypoint_y: list[float] = field(default_factory=list)
    waypoint_cos: list[float] = field(default_factory=list)
    waypoint_sin: list[float] = field(default_factory=list)
    modality_id_used: int = 0
