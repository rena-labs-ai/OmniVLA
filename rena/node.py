"""Entry point that wires Model -> Handler -> ServerTransport -> Server.

Transport is selected at startup from `config.yaml`:
  * `transport.type: ros`  -> RosServerTransport (rclpy required)
  * `transport.type: http` -> HttpServerTransport (FastAPI/uvicorn, no rclpy)

The ROS-side codec (and the `rena_msgs.srv.NavVla` import) is lazy-loaded so
HTTP mode runs on workstations that don't have a ROS environment sourced.
"""

from __future__ import annotations

import dataclasses
import io
import os
import signal
import threading
import time
from typing import Optional

import yaml
from PIL import Image as PILImage

from .handler import HandlerConfig, NavVlaHandler
from .messages import NavVlaRequest, NavVlaResponse
from .model import ModelConfig, OmniVlaModel
from .server import Server
from .transport import HttpServerTransport, RosServerTransport, ServerTransport


_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_OMNIVLA_ROOT = os.path.dirname(_PKG_DIR)
_DEFAULT_CONFIG = os.path.join(_PKG_DIR, "config.yaml")


# ---- Config helpers ---------------------------------------------------------
def _load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "model" not in cfg or "weights_path" not in cfg["model"]:
        raise ValueError(f"config {path!r} missing required 'model.weights_path'")
    return cfg


def _resolve_weights_path(weights_path: str) -> str:
    if os.path.isabs(weights_path):
        return weights_path
    return os.path.join(_OMNIVLA_ROOT, weights_path)


def _build_handler(cfg: dict) -> NavVlaHandler:
    model_cfg = ModelConfig(
        weights_path=_resolve_weights_path(cfg["model"]["weights_path"]),
        device=cfg["model"].get("device", "cuda:0"),
    )
    handler_cfg = HandlerConfig(**cfg.get("handler", {}))
    print(f"Loading OmniVLA-edge weights from {model_cfg.weights_path}")
    model = OmniVlaModel(model_cfg)
    return NavVlaHandler(model, handler_cfg)


def _warmup(handler: NavVlaHandler) -> None:
    """One dummy inference so the first real call doesn't pay JIT cost."""
    buf = io.BytesIO()
    PILImage.new("RGB", (96, 96), color=(0, 0, 0)).save(buf, format="JPEG")
    small = buf.getvalue()
    buf = io.BytesIO()
    PILImage.new("RGB", (224, 224), color=(0, 0, 0)).save(buf, format="JPEG")
    large = buf.getvalue()
    handler(NavVlaRequest(
        stamp_ns=0,
        obs_jpegs=[small] * 6,
        cur_large_jpeg=large,
        goal_x=1.0, goal_y=0.0, goal_yaw=0.0,
        modality_id=4,
    ))


# ---- HTTP codec (dict <-> dataclass) ----------------------------------------
def _decode_request_http(payload: dict) -> NavVlaRequest:
    return NavVlaRequest(**payload)


def _encode_response_http(resp: NavVlaResponse) -> dict:
    return dataclasses.asdict(resp)


# ---- ROS server path --------------------------------------------------------
def _run_ros_server(cfg: dict, handler: NavVlaHandler) -> None:
    """Spin a rclpy node hosting a NavVla service. Lazy imports keep rclpy off
    the HTTP-mode import path."""
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rena_msgs.srv import NavVla

    def decode_request(srv_req):  # NavVla.Request -> NavVlaRequest
        return NavVlaRequest(
            stamp_ns=srv_req.stamp_ns,
            obs_jpegs=[bytes(img.data) for img in srv_req.obs_images],
            cur_large_jpeg=bytes(srv_req.cur_large_image.data),
            goal_x=srv_req.goal_x,
            goal_y=srv_req.goal_y,
            goal_yaw=srv_req.goal_yaw,
            goal_jpeg=bytes(srv_req.goal_image.data) or None,
            language_prompt=srv_req.language_prompt,
            modality_id=srv_req.modality_id,
        )

    def encode_response(resp):  # NavVlaResponse -> NavVla.Response
        out = NavVla.Response()
        out.stamp_ns = resp.stamp_ns
        out.cmd_vel = Twist()
        out.cmd_vel.linear.x = resp.linear
        out.cmd_vel.angular.z = resp.angular
        out.waypoint_x = resp.waypoint_x
        out.waypoint_y = resp.waypoint_y
        out.waypoint_cos = resp.waypoint_cos
        out.waypoint_sin = resp.waypoint_sin
        out.modality_id_used = resp.modality_id_used
        return out

    class OmniVlaServerNode(Node):
        def __init__(self) -> None:
            super().__init__("omnivla_server")
            srv_name = cfg.get("transport", {}).get("ros", {}).get(
                "service_name", "/vla/predict"
            )
            transport = RosServerTransport[NavVlaRequest, NavVlaResponse](
                node=self,
                srv_type=NavVla,
                srv_name=srv_name,
                decode_request=decode_request,
                encode_response=encode_response,
            )
            self._server = Server[NavVlaRequest, NavVlaResponse](transport, handler)
            self._server.start()
            self.get_logger().info(f"OmniVLA ROS service ready on '{srv_name}'")

        def destroy_node(self) -> bool:
            try:
                self._server.stop()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"server shutdown failed: {exc!r}")
            return super().destroy_node()

    rclpy.init()
    node = OmniVlaServerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ---- HTTP server path -------------------------------------------------------
def _run_http_server(cfg: dict, handler: NavVlaHandler) -> None:
    http_cfg = cfg.get("transport", {}).get("http", {})
    host = http_cfg.get("host", "0.0.0.0")
    port = int(http_cfg.get("port", 8777))

    transport = HttpServerTransport[NavVlaRequest, NavVlaResponse](
        host=host,
        port=port,
        decode_request=_decode_request_http,
        encode_response=_encode_response_http,
    )
    server = Server[NavVlaRequest, NavVlaResponse](transport, handler)
    server.start()
    print(f"OmniVLA HTTP server ready on http://{host}:{port}/predict")

    # serve() is non-blocking (uvicorn runs on a thread); block main here on a
    # signal so Ctrl-C cleanly stops the worker.
    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda _s, _f: stop_event.set())
    signal.signal(signal.SIGTERM, lambda _s, _f: stop_event.set())
    try:
        stop_event.wait()
    finally:
        print("Stopping HTTP server...")
        server.stop()


# ---- Main -------------------------------------------------------------------
def main(args: list[str] | None = None) -> None:
    # `args` is honored if running under ros2 run (which passes --ros-args ...);
    # we accept and ignore them so the entry point signature stays compatible.
    config_path = os.environ.get("RENA_CONFIG_PATH", _DEFAULT_CONFIG)
    cfg = _load_config(config_path)
    print(f"Loaded config from {config_path}")

    handler = _build_handler(cfg)
    print("Warming up model (1 dummy inference)...")
    _warmup(handler)
    print("Warmup complete.")

    transport_type = cfg.get("transport", {}).get("type", "ros")
    if transport_type == "ros":
        _run_ros_server(cfg, handler)
    elif transport_type == "http":
        _run_http_server(cfg, handler)
    else:
        raise ValueError(f"unknown transport.type: {transport_type!r}")


if __name__ == "__main__":
    main()
