#!/usr/bin/env python3
"""Local static server with layout save endpoint for the D3 app."""

from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LAYOUT_PATH = ROOT / "data" / "space_of_concerns_layout_saved.json"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 (required override name)
        if self.path.rstrip("/") != "/save-layout":
            self._json_response(404, {"ok": False, "error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            nodes = payload.get("nodes")
            if not isinstance(nodes, list):
                raise ValueError("Missing 'nodes' list in payload.")
            validated_nodes = []
            for node in nodes:
                if not isinstance(node, dict):
                    raise ValueError("Each node must be an object.")
                node_id = str(node.get("id", ""))
                x = float(node["x"])
                y = float(node["y"])
                pinned = bool(node.get("pinned", False))
                validated_nodes.append({"id": node_id, "x": x, "y": y, "pinned": pinned})
            out = {
                "saved_at": payload.get("saved_at"),
                "charge": payload.get("charge"),
                "distance": payload.get("distance"),
                "layout_mode": payload.get("layout_mode"),
                "focus_node": payload.get("focus_node"),
                "fit_to_antarctica": payload.get("fit_to_antarctica"),
                "edge_repel_strength": payload.get("edge_repel_strength"),
                "edge_repel_radius": payload.get("edge_repel_radius"),
                "edge_scale": payload.get("edge_scale"),
                "node_scale": payload.get("node_scale"),
                "uniform_node_size": payload.get("uniform_node_size"),
                "pin_dragged": payload.get("pin_dragged"),
                "box_select": payload.get("box_select"),
                "selected_ids": payload.get("selected_ids"),
                "nodes": validated_nodes,
            }
            LAYOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            LAYOUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
            self._json_response(
                200,
                {
                    "ok": True,
                    "path": str(LAYOUT_PATH.relative_to(ROOT)),
                    "n_nodes": len(validated_nodes),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._json_response(400, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve D3 app with /save-layout endpoint.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving {ROOT} at http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
