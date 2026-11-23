"""Local web GUI for openrouter-model-picker. Run: python3 ormp.py --gui

Serves gui.html on 127.0.0.1 and a small JSON API backed by ormp's ranking code,
so the scores in the GUI and in the terminal are always computed the same way.
"""
from __future__ import annotations

import dataclasses
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import ormp

HTML = Path(__file__).with_name("gui.html")


class State:
    def __init__(self, key: str | None):
        self.lock = threading.Lock()
        self.key = key
        self.models: list[ormp.Model] = []
        self.citations: list[str] = []

    def reload(self, key: str | None = None, refresh: bool = False) -> None:
        with self.lock:
            if key is not None:
                self.key = key or None
            self.models, self.citations = ormp.load(self.key, refresh)

    def summary(self) -> dict:
        return {
            "has_key": bool(self.key),
            "n_models": len(self.models),
            "n_bench": sum(1 for m in self.models if m.metrics),
            "citations": self.citations,
            "metrics": ormp.METRICS,
            "tasks": {k: {"name": t.name, "weights": t.weights, "tokens_in": t.tokens_in,
                          "tokens_out": t.tokens_out, "min_context": t.min_context, "note": t.note,
                          "unit": t.unit, "units": t.units, "cache_hit": t.cache_hit}
                      for k, t in ormp.TASKS.items()},
        }


def model_json(m: ormp.Model) -> dict:
    return {"id": m.id, "author": m.id.split("/")[0], "name": m.name, "price_in": m.price_in * 1e6, "price_out": m.price_out * 1e6,
            "context": m.context, "free": m.free, "metrics": m.metrics,
            "price_cache": None if m.price_cache is None else m.price_cache * 1e6, "variable_price": m.variable_price,
            "cost": {k: m.cost(t) for k, t in ormp.TASKS.items()}}


def task_from(body: dict) -> ormp.Task:
    base = ormp.TASKS[body.get("task", "coding")]
    # Per-request copy: GUI tweaks never change the defaults shared with other requests.
    return dataclasses.replace(base, tokens_in=int(body.get("tokens_in", base.tokens_in)),
                               tokens_out=int(body.get("tokens_out", base.tokens_out)),
                               min_context=int(body.get("min_context", base.min_context)),
                               cache_hit=min(1.0, max(0.0, float(body.get("cache_hit", base.cache_hit)))))


def providers_json(body: dict, fetch=ormp.fetch_endpoints) -> dict:
    t = task_from(body)
    eps = ormp.parse_endpoints(fetch(body["id"]))
    eps.sort(key=lambda e: e.model.cost(t))
    return {"id": body["id"], "providers": [{
        "provider": e.provider, "tag": e.tag, "tier": e.tier, "quantization": e.quantization,
        "price_in": e.model.price_in * 1e6, "price_out": e.model.price_out * 1e6,
        "price_cache": None if e.model.price_cache is None else e.model.price_cache * 1e6,
        "variable_price": e.model.variable_price, "context": e.model.context, "cost": e.model.cost(t),
        "uptime_1d": e.uptime_1d, "latency_ms": e.latency_ms, "throughput": e.throughput,
        "tools": e.tools, "discount": e.discount} for e in eps]}


def rank_json(state: State, body: dict) -> dict:
    t = task_from(body)
    def opt(k):
        v = body.get(k)
        return float(v) if v not in (None, "") else None

    caps = {k: opt(k) for k in ("max_cost", "max_in", "max_out")}
    b = ormp.baseline_row(state.models, t, body.get("baseline") or None)
    # Units per month: calibrated from the real monthly bill of the current model when
    # given (spend / that model's cost per unit), else what the user typed.
    per_month, spend = opt("per_month"), opt("spend")
    calibrated = bool(b and spend and b.cost > 0)
    if calibrated:
        per_month = spend / b.cost
    budget = opt("budget")
    if budget is not None and per_month:
        per_unit = budget / per_month
        caps["max_cost"] = per_unit if caps["max_cost"] is None else min(caps["max_cost"], per_unit)
    min_cov = float(body.get("min_coverage", 0.5))
    rows = ormp.rank(state.models, t, min_cov, bool(body.get("include_free")), **caps)
    front = {r.model.id for r in ormp.pareto(rows)}
    # Models hidden only by the coverage threshold: shown on request, never recommended.
    ranked = {r.model.id for r in rows}
    low = [r for r in ormp.rank(state.models, t, 0.0, bool(body.get("include_free")), **caps)
           if r.model.id not in ranked]

    def row_json(r):
        return {**model_json(r.model), "quality": r.quality, "coverage": r.coverage, "task_cost": r.cost}

    return {
        "per_month": per_month, "calibrated": calibrated, "max_cost_effective": caps["max_cost"],
        "baseline": b and row_json(b),
        "vs_baseline": [{"label": label, "id": r.model.id, "quality": r.quality, "cost": r.cost}
                        for label, r in (ormp.vs_baseline(rows, b) if b else [])],
        "rows": [{**row_json(r), "pareto": r.model.id in front} for r in rows],
        "low_coverage": [{**row_json(r), "pareto": False, "low": True} for r in low],
        "recs": [{"label": label, "id": r.model.id, "quality": r.quality, "cost": r.cost}
                 for label, r in ormp.recommendations(rows, any(v is not None for v in caps.values()))],
    }


def make_handler(state: State, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _guard(self) -> bool:
            # Blocks DNS-rebinding and cross-site requests from other pages in the browser.
            if self.headers.get("Host") not in allowed_hosts:
                self._send(403, b"forbidden", "text/plain")
                return False
            origin = self.headers.get("Origin")
            if self.command == "POST" and origin and origin.split("//", 1)[-1] not in allowed_hosts:
                self._send(403, b"forbidden", "text/plain")
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            u = urlparse(self.path)
            if u.path == "/":
                self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/ids":
                self._json(sorted(m.id for m in state.models))
            elif u.path == "/api/state":
                self._json(state.summary())
            elif u.path == "/api/models":
                q = [s.strip().lower() for s in parse_qs(u.query).get("q", [""])[0].split(",") if s.strip()]
                ms = [m for m in state.models if q and any(s in m.id.lower() or s in m.name.lower() for s in q)]
                self._json([model_json(m) for m in sorted(ms, key=lambda m: m.id)[:40]])
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._guard():
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/rank":
                    self._json(rank_json(state, body))
                elif path == "/api/providers":
                    self._json(providers_json(body))
                elif path == "/api/key":
                    # No "key" in the body (e.g. "refresh") keeps the current key.
                    key = body["key"].strip() if isinstance(body.get("key"), str) else None
                    state.reload(key, refresh=bool(body.get("refresh")))
                    self._json(state.summary())
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:  # report to the page instead of dropping the connection
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def run(key: str | None, port: int = 8765, open_browser: bool = True) -> None:
    state = State(key)
    state.reload()
    srv = ThreadingHTTPServer(("127.0.0.1", port), None)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(state, port)
    url = f"http://127.0.0.1:{port}/"
    print(f"GUI en {url}  (Ctrl+C para salir)")
    if open_browser and not os.environ.get("ORMP_NO_BROWSER"):
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        srv.server_close()
