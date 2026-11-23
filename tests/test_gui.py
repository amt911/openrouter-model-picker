import http.client
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gui  # noqa: E402
from test_ormp import build  # noqa: E402


class GuiApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m, cites = build()
        cls.state = gui.State("fake-key")
        cls.state.models, cls.state.citations = list(m.values()), cites
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
        cls.port = cls.srv.server_address[1]
        cls.srv.RequestHandlerClass = gui.make_handler(cls.state, cls.port)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def req(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port)
        h = {"Content-Type": "application/json", **(headers or {})}
        c.request(method, path, json.dumps(body) if body is not None else None, h)
        r = c.getresponse()
        return r.status, r.read()

    def test_page_served(self):
        st, body = self.req("GET", "/")
        self.assertEqual(st, 200)
        self.assertIn(b"OpenRouter Model Picker", body)

    def test_rank(self):
        st, body = self.req("POST", "/api/rank", {"task": "rewrite"})
        d = json.loads(body)
        self.assertEqual(st, 200)
        self.assertEqual([r["id"] for r in d["rows"]], ["a/big", "b/mid", "c/cheap"])
        self.assertEqual(d["recs"][0]["id"], "a/big")
        self.assertTrue(all(r["pareto"] for r in d["rows"]))

    def test_rank_overrides_do_not_leak_into_defaults(self):
        import ormp
        before = ormp.TASKS["papers"].tokens_in
        self.req("POST", "/api/rank", {"task": "papers", "tokens_in": 1, "min_context": 0})
        self.assertEqual(ormp.TASKS["papers"].tokens_in, before)

    def test_rank_with_price_caps(self):
        st, body = self.req("POST", "/api/rank", {"task": "rewrite", "max_out": "5", "max_cost": "", "max_in": None})
        d = json.loads(body)
        self.assertEqual([r["id"] for r in d["rows"]], ["b/mid", "c/cheap"])
        self.assertIn("límite", d["recs"][0]["label"])

    def test_calibration_and_budget(self):
        b = self.state.models
        st, body = self.req("POST", "/api/rank", {"task": "rewrite", "baseline": "b/mid", "spend": "100"})
        d = json.loads(body)
        self.assertTrue(d["calibrated"])
        self.assertAlmostEqual(d["per_month"] * d["baseline"]["task_cost"], 100)
        # budget equal to the baseline's monthly spend keeps b/mid and drops the pricier a/big
        st, body = self.req("POST", "/api/rank", {"task": "rewrite", "baseline": "b/mid", "spend": 100, "budget": 100})
        ids = [r["id"] for r in json.loads(body)["rows"]]
        self.assertNotIn("a/big", ids)
        self.assertIn("b/mid", ids)

    def test_per_month_without_baseline(self):
        st, body = self.req("POST", "/api/rank", {"task": "rewrite", "per_month": 10, "budget": 0.0001})
        d = json.loads(body)
        self.assertFalse(d["calibrated"])
        self.assertAlmostEqual(d["max_cost_effective"], 0.00001)

    def test_low_coverage_listed_but_not_ranked(self):
        st, body = self.req("POST", "/api/rank", {"task": "coding", "min_coverage": 0.6})
        d = json.loads(body)
        self.assertNotIn("b/mid", [r["id"] for r in d["rows"]])
        self.assertIn("b/mid", [r["id"] for r in d["low_coverage"]])
        self.assertTrue(all(r["low"] and not r["pareto"] for r in d["low_coverage"]))

    def test_providers_sorted_by_cost_with_cache(self):
        raw = {"data": {"endpoints": [
            {"provider_name": "Dear", "tag": "dear/fp8", "quantization": "fp8", "model_id": "x/y", "context_length": 100,
             "pricing": {"prompt": "0.000001", "completion": "0.000001"}, "supported_parameters": ["tools"],
             "uptime_last_1d": 99.9},
            {"provider_name": "Cheap", "tag": "cheap/fp4", "quantization": "fp4", "model_id": "x/y", "context_length": 100,
             "service_tier": "flex", "pricing": {"prompt": "0.000002", "completion": "0.000001",
                                                 "input_cache_read": "0.0000001", "discount": 0.5},
             "supported_parameters": [], "uptime_last_1d": 90},
            {"provider_name": "Broken", "pricing": {"prompt": "-1", "completion": "-1"}}]}}
        d = gui.providers_json({"task": "coding", "id": "x/y"}, fetch=lambda _id: raw)
        ps = d["providers"]
        self.assertEqual([p["provider"] for p in ps], ["Cheap", "Dear"])  # cache makes the pricier input cheaper
        self.assertEqual(ps[0]["tier"], "flex")
        self.assertFalse(ps[0]["tools"])
        self.assertEqual(ps[0]["discount"], 0.5)
        self.assertAlmostEqual(ps[1]["price_in"], 1.0)

    def test_providers_many_reports_failures_per_model(self):
        ok = {"data": {"endpoints": [{"provider_name": "P", "model_id": "x/y", "context_length": 1,
                                      "pricing": {"prompt": "0.000001", "completion": "0.000001"}}]}}

        def fetch(mid):
            if mid == "bad/model":
                raise OSError("boom")
            return ok
        d = gui.providers_many_json({"task": "coding", "ids": ["x/y", "bad/model", "x/y"]}, fetch=fetch)
        self.assertEqual(len(d["providers"]["x/y"]), 1)
        self.assertEqual(d["providers"]["bad/model"], [])
        self.assertIn("boom", d["errors"]["bad/model"])
        self.assertNotIn("x/y", d["errors"])

    def test_recommend_all_tasks(self):
        fake = {"data": {"endpoints": [{"provider_name": "P", "tag": "p", "quantization": "fp8", "uptime_last_1d": 100,
                                        "context_length": 10**6, "supported_parameters": ["tools"],
                                        "pricing": {"prompt": "0.0000001", "completion": "0.0000001"}}]}}
        d = gui.recommend_json(self.state, {"baseline": "b/mid", "tasks": {
            "rewrite": {"per_month": 10}, "papers": {"spend": 50}}}, fetch=lambda mid: fake)
        self.assertEqual([t["task"] for t in d["tasks"]], ["rewrite", "papers"])
        rw = d["tasks"][0]
        self.assertEqual(rw["per_month"], 10)
        self.assertIn(rw["main"], [p["role"] for p in rw["picks"]])
        self.assertEqual(rw["picks"][0]["provider"], "P")
        pa = d["tasks"][1]
        self.assertAlmostEqual(pa["baseline"]["monthly"], 50)  # calibrated from the real bill

    def test_compare(self):
        st, body = self.req("GET", "/api/models?q=big")
        self.assertEqual({m["id"] for m in json.loads(body)}, {"a/big", "a/big:free", "a/big:batch"})

    def test_foreign_host_and_origin_rejected(self):
        self.assertEqual(self.req("GET", "/api/state", headers={"Host": "evil.example"})[0], 403)
        st, _ = self.req("POST", "/api/rank", {"task": "coding"}, {"Origin": "http://evil.example"})
        self.assertEqual(st, 403)

    def test_bad_request_reports_error(self):
        st, body = self.req("POST", "/api/rank", {"task": "nope"})
        self.assertEqual(st, 500)
        self.assertIn("error", json.loads(body))


if __name__ == "__main__":
    unittest.main()
