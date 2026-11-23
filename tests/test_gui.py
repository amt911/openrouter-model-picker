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
