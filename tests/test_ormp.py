import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ormp  # noqa: E402


def cat_model(mid, pin, pout, ctx=200_000, slug=None, out=("text",)):
    return {"id": mid, "canonical_slug": slug or mid, "name": mid, "context_length": ctx,
            "architecture": {"output_modalities": list(out)},
            "pricing": {"prompt": str(pin), "completion": str(pout)}}


CATALOG = {"data": [
    cat_model("a/big", 3e-6, 15e-6, slug="a/big-20260101"),
    cat_model("a/big:free", 0, 0),
    cat_model("a/big:batch", 1e-9, 1e-9),
    cat_model("b/mid", 1e-6, 4e-6),
    cat_model("c/cheap", 1e-7, 4e-7),
    cat_model("d/tiny-ctx", 1e-7, 1e-7, ctx=8_000),
    cat_model("openrouter/auto", -1, -1),
    cat_model("e/image", 1e-6, 1e-6, out=("image",)),
]}


def aa(slug, intel, ml=None, legal=None):
    keys = ["intelligence_index", "coding_index", "agentic_index", "finance_and_accounting_index",
            "strategy_and_ops_index", "legal_index", "healthcare_and_medical_index", "engineering_index",
            "economics_index", "cyber_index", "openness_index", "multilingual_index", "decision_index"]
    d = dict.fromkeys(keys)
    d.update(source="artificial-analysis", model_permaslug=slug, display_name=slug,
             intelligence_index=intel, multilingual_index=ml, legal_index=legal, pricing=None)
    return d


BENCH = {
    "aa": {"meta": {"citation": "Source: Artificial Analysis"}, "data": [
        aa("a/big-20260101", 70, 0.9, 60), aa("b/mid", 50, 0.8, 50), aa("c/cheap", 30, 0.5, 20),
        aa("unknown/model", 99, 0.99, 99)]},
    "da_coding": {"meta": {}, "data": [
        {"source": "design-arena", "model_permaslug": "a/big-20260101", "elo": 1400, "category": "x"},
        {"source": "design-arena", "model_permaslug": "a/big-20260101", "elo": 1200, "category": "y"},
        {"source": "design-arena", "model_permaslug": "c/cheap", "elo": 1100, "category": "x"}]},
    "or": {"meta": {}, "data": [
        {"source": "openrouter", "model_permaslug": "b/mid", "benchmark_type": "gpqa_diamond", "accuracy": 0.6},
        {"source": "openrouter", "model_permaslug": "b/mid", "benchmark_type": "tau_bench_verified_airline",
         "accuracy": 0.5}]},
    "search": {"meta": {}, "data": [
        {"source": "openrouter", "model_permaslug": "a/big", "benchmark_type": "search_hle",
         "primary_metric": "accuracy", "primary_score": 0.4},
        {"source": "openrouter", "model_permaslug": "a/big", "benchmark_type": "search_dsqa",
         "primary_metric": "accuracy", "primary_score": 0.6}]},
}


def build():
    models, index = ormp.parse_catalog(CATALOG)
    cites = ormp.attach_benchmarks(models, index, BENCH)
    ormp.percentiles(models)
    return {m.id: m for m in models}, cites


class Catalog(unittest.TestCase):
    def test_filters_routers_and_non_text(self):
        ids = {m.id for m in ormp.parse_catalog(CATALOG)[0]}
        self.assertNotIn("openrouter/auto", ids)
        self.assertNotIn("e/image", ids)
        self.assertIn("a/big:free", ids)


class Benchmarks(unittest.TestCase):
    def setUp(self):
        self.m, self.cites = build()

    def test_canonical_slug_match_and_free_variant_inherits(self):
        self.assertEqual(self.m["a/big"].metrics["aa_intelligence"], 70)
        self.assertEqual(self.m["a/big:free"].metrics["aa_intelligence"], 70)

    def test_multiple_items_are_averaged_not_overwritten(self):
        self.assertEqual(self.m["a/big"].metrics["da_coding_elo"], 1300)
        self.assertAlmostEqual(self.m["a/big"].metrics["or_search"], 50)

    def test_units(self):
        self.assertEqual(self.m["a/big"].metrics["aa_multilingual"], 90)
        self.assertEqual(self.m["b/mid"].metrics["or_gpqa"], 60)
        self.assertEqual(self.m["b/mid"].metrics["or_tau_bench"], 50)

    def test_citation_kept(self):
        self.assertEqual(self.cites, ["Source: Artificial Analysis"])

    def test_percentile_best_is_100(self):
        self.assertEqual(self.m["a/big"].pct["aa_intelligence"], 100)
        self.assertEqual(self.m["c/cheap"].pct["aa_intelligence"], 0)


class Ranking(unittest.TestCase):
    def setUp(self):
        self.m, _ = build()
        self.models = list(self.m.values())

    def test_rewrite_order_and_free_excluded(self):
        rows = ormp.rank(self.models, ormp.TASKS["rewrite"])
        self.assertEqual([r.model.id for r in rows], ["a/big", "b/mid", "c/cheap"])

    def test_free_included_on_request(self):
        rows = ormp.rank(self.models, ormp.TASKS["rewrite"], include_free=True)
        self.assertIn("a/big:free", [r.model.id for r in rows])

    def test_context_filter(self):
        rows = ormp.rank(self.models, ormp.TASKS["rewrite"], min_coverage=0)
        self.assertNotIn("d/tiny-ctx", [r.model.id for r in rows])

    def test_coverage_threshold(self):
        # b/mid has no Design Arena score: coding coverage = 0.35 + 0.20 = 0.55
        rows = {r.model.id: r for r in ormp.rank(self.models, ormp.TASKS["coding"], min_coverage=0.6)}
        self.assertNotIn("b/mid", rows)
        rows = {r.model.id: r for r in ormp.rank(self.models, ormp.TASKS["coding"], min_coverage=0.5)}
        self.assertAlmostEqual(rows["b/mid"].coverage, 0.55)

    def test_cost(self):
        t = ormp.TASKS["rewrite"]
        self.assertAlmostEqual(self.m["b/mid"].cost(t), 1e-6 * t.tokens_in + 4e-6 * t.tokens_out)

    def test_pareto_and_recommendations(self):
        rows = ormp.rank(self.models, ormp.TASKS["rewrite"])
        self.assertEqual([r.model.id for r in ormp.pareto(rows)], ["c/cheap", "b/mid", "a/big"])
        rec = dict(ormp.recommendations(rows))
        self.assertEqual(rec["Mejor calidad"].model.id, "a/big")

    def test_pareto_drops_dominated(self):
        Row, M = ormp.Row, ormp.Model
        rows = [Row(M("x", "", 0, 0, 0), 50, 1, 1.0), Row(M("y", "", 0, 0, 0), 40, 1, 2.0),
                Row(M("z", "", 0, 0, 0), 90, 1, 3.0)]
        self.assertEqual([r.model.id for r in ormp.pareto(rows)], ["x", "z"])


class PriceLimits(unittest.TestCase):
    def setUp(self):
        self.m, _ = build()
        self.models = list(self.m.values())
        self.t = ormp.TASKS["rewrite"]

    def ids(self, **caps):
        return [r.model.id for r in ormp.rank(self.models, self.t, **caps)]

    def test_max_cost_per_task(self):
        cap = self.m["b/mid"].cost(self.t)  # inclusive
        self.assertEqual(self.ids(max_cost=cap), ["b/mid", "c/cheap"])

    def test_max_in_and_out_per_million(self):
        self.assertEqual(self.ids(max_in=1.0), ["b/mid", "c/cheap"])
        self.assertEqual(self.ids(max_out=0.5), ["c/cheap"])
        self.assertEqual(self.ids(max_in=2, max_out=3), ["c/cheap"])

    def test_caps_do_not_change_scores(self):
        full = {r.model.id: r.quality for r in ormp.rank(self.models, self.t)}
        capped = {r.model.id: r.quality for r in ormp.rank(self.models, self.t, max_out=5)}
        self.assertEqual(capped, {k: full[k] for k in capped})

    def test_best_within_budget_label(self):
        rows = ormp.rank(self.models, self.t, max_cost=0.05)
        label, r = ormp.recommendations(rows, limited=True)[0]
        self.assertIn("límite", label)
        self.assertEqual(r.model.id, "b/mid")


class EndToEnd(unittest.TestCase):
    def test_show_task_prints_recommendation(self):
        import io, contextlib, argparse
        m, _ = build()
        opts = argparse.Namespace(min_coverage=0.5, include_free=False, limit=5, max_cost=None, max_in=None, max_out=None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rows = ormp.show_task(list(m.values()), ormp.TASKS["papers"], opts)
        self.assertIn("Mejor calidad: a/big", buf.getvalue())
        self.assertNotIn("a/big:batch", [r.model.id for r in rows])


if __name__ == "__main__":
    unittest.main()
