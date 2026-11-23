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


class MediaBenchIgnoredForText(unittest.TestCase):
    def test_design_arena_image_scores_do_not_become_coding_elo(self):
        models, index = ormp.parse_catalog(CATALOG)
        bench = {**BENCH, "da_image": {"meta": {}, "data": [
            {"source": "design-arena", "model_permaslug": "b/mid", "elo": 9999, "category": "image"}]}}
        ormp.attach_benchmarks(models, index, bench)
        self.assertNotIn("da_coding_elo", {m.id: m for m in models}["b/mid"].metrics)


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


class CacheAndTimeOfDay(unittest.TestCase):
    def test_cache_read_price_applies_to_cached_share(self):
        m = ormp.Model("x", "x", 1e-6, 2e-6, 0, price_cache=1e-7)
        t = ormp.Task("k", "k", {}, tokens_in=1000, tokens_out=100, min_context=0, note="", cache_hit=0.9)
        self.assertAlmostEqual(m.cost(t), 100 * 1e-6 + 900 * 1e-7 + 100 * 2e-6)

    def test_no_cache_price_pays_full_input(self):
        m = ormp.Model("x", "x", 1e-6, 0, 0)
        t = ormp.Task("k", "k", {}, tokens_in=1000, tokens_out=0, min_context=0, note="", cache_hit=0.9)
        self.assertAlmostEqual(m.cost(t), 1000 * 1e-6)

    def test_weekly_average_with_overrides(self):
        # weekdays 01:00-04:00 at double price: 5 days * 3 h out of 168 h
        p = {"prompt": "1", "completion": "2", "overrides": [
            {"utc_days": ["monday", "tuesday", "wednesday", "thursday", "friday"], "utc_start": 100, "utc_end": 400,
             "prompt": "2", "completion": "4"}]}
        avg = ormp.weekly_avg_prices(p)
        self.assertAlmostEqual(avg["prompt"], 1 + 15 / 168)
        self.assertAlmostEqual(avg["completion"], 2 * (1 + 15 / 168))

    def test_override_end_zero_means_midnight(self):
        p = {"prompt": "1", "completion": "1", "overrides": [{"utc_start": 2200, "utc_end": 0, "prompt": "3", "completion": "1"}]}
        self.assertAlmostEqual(ormp.weekly_avg_prices(p)["prompt"], 1 + 2 * 2 / 24)

    def test_catalog_marks_variable_price(self):
        raw = {"data": [{"id": "v/x", "context_length": 1, "pricing": {"prompt": "0.000001", "completion": "0.000001",
                         "input_cache_read": "0.0000001", "overrides": [{"utc_start": 0, "utc_end": 1200, "prompt": "0.000003"}]}}]}
        m = ormp.parse_catalog(raw)[0][0]
        self.assertTrue(m.variable_price)
        self.assertAlmostEqual(m.price_in, 2e-6)
        self.assertAlmostEqual(m.price_cache, 1e-7)


class Baseline(unittest.TestCase):
    def setUp(self):
        self.m, _ = build()
        self.models = list(self.m.values())
        self.t = ormp.TASKS["rewrite"]

    def test_baseline_scored_even_if_filtered(self):
        b = ormp.baseline_row(self.models, self.t, "d/tiny-ctx")  # below min context, no benchmarks
        self.assertIsNone(b)
        b = ormp.baseline_row(self.models, self.t, "b/mid")
        self.assertEqual(b.model.id, "b/mid")

    def test_better_and_cheaper(self):
        rows = ormp.rank(self.models, self.t)
        # a/big is better but pricier than b/mid; c/cheap cheaper but much worse
        self.assertEqual(ormp.vs_baseline(rows, ormp.baseline_row(self.models, self.t, "b/mid")), [])
        found = dict(ormp.vs_baseline(rows, ormp.baseline_row(self.models, self.t, "c/cheap")))
        self.assertEqual(found, {})
        # make a/big cheaper than b/mid
        self.m["a/big"].price_out = self.m["a/big"].price_in = 1e-9
        rows = ormp.rank(self.models, self.t)
        label, r = ormp.vs_baseline(rows, ormp.baseline_row(self.models, self.t, "b/mid"))[0]
        self.assertEqual(r.model.id, "a/big")
        self.assertIn("Mejor y más barato", label)


def ep(provider, pin, pout, quant="fp8", uptime=100, tools=True, cache=None):
    pricing = {"prompt": str(pin), "completion": str(pout)}
    if cache is not None:
        pricing["input_cache_read"] = str(cache)
    return {"provider_name": provider, "tag": provider.lower(), "quantization": quant, "uptime_last_1d": uptime,
            "context_length": 10**6, "pricing": pricing, "supported_parameters": ["tools"] if tools else []}


class Watermark(unittest.TestCase):
    def test_exclude_authors_in_rank_and_advise(self):
        m, _ = build()
        models = list(m.values())
        t = ormp.TASKS["rewrite"]
        self.assertNotIn("a/big", [r.model.id for r in ormp.rank(models, t, exclude_authors=("a",))])
        adv = ormp.advise(models, t, lambda mid: (_ for _ in ()).throw(OSError()), None, per_month=10,
                          exclude_authors=("a",))
        self.assertNotIn("a/big", [o.row.model.id for _, o, _ in adv.picks])
        self.assertEqual(ormp.WATERMARK_AUTHORS, ("google", "openai", "anthropic"))


class Advisor(unittest.TestCase):
    def setUp(self):
        self.m, _ = build()
        self.models = list(self.m.values())
        self.eps = {
            # b/mid: catalog 1e-6/4e-6; a cheap reliable provider and cheaper unreliable ones
            "b/mid": [ep("Good", 5e-7, 2e-6), ep("Flaky", 1e-8, 1e-8, uptime=90),
                      ep("Tiny", 1e-8, 1e-8, quant="fp4"), ep("NoTools", 1e-8, 1e-8, tools=False)],
            "a/big": [ep("Big", 3e-6, 15e-6)],
            "c/cheap": [ep("C", 1e-7, 4e-7)],
        }

    def endpoints_for(self, mid):
        return {"data": {"endpoints": self.eps.get(mid, [])}}

    def test_reliable_filter(self):
        t = ormp.TASKS["coding"]
        got = [e.provider for e in ormp.parse_endpoints(self.endpoints_for("b/mid")) if ormp.reliable(e, t)]
        self.assertEqual(got, ["Good"])
        t = ormp.TASKS["rewrite"]  # tools only matter for coding
        got = {e.provider for e in ormp.parse_endpoints(self.endpoints_for("b/mid")) if ormp.reliable(e, t)}
        self.assertEqual(got, {"Good", "NoTools"})

    def test_saving_on_own_model_wins_when_quality_is_expensive(self):
        t = ormp.TASKS["rewrite"]
        adv = ormp.advise(self.models, t, self.endpoints_for, "b/mid", per_month=10**6)
        roles = {r: o for r, o, _ in adv.picks}
        self.assertEqual(roles["ahorro"].row.model.id, "b/mid")
        self.assertEqual(roles["ahorro"].endpoint.provider, "NoTools")  # cheapest reliable for a non-coding task
        self.assertEqual(roles["calidad"].row.model.id, "a/big")
        self.assertEqual(adv.main, "ahorro")

    def test_quality_wins_when_cheap_in_absolute_terms(self):
        adv = ormp.advise(self.models, ormp.TASKS["rewrite"], self.endpoints_for, "b/mid", per_month=10)
        self.assertEqual(adv.main, "calidad")

    def test_upgrade_must_beat_quality_and_current_price(self):
        self.eps["a/big"] = [ep("BigCheap", 1e-7, 1e-7)]
        adv = ormp.advise(self.models, ormp.TASKS["rewrite"], self.endpoints_for, "b/mid", per_month=10**6)
        roles = {r: o for r, o, _ in adv.picks}
        self.assertEqual(roles["mejora"].row.model.id, "a/big")
        self.assertNotIn("calidad", roles)  # same model and provider as "mejora": deduplicated

    def test_unreliable_only_model_is_not_recommended(self):
        self.eps["a/big"] = [ep("Down", 1e-9, 1e-9, uptime=50)]
        adv = ormp.advise(self.models, ormp.TASKS["rewrite"], self.endpoints_for, None, per_month=10)
        self.assertNotIn("a/big", [o.row.model.id for _, o, _ in adv.picks])

    def test_without_baseline_offers_balance(self):
        adv = ormp.advise(self.models, ormp.TASKS["rewrite"], self.endpoints_for, None, per_month=10**6)
        self.assertIn("equilibrio", [r for r, _, _ in adv.picks])
        self.assertIsNone(adv.baseline)

    def test_price_cap_skips_dearer_providers(self):
        self.eps["b/mid"] = [ep("Cheap", 5e-7, 2e-6), ep("Dear", 9e-7, 3e-6)]
        rows = ormp.rank(self.models, ormp.TASKS["rewrite"])
        b = next(r for r in rows if r.model.id == "b/mid")
        o = ormp.best_offer(b, ormp.TASKS["rewrite"], self.endpoints_for, max_in=0.6, max_out=2.5)
        self.assertEqual(o.endpoint.provider, "Cheap")
        self.assertEqual(o.fallbacks, [])
        self.assertIsNone(ormp.best_offer(b, ormp.TASKS["rewrite"], self.endpoints_for, max_in=0.1))

    def test_routing_snippet(self):
        self.eps["b/mid"] = [ep("A", 5e-7, 2e-6), ep("B", 6e-7, 1e-6), ep("C", 7e-7, 3e-6), ep("D", 8e-7, 4e-6)]
        b = ormp.baseline_row(self.models, ormp.TASKS["rewrite"], "b/mid")
        o = ormp.best_offer(b, ormp.TASKS["rewrite"], self.endpoints_for)
        r = o.routing()
        self.assertEqual(len(r["order"]), 3)
        self.assertTrue(r["allow_fallbacks"])
        # default cap = the dearest of the three proposed providers, per million tokens
        listed = [e for e in ormp.parse_endpoints(self.endpoints_for("b/mid")) if e.tag in r["order"]]
        self.assertAlmostEqual(r["max_price"]["prompt"], max(e.model.price_in for e in listed) * 1e6)
        self.assertAlmostEqual(r["max_price"]["completion"], max(e.model.price_out for e in listed) * 1e6)
        self.assertEqual(o.routing(0.55, 9)["max_price"], {"prompt": 0.55, "completion": 9.0})

    def test_caps_and_budget_in_advise(self):
        adv = ormp.advise(self.models, ormp.TASKS["rewrite"], self.endpoints_for, "b/mid", per_month=10, max_out=1.0)
        self.assertTrue(all(o.endpoint.model.price_out * 1e6 <= 1.0 for _, o, _ in adv.picks))
        adv = ormp.advise(self.models, ormp.TASKS["rewrite"], self.endpoints_for, "b/mid", per_month=10, budget=1e-9)
        self.assertEqual(adv.picks, [])

    def test_fetch_failure_falls_back_to_catalog_price(self):
        def broken(mid):
            raise OSError("offline")
        adv = ormp.advise(self.models, ormp.TASKS["rewrite"], broken, "b/mid", per_month=10)
        self.assertTrue(adv.picks)
        self.assertTrue(all(o.endpoint is None for _, o, _ in adv.picks))


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
