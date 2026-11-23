#!/usr/bin/env python3
"""openrouter-model-picker: find the best and cheapest OpenRouter model per task.

Combines the OpenRouter model catalog (/models: prices, context) with the unified
benchmark endpoint (/benchmarks: Artificial Analysis, Design Arena, OpenRouter evals).
Standard library only. Run `python3 ormp.py` for the interactive menu.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

API = "https://openrouter.ai/api/v1"
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "openrouter-model-picker"
CACHE_TTL = 12 * 3600  # benchmarks: 500 requests/day per account, so cache hard

# Benchmark queries. Each is (cache name, query params). 4 requests cover every task.
BENCH_QUERIES = {
    "aa": {"source": "artificial-analysis"},
    "da_coding": {"source": "design-arena", "task_type": "coding"},
    "or": {"source": "openrouter"},
    "search": {"task_type": "search"},
}

# Metric -> human label. Every metric is turned into a percentile (0-100) inside the pool.
METRICS = {
    "aa_intelligence": "AA Intelligence",
    "aa_multilingual": "AA Multilingual",
    "aa_healthcare": "AA Healthcare",
    "aa_engineering": "AA Engineering",
    "aa_legal": "AA Legal",
    "da_coding_elo": "DesignArena coding ELO",
    "or_tau_bench": "tau-bench (agentes/tools)",
    "or_gpqa": "GPQA Diamond (ciencia)",
    "or_search": "Search evals (HLE/DSQA)",
}


@dataclass
class Task:
    key: str
    name: str
    weights: dict[str, float]
    tokens_in: int
    tokens_out: int
    min_context: int
    note: str
    unit: str = "tarea"  # what tokens_in/tokens_out describe
    units: str = "tareas"  # plural, for "N units per month"
    cache_hit: float = 0.0  # share of input tokens served from the prompt cache


TASKS = {
    "coding": Task(
        "coding", "Programar",
        {"da_coding_elo": 0.45, "aa_intelligence": 0.35, "or_tau_bench": 0.20},
        tokens_in=4_000_000, tokens_out=50_000, min_context=64_000,
        unit="hora de sesión", units="horas de sesión", cache_hit=0.9,
        note="Una hora de un agente de código: relee el contexto en cada paso (muchísima entrada, casi "
             "toda cacheada) y escribe poco. Estimación: calíbrala con tu gasto real.",
    ),
    "rewrite": Task(
        "rewrite", "Reescribir documentos",
        {"aa_intelligence": 0.45, "aa_multilingual": 0.40, "aa_legal": 0.15},
        tokens_in=6_000, tokens_out=6_000, min_context=32_000, unit="documento", units="documentos",
        note="No hay benchmark directo de redacción: se usa inteligencia + multilingüe (español) + legal (precisión de texto).",
    ),
    "papers": Task(
        "papers", "Cotejar papers",
        {"aa_intelligence": 0.30, "or_gpqa": 0.25, "or_search": 0.20, "aa_healthcare": 0.15, "aa_engineering": 0.10},
        tokens_in=80_000, tokens_out=2_500, min_context=128_000, unit="consulta", units="consultas",
        note="Varios papers por consulta: contexto mínimo 128k y razonamiento científico (GPQA).",
    ),
}


# ---------------------------------------------------------------- HTTP + cache
def _get(path: str, params: dict | None = None, key: str | None = None) -> dict:
    url = f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "openrouter-model-picker"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def cached(name: str, fetch, refresh: bool = False, ttl: int = CACHE_TTL) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"{name}.json"
    if not refresh and f.exists() and time.time() - f.stat().st_mtime < ttl:
        return json.loads(f.read_text())
    data = fetch()
    f.write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------- model data
@dataclass
class Model:
    id: str
    name: str
    price_in: float  # USD per token
    price_out: float
    context: int
    metrics: dict[str, float] = field(default_factory=dict)  # raw values
    pct: dict[str, float] = field(default_factory=dict)  # percentiles in pool
    price_cache: float | None = None  # USD per cached input token; None = no caching listed
    variable_price: bool = False  # price depends on the time of day (OpenRouter "overrides")

    @property
    def free(self) -> bool:
        return self.price_in == 0 and self.price_out == 0

    def cost(self, t: Task) -> float:
        """USD per task unit. Cached input is billed at the cache-read price when the model
        lists one; models without caching pay full input price for every token."""
        cached = t.tokens_in * t.cache_hit if self.price_cache is not None else 0
        return (self.price_in * (t.tokens_in - cached) + (self.price_cache or 0) * cached
                + self.price_out * t.tokens_out)


def _hhmm(v: int) -> int:
    return (v // 100) * 60 + v % 100


WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def weekly_avg_prices(p: dict) -> dict[str, float]:
    """Average of each price over a week of round-the-clock use, applying the time-of-day
    "overrides" (utc_days + utc_start/utc_end as HHMM, end 0 = midnight; first match wins)."""
    keys = ("prompt", "completion", "input_cache_read")
    base = {k: float(p[k]) for k in keys if k in p}
    ovs = p.get("overrides") or []
    if not ovs:
        return base
    total = {k: 0.0 for k in base}
    for d, day in enumerate(WEEKDAYS):
        spans = []
        for o in ovs:
            if day in (o.get("utc_days") or WEEKDAYS):
                a = _hhmm(o.get("utc_start", 0))
                b = _hhmm(o.get("utc_end", 0)) or 1440
                spans.append((a, b, o))
        for minute in range(1440):
            o = next((o for a, b, o in spans if a <= minute < b), None)
            for k in total:
                total[k] += float(o[k]) if o and k in o else base[k]
    return {k: v / (7 * 1440) for k, v in total.items()}


def parse_catalog(raw: dict) -> tuple[list[Model], dict[str, Model]]:
    """Return text-output models with valid prices, plus a slug -> model index."""
    models, index = [], {}
    for m in raw.get("data", []):
        p = m.get("pricing") or {}
        try:
            avg = weekly_avg_prices(p)
            pin, pout = avg["prompt"], avg["completion"]
        except (KeyError, ValueError, TypeError):
            continue
        if pin < 0 or pout < 0:  # routers such as openrouter/auto report -1
            continue
        outs = (m.get("architecture") or {}).get("output_modalities") or ["text"]
        if "text" not in outs:
            continue
        mod = Model(m["id"], m.get("name", m["id"]), pin, pout, int(m.get("context_length") or 0),
                    price_cache=avg.get("input_cache_read"), variable_price=bool(p.get("overrides")))
        models.append(mod)
        for slug in (m["id"], m.get("canonical_slug")):
            if slug:
                index.setdefault(slug, mod)
                index.setdefault(slug.split(":")[0], mod)
    return models, index


def _lookup(index: dict[str, Model], slug: str) -> list[Model]:
    """All catalog entries a benchmark slug refers to (base model and its :free/:variants)."""
    base = index.get(slug) or index.get(slug.split(":")[0])
    if not base:
        return []
    root = base.id.split(":")[0]
    return list({m.id: m for m in index.values() if m.id.split(":")[0] == root}.values())


def _avg_into(acc: dict, key: tuple, value) -> None:
    if value is None:
        return
    s, n = acc.get(key, (0.0, 0))
    acc[key] = (s + float(value), n + 1)


def attach_benchmarks(models: list[Model], index: dict[str, Model], bench: dict[str, dict]) -> list[str]:
    """Copy benchmark scores onto catalog models. Several items for the same model and
    metric (e.g. Design Arena categories, search benchmarks) are averaged, never
    overwritten. Returns the citations to show."""
    acc: dict[tuple, tuple[float, int]] = {}
    aa_fields = {
        "aa_intelligence": "intelligence_index",
        "aa_healthcare": "healthcare_and_medical_index",
        "aa_engineering": "engineering_index",
        "aa_legal": "legal_index",
    }
    citations = []
    for resp in bench.values():
        meta = resp.get("meta") or {}
        if meta.get("citation") and meta["citation"] not in citations:
            citations.append(meta["citation"])
        for it in resp.get("data", []):
            slug, src = it.get("model_permaslug"), it.get("source")
            if not slug:
                continue
            if src == "artificial-analysis":
                for metric, f in aa_fields.items():
                    _avg_into(acc, (slug, metric), it.get(f))
                ml = it.get("multilingual_index")
                _avg_into(acc, (slug, "aa_multilingual"), None if ml is None else ml * 100)
            elif src == "design-arena":
                _avg_into(acc, (slug, "da_coding_elo"), it.get("elo"))
            elif src == "openrouter" and "primary_score" in it:
                if it.get("benchmark_type") in ("search_hle", "search_dsqa", "search_browsecomp"):
                    _avg_into(acc, (slug, "or_search"), it["primary_score"] * 100)
            elif src == "openrouter":
                bt = it.get("benchmark_type")
                metric = {"gpqa_diamond": "or_gpqa", "tau_bench_verified_airline": "or_tau_bench"}.get(bt)
                if metric:
                    _avg_into(acc, (slug, metric), it.get("accuracy", 0) * 100)
    for (slug, metric), (s, n) in acc.items():
        for m in _lookup(index, slug):
            m.metrics[metric] = s / n
    return citations


def percentiles(models: list[Model]) -> None:
    """Percentile of each metric among the models that have it (100 = best)."""
    # Variants of one model (a/b and a/b:free) count once; ties share a mid-rank.
    for metric in METRICS:
        vals = {m.id.split(":")[0]: m.metrics[metric] for m in models if metric in m.metrics}
        allv = sorted(vals.values())
        n = len(allv)
        for m in models:
            if metric not in m.metrics:
                continue
            v = m.metrics[metric]
            lower = sum(1 for x in allv if x < v)
            equal = sum(1 for x in allv if x == v)
            m.pct[metric] = 100.0 if n == 1 else 100.0 * (lower + (equal - 1) / 2) / (n - 1)


@dataclass
class Row:
    model: Model
    quality: float
    coverage: float
    cost: float


def within_limits(m: Model, t: Task, max_cost: float | None = None, max_in: float | None = None,
                  max_out: float | None = None) -> bool:
    """Price caps: max_cost in USD per task, max_in/max_out in USD per million tokens.
    None means no cap."""
    return ((max_cost is None or m.cost(t) <= max_cost)
            and (max_in is None or m.price_in * 1e6 <= max_in)
            and (max_out is None or m.price_out * 1e6 <= max_out))


def rank(models: list[Model], t: Task, min_coverage: float = 0.5, include_free: bool = False,
         min_context: int | None = None, max_cost: float | None = None, max_in: float | None = None,
         max_out: float | None = None) -> list[Row]:
    """Score each model for the task. quality = weighted mean of the percentiles it has;
    coverage = share of the task's weight it has data for. Percentiles are computed over
    the whole catalog, so price caps remove models without changing anyone's score."""
    min_ctx = t.min_context if min_context is None else min_context
    rows = []
    for m in models:
        if m.context < min_ctx or (m.free and not include_free) or m.id.endswith(":batch"):
            continue  # :batch = asynchronous batch API, not usable interactively
        if not within_limits(m, t, max_cost, max_in, max_out):
            continue
        avail = {k: w for k, w in t.weights.items() if k in m.pct}
        cov = sum(avail.values()) / sum(t.weights.values())
        if not avail or cov < min_coverage:
            continue
        q = sum(m.pct[k] * w for k, w in avail.items()) / sum(avail.values())
        rows.append(Row(m, q, cov, m.cost(t)))
    rows.sort(key=lambda r: (-r.quality, r.cost))
    return rows


def pareto(rows: list[Row]) -> list[Row]:
    """Models no other model beats on both quality and price, cheapest first."""
    front, best = [], -1.0
    for r in sorted(rows, key=lambda r: (r.cost, -r.quality)):
        if r.quality > best:
            front.append(r)
            best = r.quality
    return front


# ---------------------------------------------------------------- providers
@dataclass
class Endpoint:
    provider: str
    tag: str
    tier: str | None
    quantization: str | None
    model: Model  # carries this provider's prices, so Model.cost works unchanged
    uptime_1d: float | None
    latency_ms: float | None
    throughput: float | None
    tools: bool
    discount: float


def fetch_endpoints(model_id: str, refresh: bool = False) -> dict:
    """Per-provider offers for one model (public endpoint, cached 1 h: uptime moves)."""
    safe = model_id.replace("/", "__").replace(":", "_")
    return cached(f"endpoints_{safe}", lambda: _get(f"/models/{model_id}/endpoints"), refresh, ttl=3600)


def parse_endpoints(raw: dict) -> list[Endpoint]:
    out = []
    for e in (raw.get("data") or {}).get("endpoints") or []:
        p = e.get("pricing") or {}
        try:
            avg = weekly_avg_prices(p)
            pin, pout = avg["prompt"], avg["completion"]
        except (KeyError, ValueError, TypeError):
            continue
        if pin < 0 or pout < 0:
            continue
        m = Model(e.get("model_id", ""), e.get("model_name", ""), pin, pout, int(e.get("context_length") or 0),
                  price_cache=avg.get("input_cache_read"), variable_price=bool(p.get("overrides")))
        out.append(Endpoint(e.get("provider_name", "?"), e.get("tag", ""), e.get("service_tier"),
                            e.get("quantization"), m, e.get("uptime_last_1d"), e.get("latency_last_30m"),
                            e.get("throughput_last_30m"), "tools" in (e.get("supported_parameters") or []),
                            float(p.get("discount") or 0)))
    return out


def baseline_row(models: list[Model], t: Task, model_id: str | None) -> Row | None:
    """Score the user's current model on the same scale, ignoring every filter."""
    m = next((m for m in models if m.id == model_id), None) if model_id else None
    rows = rank([m], t, 0.0, True, 0) if m else []
    return rows[0] if rows else None


def vs_baseline(rows: list[Row], base: Row) -> list[tuple[str, Row]]:
    """Answers "is there something better and cheaper than what I use?"."""
    out = []
    better = [r for r in rows if r.quality > base.quality and r.cost < base.cost and r.model.id != base.model.id]
    if better:
        out.append((f"Mejor y más barato que {base.model.id}", max(better, key=lambda r: r.quality)))
    similar = [r for r in rows if r.quality >= base.quality - 5 and r.cost < base.cost and r.model.id != base.model.id]
    if similar:
        out.append((f"Calidad parecida (±5) y lo más barato posible", min(similar, key=lambda r: r.cost)))
    return out


def recommendations(rows: list[Row], limited: bool = False) -> list[tuple[str, Row]]:
    if not rows:
        return []
    top = rows[0]
    out = [("Mejor calidad dentro de tu límite de precio" if limited else "Mejor calidad", top)]
    front = pareto(rows)
    for frac, label in ((0.9, "Más barato con ≥90% de la calidad máxima"),
                        (0.75, "Más barato con ≥75% de la calidad máxima")):
        cand = next((r for r in front if r.quality >= frac * top.quality), None)
        if cand:
            out.append((label, cand))
    return out


# ---------------------------------------------------------------- advisor
NEGLIGIBLE_USD = 20.0  # a monthly cost under this is not worth trading quality for
DEFAULT_PER_MONTH = {"coding": 176, "rewrite": 100, "papers": 100}  # 1 session x 8 h x 22 days


def reliable(e: Endpoint, t: Task) -> bool:
    """Provider good enough to recommend: uptime >= 99% over 24 h, not 4-bit, and with tool
    calling when the task is coding (agents need it)."""
    return ((e.uptime_1d or 0) >= 99 and not re.search(r"(fp|int)4", e.quantization or "")
            and (t.key != "coding" or e.tools))


@dataclass
class Offer:
    row: Row
    endpoint: Endpoint | None  # cheapest reliable provider; None = catalog price (no data)
    cost: float  # USD per unit with that provider


def best_offer(row: Row, t: Task, endpoints_for) -> Offer | None:
    try:
        eps = [e for e in parse_endpoints(endpoints_for(row.model.id)) if reliable(e, t)]
    except Exception:  # noqa: BLE001 - no provider data: fall back to the catalog price
        return Offer(row, None, row.cost)
    if not eps:
        return None  # nobody serves it reliably: do not recommend it
    e = min(eps, key=lambda e: e.model.cost(t))
    return Offer(row, e, e.model.cost(t))


@dataclass
class Advice:
    task: Task
    per_month: float
    baseline: Row | None
    picks: list[tuple[str, Offer, str]]  # (role, offer, why)
    main: str | None  # role of the recommended pick

    def monthly(self, cost: float) -> float:
        return cost * self.per_month


def advise(models: list[Model], t: Task, endpoints_for, baseline_id: str | None = None,
           per_month: float | None = None, negligible: float = NEGLIGIBLE_USD,
           min_coverage: float = 0.5) -> Advice:
    """The recommendation algorithm, per task:
    1. Rank models (benchmarks, coverage >= min_coverage, no :free) and price each one with
       its cheapest reliable provider (see `reliable`).
    2. Candidates: "ahorro" = your own model with that provider; "mejora" = the best model
       that beats yours and costs less than you pay today; "calidad" = the cheapest model
       reaching 90% of the best quality available; without a current model, "equilibrio" =
       the cheapest one reaching 75%.
    3. Pick: if "calidad" costs under `negligible` USD a month at your volume, quality wins
       (saving cents is not worth worse answers); otherwise the saving on your own model,
       then the upgrade, then the balance."""
    per_month = per_month or DEFAULT_PER_MONTH.get(t.key, 100)
    rows = rank(models, t, min_coverage)
    offers = [o for o in (best_offer(r, t, endpoints_for) for r in rows) if o]
    adv = Advice(t, per_month, baseline_row(models, t, baseline_id), [], None)
    if not offers:
        return adv
    top = max(o.row.quality for o in offers)
    front, best = [], -1.0
    for o in sorted(offers, key=lambda o: (o.cost, -o.row.quality)):
        if o.row.quality > best:
            front.append(o)
            best = o.row.quality

    def cheapest_reaching(frac):
        return next((o for o in front if o.row.quality >= frac * top), None)

    b = adv.baseline
    if b:
        mine = best_offer(b, t, endpoints_for)
        if mine and mine.endpoint and mine.cost < b.cost * 0.95:
            adv.picks.append(("ahorro", mine, "Tu mismo modelo con el proveedor fiable más barato: mismos "
                              f"resultados, {(1 - mine.cost / b.cost) * 100:.0f}% menos que el precio de catálogo."))
        better = [o for o in offers if o.row.quality > b.quality and o.cost < b.cost and o.row.model.id != b.model.id]
        if better:
            up = max(better, key=lambda o: (o.row.quality, -o.cost))
            adv.picks.append(("mejora", up, f"Mejor que tu modelo (+{up.row.quality - b.quality:.1f} de calidad) y "
                              f"más barato que el precio de catálogo del tuyo ({(up.cost / b.cost - 1) * 100:+.0f}%)."))
    else:
        bal = cheapest_reaching(0.75)
        if bal:
            adv.picks.append(("equilibrio", bal, "El más barato con al menos el 75% de la mejor calidad disponible."))
    q = cheapest_reaching(0.9)
    if q:
        adv.picks.append(("calidad", q, "El más barato con al menos el 90% de la mejor calidad disponible."))
    # the same model and provider twice is one pick: keep the first role
    seen, uniq = set(), []
    for role, o, why in adv.picks:
        k = (o.row.model.id, o.endpoint and o.endpoint.tag)
        if k not in seen:
            seen.add(k)
            uniq.append((role, o, why))
    adv.picks = uniq
    roles = {r: o for r, o, _ in adv.picks}
    if "calidad" in roles and adv.monthly(roles["calidad"].cost) <= negligible:
        adv.main = "calidad"
    else:
        adv.main = next((r for r in ("ahorro", "mejora", "equilibrio", "calidad") if r in roles), None)
    return adv


# ---------------------------------------------------------------- output
BOLD, DIM, GREEN, CYAN, YELLOW, RESET = ("\033[1m", "\033[2m", "\033[32m", "\033[36m", "\033[33m", "\033[0m") \
    if sys.stdout.isatty() and not os.environ.get("NO_COLOR") else ("",) * 6


def usd(x: float) -> str:
    if x == 0:
        return "gratis"
    return f"${x:.4f}" if x < 0.1 else f"${x:.3f}"


def per_m(x: float) -> str:
    return f"{x * 1e6:.2f}"


def print_table(rows: list[Row], t: Task, limit: int, mark: set[str] = frozenset()) -> None:
    hdr = f"{'#':>3}  {'modelo':<44} {'calidad':>7} {'cobert':>6} {'$/tarea':>9} {'$in/M':>7} {'$out/M':>7} {'ctx':>6}"
    print(BOLD + hdr + RESET)
    for i, r in enumerate(rows[:limit], 1):
        m = r.model
        star = GREEN + "★" + RESET if m.id in mark else " "
        print(f"{i:>3}{star} {m.id[:44]:<44} {r.quality:>7.1f} {r.coverage * 100:>5.0f}% {usd(r.cost):>9} "
              f"{per_m(m.price_in):>7} {per_m(m.price_out):>7} {m.context // 1000:>5}k")
    if not rows:
        print(YELLOW + "  (ningún modelo cumple los filtros)" + RESET)


def show_task(models, t: Task, opts) -> list[Row]:
    rows = rank(models, t, opts.min_coverage, opts.include_free,
                max_cost=opts.max_cost, max_in=opts.max_in, max_out=opts.max_out)
    limited = any(x is not None for x in (opts.max_cost, opts.max_in, opts.max_out))
    front = pareto(rows)
    print(f"\n{BOLD}{CYAN}== {t.name} =={RESET}  perfil por {t.unit}: {t.tokens_in:,} tokens entrada "
          f"({t.cache_hit:.0%} cacheada) + {t.tokens_out:,} salida, contexto ≥ {t.min_context // 1000}k")
    print(DIM + t.note + RESET)
    print(DIM + "Pesos: " + ", ".join(f"{METRICS[k]} {w:.0%}" for k, w in t.weights.items()) + RESET)
    print(f"\n{BOLD}Recomendación{RESET}")
    if limited:
        print(YELLOW + "Límites de precio: " + ", ".join(
            f"{n} ≤ {v}" for n, v in (("$/tarea", opts.max_cost), ("$in/M", opts.max_in), ("$out/M", opts.max_out))
            if v is not None) + RESET)
    per_month = getattr(opts, "per_month", None)
    def money(r):
        return f"{usd(r.cost)}/{t.unit}" + (f", ≈ ${r.cost * per_month:,.0f}/mes" if per_month else "")
    for label, r in recommendations(rows, limited):
        print(f"  {GREEN}{label}:{RESET} {r.model.id}  (calidad {r.quality:.1f}, {money(r)})")
    base = baseline_row(models, t, getattr(opts, "baseline", None))
    if getattr(opts, "baseline", None) and not base:
        print(YELLOW + f"  No encuentro {opts.baseline} con benchmarks para esta tarea." + RESET)
    if base:
        print(f"\n{BOLD}Frente a tu modelo actual{RESET}: {base.model.id} (calidad {base.quality:.1f}, {money(base)})")
        found = vs_baseline(rows, base)
        for label, r in found:
            delta = f", {(r.cost / base.cost - 1) * 100:+.0f}% coste" if base.cost else ""
            print(f"  {GREEN}{label}:{RESET} {r.model.id}  (calidad {r.quality:.1f}, {money(r)}{delta})")
        if not found:
            print(YELLOW + "  Ningún modelo es mejor y más barato a la vez: tu modelo ya está en la frontera." + RESET)
    print(f"\n{BOLD}Top {opts.limit} por calidad{RESET}  (★ = frontera calidad/precio: nadie es mejor y más barato)")
    print_table(rows, t, opts.limit, {r.model.id for r in front})
    print(f"\n{BOLD}Frontera calidad/precio{RESET} (del más barato al mejor)")
    print_table(front, t, 50)
    return rows


ROLE_LABEL = {"ahorro": "Ahorro", "mejora": "Mejora", "equilibrio": "Equilibrio", "calidad": "Calidad"}


def print_advice(models, opts) -> None:
    total_now = total_new = 0.0
    for t in TASKS.values():
        pm = opts.per_month if (opts.per_month and t.key == "coding") else None
        adv = advise(models, t, fetch_endpoints, opts.baseline, pm)
        print(f"\n{BOLD}{CYAN}== {t.name} =={RESET}  ({adv.per_month:,.0f} {t.units}/mes)")
        if adv.baseline:
            now = adv.monthly(adv.baseline.cost)
            total_now += now
            print(DIM + f"Ahora: {adv.baseline.model.id}, calidad {adv.baseline.quality:.1f}, ≈ ${now:,.2f}/mes "
                  "(precio de catálogo)" + RESET)
        if not adv.picks:
            print(YELLOW + "  Sin recomendación: faltan benchmarks o proveedores fiables." + RESET)
            continue
        for role, o, why in adv.picks:
            star = GREEN + "► RECOMENDADO " + RESET if role == adv.main else "  "
            prov = f"{o.endpoint.provider} ({o.endpoint.tag})" if o.endpoint else "precio de catálogo"
            print(f"  {star}{BOLD}{ROLE_LABEL[role]}:{RESET} {o.row.model.id} vía {prov}  "
                  f"calidad {o.row.quality:.1f}, ≈ ${adv.monthly(o.cost):,.2f}/mes")
            print(DIM + "      " + why + RESET)
            if role == adv.main:
                total_new += adv.monthly(o.cost)
    if total_now:
        print(f"\n{BOLD}Total:{RESET} ≈ ${total_now:,.2f}/mes ahora → ≈ ${total_new:,.2f}/mes con lo recomendado")
    print(DIM + "Para fijar un proveedor en OpenRouter: \"provider\": {\"order\": [\"<variante>\"]}" + RESET)


def compare(models: list[Model], query: str) -> None:
    terms = [q.strip().lower() for q in query.split(",") if q.strip()]
    chosen = sorted((m for m in models if any(q in m.id.lower() for q in terms)), key=lambda m: m.id)[:30]
    if not chosen:
        print(YELLOW + "Ningún modelo coincide." + RESET)
        return
    short = {"aa_intelligence": "AAint", "aa_multilingual": "AAml", "aa_healthcare": "AAhc",
             "aa_engineering": "AAeng", "aa_legal": "AAleg", "da_coding_elo": "DAelo",
             "or_tau_bench": "tau", "or_gpqa": "GPQA", "or_search": "search"}
    print(BOLD + f"{'modelo':<40} {'$in/M':>7} {'$out/M':>7} {'ctx':>6}" + "".join(f"{v:>7}" for v in short.values())
          + "".join(f"{'$' + k:>10}" for k in TASKS) + RESET)
    for m in chosen:
        print(f"{m.id[:40]:<40} {per_m(m.price_in):>7} {per_m(m.price_out):>7} {m.context // 1000:>5}k"
              + "".join(f"{m.metrics[k]:>7.0f}" if k in m.metrics else f"{'-':>7}" for k in short)
              + "".join(f"{usd(m.cost(t)):>10}" for t in TASKS.values()))
    print(DIM + "Leyenda: " + ", ".join(f"{v}={METRICS[k]}" for k, v in short.items())
          + ". $tarea = coste con el perfil de tokens de cada tarea." + RESET)


def export_csv(models: list[Model], path: str) -> None:
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "price_in_per_M", "price_out_per_M", "context", *METRICS,
                     *(f"quality_{k}" for k in TASKS), *(f"cost_{k}" for k in TASKS)])
        scores = {k: {r.model.id: r.quality for r in rank(models, t, 0.0, True, 0)} for k, t in TASKS.items()}
        for m in models:
            wr.writerow([m.id, per_m(m.price_in), per_m(m.price_out), m.context,
                         *(m.metrics.get(k, "") for k in METRICS),
                         *(f"{scores[k].get(m.id):.1f}" if m.id in scores[k] else "" for k in TASKS),
                         *(f"{m.cost(t):.6f}" for t in TASKS.values())])
    print(GREEN + f"CSV guardado en {path}" + RESET)


# ---------------------------------------------------------------- loading
def load(key: str | None, refresh: bool) -> tuple[list[Model], list[str]]:
    raw = cached("models", lambda: _get("/models"), refresh)
    models, index = parse_catalog(raw)
    bench = {}
    if key:
        for name, params in BENCH_QUERIES.items():
            try:
                bench[name] = cached(f"bench_{name}", lambda p=params: _get("/benchmarks", p, key), refresh)
            except urllib.error.HTTPError as e:
                print(YELLOW + f"Aviso: /benchmarks {params} -> HTTP {e.code}" + RESET, file=sys.stderr)
                if e.code == 401:
                    break
    citations = attach_benchmarks(models, index, bench)
    percentiles(models)
    n = sum(1 for m in models if m.metrics)
    print(DIM + f"{len(models)} modelos en el catálogo, {n} con benchmarks." + RESET)
    if not bench:
        print(YELLOW + "Sin API key no hay benchmarks: solo se puede comparar precio/contexto (opción 4)." + RESET)
    return models, citations


def ask(prompt: str, default: str = "") -> str:
    try:
        v = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    except EOFError:
        sys.exit(0)
    return v or default


def edit_profile(opts) -> None:
    t = TASKS.get(ask("Tarea (coding/rewrite/papers)", "coding"))
    if not t:
        return
    t.tokens_in = int(ask("Tokens de entrada por tarea", str(t.tokens_in)))
    t.tokens_out = int(ask("Tokens de salida por tarea", str(t.tokens_out)))
    t.min_context = int(ask("Contexto mínimo (tokens)", str(t.min_context)))
    opts.min_coverage = float(ask("Cobertura mínima de benchmarks 0-1", str(opts.min_coverage)))
    opts.include_free = ask("Incluir variantes gratis (con límites de uso)? s/n", "s" if opts.include_free else "n") == "s"
    opts.limit = int(ask("Filas a mostrar", str(opts.limit)))
    def cap(prompt, cur):
        v = ask(prompt + " (vacío = sin límite)", "" if cur is None else str(cur))
        return float(v) if v else None
    opts.max_cost = cap("Precio máximo por tarea en USD", opts.max_cost)
    opts.max_in = cap("Precio máximo de entrada, USD por millón de tokens", opts.max_in)
    opts.max_out = cap("Precio máximo de salida, USD por millón de tokens", opts.max_out)


def menu(models, citations, opts) -> None:
    while True:
        print(f"\n{BOLD}¿Para qué buscas modelo?{RESET}")
        for i, t in enumerate(TASKS.values(), 1):
            print(f"  {i}) {t.name}")
        print("  4) Comparar modelos concretos (ej: claude, gpt-5, deepseek)")
        print("  5) Ajustar perfil / filtros / límites de precio")
        print("  6) Exportar todo a CSV")
        print("  q) Salir")
        c = ask(">")
        if c in ("1", "2", "3"):
            show_task(models, list(TASKS.values())[int(c) - 1], opts)
            for cit in citations:
                print(DIM + cit + RESET)
        elif c == "4":
            compare(models, ask("Nombres separados por coma"))
        elif c == "5":
            edit_profile(opts)
        elif c == "6":
            export_csv(models, ask("Fichero", "modelos.csv"))
        elif c.lower() in ("q", "salir", "exit"):
            return


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Compara modelos de OpenRouter por tarea, calidad y precio.")
    ap.add_argument("task", nargs="?", choices=list(TASKS), help="muestra una tarea y sale (sin menú)")
    ap.add_argument("--compare", help="modelos a comparar, separados por coma")
    ap.add_argument("--csv", help="exporta todo a este CSV y sale")
    ap.add_argument("--refresh", action="store_true", help="ignora la caché (12 h)")
    ap.add_argument("--free", dest="include_free", action="store_true", help="incluye variantes :free")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--max-cost", type=float, help="precio máximo por tarea (USD)")
    ap.add_argument("--baseline", help="tu modelo actual, p. ej. deepseek/deepseek-v4.1-flash-1017")
    ap.add_argument("--per-month", type=float, help="tareas (u horas de sesión en coding) al mes")
    ap.add_argument("--max-in", type=float, help="precio máximo de entrada (USD por millón de tokens)")
    ap.add_argument("--max-out", type=float, help="precio máximo de salida (USD por millón de tokens)")
    ap.add_argument("--gui", action="store_true", help="abre la interfaz gráfica en el navegador")
    ap.add_argument("--recommend", action="store_true",
                    help="recomendación para todas las tareas (usa --baseline y --per-month para programar)")
    ap.add_argument("--port", type=int, default=8765, help="puerto de la GUI (0 = cualquiera libre)")
    opts = ap.parse_args(argv)

    if opts.gui:
        import gui
        gui.run(os.environ.get("OPENROUTER_API_KEY"), opts.port)
        return

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and sys.stdin.isatty() and not (opts.task or opts.compare or opts.csv or opts.recommend):
        key = getpass.getpass("OPENROUTER_API_KEY (Enter para seguir sin benchmarks): ").strip() or None
    models, citations = load(key, opts.refresh)

    if opts.recommend:
        print_advice(models, opts)
        return
    if opts.task or opts.compare or opts.csv:
        if opts.task:
            show_task(models, TASKS[opts.task], opts)
            for cit in citations:
                print(DIM + cit + RESET)
        if opts.compare:
            compare(models, opts.compare)
        if opts.csv:
            export_csv(models, opts.csv)
        return
    menu(models, citations, opts)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
