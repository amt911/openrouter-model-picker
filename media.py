"""Images, video and transcription: catalogs, price normalization and recommendations.

OpenRouter lists these outside the chat catalog and prices them in different units, so
everything is normalized to one comparable unit per kind:
  image          USD per 1024x1024 image (~1 MP)
  video          USD per second at the chosen resolution, with or without audio
  transcription  USD per minute of audio
Prices that cannot be converted honestly (unknown units) are kept, flagged, and never ranked.
Quality comes from Design Arena's image/video arenas (ELO); there is no transcription
benchmark in OpenRouter's API, so transcription is ranked by price only.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import ormp

KINDS = {
    "image": {"name": "Imágenes", "unit": "imagen", "units": "imágenes", "per_month": 100},
    "video": {"name": "Vídeo", "unit": "segundo de vídeo", "units": "segundos de vídeo", "per_month": 60},
    "transcription": {"name": "Transcripción", "unit": "minuto de audio", "units": "minutos de audio",
                      "per_month": 600},
}
IMAGE_TOKENS = 1290  # output tokens of a ~1 MP image (Gemini's documented figure); token-priced = estimate
MP_PER_IMAGE = 1024 * 1024 / 1e6
RESOLUTIONS = ["360p", "480p", "720p", "768p", "1080p", "2k", "4k"]


@dataclass
class MediaModel:
    kind: str
    id: str
    name: str
    unit_cost: float | None  # USD per normalized unit; None = not comparable
    provider: str | None = None
    tag: str | None = None
    estimated: bool = False  # converted with an assumption (e.g. tokens per image)
    note: str = ""
    elo: float | None = None
    quality: float | None = None  # percentile of elo among this kind's models
    providers: list[dict] = field(default_factory=list)  # all offers, cheapest first
    special: bool = False  # niche use (e.g. talking avatars): listed, never recommended


# ---------------------------------------------------------------- fetching (cached)
def _endpoints_many(paths: dict[str, str], prefix: str, refresh: bool) -> dict:
    def one(item):
        mid, path = item
        safe = mid.replace("/", "__").replace(":", "_")
        try:
            return mid, ormp.cached(f"{prefix}_{safe}", lambda: ormp._get(path), refresh)
        except Exception as e:  # noqa: BLE001 - per model, reported as missing prices
            return mid, e
    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(ex.map(one, paths.items()))


def fetch_all(refresh: bool = False) -> dict:
    """Raw catalogs for the three kinds. Public endpoints, no key needed."""
    imgs = ormp.cached("images_models", lambda: ormp._get("/images/models"), refresh)
    vids = ormp.cached("videos_models", lambda: ormp._get("/videos/models"), refresh)
    stt = ormp.cached("stt_models", lambda: ormp._get("/models", {"output_modalities": "transcription"}), refresh)
    img_list = imgs.get("data", imgs) if isinstance(imgs, dict) else imgs
    stt_list = stt.get("data", [])
    img_eps = _endpoints_many({m["id"]: m["endpoints"].replace("/api/v1", "") for m in img_list
                               if isinstance(m.get("endpoints"), str)}, "imgeps", refresh)
    stt_eps = _endpoints_many({m["id"]: f"/models/{m['id']}/endpoints" for m in stt_list}, "stteps", refresh)
    return {"images": img_list, "image_endpoints": img_eps,
            "videos": vids.get("data", vids) if isinstance(vids, dict) else vids,
            "stt": stt_list, "stt_endpoints": stt_eps}


# ---------------------------------------------------------------- normalization
def image_cost(pricing: list[dict]) -> tuple[float | None, bool, str]:
    """USD per ~1 MP image from an image endpoint's pricing list."""
    outs = [p for p in pricing if p.get("billable") == "output_image"]
    if not outs:
        return None, False, "sin precio de salida"
    # standard 1K output: no variant or a 1k variant; never surcharges or style add-ons
    rank = {"1k": 0, None: 1, "medium_1k": 2, "low_1k": 3}
    std = sorted((p for p in outs if p.get("variant") in rank), key=lambda p: rank[p.get("variant")])
    if not std:
        return None, False, "solo variantes especiales"
    p = std[0]
    c, unit = float(p["cost_usd"]), p.get("unit")
    if unit in ("image", "request"):
        return c, False, ""
    if unit == "megapixel":
        return c * MP_PER_IMAGE, False, ""
    if unit == "token":
        return c * IMAGE_TOKENS, True, f"≈ estimado con {IMAGE_TOKENS} tokens por imagen"
    return None, False, f"unidad desconocida: {unit}"


def _sku_parts(key: str):
    res = next((r for r in RESOLUTIONS if re.search(rf"(^|_){r}($|_)", key)), None)
    audio = "with" if "with_audio" in key else "without" if "without_audio" in key else None
    return res, audio


def video_cost(m: dict, resolution: str = "1080p", audio: bool = False) -> tuple[float | None, bool, str]:
    """USD per second at `resolution` from a video model's pricing_skus."""
    skus = m.get("pricing_skus") or {}
    if audio and m.get("generate_audio") is False:
        return None, False, "no genera audio"
    supported = [r.lower() for r in (m.get("supported_resolutions") or [])]
    if supported and resolution.lower() not in supported:
        return None, False, f"no ofrece {resolution}"
    per_sec = {}
    for k, v in skus.items():
        if any(w in k for w in ("input", "reference", "continuation", "long_", "minimum", "megapixel",
                                "image_to_video")):
            continue
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        if k.startswith("cents_per"):
            val /= 100
        elif k.startswith("video_tokens"):
            continue  # handled below
        elif not (k.startswith("duration_seconds") or k.startswith("text_to_video_duration_seconds")):
            continue
        per_sec[_sku_parts(k)] = val
    want_res = resolution.lower()

    def pick(table):
        audio_order = ["with", None] if audio else ["without", None, "with"]
        for res in (want_res, None):
            for a in audio_order:
                if (res, a) in table:
                    return table[(res, a)]
        return None
    c = pick(per_sec)
    if c is not None:
        return c, False, ""
    # token-priced (ByteDance Seedance): tokens = width * height * fps * seconds / 1024
    toks = {}
    for k, v in skus.items():
        if k.startswith("video_tokens") and "input" not in k:
            res, a = _sku_parts(k)
            toks[(res, a)] = float(v)
    t = pick(toks)
    if t is not None:
        w, h = {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}.get(
            want_res, (1280, 720))
        return t * w * h * 24 / 1024, True, "≈ estimado: tokens = ancho × alto × 24 fps / 1024"
    return None, False, "precio en unidades no comparables"


def stt_cost(pricing: dict) -> tuple[float | None, bool, str]:
    """USD per minute of audio. Duration-priced models list USD per second as `prompt`."""
    try:
        pin, pout = float(pricing.get("prompt", "x")), float(pricing.get("completion", 0) or 0)
    except ValueError:
        return None, False, "sin precio"
    if pout == 0 and 0 < pin < 0.01:
        return pin * 60, False, ""
    if pout > 0:
        return None, False, "cobra por tokens: no comparable por minuto"
    return None, False, "unidad de precio no clara en la API"


# ---------------------------------------------------------------- build offers
def _eps(raw) -> list[dict]:
    if isinstance(raw, Exception) or not raw:
        return []
    d = raw.get("data", raw)
    return d.get("endpoints") or [] if isinstance(d, dict) else []


def build(raw: dict, elo: dict[str, dict[str, float]] | None = None, resolution: str = "1080p",
          audio: bool = False) -> dict[str, list[MediaModel]]:
    """Normalized offers per kind, cheapest provider first. `elo` = {kind: {model id: ELO}}."""
    elo = elo or {}
    out = {"image": [], "video": [], "transcription": []}
    for m in raw.get("images") or []:
        offers = []
        for e in _eps(raw.get("image_endpoints", {}).get(m["id"])):
            c, est, note = image_cost(e.get("pricing") or [])
            offers.append({"provider": e.get("provider_name"), "tag": e.get("provider_tag"), "cost": c,
                           "estimated": est, "note": note})
        mm = _from_offers("image", m["id"], m.get("name", m["id"]), offers)
        if re.search(r"panorama|world", m["id"]):
            mm.special = True
            mm.note = (mm.note + "; " if mm.note else "") + "panoramas 360°, no imágenes normales"
        out["image"].append(mm)
    for m in raw.get("videos") or []:
        if re.search(r"upscale|edit", m["id"]):
            continue  # not text-to-video generation
        c, est, note = video_cost(m, resolution, audio)
        mm = MediaModel("video", m["id"], m.get("name", m["id"]), c, estimated=est, note=note)
        if "avatar" in m["id"]:
            mm.special = True
            mm.note = (mm.note + "; " if mm.note else "") + "avatar que habla (lip-sync), no texto a vídeo general"
        out["video"].append(mm)
    for m in raw.get("stt") or []:
        offers = []
        for e in _eps(raw.get("stt_endpoints", {}).get(m["id"])):
            c, est, note = stt_cost(e.get("pricing") or {})
            offers.append({"provider": e.get("provider_name"), "tag": e.get("tag"), "cost": c,
                           "estimated": est, "note": note, "uptime_1d": e.get("uptime_last_1d")})
        if not offers:  # no endpoint data: catalog price
            c, est, note = stt_cost(m.get("pricing") or {})
            offers = [{"provider": None, "tag": None, "cost": c, "estimated": est, "note": note}]
        out["transcription"].append(_from_offers("transcription", m["id"], m.get("name", m["id"]), offers))
    for kind, ms in out.items():
        scores = elo.get(kind, {})
        for mm in ms:
            mm.elo = scores.get(mm.id)
        have = sorted(mm.elo for mm in ms if mm.elo is not None)
        for mm in ms:
            if mm.elo is not None:
                lower = sum(1 for x in have if x < mm.elo)
                equal = sum(1 for x in have if x == mm.elo)
                mm.quality = 100.0 if len(have) == 1 else 100.0 * (lower + (equal - 1) / 2) / (len(have) - 1)
    return out


def _from_offers(kind, mid, name, offers) -> MediaModel:
    offers.sort(key=lambda o: (o["cost"] is None, o["cost"] or 0))
    best = offers[0] if offers else {"cost": None, "estimated": False, "note": "sin proveedores"}
    return MediaModel(kind, mid, name, best["cost"], best.get("provider"), best.get("tag"),
                      best["estimated"], best["note"], providers=offers)


def elo_from_bench(bench: dict) -> dict[str, dict[str, float]]:
    """Design Arena image/video ELO per OpenRouter model id (mean over categories)."""
    out = {}
    for kind, key in (("image", "da_image"), ("video", "da_video")):
        acc = {}
        for it in (bench.get(key) or {}).get("data", []):
            if it.get("model_permaslug") and it.get("elo") is not None:
                acc.setdefault(it["model_permaslug"], []).append(float(it["elo"]))
        out[kind] = {k: sum(v) / len(v) for k, v in acc.items()}
    return out


# ---------------------------------------------------------------- advisor
@dataclass
class MediaAdvice:
    kind: str
    per_month: float
    picks: list[tuple[str, MediaModel, str]]
    main: str | None
    has_quality: bool


def advise(models: list[MediaModel], per_month: float, cap: float | None = None,
           negligible: float = ormp.NEGLIGIBLE_USD) -> MediaAdvice:
    """Same rules as the text advisor: "ahorro" = cheapest; "calidad" = cheapest reaching 90%
    of the best Design Arena quality; quality wins when it costs under `negligible` a month.
    `cap` = max USD per unit. Without quality data only the cheapest is offered."""
    kind = models[0].kind if models else "?"
    ok = [m for m in models if m.unit_cost is not None and m.unit_cost > 0 and not m.special
          and (cap is None or m.unit_cost <= cap + 1e-12)]
    if not ok:
        return MediaAdvice(kind, per_month, [], None, False)
    cheapest = min(ok, key=lambda m: m.unit_cost)
    picks = []
    rated = [m for m in ok if m.quality is not None]
    if rated:
        top = max(m.quality for m in rated)
        cheap_rated = min(rated, key=lambda m: m.unit_cost)
        picks.append(("ahorro", cheap_rated, "El más barato con nota de calidad en Design Arena."))
        q = min((m for m in rated if m.quality >= 0.9 * top), key=lambda m: m.unit_cost)
        picks.append(("calidad", q, "El más barato con al menos el 90% de la mejor nota de Design Arena."))
        if cheapest.id != cheap_rated.id:
            picks.append(("mas_barato", cheapest, "El más barato de todos, pero sin nota de calidad: pruébalo antes."))
    else:
        why = ("El más barato. OpenRouter no publica benchmarks de transcripción: pruébalo con tus audios "
               "antes de decidir." if kind == "transcription" else
               "El más barato. No hay notas de Design Arena para comparar calidad (llegan con tu API key): "
               "pruébalo antes de decidir.")
        picks.append(("mas_barato", cheapest, why))
    seen, uniq = set(), []
    for role, m, why in picks:
        if m.id not in seen:
            seen.add(m.id)
            uniq.append((role, m, why))
    roles = {r: m for r, m, _ in uniq}
    if "calidad" in roles and roles["calidad"].unit_cost * per_month <= negligible:
        main = "calidad"
    else:
        main = next(r for r in ("ahorro", "mas_barato", "calidad") if r in roles)
    return MediaAdvice(kind, per_month, uniq, main, bool(rated))
