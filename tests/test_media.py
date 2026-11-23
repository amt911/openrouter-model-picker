import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import media  # noqa: E402


class ImageCost(unittest.TestCase):
    def test_units(self):
        self.assertEqual(media.image_cost([{"billable": "output_image", "unit": "image", "cost_usd": 0.04}]), (0.04, False, ""))
        c, est, _ = media.image_cost([{"billable": "output_image", "unit": "megapixel", "cost_usd": 0.01}])
        self.assertAlmostEqual(c, 0.01 * 1.048576)
        self.assertFalse(est)
        c, est, note = media.image_cost([{"billable": "output_image", "unit": "token", "cost_usd": 1e-5}])
        self.assertAlmostEqual(c, 1e-5 * media.IMAGE_TOKENS)
        self.assertTrue(est)

    def test_picks_1k_and_ignores_surcharges(self):
        p = [{"billable": "output_image", "unit": "image", "cost_usd": 0.6, "variant": "4k"},
             {"billable": "output_image", "unit": "image", "cost_usd": 0.05, "variant": "1k"},
             {"billable": "output_image", "unit": "image", "cost_usd": 0.02, "variant": "high_thinking_surcharge"},
             {"billable": "input_text", "unit": "token", "cost_usd": 1e-6}]
        self.assertEqual(media.image_cost(p)[0], 0.05)
        self.assertIsNone(media.image_cost([p[0]])[0])


class VideoCost(unittest.TestCase):
    def test_resolution_audio_and_cents(self):
        m = {"supported_resolutions": ["720p", "1080p"], "generate_audio": True, "pricing_skus": {
            "duration_seconds_with_audio": "0.4", "duration_seconds_without_audio": "0.2",
            "duration_seconds_with_audio_720p": "0.3", "reference_input_images": "9"}}
        self.assertEqual(media.video_cost(m, "1080p", False)[0], 0.2)
        self.assertEqual(media.video_cost(m, "1080p", True)[0], 0.4)
        self.assertEqual(media.video_cost(m, "720p", True)[0], 0.3)
        self.assertIsNone(media.video_cost(m, "4k")[0])
        cents = {"supported_resolutions": ["720p"], "pricing_skus": {"cents_per_video_output_second_720p": "14"}}
        self.assertAlmostEqual(media.video_cost(cents, "720p")[0], 0.14)

    def test_no_audio_model_excluded_when_audio_wanted(self):
        m = {"generate_audio": False, "pricing_skus": {"duration_seconds": "0.1"}}
        self.assertIsNone(media.video_cost(m, "1080p", True)[0])

    def test_tokens_are_estimated(self):
        m = {"supported_resolutions": ["720p"], "pricing_skus": {"video_tokens": "0.000007"}}
        c, est, _ = media.video_cost(m, "720p")
        self.assertAlmostEqual(c, 0.000007 * 1280 * 720 * 24 / 1024)
        self.assertTrue(est)


class SttCost(unittest.TestCase):
    def test_units(self):
        self.assertAlmostEqual(media.stt_cost({"prompt": "0.0001", "completion": "0"})[0], 0.006)
        self.assertIsNone(media.stt_cost({"prompt": "0.000002", "completion": "0.000012"})[0])
        self.assertIsNone(media.stt_cost({"prompt": "0.36", "completion": "0"})[0])


RAW = {
    "images": [{"id": "a/img", "name": "A", "endpoints": "/x"}, {"id": "b/img", "name": "B", "endpoints": "/y"},
               {"id": "c/img", "name": "C", "endpoints": "/z"}],
    "image_endpoints": {
        "a/img": {"endpoints": [{"provider_name": "P1", "provider_tag": "p1", "pricing": [
            {"billable": "output_image", "unit": "image", "cost_usd": 0.10}]},
            {"provider_name": "P2", "provider_tag": "p2", "pricing": [
                {"billable": "output_image", "unit": "image", "cost_usd": 0.08}]}]},
        "b/img": {"endpoints": [{"provider_name": "Q", "provider_tag": "q", "pricing": [
            {"billable": "output_image", "unit": "image", "cost_usd": 0.01}]}]},
        "c/img": OSError("down")},
    "videos": [{"id": "x/avatar-1", "pricing_skus": {"duration_seconds": "0.01"}},
               {"id": "x/gen", "pricing_skus": {"duration_seconds": "0.2"}},
               {"id": "x/video-upscale", "pricing_skus": {"duration_seconds": "0.001"}}],
    "stt": [{"id": "s/a", "pricing": {"prompt": "0.0001", "completion": "0"}}],
    "stt_endpoints": {"s/a": {"data": {"endpoints": [
        {"provider_name": "S1", "tag": "s1", "pricing": {"prompt": "0.0001", "completion": "0"}},
        {"provider_name": "S2", "tag": "s2", "pricing": {"prompt": "0.00005", "completion": "0"}}]}}},
}


class BuildAndAdvise(unittest.TestCase):
    def test_build(self):
        out = media.build(RAW, {"image": {"a/img": 1300, "b/img": 1000}})
        img = {m.id: m for m in out["image"]}
        self.assertEqual((img["a/img"].provider, img["a/img"].unit_cost), ("P2", 0.08))
        self.assertEqual(img["a/img"].quality, 100)
        self.assertIsNone(img["c/img"].unit_cost)  # endpoint fetch failed: no price, no crash
        vid = {m.id: m for m in out["video"]}
        self.assertNotIn("x/video-upscale", vid)
        self.assertTrue(vid["x/avatar-1"].special)
        stt = out["transcription"][0]
        self.assertEqual(stt.provider, "S2")
        self.assertAlmostEqual(stt.unit_cost, 0.003)

    def test_advise_quality_vs_cheap(self):
        out = media.build(RAW, {"image": {"a/img": 1300, "b/img": 1000}})
        adv = media.advise(out["image"], per_month=100)  # 100 x 0.08 = $8 < $20: quality wins
        roles = {r: m.id for r, m, _ in adv.picks}
        self.assertEqual(roles, {"ahorro": "b/img", "calidad": "a/img"})
        self.assertEqual(adv.main, "calidad")
        adv = media.advise(out["image"], per_month=10_000)
        self.assertEqual(adv.main, "ahorro")
        adv = media.advise(out["image"], per_month=100, cap=0.05)
        self.assertEqual([m.id for _, m, _ in adv.picks], ["b/img"])

    def test_avatars_never_recommended_and_no_quality_means_cheapest(self):
        out = media.build(RAW)
        adv = media.advise(out["video"], per_month=60)
        self.assertEqual([(r, m.id) for r, m, _ in adv.picks], [("mas_barato", "x/gen")])
        self.assertFalse(adv.has_quality)

    def test_elo_from_bench(self):
        b = {"da_image": {"data": [{"model_permaslug": "a/img", "elo": 1200}, {"model_permaslug": "a/img", "elo": 1400}]}}
        self.assertEqual(media.elo_from_bench(b)["image"], {"a/img": 1300})


if __name__ == "__main__":
    unittest.main()
