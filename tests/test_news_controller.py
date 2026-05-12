import csv
import io
import json
import os
import shutil
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from src.api.news_controller import NewsController
from src.services.rss_digest import RssDigestClient


NOW_UTC_ISO = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


SAMPLE_PAYLOAD = {
    "schema_version": "1.0",
    "generated_at": NOW_UTC_ISO,
    "contract": "rss_pipeline_precomputed",
    "digest": {"generated_at": NOW_UTC_ISO, "run_id": "digest-controller"},
    "summary": {"articles": 1, "scored_articles": 1},
    "analysis": {},
    "articles": [
        {
            "id": "a-1",
            "title": "Controller Story",
            "link": "https://example.com/controller",
            "published": NOW_UTC_ISO,
            "summary": "Summary",
            "ai_summary": "AI Summary",
            "ai_tags": ["OpenAI"],
            "topic_tags": ["General"],
            "source": {"id": "source-a", "name": "Source A"},
            "feed": {"name": "Feed", "url": "https://example.com/feed"},
            "scraped": {"title": "Controller Story", "body_text": "Body"},
            "scrape_error": None,
            "score": {"value": 12.0, "max_value": 20.0, "percent": 60.0, "rubric_count": 2},
        }
    ],
}


TEMPORAL_EXPORT_PAYLOAD = {
    "meta": {"source_mode": "test"},
    "data": {
        "derived": {
            "group_temporal_latent_space": {
                "config": {
                    "bucket_granularity": "week",
                    "min_articles_per_bucket": 2,
                    "min_buckets_per_group": 2,
                },
                "groups": {
                    "source": [
                        {
                            "group": "Source A",
                            "group_key": "source-a",
                            "status": "ok",
                            "reason": "",
                            "n_articles": 5,
                            "n_buckets": 3,
                            "date_start": "2026-04-01",
                            "date_end": "2026-04-26",
                            "buckets": [
                                {
                                    "bucket_start": "2026-03-30",
                                    "bucket_end": "2026-04-05",
                                    "bucket_label": "2026-03-30",
                                    "status": "ok",
                                    "n_articles": 2,
                                    "n_sources": 1,
                                    "corpus_share": 0.5,
                                    "pc1": 0.11,
                                    "pc2": -0.22,
                                    "mds1": -0.05,
                                    "mds2": 0.18,
                                    "dispersion_pca": 0.07,
                                    "dispersion_mds": 0.05,
                                    "source_counts": {"Source A": 2},
                                    "top_lens_deviations": [
                                        {"lens": "Evidence", "delta": 8.0},
                                        {"lens": "Impact", "delta": -6.0},
                                    ],
                                },
                                {
                                    "bucket_start": "2026-04-06",
                                    "bucket_end": "2026-04-12",
                                    "bucket_label": "2026-04-06",
                                    "status": "sparse",
                                    "n_articles": 1,
                                    "n_sources": 1,
                                    "corpus_share": 0.25,
                                    "pc1": 0.19,
                                    "pc2": -0.17,
                                    "mds1": -0.01,
                                    "mds2": 0.11,
                                    "dispersion_pca": 0.0,
                                    "dispersion_mds": 0.0,
                                    "source_counts": {"Source A": 1},
                                    "top_lens_deviations": [
                                        {"lens": "Novelty", "delta": 4.0},
                                    ],
                                },
                                {
                                    "bucket_start": "2026-04-20",
                                    "bucket_end": "2026-04-26",
                                    "bucket_label": "2026-04-20",
                                    "status": "ok",
                                    "n_articles": 2,
                                    "n_sources": 1,
                                    "corpus_share": 0.5,
                                    "pc1": 0.31,
                                    "pc2": -0.08,
                                    "mds1": 0.06,
                                    "mds2": 0.26,
                                    "dispersion_pca": 0.09,
                                    "dispersion_mds": 0.06,
                                    "source_counts": {"Source A": 2},
                                    "top_lens_deviations": [
                                        {"lens": "Impact", "delta": -9.0},
                                        {"lens": "Evidence", "delta": 6.0},
                                    ],
                                },
                            ],
                            "path_summary": {
                                "bucket_count": 3,
                                "valid_pca_bucket_count": 2,
                                "valid_mds_bucket_count": 2,
                                "sparse_bucket_count": 1,
                                "coverage_gap_count": 1,
                                "coverage_gap_ranges": [
                                    {
                                        "start_bucket": "2026-04-13",
                                        "end_bucket": "2026-04-13",
                                        "missing_bucket_count": 1,
                                        "label": "2026-04-13",
                                    }
                                ],
                                "start_bucket": "2026-03-30",
                                "end_bucket": "2026-04-20",
                                "total_movement_pca": 0.33,
                                "largest_jump_pca": 0.33,
                                "direction_pca": {"pc1_delta": 0.2, "pc2_delta": -0.1},
                                "total_movement_mds": 0.21,
                                "largest_jump_mds": 0.21,
                                "direction_mds": {"mds1_delta": -0.05, "mds2_delta": 0.08},
                            },
                            "popularity_summary": {
                                "peak_bucket": "2026-03-30",
                                "peak_articles": 2,
                                "first_share": 0.5,
                                "latest_share": 0.5,
                                "share_delta": 0.0,
                                "recent_share_delta": 0.0,
                            },
                        }
                    ],
                    "topic": [
                        {
                            "group": "Policy",
                            "group_key": "policy",
                            "status": "ok",
                            "reason": "",
                            "n_articles": 2,
                            "n_buckets": 1,
                            "date_start": "2026-04-01",
                            "date_end": "2026-04-05",
                            "buckets": [
                                {
                                    "bucket_start": "2026-03-30",
                                    "bucket_end": "2026-04-05",
                                    "bucket_label": "2026-03-30",
                                    "status": "ok",
                                    "n_articles": 2,
                                    "n_sources": 1,
                                    "corpus_share": 0.2,
                                    "pc1": 0.41,
                                    "pc2": 0.12,
                                    "mds1": 0.02,
                                    "mds2": -0.14,
                                    "dispersion_pca": 0.04,
                                    "dispersion_mds": 0.03,
                                    "source_counts": {"Source A": 2},
                                    "top_lens_deviations": [
                                        {"lens": "Risk", "delta": 5.0},
                                    ],
                                }
                            ],
                            "path_summary": {
                                "bucket_count": 1,
                                "valid_pca_bucket_count": 1,
                                "valid_mds_bucket_count": 1,
                                "sparse_bucket_count": 0,
                                "coverage_gap_count": 0,
                                "coverage_gap_ranges": [],
                                "start_bucket": "2026-03-30",
                                "end_bucket": "2026-03-30",
                                "total_movement_pca": 0.0,
                                "largest_jump_pca": 0.0,
                                "direction_pca": {"pc1_delta": 0.0, "pc2_delta": 0.0},
                                "total_movement_mds": 0.0,
                                "largest_jump_mds": 0.0,
                                "direction_mds": {"mds1_delta": 0.0, "mds2_delta": 0.0},
                            },
                            "popularity_summary": {
                                "peak_bucket": "2026-03-30",
                                "peak_articles": 2,
                                "first_share": 0.2,
                                "latest_share": 0.2,
                                "share_delta": 0.0,
                                "recent_share_delta": None,
                            },
                        }
                    ],
                    "tag": [
                        {
                            "group": "Risk",
                            "group_key": "risk",
                            "status": "ok",
                            "reason": "",
                            "n_articles": 3,
                            "n_buckets": 2,
                            "date_start": "2026-04-01",
                            "date_end": "2026-04-20",
                            "buckets": [
                                {
                                    "bucket_start": "2026-03-30",
                                    "bucket_end": "2026-04-05",
                                    "bucket_label": "2026-03-30",
                                    "status": "ok",
                                    "n_articles": 1,
                                    "n_sources": 1,
                                    "corpus_share": 0.1,
                                    "pc1": 0.18,
                                    "pc2": 0.04,
                                    "mds1": -0.02,
                                    "mds2": -0.06,
                                    "dispersion_pca": 0.03,
                                    "dispersion_mds": 0.02,
                                    "source_counts": {"Source A": 1},
                                    "top_lens_deviations": [
                                        {"lens": "Risk", "delta": 5.0},
                                    ],
                                },
                                {
                                    "bucket_start": "2026-04-20",
                                    "bucket_end": "2026-04-26",
                                    "bucket_label": "2026-04-20",
                                    "status": "ok",
                                    "n_articles": 2,
                                    "n_sources": 1,
                                    "corpus_share": 0.25,
                                    "pc1": 0.23,
                                    "pc2": 0.06,
                                    "mds1": 0.01,
                                    "mds2": -0.02,
                                    "dispersion_pca": 0.04,
                                    "dispersion_mds": 0.03,
                                    "source_counts": {"Source A": 2},
                                    "top_lens_deviations": [
                                        {"lens": "Risk", "delta": 6.0},
                                    ],
                                },
                            ],
                            "path_summary": {
                                "bucket_count": 2,
                                "valid_pca_bucket_count": 2,
                                "valid_mds_bucket_count": 2,
                                "sparse_bucket_count": 0,
                                "coverage_gap_count": 0,
                                "coverage_gap_ranges": [],
                                "start_bucket": "2026-03-30",
                                "end_bucket": "2026-04-20",
                                "total_movement_pca": 0.05,
                                "largest_jump_pca": 0.05,
                                "direction_pca": {"pc1_delta": 0.05, "pc2_delta": 0.02},
                                "total_movement_mds": 0.05,
                                "largest_jump_mds": 0.05,
                                "direction_mds": {"mds1_delta": 0.03, "mds2_delta": 0.04},
                            },
                            "popularity_summary": {
                                "peak_bucket": "2026-04-20",
                                "peak_articles": 2,
                                "first_share": 0.1,
                                "latest_share": 0.25,
                                "share_delta": 0.15,
                                "recent_share_delta": 0.15,
                            },
                        }
                    ],
                },
            }
        }
    },
}


class StubClient:
    def __init__(self, payload):
        self.payload = payload

    def get_payload(self, *, force_refresh: bool, snapshot_date: str | None = None):
        return self.payload


class RecordingStubClient(StubClient):
    def __init__(self, payload):
        super().__init__(payload)
        self.calls = []

    def get_payload(self, *, force_refresh: bool, snapshot_date: str | None = None):
        self.calls.append({"force_refresh": force_refresh, "snapshot_date": snapshot_date})
        return self.payload


def _snapshot_temporal_export_payload(snapshot_date: str) -> dict:
    return {
        **TEMPORAL_EXPORT_PAYLOAD,
        "meta": {
            **(TEMPORAL_EXPORT_PAYLOAD.get("meta") or {}),
            "source_mode": "snapshot",
            "snapshot_date": snapshot_date,
        },
    }


def _snapshot_sparse_temporal_export_payload(snapshot_date: str) -> dict:
    payload = deepcopy(_snapshot_temporal_export_payload(snapshot_date))
    groups = payload["data"]["derived"]["group_temporal_latent_space"]["groups"]

    source_row = groups["source"][0]
    source_row["group_key"] = "source a"
    source_row["popularity_summary"] = {
        "peak_bucket": "2026-03-30",
        "peak_articles": 2,
        "first_share": 0.5,
        "latest_share": 0.5,
        "share_delta": 0.0,
        "recent_share_delta": 1 / 6,
    }

    topic_row = groups["topic"][0]
    topic_row["popularity_summary"] = {
        "peak_bucket": "2026-03-30",
        "peak_articles": 2,
        "first_share": 0.5,
        "latest_share": 0.5,
        "share_delta": 0.0,
        "recent_share_delta": 1 / 6,
    }

    tag_row = groups["tag"][0]
    tag_row["popularity_summary"] = {
        "peak_bucket": "2026-03-30",
        "peak_articles": 2,
        "first_share": 0.1,
        "latest_share": 0.1,
        "share_delta": 0.0,
        "recent_share_delta": 1 / 6,
    }

    return payload


def _current_sparse_temporal_export_payload() -> dict:
    payload = _snapshot_sparse_temporal_export_payload("2026-04-22")
    payload["meta"]["source_mode"] = "current"
    payload["meta"]["snapshot_date"] = None
    return payload


def _current_sparse_temporal_export_payload_with_multi_word_topic() -> dict:
    payload = deepcopy(_current_sparse_temporal_export_payload())
    topic_row = payload["data"]["derived"]["group_temporal_latent_space"]["groups"]["topic"][0]
    topic_row["group"] = "Climate Risk"
    topic_row["group_key"] = "climate risk"
    return payload


def _snapshot_sparse_temporal_export_payload_with_multi_word_topic(snapshot_date: str) -> dict:
    payload = deepcopy(_snapshot_sparse_temporal_export_payload(snapshot_date))
    topic_row = payload["data"]["derived"]["group_temporal_latent_space"]["groups"]["topic"][0]
    topic_row["group"] = "Climate Risk"
    topic_row["group_key"] = "climate risk"
    return payload


class NewsControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="rss-news-controller-"))
        cls.current_payload_path = cls.temp_dir / "rss_openai_precomputed.json"
        cls.current_payload_path.write_text(json.dumps(SAMPLE_PAYLOAD), encoding="utf-8")

        os.environ["RSS_DAILY_JSON_URL"] = f"file://{cls.current_payload_path}"
        os.environ["RSS_CACHE_TTL_SECONDS"] = "60"
        os.environ["RSS_HTTP_TIMEOUT_SECONDS"] = "5"
        os.environ["RSS_MAX_AGE_SECONDS"] = "172800"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_digest_rejects_invalid_limit(self):
        controller = NewsController(RssDigestClient())
        response = controller.get_digest(
            refresh=None,
            date=None,
            tag=None,
            source=None,
            limit="0",
            snapshot_date=None,
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body.get("status"), "bad_request")

    def test_digest_success_sets_cache_headers_and_refresh_disables_cache(self):
        controller = NewsController(RssDigestClient())
        cached = controller.get_digest(
            refresh=None,
            date=None,
            tag=None,
            source=None,
            limit="1",
            snapshot_date=None,
        )
        self.assertEqual(cached.status_code, 200)
        self.assertIn("public, max-age=", cached.headers.get("Cache-Control", ""))

        refreshed = controller.get_digest(
            refresh="1",
            date=None,
            tag=None,
            source=None,
            limit="1",
            snapshot_date=None,
        )
        self.assertEqual(refreshed.status_code, 200)
        self.assertEqual(refreshed.headers.get("Cache-Control"), "no-store")

    def test_export_csv_uses_csv_response_contract(self):
        controller = NewsController(RssDigestClient())
        response = controller.export_artifact(
            refresh=None,
            artifact="event_control_summary",
            export_format="csv",
            snapshot_date=None,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertIn("Content-Disposition", response.headers)
        self.assertIsInstance(response.body, str)
        self.assertIn("event_control_summary.csv", response.headers["Content-Disposition"])
        self.assertIn("event_count", response.body)

    def test_export_json_surfaces_group_temporal_path_summary_rows(self):
        controller = NewsController(StubClient(TEMPORAL_EXPORT_PAYLOAD))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_path_summary",
            export_format="json",
            snapshot_date=None,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body["artifact"], "group_temporal_path_summary")
        self.assertEqual(len(response.body["rows"]), 3)

        row = response.body["rows"][0]
        self.assertEqual(row["group_type"], "source")
        self.assertEqual(row["group"], "Source A")
        self.assertEqual(row["coverage_gap_count"], 1)
        self.assertEqual(row["coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"missing_bucket_count\": 1", row["coverage_gap_ranges_json"])
        self.assertEqual(row["direction_pca_pc1_delta"], 0.2)
        self.assertEqual(row["direction_mds_mds2_delta"], 0.08)
        self.assertEqual(row["movement_pattern_pca"], "jump-led")
        self.assertEqual(row["largest_jump_share_pca"], 1.0)
        self.assertEqual(row["movement_pattern_mds"], "jump-led")
        self.assertEqual(row["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(row["popularity_peak_articles"], 2)
        self.assertEqual(row["popularity_share_delta"], 0.0)

        policy_row = next(row for row in response.body["rows"] if row["group_type"] == "topic")
        self.assertEqual(policy_row["movement_pattern_pca"], "no measurable movement")
        self.assertIsNone(policy_row["largest_jump_share_pca"])
        self.assertEqual(policy_row["popularity_first_share"], 0.2)
        self.assertEqual(policy_row["popularity_latest_share"], 0.2)

    def test_export_json_surfaces_group_temporal_bucket_rows(self):
        controller = NewsController(StubClient(TEMPORAL_EXPORT_PAYLOAD))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_buckets",
            export_format="json",
            snapshot_date=None,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body["artifact"], "group_temporal_buckets")
        self.assertEqual(len(response.body["rows"]), 6)

        sparse_row = next(row for row in response.body["rows"] if row["bucket_status"] == "sparse")
        self.assertEqual(sparse_row["group_type"], "source")
        self.assertEqual(sparse_row["group"], "Source A")
        self.assertEqual(sparse_row["group_coverage_gap_count"], 1)
        self.assertEqual(sparse_row["group_coverage_gap_labels"], "2026-04-13")
        self.assertEqual(sparse_row["bucket_start"], "2026-04-06")
        self.assertEqual(sparse_row["bucket_n_articles"], 1)
        self.assertEqual(sparse_row["bucket_pc1"], 0.19)
        self.assertEqual(sparse_row["group_movement_pattern_pca"], "jump-led")
        self.assertEqual(sparse_row["group_largest_jump_share_pca"], 1.0)
        self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(sparse_row["group_popularity_peak_articles"], 2)
        self.assertEqual(sparse_row["group_popularity_recent_share_delta"], 0.0)
        self.assertEqual(sparse_row["bucket_top_lens_labels"], "Novelty")
        self.assertIn("\"Source A\": 1", sparse_row["bucket_source_counts_json"])

    def test_export_json_filters_group_temporal_buckets_by_group_type_and_group_key(self):
        controller = NewsController(StubClient(TEMPORAL_EXPORT_PAYLOAD))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_buckets",
            export_format="json",
            snapshot_date=None,
            group_type="source",
            group_key="source-a",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body["filters"], {"group_type": "source", "group_key": "source-a"})
        self.assertEqual(len(response.body["rows"]), 3)
        self.assertTrue(all(row["group_type"] == "source" for row in response.body["rows"]))
        self.assertTrue(all(row["group_key"] == "source-a" for row in response.body["rows"]))

    def test_export_json_filters_group_temporal_buckets_normalizes_source_group_key_variants_in_current_mode(self):
        controller = NewsController(StubClient(_current_sparse_temporal_export_payload()))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_buckets",
            export_format="json",
            snapshot_date=None,
            group_type="source",
            group_key="Source_A",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body["filters"], {"group_type": "source", "group_key": "Source_A"})
        self.assertEqual(response.body["meta"]["source_mode"], "current")
        self.assertIsNone(response.body["meta"]["snapshot_date"])
        self.assertEqual(len(response.body["rows"]), 3)
        self.assertTrue(all(row["group_type"] == "source" for row in response.body["rows"]))
        self.assertTrue(all(row["group_key"] == "source a" for row in response.body["rows"]))

    def test_export_json_filters_group_temporal_buckets_by_source_variant_and_bucket_label_in_current_mode(self):
        controller = NewsController(StubClient(_current_sparse_temporal_export_payload()))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_buckets",
            export_format="json",
            snapshot_date=None,
            group_type="source",
            group_key="Source_A",
            bucket_label="2026-04-06",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(
            response.body["filters"],
            {"group_type": "source", "group_key": "Source_A", "bucket_label": "2026-04-06"},
        )
        self.assertEqual(response.body["meta"]["source_mode"], "current")
        self.assertIsNone(response.body["meta"]["snapshot_date"])
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertEqual(response.body["rows"][0]["group_type"], "source")
        self.assertEqual(response.body["rows"][0]["group_key"], "source a")
        self.assertEqual(response.body["rows"][0]["group"], "Source A")
        self.assertEqual(response.body["rows"][0]["bucket_label"], "2026-04-06")
        self.assertEqual(response.body["rows"][0]["bucket_status"], "sparse")
        self.assertEqual(response.body["rows"][0]["bucket_n_articles"], 1)

    def test_export_json_filters_group_temporal_buckets_by_bucket_label_without_group_filters_in_current_mode(self):
        controller = NewsController(StubClient(_current_sparse_temporal_export_payload()))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_buckets",
            export_format="json",
            snapshot_date=None,
            bucket_label="2026-04-06",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body["filters"], {"bucket_label": "2026-04-06"})
        self.assertEqual(response.body["meta"]["source_mode"], "current")
        self.assertIsNone(response.body["meta"]["snapshot_date"])
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertEqual({row["group_type"] for row in response.body["rows"]}, {"source"})
        self.assertTrue(all(row["bucket_label"] == "2026-04-06" for row in response.body["rows"]))
        self.assertTrue(all(row["bucket_status"] == "sparse" for row in response.body["rows"]))

    def test_export_json_filters_group_temporal_path_summary_by_group_type_and_group_key(self):
        controller = NewsController(StubClient(TEMPORAL_EXPORT_PAYLOAD))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_path_summary",
            export_format="json",
            snapshot_date=None,
            group_type="source",
            group_key="source-a",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body["filters"], {"group_type": "source", "group_key": "source-a"})
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertTrue(all(row["group_type"] == "source" for row in response.body["rows"]))
        self.assertTrue(all(row["group_key"] == "source-a" for row in response.body["rows"]))

    def test_export_json_filters_group_temporal_path_summary_normalizes_source_group_key_variants(self):
        controller = NewsController(StubClient(TEMPORAL_EXPORT_PAYLOAD))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_path_summary",
            export_format="json",
            snapshot_date=None,
            group_type="source",
            group_key="Source_A",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body["filters"], {"group_type": "source", "group_key": "Source_A"})
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertTrue(all(row["group_type"] == "source" for row in response.body["rows"]))
        self.assertTrue(all(row["group_key"] == "source-a" for row in response.body["rows"]))

    def test_export_json_filters_group_temporal_path_summary_normalizes_source_group_key_variants_in_current_mode(self):
        controller = NewsController(StubClient(_current_sparse_temporal_export_payload()))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_path_summary",
            export_format="json",
            snapshot_date=None,
            group_type="source",
            group_key="Source_A",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body["filters"], {"group_type": "source", "group_key": "Source_A"})
        self.assertEqual(response.body["meta"]["source_mode"], "current")
        self.assertIsNone(response.body["meta"]["snapshot_date"])
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertEqual(response.body["rows"][0]["group_type"], "source")
        self.assertEqual(response.body["rows"][0]["group_key"], "source a")
        self.assertEqual(response.body["rows"][0]["group"], "Source A")
        self.assertEqual(response.body["rows"][0]["coverage_gap_labels"], "2026-04-13")
        self.assertAlmostEqual(response.body["rows"][0]["popularity_recent_share_delta"], 1 / 6)

    def test_export_json_filters_group_temporal_path_summary_by_topic_and_includes_popularity_fields(self):
        controller = NewsController(StubClient(TEMPORAL_EXPORT_PAYLOAD))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_path_summary",
            export_format="json",
            snapshot_date=None,
            group_type="topic",
            group_key="policy",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body["filters"], {"group_type": "topic", "group_key": "policy"})
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertEqual(response.body["rows"][0]["group"], "Policy")
        self.assertEqual(response.body["rows"][0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(response.body["rows"][0]["popularity_peak_articles"], 2)
        self.assertEqual(response.body["rows"][0]["popularity_first_share"], 0.2)
        self.assertEqual(response.body["rows"][0]["popularity_latest_share"], 0.2)

    def test_export_snapshot_group_temporal_buckets_keeps_group_filters(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_buckets",
            export_format="json",
            snapshot_date=snapshot_date,
            group_type="source",
            group_key="source-a",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertEqual(response.body["filters"], {"group_type": "source", "group_key": "source-a"})
        self.assertEqual(response.body["meta"]["source_mode"], "snapshot")
        self.assertEqual(response.body["meta"]["snapshot_date"], snapshot_date)
        self.assertEqual(len(response.body["rows"]), 3)
        self.assertTrue(all(row["group_type"] == "source" for row in response.body["rows"]))
        self.assertTrue(all(row["group_key"] == "source-a" for row in response.body["rows"]))
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": snapshot_date}])

    def test_export_snapshot_group_temporal_buckets_normalizes_source_group_key_variants(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_sparse_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_buckets",
            export_format="json",
            snapshot_date=snapshot_date,
            group_type="source",
            group_key="Source_A",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertEqual(response.body["filters"], {"group_type": "source", "group_key": "Source_A"})
        self.assertEqual(response.body["meta"]["source_mode"], "snapshot")
        self.assertEqual(response.body["meta"]["snapshot_date"], snapshot_date)
        self.assertEqual(len(response.body["rows"]), 3)
        self.assertTrue(all(row["group_type"] == "source" for row in response.body["rows"]))
        self.assertTrue(all(row["group_key"] == "source a" for row in response.body["rows"]))
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": snapshot_date}])

    def test_export_snapshot_group_temporal_path_summary_keeps_group_filters(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_path_summary",
            export_format="json",
            snapshot_date=snapshot_date,
            group_type="topic",
            group_key="policy",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIn("public, max-age=", response.headers.get("Cache-Control", ""))
        self.assertEqual(response.body["filters"], {"group_type": "topic", "group_key": "policy"})
        self.assertEqual(response.body["meta"]["source_mode"], "snapshot")
        self.assertEqual(response.body["meta"]["snapshot_date"], snapshot_date)
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertEqual(response.body["rows"][0]["group"], "Policy")
        self.assertEqual(response.body["rows"][0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(client.calls, [{"force_refresh": False, "snapshot_date": snapshot_date}])

    def test_export_snapshot_group_temporal_path_summary_normalizes_source_group_key_variants(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_sparse_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_path_summary",
            export_format="json",
            snapshot_date=snapshot_date,
            group_type="source",
            group_key="Source_A",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIn("public, max-age=", response.headers.get("Cache-Control", ""))
        self.assertEqual(response.body["filters"], {"group_type": "source", "group_key": "Source_A"})
        self.assertEqual(response.body["meta"]["source_mode"], "snapshot")
        self.assertEqual(response.body["meta"]["snapshot_date"], snapshot_date)
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertEqual(response.body["rows"][0]["group_type"], "source")
        self.assertEqual(response.body["rows"][0]["group_key"], "source a")
        self.assertEqual(response.body["rows"][0]["group"], "Source A")
        self.assertAlmostEqual(response.body["rows"][0]["popularity_recent_share_delta"], 1 / 6)
        self.assertEqual(client.calls, [{"force_refresh": False, "snapshot_date": snapshot_date}])

    def test_export_json_surfaces_group_temporal_popularity_momentum_rows(self):
        controller = NewsController(StubClient(TEMPORAL_EXPORT_PAYLOAD))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_popularity_momentum",
            export_format="json",
            snapshot_date=None,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body["artifact"], "group_temporal_popularity_momentum")
        self.assertEqual(len(response.body["rows"]), 3)

        risk_row = next(row for row in response.body["rows"] if row["group_type"] == "tag")
        self.assertEqual(risk_row["group"], "Risk")
        self.assertEqual(risk_row["popularity_share_direction"], "rising")
        self.assertEqual(risk_row["popularity_recent_share_direction"], "rising")
        self.assertEqual(risk_row["popularity_momentum_label"], "rising now")
        self.assertEqual(risk_row["popularity_first_share_rank_within_type"], 1)
        self.assertEqual(risk_row["popularity_latest_share_rank_within_type"], 1)
        self.assertEqual(risk_row["popularity_rank_change_within_type"], 0)
        self.assertEqual(risk_row["popularity_share_delta_rank_within_type"], 1)
        self.assertEqual(risk_row["popularity_recent_share_delta_rank_within_type"], 1)
        self.assertEqual(risk_row["popularity_peak_bucket"], "2026-04-20")
        self.assertEqual(risk_row["popularity_peak_articles"], 2)

    def test_export_json_filters_group_temporal_popularity_momentum_by_tag(self):
        controller = NewsController(StubClient(TEMPORAL_EXPORT_PAYLOAD))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_popularity_momentum",
            export_format="json",
            snapshot_date=None,
            group_type="tag",
            group_key="risk",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body["filters"], {"group_type": "tag", "group_key": "risk"})
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertEqual(response.body["rows"][0]["group"], "Risk")
        self.assertEqual(response.body["rows"][0]["popularity_share_delta"], 0.15)
        self.assertEqual(response.body["rows"][0]["popularity_recent_share_delta"], 0.15)
        self.assertEqual(response.body["rows"][0]["popularity_momentum_label"], "rising now")
        self.assertEqual(response.body["rows"][0]["popularity_first_share_rank_within_type"], 1)
        self.assertEqual(response.body["rows"][0]["popularity_latest_share_rank_within_type"], 1)
        self.assertEqual(response.body["rows"][0]["popularity_rank_change_within_type"], 0)

    def test_export_json_filters_group_temporal_popularity_momentum_normalizes_source_group_key_variants_in_current_mode(self):
        controller = NewsController(StubClient(_current_sparse_temporal_export_payload()))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_popularity_momentum",
            export_format="json",
            snapshot_date=None,
            group_type="source",
            group_key="Source_A",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIsInstance(response.body, dict)
        self.assertEqual(response.body["filters"], {"group_type": "source", "group_key": "Source_A"})
        self.assertEqual(response.body["meta"]["source_mode"], "current")
        self.assertIsNone(response.body["meta"]["snapshot_date"])
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertEqual(response.body["rows"][0]["group_type"], "source")
        self.assertEqual(response.body["rows"][0]["group_key"], "source a")
        self.assertEqual(response.body["rows"][0]["group"], "Source A")
        self.assertAlmostEqual(response.body["rows"][0]["popularity_recent_share_delta"], 1 / 6)
        self.assertEqual(response.body["rows"][0]["popularity_momentum_label"], "rebounding to baseline")

    def test_export_snapshot_group_temporal_popularity_momentum_keeps_group_filters(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_popularity_momentum",
            export_format="json",
            snapshot_date=snapshot_date,
            group_type="tag",
            group_key="risk",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIn("public, max-age=", response.headers.get("Cache-Control", ""))
        self.assertEqual(response.body["filters"], {"group_type": "tag", "group_key": "risk"})
        self.assertEqual(response.body["meta"]["source_mode"], "snapshot")
        self.assertEqual(response.body["meta"]["snapshot_date"], snapshot_date)
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertEqual(response.body["rows"][0]["group"], "Risk")
        self.assertEqual(response.body["rows"][0]["popularity_momentum_label"], "rising now")
        self.assertEqual(response.body["rows"][0]["popularity_share_delta"], 0.15)
        self.assertEqual(response.body["rows"][0]["popularity_recent_share_delta"], 0.15)
        self.assertEqual(client.calls, [{"force_refresh": False, "snapshot_date": snapshot_date}])

    def test_export_snapshot_group_temporal_popularity_momentum_keeps_source_filters(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_sparse_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_popularity_momentum",
            export_format="json",
            snapshot_date=snapshot_date,
            group_type="source",
            group_key="source-a",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIn("public, max-age=", response.headers.get("Cache-Control", ""))
        self.assertEqual(response.body["filters"], {"group_type": "source", "group_key": "source-a"})
        self.assertEqual(response.body["meta"]["source_mode"], "snapshot")
        self.assertEqual(response.body["meta"]["snapshot_date"], snapshot_date)
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertEqual(response.body["rows"][0]["group_type"], "source")
        self.assertEqual(response.body["rows"][0]["group_key"], "source a")
        self.assertEqual(response.body["rows"][0]["group"], "Source A")
        self.assertEqual(response.body["rows"][0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(response.body["rows"][0]["popularity_peak_articles"], 2)
        self.assertEqual(response.body["rows"][0]["popularity_first_share"], 0.5)
        self.assertEqual(response.body["rows"][0]["popularity_latest_share"], 0.5)
        self.assertEqual(response.body["rows"][0]["popularity_share_delta"], 0.0)
        self.assertAlmostEqual(response.body["rows"][0]["popularity_recent_share_delta"], 1 / 6)
        self.assertEqual(response.body["rows"][0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertEqual(response.body["rows"][0]["popularity_share_delta_rank_within_type"], 1)
        self.assertEqual(response.body["rows"][0]["popularity_recent_share_delta_rank_within_type"], 1)
        self.assertEqual(client.calls, [{"force_refresh": False, "snapshot_date": snapshot_date}])

    def test_export_snapshot_group_temporal_popularity_momentum_normalizes_source_group_key_variants(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_sparse_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_popularity_momentum",
            export_format="json",
            snapshot_date=snapshot_date,
            group_type="source",
            group_key="Source_A",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIn("public, max-age=", response.headers.get("Cache-Control", ""))
        self.assertEqual(response.body["filters"], {"group_type": "source", "group_key": "Source_A"})
        self.assertEqual(response.body["meta"]["source_mode"], "snapshot")
        self.assertEqual(response.body["meta"]["snapshot_date"], snapshot_date)
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertEqual(response.body["rows"][0]["group_type"], "source")
        self.assertEqual(response.body["rows"][0]["group_key"], "source a")
        self.assertEqual(response.body["rows"][0]["group"], "Source A")
        self.assertAlmostEqual(response.body["rows"][0]["popularity_recent_share_delta"], 1 / 6)
        self.assertEqual(response.body["rows"][0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertEqual(client.calls, [{"force_refresh": False, "snapshot_date": snapshot_date}])

    def test_export_snapshot_group_temporal_popularity_momentum_keeps_topic_filters(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_sparse_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_popularity_momentum",
            export_format="json",
            snapshot_date=snapshot_date,
            group_type="topic",
            group_key="policy",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertIn("public, max-age=", response.headers.get("Cache-Control", ""))
        self.assertEqual(response.body["filters"], {"group_type": "topic", "group_key": "policy"})
        self.assertEqual(response.body["meta"]["source_mode"], "snapshot")
        self.assertEqual(response.body["meta"]["snapshot_date"], snapshot_date)
        self.assertEqual(len(response.body["rows"]), 1)
        self.assertEqual(response.body["rows"][0]["group_type"], "topic")
        self.assertEqual(response.body["rows"][0]["group_key"], "policy")
        self.assertEqual(response.body["rows"][0]["group"], "Policy")
        self.assertEqual(response.body["rows"][0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(response.body["rows"][0]["popularity_peak_articles"], 2)
        self.assertEqual(response.body["rows"][0]["popularity_first_share"], 0.5)
        self.assertEqual(response.body["rows"][0]["popularity_latest_share"], 0.5)
        self.assertEqual(response.body["rows"][0]["popularity_share_delta"], 0.0)
        self.assertAlmostEqual(response.body["rows"][0]["popularity_recent_share_delta"], 1 / 6)
        self.assertEqual(response.body["rows"][0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertEqual(response.body["rows"][0]["popularity_share_delta_rank_within_type"], 1)
        self.assertEqual(response.body["rows"][0]["popularity_recent_share_delta_rank_within_type"], 1)
        self.assertEqual(client.calls, [{"force_refresh": False, "snapshot_date": snapshot_date}])

    def test_export_snapshot_group_temporal_popularity_momentum_csv_keeps_group_filters(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_popularity_momentum",
            export_format="csv",
            snapshot_date=snapshot_date,
            group_type="tag",
            group_key="risk",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_popularity_momentum.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("popularity_momentum_label", reader.fieldnames)
        self.assertIn("popularity_share_delta_rank_within_type", reader.fieldnames)
        self.assertIn("popularity_recent_share_delta_rank_within_type", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "tag")
        self.assertEqual(rows[0]["group_key"], "risk")
        self.assertEqual(rows[0]["group"], "Risk")
        self.assertEqual(rows[0]["popularity_momentum_label"], "rising now")
        self.assertEqual(rows[0]["popularity_share_delta"], "0.15")
        self.assertEqual(rows[0]["popularity_recent_share_delta"], "0.15")
        self.assertEqual(rows[0]["popularity_share_delta_rank_within_type"], "1")
        self.assertEqual(rows[0]["popularity_recent_share_delta_rank_within_type"], "1")
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": snapshot_date}])

    def test_export_snapshot_group_temporal_popularity_momentum_csv_keeps_source_filters(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_sparse_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_popularity_momentum",
            export_format="csv",
            snapshot_date=snapshot_date,
            group_type="source",
            group_key="source-a",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_popularity_momentum.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("popularity_peak_bucket", reader.fieldnames)
        self.assertIn("popularity_momentum_label", reader.fieldnames)
        self.assertIn("popularity_recent_share_delta_rank_within_type", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "source")
        self.assertEqual(rows[0]["group_key"], "source a")
        self.assertEqual(rows[0]["group"], "Source A")
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_peak_articles"], "2")
        self.assertEqual(rows[0]["popularity_first_share"], "0.5")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
        self.assertEqual(rows[0]["popularity_share_delta"], "0.0")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(rows[0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertEqual(rows[0]["popularity_share_delta_rank_within_type"], "1")
        self.assertEqual(rows[0]["popularity_recent_share_delta_rank_within_type"], "1")
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": snapshot_date}])

    def test_export_snapshot_group_temporal_popularity_momentum_csv_keeps_topic_filters(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_sparse_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_popularity_momentum",
            export_format="csv",
            snapshot_date=snapshot_date,
            group_type="topic",
            group_key="policy",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_popularity_momentum.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("popularity_peak_bucket", reader.fieldnames)
        self.assertIn("popularity_momentum_label", reader.fieldnames)
        self.assertIn("popularity_recent_share_delta_rank_within_type", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "topic")
        self.assertEqual(rows[0]["group_key"], "policy")
        self.assertEqual(rows[0]["group"], "Policy")
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_peak_articles"], "2")
        self.assertEqual(rows[0]["popularity_first_share"], "0.5")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
        self.assertEqual(rows[0]["popularity_share_delta"], "0.0")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(rows[0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertEqual(rows[0]["popularity_share_delta_rank_within_type"], "1")
        self.assertEqual(rows[0]["popularity_recent_share_delta_rank_within_type"], "1")
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": snapshot_date}])

    def test_export_snapshot_group_temporal_path_summary_csv_normalizes_source_group_key_variants(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_sparse_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_path_summary",
            export_format="csv",
            snapshot_date=snapshot_date,
            group_type="source",
            group_key="Source_A",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_path_summary.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("coverage_gap_labels", reader.fieldnames)
        self.assertIn("coverage_gap_ranges_json", reader.fieldnames)
        self.assertIn("popularity_recent_share_delta", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "source")
        self.assertEqual(rows[0]["group_key"], "source a")
        self.assertEqual(rows[0]["group"], "Source A")
        self.assertEqual(rows[0]["coverage_gap_count"], "1")
        self.assertEqual(rows[0]["coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"missing_bucket_count\": 1", rows[0]["coverage_gap_ranges_json"])
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_peak_articles"], "2")
        self.assertEqual(rows[0]["popularity_first_share"], "0.5")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": snapshot_date}])

    def test_export_current_group_temporal_path_summary_csv_keeps_group_filters(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_path_summary",
            export_format="csv",
            snapshot_date=None,
            group_type="tag",
            group_key="risk",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_path_summary.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("coverage_gap_labels", reader.fieldnames)
        self.assertIn("coverage_gap_ranges_json", reader.fieldnames)
        self.assertIn("popularity_recent_share_delta", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "tag")
        self.assertEqual(rows[0]["group_key"], "risk")
        self.assertEqual(rows[0]["group"], "Risk")
        self.assertEqual(rows[0]["coverage_gap_count"], "0")
        self.assertEqual(rows[0]["coverage_gap_labels"], "")
        self.assertEqual(rows[0]["coverage_gap_ranges_json"], "[]")
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_first_share"], "0.1")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.1")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_current_group_temporal_path_summary_csv_keeps_source_filters(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_path_summary",
            export_format="csv",
            snapshot_date=None,
            group_type="source",
            group_key="source-a",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_path_summary.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("coverage_gap_labels", reader.fieldnames)
        self.assertIn("coverage_gap_ranges_json", reader.fieldnames)
        self.assertIn("popularity_recent_share_delta", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "source")
        self.assertEqual(rows[0]["group_key"], "source a")
        self.assertEqual(rows[0]["group"], "Source A")
        self.assertEqual(rows[0]["coverage_gap_count"], "1")
        self.assertEqual(rows[0]["coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"missing_bucket_count\": 1", rows[0]["coverage_gap_ranges_json"])
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_peak_articles"], "2")
        self.assertEqual(rows[0]["popularity_first_share"], "0.5")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_current_group_temporal_path_summary_csv_normalizes_source_group_key_variants(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_path_summary",
            export_format="csv",
            snapshot_date=None,
            group_type="source",
            group_key="Source_A",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_path_summary.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("coverage_gap_labels", reader.fieldnames)
        self.assertIn("coverage_gap_ranges_json", reader.fieldnames)
        self.assertIn("popularity_recent_share_delta", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "source")
        self.assertEqual(rows[0]["group_key"], "source a")
        self.assertEqual(rows[0]["group"], "Source A")
        self.assertEqual(rows[0]["coverage_gap_count"], "1")
        self.assertEqual(rows[0]["coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"missing_bucket_count\": 1", rows[0]["coverage_gap_ranges_json"])
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_peak_articles"], "2")
        self.assertEqual(rows[0]["popularity_first_share"], "0.5")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_current_group_temporal_path_summary_csv_keeps_topic_filters(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_path_summary",
            export_format="csv",
            snapshot_date=None,
            group_type="topic",
            group_key="policy",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_path_summary.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("coverage_gap_labels", reader.fieldnames)
        self.assertIn("coverage_gap_ranges_json", reader.fieldnames)
        self.assertIn("popularity_recent_share_delta", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "topic")
        self.assertEqual(rows[0]["group_key"], "policy")
        self.assertEqual(rows[0]["group"], "Policy")
        self.assertEqual(rows[0]["coverage_gap_count"], "0")
        self.assertEqual(rows[0]["coverage_gap_labels"], "")
        self.assertEqual(rows[0]["coverage_gap_ranges_json"], "[]")
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_peak_articles"], "2")
        self.assertEqual(rows[0]["popularity_first_share"], "0.5")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_current_group_temporal_buckets_csv_keeps_group_filters(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_buckets",
            export_format="csv",
            snapshot_date=None,
            group_type="tag",
            group_key="risk",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_buckets.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("group_coverage_gap_ranges_json", reader.fieldnames)
        self.assertIn("bucket_top_lens_deviations_json", reader.fieldnames)
        self.assertIn("group_popularity_recent_share_delta", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["group_type"] == "tag" for row in rows))
        self.assertTrue(all(row["group_key"] == "risk" for row in rows))
        self.assertEqual([row["bucket_label"] for row in rows], ["2026-03-30", "2026-04-20"])
        self.assertEqual(rows[0]["group"], "Risk")
        self.assertEqual(rows[0]["group_coverage_gap_ranges_json"], "[]")
        self.assertEqual(rows[0]["group_popularity_share_delta"], "0.0")
        self.assertAlmostEqual(float(rows[0]["group_popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_current_group_temporal_buckets_csv_keeps_source_filters(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_buckets",
            export_format="csv",
            snapshot_date=None,
            group_type="source",
            group_key="source-a",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_buckets.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("group_coverage_gap_labels", reader.fieldnames)
        self.assertIn("bucket_source_counts_json", reader.fieldnames)
        self.assertIn("group_popularity_peak_bucket", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["group_type"] == "source" for row in rows))
        self.assertTrue(all(row["group_key"] == "source a" for row in rows))
        sparse_row = next(row for row in rows if row["bucket_label"] == "2026-04-06")
        self.assertEqual(sparse_row["group"], "Source A")
        self.assertEqual(sparse_row["bucket_status"], "sparse")
        self.assertEqual(sparse_row["group_coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"Source A\": 1", sparse_row["bucket_source_counts_json"])
        self.assertEqual(sparse_row["bucket_top_lens_labels"], "Novelty")
        self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(sparse_row["group_popularity_peak_articles"], "2")
        self.assertAlmostEqual(float(sparse_row["group_popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_current_group_temporal_buckets_csv_normalizes_source_group_key_variants(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_buckets",
            export_format="csv",
            snapshot_date=None,
            group_type="source",
            group_key="Source_A",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_buckets.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        rows = list(reader)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["group_type"] == "source" for row in rows))
        self.assertTrue(all(row["group_key"] == "source a" for row in rows))
        sparse_row = next(row for row in rows if row["bucket_label"] == "2026-04-06")
        self.assertEqual(sparse_row["group"], "Source A")
        self.assertEqual(sparse_row["group_coverage_gap_labels"], "2026-04-13")
        self.assertAlmostEqual(float(sparse_row["group_popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_current_group_temporal_buckets_csv_keeps_topic_filters(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_buckets",
            export_format="csv",
            snapshot_date=None,
            group_type="topic",
            group_key="policy",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_buckets.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("bucket_top_lens_labels", reader.fieldnames)
        self.assertIn("group_popularity_latest_share", reader.fieldnames)
        self.assertIn("group_popularity_recent_share_delta", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "topic")
        self.assertEqual(rows[0]["group_key"], "policy")
        self.assertEqual(rows[0]["group"], "Policy")
        self.assertEqual(rows[0]["bucket_label"], "2026-03-30")
        self.assertEqual(rows[0]["bucket_top_lens_labels"], "Risk")
        self.assertEqual(rows[0]["group_popularity_latest_share"], "0.5")
        self.assertAlmostEqual(float(rows[0]["group_popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_current_group_temporal_buckets_json_supports_bucket_label_filters_for_multi_word_topic_keys(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload_with_multi_word_topic())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_buckets",
            export_format="json",
            snapshot_date=None,
            group_type="topic",
            group_key="Climate_Risk",
            bucket_label="2026-03-30",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertEqual(
            response.body["filters"],
            {"group_type": "topic", "group_key": "Climate_Risk", "bucket_label": "2026-03-30"},
        )

        rows = response.body["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "topic")
        self.assertEqual(rows[0]["group_key"], "climate risk")
        self.assertEqual(rows[0]["group"], "Climate Risk")
        self.assertEqual(rows[0]["bucket_label"], "2026-03-30")
        self.assertEqual(rows[0]["bucket_status"], "ok")
        self.assertEqual(rows[0]["bucket_n_articles"], 2)
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_current_group_temporal_buckets_csv_supports_bucket_label_filters_for_multi_word_topic_keys(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload_with_multi_word_topic())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_buckets",
            export_format="csv",
            snapshot_date=None,
            group_type="topic",
            group_key="Climate_Risk",
            bucket_label="2026-03-30",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_buckets.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "topic")
        self.assertEqual(rows[0]["group_key"], "climate risk")
        self.assertEqual(rows[0]["group"], "Climate Risk")
        self.assertEqual(rows[0]["bucket_label"], "2026-03-30")
        self.assertEqual(rows[0]["bucket_status"], "ok")
        self.assertEqual(rows[0]["bucket_n_articles"], "2")
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_snapshot_group_temporal_buckets_json_supports_bucket_label_filters_for_multi_word_topic_keys(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_sparse_temporal_export_payload_with_multi_word_topic(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_buckets",
            export_format="json",
            snapshot_date=snapshot_date,
            group_type="topic",
            group_key="Climate_Risk",
            bucket_label="2026-03-30",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertEqual(
            response.body["filters"],
            {"group_type": "topic", "group_key": "Climate_Risk", "bucket_label": "2026-03-30"},
        )
        self.assertEqual(response.body["meta"]["source_mode"], "snapshot")
        self.assertEqual(response.body["meta"]["snapshot_date"], snapshot_date)

        rows = response.body["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "topic")
        self.assertEqual(rows[0]["group_key"], "climate risk")
        self.assertEqual(rows[0]["group"], "Climate Risk")
        self.assertEqual(rows[0]["bucket_label"], "2026-03-30")
        self.assertEqual(rows[0]["bucket_status"], "ok")
        self.assertEqual(rows[0]["bucket_n_articles"], 2)
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": snapshot_date}])

    def test_export_snapshot_group_temporal_buckets_csv_supports_bucket_label_filters_for_multi_word_topic_keys(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_sparse_temporal_export_payload_with_multi_word_topic(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_buckets",
            export_format="csv",
            snapshot_date=snapshot_date,
            group_type="topic",
            group_key="Climate_Risk",
            bucket_label="2026-03-30",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_buckets.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "topic")
        self.assertEqual(rows[0]["group_key"], "climate risk")
        self.assertEqual(rows[0]["group"], "Climate Risk")
        self.assertEqual(rows[0]["bucket_label"], "2026-03-30")
        self.assertEqual(rows[0]["bucket_status"], "ok")
        self.assertEqual(rows[0]["bucket_n_articles"], "2")
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": snapshot_date}])

    def test_export_snapshot_group_temporal_buckets_csv_supports_bucket_label_filters_without_group_key(self):
        snapshot_date = "2026-04-22"
        client = RecordingStubClient(_snapshot_sparse_temporal_export_payload(snapshot_date))
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_buckets",
            export_format="csv",
            snapshot_date=snapshot_date,
            bucket_label="2026-04-06",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_buckets.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual({row["group_type"] for row in rows}, {"source"})
        self.assertTrue(all(row["bucket_label"] == "2026-04-06" for row in rows))
        self.assertTrue(all(row["bucket_status"] == "sparse" for row in rows))
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": snapshot_date}])

    def test_export_current_group_temporal_popularity_momentum_csv_keeps_group_filters(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_popularity_momentum",
            export_format="csv",
            snapshot_date=None,
            group_type="tag",
            group_key="risk",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_popularity_momentum.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("popularity_momentum_label", reader.fieldnames)
        self.assertIn("popularity_share_delta_rank_within_type", reader.fieldnames)
        self.assertIn("popularity_recent_share_delta_rank_within_type", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "tag")
        self.assertEqual(rows[0]["group_key"], "risk")
        self.assertEqual(rows[0]["group"], "Risk")
        self.assertEqual(rows[0]["popularity_share_delta"], "0.0")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(rows[0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertEqual(rows[0]["popularity_share_delta_rank_within_type"], "1")
        self.assertEqual(rows[0]["popularity_recent_share_delta_rank_within_type"], "1")
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_current_group_temporal_popularity_momentum_csv_keeps_source_filters(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_popularity_momentum",
            export_format="csv",
            snapshot_date=None,
            group_type="source",
            group_key="source-a",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_popularity_momentum.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("popularity_peak_bucket", reader.fieldnames)
        self.assertIn("popularity_momentum_label", reader.fieldnames)
        self.assertIn("popularity_recent_share_delta_rank_within_type", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "source")
        self.assertEqual(rows[0]["group_key"], "source a")
        self.assertEqual(rows[0]["group"], "Source A")
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_peak_articles"], "2")
        self.assertEqual(rows[0]["popularity_first_share"], "0.5")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
        self.assertEqual(rows[0]["popularity_share_delta"], "0.0")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(rows[0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertEqual(rows[0]["popularity_share_delta_rank_within_type"], "1")
        self.assertEqual(rows[0]["popularity_recent_share_delta_rank_within_type"], "1")
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_current_group_temporal_popularity_momentum_csv_normalizes_source_group_key_variants(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_popularity_momentum",
            export_format="csv",
            snapshot_date=None,
            group_type="source",
            group_key="Source_A",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_popularity_momentum.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "source")
        self.assertEqual(rows[0]["group_key"], "source a")
        self.assertEqual(rows[0]["group"], "Source A")
        self.assertEqual(rows[0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_current_group_temporal_popularity_momentum_csv_keeps_topic_filters(self):
        client = RecordingStubClient(_current_sparse_temporal_export_payload())
        controller = NewsController(client)
        response = controller.export_artifact(
            refresh="1",
            artifact="group_temporal_popularity_momentum",
            export_format="csv",
            snapshot_date=None,
            group_type="topic",
            group_key="policy",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/csv; charset=utf-8")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        self.assertIn("group_temporal_popularity_momentum.csv", response.headers.get("Content-Disposition", ""))

        reader = csv.DictReader(io.StringIO(response.body))
        self.assertIsNotNone(reader.fieldnames)
        self.assertIn("group_type", reader.fieldnames)
        self.assertIn("group_key", reader.fieldnames)
        self.assertIn("popularity_peak_bucket", reader.fieldnames)
        self.assertIn("popularity_momentum_label", reader.fieldnames)
        self.assertIn("popularity_recent_share_delta_rank_within_type", reader.fieldnames)

        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "topic")
        self.assertEqual(rows[0]["group_key"], "policy")
        self.assertEqual(rows[0]["group"], "Policy")
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_peak_articles"], "2")
        self.assertEqual(rows[0]["popularity_first_share"], "0.5")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
        self.assertEqual(rows[0]["popularity_share_delta"], "0.0")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(rows[0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertEqual(rows[0]["popularity_share_delta_rank_within_type"], "1")
        self.assertEqual(rows[0]["popularity_recent_share_delta_rank_within_type"], "1")
        self.assertEqual(client.calls, [{"force_refresh": True, "snapshot_date": None}])

    def test_export_rejects_group_key_without_group_type(self):
        controller = NewsController(StubClient(TEMPORAL_EXPORT_PAYLOAD))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_buckets",
            export_format="json",
            snapshot_date=None,
            group_key="source-a",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.body["status"], "bad_request")
        self.assertIn("group_key requires group_type", response.body["error"])

    def test_export_rejects_bucket_label_for_non_bucket_artifacts(self):
        controller = NewsController(StubClient(TEMPORAL_EXPORT_PAYLOAD))
        response = controller.export_artifact(
            refresh=None,
            artifact="group_temporal_path_summary",
            export_format="json",
            snapshot_date=None,
            bucket_label="2026-04-06",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.body["status"], "bad_request")
        self.assertIn("bucket_label is only supported for artifact=group_temporal_buckets", response.body["error"])


if __name__ == "__main__":
    unittest.main()
