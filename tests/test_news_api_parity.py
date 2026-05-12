import csv
import io
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from flask import Flask

from src.api.fastapi_news import register_fastapi_news_endpoints
from src.api.news_endpoints import register_news_endpoints
from src.api.news_schemas import NewsApiEnvelope


NOW_UTC = datetime.now(timezone.utc)
NOW_UTC_ISO = NOW_UTC.isoformat().replace("+00:00", "Z")
DIGEST_UTC_ISO = (NOW_UTC - timedelta(minutes=3)).isoformat().replace("+00:00", "Z")


SAMPLE_PAYLOAD = {
    "schema_version": "1.0",
    "generated_at": NOW_UTC_ISO,
    "contract": "rss_pipeline_precomputed",
    "digest": {
        "generated_at": DIGEST_UTC_ISO,
        "run_id": "digest-parity-abc123",
    },
    "summary": {"articles": 3, "scored_articles": 3},
    "analysis": {"lens_summary": {}, "source_differentiation": {}},
    "articles": [
        {
            "id": "a-1",
            "title": "Latest Story",
            "link": "https://example.com/latest",
            "published": "Mon, 02 Mar 2026 15:25:29 -0500",
            "summary": "Summary 1",
            "ai_summary": "AI Summary 1",
            "ai_tags": ["OpenAI", "Policy"],
            "topic_tags": ["General"],
            "source": {"id": "pbs-newshour", "name": "PBS NewsHour"},
            "feed": {"name": "Headlines", "url": "https://example.com/feed"},
            "scraped": {"title": "Latest Story", "body_text": "Body"},
            "scrape_error": None,
            "score": {
                "value": 14.0,
                "max_value": 20.0,
                "percent": 70.0,
                "rubric_count": 3,
                "lens_scores": {"L1": {"percent": 70.0}},
            },
        },
        {
            "id": "a-2",
            "title": "Older Story",
            "link": "https://example.com/older",
            "published": "2026-03-01T03:10:00Z",
            "summary": "Summary 2",
            "ai_summary": "AI Summary 2",
            "ai_tags": ["Science"],
            "topic_tags": ["OpenAI"],
            "source": {"id": "npr", "name": "NPR"},
            "feed": {"name": "World", "url": "https://example.com/world"},
            "scraped": {"title": "Older Story", "body_text": "Body"},
            "scrape_error": None,
            "score": {"value": 8.0, "max_value": 20.0, "percent": 40.0, "rubric_count": 3},
        },
        {
            "id": "a-3",
            "title": "Failed Scrape Story",
            "link": "https://example.com/failed",
            "published": "2026-03-03T03:10:00Z",
            "summary": "Summary 3",
            "ai_summary": "AI Summary 3",
            "ai_tags": ["OpenAI"],
            "topic_tags": ["General"],
            "source": {"id": "failed-source", "name": "Failed Source"},
            "feed": {"name": "Errors", "url": "https://example.com/errors"},
            "scraped": None,
            "scrape_error": "HTTP 500",
            "score": {"value": 20.0, "max_value": 20.0, "percent": 100.0, "rubric_count": 3},
        },
    ],
}


def _sparse_bucket_temporal_payload() -> dict:
    articles = []
    article_specs = [
        ("2026-03-31", "Source A", ["Policy"], ["Risk"], 82.0, 24.0, 28.0),
        ("2026-04-02", "Source A", ["Policy"], ["Risk"], 80.0, 26.0, 30.0),
        ("2026-04-08", "Source A", ["Policy"], ["Risk"], 78.0, 28.0, 32.0),
        ("2026-04-22", "Source A", ["Policy"], ["Risk"], 76.0, 30.0, 34.0),
        ("2026-04-24", "Source A", ["Policy"], ["Risk"], 74.0, 32.0, 36.0),
        ("2026-04-01", "Source B", ["Markets"], ["Growth"], 44.0, 76.0, 38.0),
        ("2026-04-03", "Source B", ["Markets"], ["Growth"], 42.0, 78.0, 40.0),
        ("2026-04-07", "Source B", ["Markets"], ["Growth"], 46.0, 74.0, 42.0),
        ("2026-04-09", "Source B", ["Markets"], ["Growth"], 48.0, 72.0, 44.0),
        ("2026-04-14", "Source B", ["Markets"], ["Growth"], 50.0, 70.0, 46.0),
        ("2026-04-16", "Source B", ["Markets"], ["Growth"], 52.0, 68.0, 48.0),
        ("2026-04-21", "Source B", ["Markets"], ["Growth"], 54.0, 66.0, 50.0),
        ("2026-04-23", "Source B", ["Markets"], ["Growth"], 56.0, 64.0, 52.0),
    ]
    for idx, (published, source_name, topic_tags, ai_tags, evidence, impact, novelty) in enumerate(article_specs):
        articles.append(
            {
                "id": f"sparse-bucket-parity-{idx}",
                "title": f"Sparse bucket parity {idx}",
                "link": f"https://example.com/sparse-bucket/{idx}",
                "published": f"{published}T00:00:00Z",
                "summary": f"Summary {idx}",
                "ai_summary": f"AI Summary {idx}",
                "ai_tags": ai_tags,
                "topic_tags": topic_tags,
                "source": {"id": source_name.lower().replace(" ", "-"), "name": source_name},
                "feed": {"name": "Feed", "url": "https://example.com/feed"},
                "scraped": {"title": f"Sparse bucket parity {idx}", "body_text": "Body"},
                "scrape_error": None,
                "score": {
                    "value": 15.0,
                    "max_value": 20.0,
                    "percent": 75.0,
                    "rubric_count": 3,
                    "lens_scores": {
                        "Evidence": {"percent": evidence},
                        "Impact": {"percent": impact},
                        "Novelty": {"percent": novelty},
                    },
                },
            }
        )

    return {
        "schema_version": "1.0",
        "generated_at": NOW_UTC_ISO,
        "contract": "rss_pipeline_precomputed",
        "digest": {
            "generated_at": DIGEST_UTC_ISO,
            "run_id": "digest-parity-sparse-bucket",
        },
        "summary": {"articles": len(articles), "scored_articles": len(articles)},
        "analysis": {
            "lens_summary": {
                "lenses": [
                    {"name": "Evidence", "max_total": 10.0},
                    {"name": "Impact", "max_total": 10.0},
                    {"name": "Novelty", "max_total": 10.0},
                ]
            },
            "source_differentiation": {},
        },
        "articles": articles,
    }


def _sparse_bucket_temporal_payload_with_multi_word_topic() -> dict:
    payload = _sparse_bucket_temporal_payload()
    for article in payload.get("articles", []):
        if not isinstance(article, dict):
            continue
        topic_tags = article.get("topic_tags")
        if topic_tags == ["Policy"]:
            article["topic_tags"] = ["Climate Risk"]
    return payload


VOLATILE_KEYS = {"fetched_at", "from_cache", "age_seconds"}


def _strip_volatile(value: Any):
    if isinstance(value, dict):
        return {key: _strip_volatile(val) for key, val in value.items() if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


class NewsApiParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="rss-news-parity-"))
        cls.current_payload_path = cls.temp_dir / "rss_openai_precomputed.json"
        cls.current_payload_path.write_text(json.dumps(SAMPLE_PAYLOAD), encoding="utf-8")

        cls.snapshot_date = "2026-03-10"
        cls.snapshot_payload_path = cls.temp_dir / f"rss_openai_daily_{cls.snapshot_date}.json"
        cls.snapshot_payload_path.write_text(json.dumps(SAMPLE_PAYLOAD), encoding="utf-8")

        os.environ["RSS_DAILY_JSON_URL"] = f"file://{cls.current_payload_path}"
        os.environ["RSS_HISTORY_JSON_URL_TEMPLATE"] = f"file://{cls.temp_dir}/rss_openai_daily_{{date}}.json"
        os.environ["RSS_CACHE_TTL_SECONDS"] = "60"
        os.environ["RSS_HTTP_TIMEOUT_SECONDS"] = "5"
        os.environ["RSS_MAX_AGE_SECONDS"] = "172800"

        flask_app = Flask(__name__)
        register_news_endpoints(flask_app)
        cls.flask_client = flask_app.test_client()

        fastapi_app = FastAPI()
        register_fastapi_news_endpoints(fastapi_app)
        cls.fastapi_client = TestClient(fastapi_app)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _assert_json_parity(self, path: str):
        flask_response = self.flask_client.get(path)
        fastapi_response = self.fastapi_client.get(path)

        self.assertEqual(flask_response.status_code, fastapi_response.status_code, msg=path)

        flask_payload = flask_response.get_json()
        fastapi_payload = fastapi_response.json()
        NewsApiEnvelope.model_validate(fastapi_payload)
        NewsApiEnvelope.model_validate(flask_payload)

        self.assertEqual(_strip_volatile(flask_payload), _strip_volatile(fastapi_payload), msg=path)
        return flask_payload

    def _assert_csv_parity(self, path: str):
        flask_response = self.flask_client.get(path)
        fastapi_response = self.fastapi_client.get(path)

        self.assertEqual(flask_response.status_code, fastapi_response.status_code, msg=path)
        self.assertEqual(flask_response.headers.get("Content-Disposition"), fastapi_response.headers.get("Content-Disposition"), msg=path)
        self.assertEqual(flask_response.get_data(as_text=True), fastapi_response.text, msg=path)

        csv_payload = flask_response.get_data(as_text=True)
        rows = list(csv.DictReader(io.StringIO(csv_payload)))
        return csv_payload, rows

    def _assert_snapshot_group_temporal_popularity_momentum_filter_parity(
        self,
        *,
        group_type: str,
        group_key: str,
        expected_row_key: str,
        expected_group: str,
    ):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            payload = self._assert_json_parity(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=json&refresh=1"
                f"&group_type={group_type}&group_key={group_key}&snapshot_date={self.snapshot_date}"
            )
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")
            self.fastapi_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

        self.assertEqual(payload["artifact"], "group_temporal_popularity_momentum")
        self.assertEqual(payload["filters"], {"group_type": group_type, "group_key": group_key})
        self.assertEqual(payload["meta"]["source_mode"], "snapshot")
        self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["group_type"], group_type)
        self.assertEqual(payload["rows"][0]["group_key"], expected_row_key)
        self.assertEqual(payload["rows"][0]["group"], expected_group)
        self.assertEqual(payload["rows"][0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(payload["rows"][0]["popularity_peak_articles"], 2)
        self.assertEqual(payload["rows"][0]["popularity_first_share"], 0.5)
        self.assertEqual(payload["rows"][0]["popularity_latest_share"], 0.5)
        self.assertEqual(payload["rows"][0]["popularity_share_delta"], 0.0)
        self.assertAlmostEqual(payload["rows"][0]["popularity_recent_share_delta"], 1 / 6)
        self.assertEqual(payload["rows"][0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertEqual(payload["rows"][0]["popularity_share_delta_rank_within_type"], 1)
        self.assertEqual(payload["rows"][0]["popularity_recent_share_delta_rank_within_type"], 1)

    def _assert_current_group_temporal_popularity_momentum_filter_parity(
        self,
        *,
        group_type: str,
        group_key: str,
        expected_row_key: str,
        expected_group: str,
    ):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            payload = self._assert_json_parity(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=json&refresh=1"
                f"&group_type={group_type}&group_key={group_key}"
            )
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get("/api/news/stats?refresh=1")
            self.fastapi_client.get("/api/news/stats?refresh=1")

        self.assertEqual(payload["artifact"], "group_temporal_popularity_momentum")
        self.assertEqual(payload["filters"], {"group_type": group_type, "group_key": group_key})
        self.assertEqual(payload["meta"]["source_mode"], "current")
        self.assertIsNone(payload["meta"]["snapshot_date"])
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["group_type"], group_type)
        self.assertEqual(payload["rows"][0]["group_key"], expected_row_key)
        self.assertEqual(payload["rows"][0]["group"], expected_group)
        self.assertEqual(payload["rows"][0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(payload["rows"][0]["popularity_peak_articles"], 2)
        self.assertEqual(payload["rows"][0]["popularity_first_share"], 0.5)
        self.assertEqual(payload["rows"][0]["popularity_latest_share"], 0.5)
        self.assertEqual(payload["rows"][0]["popularity_share_delta"], 0.0)
        self.assertAlmostEqual(payload["rows"][0]["popularity_recent_share_delta"], 1 / 6)
        self.assertEqual(payload["rows"][0]["popularity_share_direction"], "flat")
        self.assertEqual(payload["rows"][0]["popularity_recent_share_direction"], "rising")
        self.assertEqual(payload["rows"][0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertEqual(payload["rows"][0]["popularity_first_share_rank_within_type"], 1)
        self.assertEqual(payload["rows"][0]["popularity_latest_share_rank_within_type"], 1)
        self.assertEqual(payload["rows"][0]["popularity_rank_change_within_type"], 0)
        self.assertEqual(payload["rows"][0]["popularity_share_delta_rank_within_type"], 1)
        self.assertEqual(payload["rows"][0]["popularity_recent_share_delta_rank_within_type"], 1)

    def _assert_snapshot_group_temporal_path_summary_filter_parity(
        self,
        *,
        group_type: str,
        group_key: str,
        expected_row_key: str,
        expected_group: str,
    ):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            payload = self._assert_json_parity(
                f"/api/news/export?artifact=group_temporal_path_summary&format=json&refresh=1"
                f"&group_type={group_type}&group_key={group_key}&snapshot_date={self.snapshot_date}"
            )
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")
            self.fastapi_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

        self.assertEqual(payload["artifact"], "group_temporal_path_summary")
        self.assertEqual(payload["filters"], {"group_type": group_type, "group_key": group_key})
        self.assertEqual(payload["meta"]["source_mode"], "snapshot")
        self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["group_type"], group_type)
        self.assertEqual(payload["rows"][0]["group_key"], expected_row_key)
        self.assertEqual(payload["rows"][0]["group"], expected_group)
        self.assertEqual(payload["rows"][0]["coverage_gap_count"], 1)
        self.assertEqual(payload["rows"][0]["coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"missing_bucket_count\": 1", payload["rows"][0]["coverage_gap_ranges_json"])
        self.assertEqual(payload["rows"][0]["movement_pattern_pca"], "jump-led")
        self.assertEqual(payload["rows"][0]["largest_jump_share_pca"], 1.0)
        self.assertEqual(payload["rows"][0]["movement_pattern_mds"], "jump-led")
        self.assertEqual(payload["rows"][0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(payload["rows"][0]["popularity_peak_articles"], 2)
        self.assertEqual(payload["rows"][0]["popularity_first_share"], 0.5)
        self.assertEqual(payload["rows"][0]["popularity_latest_share"], 0.5)
        self.assertEqual(payload["rows"][0]["popularity_share_delta"], 0.0)
        self.assertAlmostEqual(payload["rows"][0]["popularity_recent_share_delta"], 1 / 6)

    def _assert_current_group_temporal_path_summary_filter_parity(
        self,
        *,
        group_type: str,
        group_key: str,
        expected_row_key: str,
        expected_group: str,
    ):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            payload = self._assert_json_parity(
                f"/api/news/export?artifact=group_temporal_path_summary&format=json&refresh=1"
                f"&group_type={group_type}&group_key={group_key}"
            )
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get("/api/news/stats?refresh=1")
            self.fastapi_client.get("/api/news/stats?refresh=1")

        self.assertEqual(payload["artifact"], "group_temporal_path_summary")
        self.assertEqual(payload["filters"], {"group_type": group_type, "group_key": group_key})
        self.assertEqual(payload["meta"]["source_mode"], "current")
        self.assertIsNone(payload["meta"]["snapshot_date"])
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["group_type"], group_type)
        self.assertEqual(payload["rows"][0]["group_key"], expected_row_key)
        self.assertEqual(payload["rows"][0]["group"], expected_group)
        self.assertEqual(payload["rows"][0]["coverage_gap_count"], 1)
        self.assertEqual(payload["rows"][0]["coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"missing_bucket_count\": 1", payload["rows"][0]["coverage_gap_ranges_json"])
        self.assertEqual(payload["rows"][0]["movement_pattern_pca"], "jump-led")
        self.assertEqual(payload["rows"][0]["largest_jump_share_pca"], 1.0)
        self.assertEqual(payload["rows"][0]["movement_pattern_mds"], "jump-led")
        self.assertEqual(payload["rows"][0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(payload["rows"][0]["popularity_peak_articles"], 2)
        self.assertEqual(payload["rows"][0]["popularity_first_share"], 0.5)
        self.assertEqual(payload["rows"][0]["popularity_latest_share"], 0.5)
        self.assertEqual(payload["rows"][0]["popularity_share_delta"], 0.0)
        self.assertAlmostEqual(payload["rows"][0]["popularity_recent_share_delta"], 1 / 6)

    def _assert_snapshot_group_temporal_path_summary_csv_filter_parity(
        self,
        *,
        group_type: str,
        group_key: str,
        expected_row_key: str,
        expected_group: str,
    ):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            csv_payload, rows = self._assert_csv_parity(
                f"/api/news/export?artifact=group_temporal_path_summary&format=csv&refresh=1"
                f"&group_type={group_type}&group_key={group_key}&snapshot_date={self.snapshot_date}"
            )
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")
            self.fastapi_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

        self.assertTrue(csv_payload.startswith("group_type,group,group_key,status,reason,"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], group_type)
        self.assertEqual(rows[0]["group_key"], expected_row_key)
        self.assertEqual(rows[0]["group"], expected_group)
        self.assertEqual(rows[0]["coverage_gap_count"], "1")
        self.assertEqual(rows[0]["coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"missing_bucket_count\": 1", rows[0]["coverage_gap_ranges_json"])
        self.assertEqual(rows[0]["movement_pattern_pca"], "jump-led")
        self.assertEqual(rows[0]["largest_jump_share_pca"], "1.0")
        self.assertEqual(rows[0]["movement_pattern_mds"], "jump-led")
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_peak_articles"], "2")
        self.assertEqual(rows[0]["popularity_first_share"], "0.5")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
        self.assertEqual(rows[0]["popularity_share_delta"], "0.0")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)

    def _assert_current_group_temporal_path_summary_csv_filter_parity(
        self,
        *,
        group_type: str,
        group_key: str,
        expected_row_key: str,
        expected_group: str,
    ):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            csv_payload, rows = self._assert_csv_parity(
                f"/api/news/export?artifact=group_temporal_path_summary&format=csv&refresh=1"
                f"&group_type={group_type}&group_key={group_key}"
            )
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get("/api/news/stats?refresh=1")
            self.fastapi_client.get("/api/news/stats?refresh=1")

        self.assertTrue(csv_payload.startswith("group_type,group,group_key,status,reason,"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], group_type)
        self.assertEqual(rows[0]["group_key"], expected_row_key)
        self.assertEqual(rows[0]["group"], expected_group)
        self.assertEqual(rows[0]["coverage_gap_count"], "1")
        self.assertEqual(rows[0]["coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"missing_bucket_count\": 1", rows[0]["coverage_gap_ranges_json"])
        self.assertEqual(rows[0]["movement_pattern_pca"], "jump-led")
        self.assertEqual(rows[0]["largest_jump_share_pca"], "1.0")
        self.assertEqual(rows[0]["movement_pattern_mds"], "jump-led")
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_peak_articles"], "2")
        self.assertEqual(rows[0]["popularity_first_share"], "0.5")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
        self.assertEqual(rows[0]["popularity_share_delta"], "0.0")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)

    def _assert_snapshot_group_temporal_buckets_filter_parity(
        self,
        *,
        group_type: str,
        group_key: str,
        expected_row_key: str,
        expected_group: str,
    ):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            payload = self._assert_json_parity(
                f"/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                f"&group_type={group_type}&group_key={group_key}&snapshot_date={self.snapshot_date}"
            )
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")
            self.fastapi_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

        self.assertEqual(payload["artifact"], "group_temporal_buckets")
        self.assertEqual(payload["filters"], {"group_type": group_type, "group_key": group_key})
        self.assertEqual(payload["meta"]["source_mode"], "snapshot")
        self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
        self.assertEqual(len(payload["rows"]), 3)
        self.assertTrue(all(row["group_type"] == group_type for row in payload["rows"]))
        self.assertTrue(all(row["group_key"] == expected_row_key for row in payload["rows"]))
        self.assertTrue(all(row["group"] == expected_group for row in payload["rows"]))

        sparse_row = next(row for row in payload["rows"] if row["bucket_status"] == "sparse")
        self.assertEqual(sparse_row["bucket_start"], "2026-04-06")
        self.assertEqual(sparse_row["bucket_n_articles"], 1)
        self.assertEqual(sparse_row["group_sparse_bucket_count"], 1)
        self.assertEqual(sparse_row["group_coverage_gap_count"], 1)
        self.assertEqual(sparse_row["group_coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"missing_bucket_count\": 1", sparse_row["group_coverage_gap_ranges_json"])
        self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
        self.assertAlmostEqual(sparse_row["group_popularity_recent_share_delta"], 1 / 6)
        self.assertIn("\"Source A\": 1", sparse_row["bucket_source_counts_json"])

    def _assert_current_group_temporal_buckets_filter_parity(
        self,
        *,
        group_type: str,
        group_key: str,
        expected_row_key: str,
        expected_group: str,
    ):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            payload = self._assert_json_parity(
                f"/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                f"&group_type={group_type}&group_key={group_key}"
            )
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get("/api/news/stats?refresh=1")
            self.fastapi_client.get("/api/news/stats?refresh=1")

        self.assertEqual(payload["artifact"], "group_temporal_buckets")
        self.assertEqual(payload["filters"], {"group_type": group_type, "group_key": group_key})
        self.assertEqual(payload["meta"]["source_mode"], "current")
        self.assertIsNone(payload["meta"]["snapshot_date"])
        self.assertEqual(len(payload["rows"]), 3)
        self.assertTrue(all(row["group_type"] == group_type for row in payload["rows"]))
        self.assertTrue(all(row["group_key"] == expected_row_key for row in payload["rows"]))
        self.assertTrue(all(row["group"] == expected_group for row in payload["rows"]))

        sparse_row = next(row for row in payload["rows"] if row["bucket_status"] == "sparse")
        self.assertEqual(sparse_row["bucket_start"], "2026-04-06")
        self.assertEqual(sparse_row["bucket_n_articles"], 1)
        self.assertEqual(sparse_row["group_sparse_bucket_count"], 1)
        self.assertEqual(sparse_row["group_coverage_gap_count"], 1)
        self.assertEqual(sparse_row["group_coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"missing_bucket_count\": 1", sparse_row["group_coverage_gap_ranges_json"])
        self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
        self.assertAlmostEqual(sparse_row["group_popularity_recent_share_delta"], 1 / 6)
        self.assertIn("\"Source A\": 1", sparse_row["bucket_source_counts_json"])

    def _assert_snapshot_group_temporal_buckets_csv_filter_parity(
        self,
        *,
        group_type: str,
        group_key: str,
        expected_row_key: str,
        expected_group: str,
    ):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            csv_payload, rows = self._assert_csv_parity(
                f"/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                f"&group_type={group_type}&group_key={group_key}&snapshot_date={self.snapshot_date}"
            )
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")
            self.fastapi_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

        self.assertTrue(csv_payload.startswith("group_type,group,group_key,group_status,group_reason,"))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["group_type"] == group_type for row in rows))
        self.assertTrue(all(row["group_key"] == expected_row_key for row in rows))
        self.assertTrue(all(row["group"] == expected_group for row in rows))

        sparse_row = next(row for row in rows if row["bucket_status"] == "sparse")
        self.assertEqual(sparse_row["bucket_start"], "2026-04-06")
        self.assertEqual(sparse_row["bucket_n_articles"], "1")
        self.assertEqual(sparse_row["group_sparse_bucket_count"], "1")
        self.assertEqual(sparse_row["group_coverage_gap_count"], "1")
        self.assertEqual(sparse_row["group_coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"missing_bucket_count\": 1", sparse_row["group_coverage_gap_ranges_json"])
        self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
        self.assertAlmostEqual(float(sparse_row["group_popularity_recent_share_delta"]), 1 / 6)
        self.assertIn("\"Source A\": 1", sparse_row["bucket_source_counts_json"])

    def _assert_current_group_temporal_buckets_csv_filter_parity(
        self,
        *,
        group_type: str,
        group_key: str,
        expected_row_key: str,
        expected_group: str,
    ):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            csv_payload, rows = self._assert_csv_parity(
                f"/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                f"&group_type={group_type}&group_key={group_key}"
            )
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get("/api/news/stats?refresh=1")
            self.fastapi_client.get("/api/news/stats?refresh=1")

        self.assertTrue(csv_payload.startswith("group_type,group,group_key,group_status,group_reason,"))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["group_type"] == group_type for row in rows))
        self.assertTrue(all(row["group_key"] == expected_row_key for row in rows))
        self.assertTrue(all(row["group"] == expected_group for row in rows))

        sparse_row = next(row for row in rows if row["bucket_status"] == "sparse")
        self.assertEqual(sparse_row["bucket_start"], "2026-04-06")
        self.assertEqual(sparse_row["bucket_n_articles"], "1")
        self.assertEqual(sparse_row["group_sparse_bucket_count"], "1")
        self.assertEqual(sparse_row["group_coverage_gap_count"], "1")
        self.assertEqual(sparse_row["group_coverage_gap_labels"], "2026-04-13")
        self.assertIn("\"missing_bucket_count\": 1", sparse_row["group_coverage_gap_ranges_json"])
        self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
        self.assertAlmostEqual(float(sparse_row["group_popularity_recent_share_delta"]), 1 / 6)
        self.assertIn("\"Source A\": 1", sparse_row["bucket_source_counts_json"])

    def _assert_snapshot_group_temporal_popularity_momentum_csv_filter_parity(
        self,
        *,
        group_type: str,
        group_key: str,
        expected_row_key: str,
        expected_group: str,
    ):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            csv_payload, rows = self._assert_csv_parity(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=csv&refresh=1"
                f"&group_type={group_type}&group_key={group_key}&snapshot_date={self.snapshot_date}"
            )
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")
            self.fastapi_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

        self.assertTrue(csv_payload.startswith("group_type,group,group_key,status,reason,"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], group_type)
        self.assertEqual(rows[0]["group_key"], expected_row_key)
        self.assertEqual(rows[0]["group"], expected_group)
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_peak_articles"], "2")
        self.assertEqual(rows[0]["popularity_first_share"], "0.5")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
        self.assertEqual(rows[0]["popularity_share_delta"], "0.0")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(rows[0]["popularity_share_direction"], "flat")
        self.assertEqual(rows[0]["popularity_recent_share_direction"], "rising")
        self.assertEqual(rows[0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertEqual(rows[0]["popularity_first_share_rank_within_type"], "1")
        self.assertEqual(rows[0]["popularity_latest_share_rank_within_type"], "1")
        self.assertEqual(rows[0]["popularity_rank_change_within_type"], "0")
        self.assertEqual(rows[0]["popularity_share_delta_rank_within_type"], "1")
        self.assertEqual(rows[0]["popularity_recent_share_delta_rank_within_type"], "1")

    def _assert_current_group_temporal_popularity_momentum_csv_filter_parity(
        self,
        *,
        group_type: str,
        group_key: str,
        expected_row_key: str,
        expected_group: str,
    ):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            csv_payload, rows = self._assert_csv_parity(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=csv&refresh=1"
                f"&group_type={group_type}&group_key={group_key}"
            )
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get("/api/news/stats?refresh=1")
            self.fastapi_client.get("/api/news/stats?refresh=1")

        self.assertTrue(csv_payload.startswith("group_type,group,group_key,status,reason,"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], group_type)
        self.assertEqual(rows[0]["group_key"], expected_row_key)
        self.assertEqual(rows[0]["group"], expected_group)
        self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(rows[0]["popularity_peak_articles"], "2")
        self.assertEqual(rows[0]["popularity_first_share"], "0.5")
        self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
        self.assertEqual(rows[0]["popularity_share_delta"], "0.0")
        self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        self.assertEqual(rows[0]["popularity_share_direction"], "flat")
        self.assertEqual(rows[0]["popularity_recent_share_direction"], "rising")
        self.assertEqual(rows[0]["popularity_momentum_label"], "rebounding to baseline")
        self.assertEqual(rows[0]["popularity_first_share_rank_within_type"], "1")
        self.assertEqual(rows[0]["popularity_latest_share_rank_within_type"], "1")
        self.assertEqual(rows[0]["popularity_rank_change_within_type"], "0")
        self.assertEqual(rows[0]["popularity_share_delta_rank_within_type"], "1")
        self.assertEqual(rows[0]["popularity_recent_share_delta_rank_within_type"], "1")

    def test_json_route_parity(self):
        paths = [
            "/api/news/digest?refresh=1",
            "/api/news/digest/latest?refresh=1",
            "/api/news/stats?refresh=1",
            "/api/news/upstream?refresh=1",
            "/api/news/export?artifact=source_differentiation_summary&format=json&refresh=1",
            "/health/news-freshness",
            f"/api/news/stats?snapshot_date={self.snapshot_date}",
        ]
        for path in paths:
            with self.subTest(path=path):
                self._assert_json_parity(path)

    def test_snapshot_group_temporal_popularity_momentum_source_filter_parity(self):
        self._assert_snapshot_group_temporal_popularity_momentum_filter_parity(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_popularity_momentum_source_filter_parity(self):
        self._assert_current_group_temporal_popularity_momentum_filter_parity(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_snapshot_group_temporal_popularity_momentum_normalizes_source_group_key_variants(self):
        self._assert_snapshot_group_temporal_popularity_momentum_filter_parity(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_popularity_momentum_normalizes_source_group_key_variants(self):
        self._assert_current_group_temporal_popularity_momentum_filter_parity(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_snapshot_group_temporal_popularity_momentum_tag_filter_parity(self):
        self._assert_snapshot_group_temporal_popularity_momentum_filter_parity(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_current_group_temporal_popularity_momentum_tag_filter_parity(self):
        self._assert_current_group_temporal_popularity_momentum_filter_parity(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_snapshot_group_temporal_popularity_momentum_topic_filter_parity(self):
        self._assert_snapshot_group_temporal_popularity_momentum_filter_parity(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_current_group_temporal_popularity_momentum_topic_filter_parity(self):
        self._assert_current_group_temporal_popularity_momentum_filter_parity(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_current_group_temporal_popularity_momentum_normalizes_multi_word_topic_group_key_parity(self):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(
                json.dumps(_sparse_bucket_temporal_payload_with_multi_word_topic()),
                encoding="utf-8",
            )
            payload = self._assert_json_parity(
                "/api/news/export?artifact=group_temporal_popularity_momentum&format=json&refresh=1"
                "&group_type=topic&group_key=Climate_Risk"
            )
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get("/api/news/stats?refresh=1")
            self.fastapi_client.get("/api/news/stats?refresh=1")

        self.assertEqual(payload["artifact"], "group_temporal_popularity_momentum")
        self.assertEqual(payload["filters"], {"group_type": "topic", "group_key": "Climate_Risk"})
        self.assertEqual(payload["meta"]["source_mode"], "current")
        self.assertIsNone(payload["meta"]["snapshot_date"])
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["group_type"], "topic")
        self.assertEqual(payload["rows"][0]["group_key"], "climate risk")
        self.assertEqual(payload["rows"][0]["group"], "Climate Risk")
        self.assertEqual(payload["rows"][0]["popularity_peak_bucket"], "2026-03-30")
        self.assertEqual(payload["rows"][0]["popularity_peak_articles"], 2)
        self.assertEqual(payload["rows"][0]["popularity_first_share"], 0.5)
        self.assertEqual(payload["rows"][0]["popularity_latest_share"], 0.5)
        self.assertEqual(payload["rows"][0]["popularity_share_delta"], 0.0)
        self.assertAlmostEqual(payload["rows"][0]["popularity_recent_share_delta"], 1 / 6)
        self.assertEqual(payload["rows"][0]["popularity_momentum_label"], "rebounding to baseline")

    def test_snapshot_group_temporal_path_summary_source_filter_parity(self):
        self._assert_snapshot_group_temporal_path_summary_filter_parity(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_snapshot_group_temporal_path_summary_normalizes_source_group_key_variants(self):
        self._assert_snapshot_group_temporal_path_summary_filter_parity(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_path_summary_source_filter_parity(self):
        self._assert_current_group_temporal_path_summary_filter_parity(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_path_summary_normalizes_source_group_key_variants(self):
        self._assert_current_group_temporal_path_summary_filter_parity(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_snapshot_group_temporal_path_summary_tag_filter_parity(self):
        self._assert_snapshot_group_temporal_path_summary_filter_parity(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_current_group_temporal_path_summary_tag_filter_parity(self):
        self._assert_current_group_temporal_path_summary_filter_parity(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_snapshot_group_temporal_path_summary_topic_filter_parity(self):
        self._assert_snapshot_group_temporal_path_summary_filter_parity(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_current_group_temporal_path_summary_topic_filter_parity(self):
        self._assert_current_group_temporal_path_summary_filter_parity(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_snapshot_group_temporal_buckets_source_filter_parity(self):
        self._assert_snapshot_group_temporal_buckets_filter_parity(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_snapshot_group_temporal_buckets_normalizes_source_group_key_variants(self):
        self._assert_snapshot_group_temporal_buckets_filter_parity(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_buckets_source_filter_parity(self):
        self._assert_current_group_temporal_buckets_filter_parity(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_buckets_normalizes_source_group_key_variants(self):
        self._assert_current_group_temporal_buckets_filter_parity(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_buckets_supports_bucket_label_filter_parity(self):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            payload = self._assert_json_parity(
                "/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                "&group_type=source&group_key=Source_A&bucket_label=2026-04-06"
            )
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get("/api/news/stats?refresh=1")
            self.fastapi_client.get("/api/news/stats?refresh=1")

        self.assertEqual(payload["artifact"], "group_temporal_buckets")
        self.assertEqual(
            payload["filters"],
            {"group_type": "source", "group_key": "Source_A", "bucket_label": "2026-04-06"},
        )
        self.assertEqual(payload["meta"]["source_mode"], "current")
        self.assertIsNone(payload["meta"]["snapshot_date"])
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["group_type"], "source")
        self.assertEqual(payload["rows"][0]["group_key"], "source a")
        self.assertEqual(payload["rows"][0]["group"], "Source A")
        self.assertEqual(payload["rows"][0]["bucket_label"], "2026-04-06")
        self.assertEqual(payload["rows"][0]["bucket_status"], "sparse")
        self.assertEqual(payload["rows"][0]["bucket_n_articles"], 1)

    def test_current_group_temporal_buckets_supports_bucket_only_label_filter_parity(self):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            payload = self._assert_json_parity(
                "/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                "&bucket_label=2026-04-06"
            )
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get("/api/news/stats?refresh=1")
            self.fastapi_client.get("/api/news/stats?refresh=1")

        self.assertEqual(payload["artifact"], "group_temporal_buckets")
        self.assertEqual(payload["filters"], {"bucket_label": "2026-04-06"})
        self.assertEqual(payload["meta"]["source_mode"], "current")
        self.assertIsNone(payload["meta"]["snapshot_date"])
        self.assertEqual(len(payload["rows"]), 6)
        self.assertEqual({row["group_type"] for row in payload["rows"]}, {"source", "topic", "tag"})
        self.assertEqual({row["group_key"] for row in payload["rows"]}, {"source a", "source b", "policy", "markets", "risk", "growth"})
        self.assertTrue(all(row["bucket_label"] == "2026-04-06" for row in payload["rows"]))
        self.assertEqual(sum(1 for row in payload["rows"] if row["bucket_status"] == "sparse"), 3)

    def test_current_group_temporal_buckets_csv_supports_bucket_only_label_filter_parity(self):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            _csv_payload, rows = self._assert_csv_parity(
                "/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                "&bucket_label=2026-04-06"
            )
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get("/api/news/stats?refresh=1")
            self.fastapi_client.get("/api/news/stats?refresh=1")

        self.assertEqual(len(rows), 6)
        self.assertEqual({row["group_type"] for row in rows}, {"source", "topic", "tag"})
        self.assertEqual({row["group_key"] for row in rows}, {"source a", "source b", "policy", "markets", "risk", "growth"})
        self.assertTrue(all(row["bucket_label"] == "2026-04-06" for row in rows))
        self.assertEqual(sum(1 for row in rows if row["bucket_status"] == "sparse"), 3)

    def test_current_group_temporal_buckets_supports_bucket_label_filter_for_multi_word_topic_keys_parity(self):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(
                json.dumps(_sparse_bucket_temporal_payload_with_multi_word_topic()),
                encoding="utf-8",
            )
            payload = self._assert_json_parity(
                "/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                "&group_type=topic&group_key=Climate_Risk&bucket_label=2026-03-30"
            )
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get("/api/news/stats?refresh=1")
            self.fastapi_client.get("/api/news/stats?refresh=1")

        self.assertEqual(payload["artifact"], "group_temporal_buckets")
        self.assertEqual(
            payload["filters"],
            {"group_type": "topic", "group_key": "Climate_Risk", "bucket_label": "2026-03-30"},
        )
        self.assertEqual(payload["meta"]["source_mode"], "current")
        self.assertIsNone(payload["meta"]["snapshot_date"])
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["group_type"], "topic")
        self.assertEqual(payload["rows"][0]["group_key"], "climate risk")
        self.assertEqual(payload["rows"][0]["group"], "Climate Risk")
        self.assertEqual(payload["rows"][0]["bucket_label"], "2026-03-30")
        self.assertEqual(payload["rows"][0]["bucket_status"], "ok")
        self.assertEqual(payload["rows"][0]["bucket_n_articles"], 2)

    def test_current_group_temporal_buckets_csv_supports_bucket_label_filter_for_multi_word_topic_keys_parity(self):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(
                json.dumps(_sparse_bucket_temporal_payload_with_multi_word_topic()),
                encoding="utf-8",
            )
            _csv_payload, rows = self._assert_csv_parity(
                "/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                "&group_type=topic&group_key=Climate_Risk&bucket_label=2026-03-30"
            )
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get("/api/news/stats?refresh=1")
            self.fastapi_client.get("/api/news/stats?refresh=1")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "topic")
        self.assertEqual(rows[0]["group_key"], "climate risk")
        self.assertEqual(rows[0]["group"], "Climate Risk")
        self.assertEqual(rows[0]["bucket_label"], "2026-03-30")
        self.assertEqual(rows[0]["bucket_status"], "ok")
        self.assertEqual(rows[0]["bucket_n_articles"], "2")

    def test_snapshot_group_temporal_buckets_tag_filter_parity(self):
        self._assert_snapshot_group_temporal_buckets_filter_parity(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_current_group_temporal_buckets_tag_filter_parity(self):
        self._assert_current_group_temporal_buckets_filter_parity(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_snapshot_group_temporal_buckets_topic_filter_parity(self):
        self._assert_snapshot_group_temporal_buckets_filter_parity(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_snapshot_group_temporal_buckets_supports_bucket_label_filter_for_multi_word_topic_keys_parity(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(
                json.dumps(_sparse_bucket_temporal_payload_with_multi_word_topic()),
                encoding="utf-8",
            )
            payload = self._assert_json_parity(
                f"/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                f"&group_type=topic&group_key=Climate_Risk&bucket_label=2026-03-30&snapshot_date={self.snapshot_date}"
            )
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")
            self.fastapi_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

        self.assertEqual(payload["artifact"], "group_temporal_buckets")
        self.assertEqual(
            payload["filters"],
            {"group_type": "topic", "group_key": "Climate_Risk", "bucket_label": "2026-03-30"},
        )
        self.assertEqual(payload["meta"]["source_mode"], "snapshot")
        self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["group_type"], "topic")
        self.assertEqual(payload["rows"][0]["group_key"], "climate risk")
        self.assertEqual(payload["rows"][0]["group"], "Climate Risk")
        self.assertEqual(payload["rows"][0]["bucket_label"], "2026-03-30")
        self.assertEqual(payload["rows"][0]["bucket_status"], "ok")
        self.assertEqual(payload["rows"][0]["bucket_n_articles"], 2)

    def test_snapshot_group_temporal_buckets_supports_bucket_only_label_filter_parity(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            payload = self._assert_json_parity(
                f"/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                f"&bucket_label=2026-04-06&snapshot_date={self.snapshot_date}"
            )
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")
            self.fastapi_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

        self.assertEqual(payload["artifact"], "group_temporal_buckets")
        self.assertEqual(payload["filters"], {"bucket_label": "2026-04-06"})
        self.assertEqual(payload["meta"]["source_mode"], "snapshot")
        self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
        self.assertEqual(len(payload["rows"]), 6)
        self.assertEqual({row["group_type"] for row in payload["rows"]}, {"source", "topic", "tag"})
        self.assertEqual({row["group_key"] for row in payload["rows"]}, {"source a", "source b", "policy", "markets", "risk", "growth"})
        self.assertTrue(all(row["bucket_label"] == "2026-04-06" for row in payload["rows"]))
        self.assertEqual(sum(1 for row in payload["rows"] if row["bucket_status"] == "sparse"), 3)

    def test_current_group_temporal_buckets_topic_filter_parity(self):
        self._assert_current_group_temporal_buckets_filter_parity(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_snapshot_group_temporal_buckets_source_filter_csv_parity(self):
        self._assert_snapshot_group_temporal_buckets_csv_filter_parity(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_snapshot_group_temporal_buckets_csv_normalizes_source_group_key_variants(self):
        self._assert_snapshot_group_temporal_buckets_csv_filter_parity(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_buckets_source_filter_csv_parity(self):
        self._assert_current_group_temporal_buckets_csv_filter_parity(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_buckets_csv_normalizes_source_group_key_variants(self):
        self._assert_current_group_temporal_buckets_csv_filter_parity(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_snapshot_group_temporal_buckets_tag_filter_csv_parity(self):
        self._assert_snapshot_group_temporal_buckets_csv_filter_parity(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_current_group_temporal_buckets_tag_filter_csv_parity(self):
        self._assert_current_group_temporal_buckets_csv_filter_parity(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_snapshot_group_temporal_buckets_topic_filter_csv_parity(self):
        self._assert_snapshot_group_temporal_buckets_csv_filter_parity(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_current_group_temporal_buckets_topic_filter_csv_parity(self):
        self._assert_current_group_temporal_buckets_csv_filter_parity(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_snapshot_group_temporal_buckets_csv_supports_bucket_label_filter_for_multi_word_topic_keys_parity(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(
                json.dumps(_sparse_bucket_temporal_payload_with_multi_word_topic()),
                encoding="utf-8",
            )
            _csv_payload, rows = self._assert_csv_parity(
                f"/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                f"&group_type=topic&group_key=Climate_Risk&bucket_label=2026-03-30&snapshot_date={self.snapshot_date}"
            )
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")
            self.fastapi_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_type"], "topic")
        self.assertEqual(rows[0]["group_key"], "climate risk")
        self.assertEqual(rows[0]["group"], "Climate Risk")
        self.assertEqual(rows[0]["bucket_label"], "2026-03-30")
        self.assertEqual(rows[0]["bucket_status"], "ok")
        self.assertEqual(rows[0]["bucket_n_articles"], "2")

    def test_snapshot_group_temporal_buckets_csv_supports_bucket_only_label_filter_parity(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            _csv_payload, rows = self._assert_csv_parity(
                f"/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                f"&bucket_label=2026-04-06&snapshot_date={self.snapshot_date}"
            )
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.flask_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")
            self.fastapi_client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

        self.assertEqual(len(rows), 6)
        self.assertEqual({row["group_type"] for row in rows}, {"source", "topic", "tag"})
        self.assertEqual({row["group_key"] for row in rows}, {"source a", "source b", "policy", "markets", "risk", "growth"})
        self.assertTrue(all(row["bucket_label"] == "2026-04-06" for row in rows))
        self.assertEqual(sum(1 for row in rows if row["bucket_status"] == "sparse"), 3)

    def test_snapshot_group_temporal_path_summary_source_filter_csv_parity(self):
        self._assert_snapshot_group_temporal_path_summary_csv_filter_parity(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_snapshot_group_temporal_path_summary_csv_normalizes_source_group_key_variants(self):
        self._assert_snapshot_group_temporal_path_summary_csv_filter_parity(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_path_summary_source_filter_csv_parity(self):
        self._assert_current_group_temporal_path_summary_csv_filter_parity(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_path_summary_csv_normalizes_source_group_key_variants(self):
        self._assert_current_group_temporal_path_summary_csv_filter_parity(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_snapshot_group_temporal_path_summary_tag_filter_csv_parity(self):
        self._assert_snapshot_group_temporal_path_summary_csv_filter_parity(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_current_group_temporal_path_summary_tag_filter_csv_parity(self):
        self._assert_current_group_temporal_path_summary_csv_filter_parity(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_snapshot_group_temporal_path_summary_topic_filter_csv_parity(self):
        self._assert_snapshot_group_temporal_path_summary_csv_filter_parity(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_current_group_temporal_path_summary_topic_filter_csv_parity(self):
        self._assert_current_group_temporal_path_summary_csv_filter_parity(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_snapshot_group_temporal_popularity_momentum_source_filter_csv_parity(self):
        self._assert_snapshot_group_temporal_popularity_momentum_csv_filter_parity(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_popularity_momentum_source_filter_csv_parity(self):
        self._assert_current_group_temporal_popularity_momentum_csv_filter_parity(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_snapshot_group_temporal_popularity_momentum_csv_normalizes_source_group_key_variants(self):
        self._assert_snapshot_group_temporal_popularity_momentum_csv_filter_parity(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_current_group_temporal_popularity_momentum_csv_normalizes_source_group_key_variants(self):
        self._assert_current_group_temporal_popularity_momentum_csv_filter_parity(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_snapshot_group_temporal_popularity_momentum_tag_filter_csv_parity(self):
        self._assert_snapshot_group_temporal_popularity_momentum_csv_filter_parity(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_current_group_temporal_popularity_momentum_tag_filter_csv_parity(self):
        self._assert_current_group_temporal_popularity_momentum_csv_filter_parity(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_snapshot_group_temporal_popularity_momentum_topic_filter_csv_parity(self):
        self._assert_snapshot_group_temporal_popularity_momentum_csv_filter_parity(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_current_group_temporal_popularity_momentum_topic_filter_csv_parity(self):
        self._assert_current_group_temporal_popularity_momentum_csv_filter_parity(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )


if __name__ == "__main__":
    unittest.main()
