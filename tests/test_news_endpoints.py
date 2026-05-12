import csv
import io
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask

from src.api.news_endpoints import register_news_endpoints


NOW_UTC = datetime.now(timezone.utc)
NOW_UTC_ISO = NOW_UTC.isoformat().replace("+00:00", "Z")
DIGEST_UTC_ISO = (NOW_UTC - timedelta(minutes=3)).isoformat().replace("+00:00", "Z")


SAMPLE_PAYLOAD = {
    "schema_version": "1.0",
    "generated_at": NOW_UTC_ISO,
    "contract": "rss_pipeline_precomputed",
    "digest": {
        "generated_at": DIGEST_UTC_ISO,
        "run_id": "digest-abc123",
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
                "id": f"sparse-bucket-flask-{idx}",
                "title": f"Sparse bucket flask {idx}",
                "link": f"https://example.com/sparse-bucket/{idx}",
                "published": f"{published}T00:00:00Z",
                "summary": f"Summary {idx}",
                "ai_summary": f"AI Summary {idx}",
                "ai_tags": ai_tags,
                "topic_tags": topic_tags,
                "source": {"id": source_name.lower().replace(" ", "-"), "name": source_name},
                "feed": {"name": "Feed", "url": "https://example.com/feed"},
                "scraped": {"title": f"Sparse bucket flask {idx}", "body_text": "Body"},
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
            "run_id": "digest-flask-sparse-bucket",
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
        if article.get("topic_tags") == ["Policy"]:
            article["topic_tags"] = ["Climate Risk"]
    return payload


class NewsEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="rss-news-endpoints-"))
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

        app = Flask(__name__)
        register_news_endpoints(app)
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_digest_and_filters(self):
        response = self.client.get("/api/news/digest")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["data"]), 2)
        self.assertEqual(payload["meta"]["input_articles_count"], 3)
        self.assertEqual(payload["meta"]["excluded_unscraped_articles"], 1)

        response = self.client.get("/api/news/digest?tag=openai")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()["data"]), 2)

        response = self.client.get("/api/news/digest?source=pbs")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()["data"]), 1)

    def test_latest_and_bad_limit(self):
        latest = self.client.get("/api/news/digest/latest")
        self.assertEqual(latest.status_code, 200)
        self.assertEqual(latest.get_json()["data"]["id"], "a-1")
        self.assertEqual(latest.get_json()["meta"]["excluded_unscraped_articles"], 1)

        bad_limit = self.client.get("/api/news/digest?limit=0")
        self.assertEqual(bad_limit.status_code, 400)

    def test_snapshot_date_routing_and_meta(self):
        response = self.client.get(f"/api/news/digest?snapshot_date={self.snapshot_date}")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["filters"]["snapshot_date"], self.snapshot_date)
        self.assertEqual(payload["meta"]["source_mode"], "snapshot")
        self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
        self.assertIn(f"rss_openai_daily_{self.snapshot_date}.json", payload["meta"]["source_url"])

        latest = self.client.get(f"/api/news/digest/latest?snapshot_date={self.snapshot_date}")
        self.assertEqual(latest.status_code, 200)
        self.assertEqual(latest.get_json()["meta"]["source_mode"], "snapshot")

        stats = self.client.get(f"/api/news/stats?snapshot_date={self.snapshot_date}")
        self.assertEqual(stats.status_code, 200)
        self.assertEqual(stats.get_json()["meta"]["source_mode"], "snapshot")

    def test_snapshot_date_validation_and_missing_file(self):
        bad_date = self.client.get("/api/news/digest?snapshot_date=2026/03/10")
        self.assertEqual(bad_date.status_code, 400)
        self.assertEqual(bad_date.get_json()["status"], "bad_request")

        missing_snapshot = self.client.get("/api/news/digest?snapshot_date=2026-03-09")
        self.assertEqual(missing_snapshot.status_code, 404)
        self.assertEqual(missing_snapshot.get_json()["status"], "not_found")

        missing_stats = self.client.get("/api/news/stats?snapshot_date=2026-03-09")
        self.assertEqual(missing_stats.status_code, 404)
        self.assertEqual(missing_stats.get_json()["status"], "not_found")

    def _assert_export_current_group_temporal_popularity_momentum_filter(
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
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=json&refresh=1"
                f"&group_type={group_type}&group_key={group_key}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
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
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get("/api/news/stats?refresh=1")

    def _assert_export_current_group_temporal_popularity_momentum_csv_filter(
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
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=csv&refresh=1"
                f"&group_type={group_type}&group_key={group_key}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_popularity_momentum.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
            self.assertIsNotNone(reader.fieldnames)
            self.assertIn("group_type", reader.fieldnames)
            self.assertIn("group_key", reader.fieldnames)
            self.assertIn("popularity_peak_bucket", reader.fieldnames)
            self.assertIn("popularity_momentum_label", reader.fieldnames)
            self.assertIn("popularity_recent_share_delta_rank_within_type", reader.fieldnames)

            rows = list(reader)
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
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get("/api/news/stats?refresh=1")

    def _assert_export_current_group_temporal_buckets_filter(
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
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                f"&group_type={group_type}&group_key={group_key}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
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
            self.assertIn('"label": "2026-04-13"', sparse_row["group_coverage_gap_ranges_json"])
            self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
            self.assertAlmostEqual(sparse_row["group_popularity_recent_share_delta"], 1 / 6)
            self.assertIn('"Source A": 1', sparse_row["bucket_source_counts_json"])
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get("/api/news/stats?refresh=1")

    def _assert_export_current_group_temporal_buckets_csv_filter(
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
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                f"&group_type={group_type}&group_key={group_key}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_buckets.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
            self.assertIsNotNone(reader.fieldnames)
            self.assertIn("group_type", reader.fieldnames)
            self.assertIn("group_key", reader.fieldnames)
            self.assertIn("group_sparse_bucket_count", reader.fieldnames)
            self.assertIn("group_coverage_gap_labels", reader.fieldnames)
            self.assertIn("group_popularity_recent_share_delta", reader.fieldnames)
            self.assertIn("bucket_source_counts_json", reader.fieldnames)

            rows = list(reader)
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
            self.assertIn('"label": "2026-04-13"', sparse_row["group_coverage_gap_ranges_json"])
            self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
            self.assertAlmostEqual(float(sparse_row["group_popularity_recent_share_delta"]), 1 / 6)
            self.assertIn('"Source A": 1', sparse_row["bucket_source_counts_json"])
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get("/api/news/stats?refresh=1")

    def _assert_export_current_group_temporal_path_summary_filter(
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
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_path_summary&format=json&refresh=1"
                f"&group_type={group_type}&group_key={group_key}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
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
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get("/api/news/stats?refresh=1")

    def _assert_export_current_group_temporal_path_summary_csv_filter(
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
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_path_summary&format=csv&refresh=1"
                f"&group_type={group_type}&group_key={group_key}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_path_summary.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
            self.assertIsNotNone(reader.fieldnames)
            self.assertIn("group_type", reader.fieldnames)
            self.assertIn("group_key", reader.fieldnames)
            self.assertIn("coverage_gap_labels", reader.fieldnames)
            self.assertIn("coverage_gap_ranges_json", reader.fieldnames)
            self.assertIn("movement_pattern_pca", reader.fieldnames)
            self.assertIn("popularity_recent_share_delta", reader.fieldnames)

            rows = list(reader)
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
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get("/api/news/stats?refresh=1")

    def _assert_export_snapshot_group_temporal_path_summary_filter(
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
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_path_summary&format=json&refresh=1"
                f"&group_type={group_type}&group_key={group_key}&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
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
            self.assertIn('"label": "2026-04-13"', payload["rows"][0]["coverage_gap_ranges_json"])
            self.assertEqual(payload["rows"][0]["bucket_granularity"], "week")
            self.assertEqual(payload["rows"][0]["popularity_peak_bucket"], "2026-03-30")
            self.assertEqual(payload["rows"][0]["popularity_peak_articles"], 2)
            self.assertEqual(payload["rows"][0]["popularity_first_share"], 0.5)
            self.assertEqual(payload["rows"][0]["popularity_latest_share"], 0.5)
            self.assertAlmostEqual(payload["rows"][0]["popularity_recent_share_delta"], 1 / 6)
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_popularity_momentum_keeps_group_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=json&refresh=1"
                f"&group_type=tag&group_key=risk&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
            self.assertEqual(payload["artifact"], "group_temporal_popularity_momentum")
            self.assertEqual(payload["filters"], {"group_type": "tag", "group_key": "risk"})
            self.assertEqual(payload["meta"]["source_mode"], "snapshot")
            self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
            self.assertEqual(len(payload["rows"]), 1)
            self.assertEqual(payload["rows"][0]["group"], "Risk")
            self.assertEqual(payload["rows"][0]["popularity_momentum_label"], "rebounding to baseline")
            self.assertAlmostEqual(payload["rows"][0]["popularity_recent_share_delta"], 1 / 6)
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_popularity_momentum_keeps_source_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=json&refresh=1"
                f"&group_type=source&group_key=source-a&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
            self.assertEqual(payload["artifact"], "group_temporal_popularity_momentum")
            self.assertEqual(payload["filters"], {"group_type": "source", "group_key": "source-a"})
            self.assertEqual(payload["meta"]["source_mode"], "snapshot")
            self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
            self.assertEqual(len(payload["rows"]), 1)
            self.assertEqual(payload["rows"][0]["group_type"], "source")
            self.assertEqual(payload["rows"][0]["group_key"], "source a")
            self.assertEqual(payload["rows"][0]["group"], "Source A")
            self.assertEqual(payload["rows"][0]["popularity_peak_bucket"], "2026-03-30")
            self.assertEqual(payload["rows"][0]["popularity_peak_articles"], 2)
            self.assertEqual(payload["rows"][0]["popularity_first_share"], 0.5)
            self.assertEqual(payload["rows"][0]["popularity_latest_share"], 0.5)
            self.assertEqual(payload["rows"][0]["popularity_share_delta"], 0.0)
            self.assertAlmostEqual(payload["rows"][0]["popularity_recent_share_delta"], 1 / 6)
            self.assertEqual(payload["rows"][0]["popularity_momentum_label"], "rebounding to baseline")
            self.assertEqual(payload["rows"][0]["popularity_share_delta_rank_within_type"], 1)
            self.assertEqual(payload["rows"][0]["popularity_recent_share_delta_rank_within_type"], 1)
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_popularity_momentum_normalizes_source_group_key_variants(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=json&refresh=1"
                f"&group_type=source&group_key=Source_A&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
            self.assertEqual(payload["artifact"], "group_temporal_popularity_momentum")
            self.assertEqual(payload["filters"], {"group_type": "source", "group_key": "Source_A"})
            self.assertEqual(payload["meta"]["source_mode"], "snapshot")
            self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
            self.assertEqual(len(payload["rows"]), 1)
            self.assertEqual(payload["rows"][0]["group_type"], "source")
            self.assertEqual(payload["rows"][0]["group_key"], "source a")
            self.assertEqual(payload["rows"][0]["group"], "Source A")
            self.assertEqual(payload["rows"][0]["popularity_peak_bucket"], "2026-03-30")
            self.assertEqual(payload["rows"][0]["popularity_peak_articles"], 2)
            self.assertEqual(payload["rows"][0]["popularity_first_share"], 0.5)
            self.assertEqual(payload["rows"][0]["popularity_latest_share"], 0.5)
            self.assertEqual(payload["rows"][0]["popularity_share_delta"], 0.0)
            self.assertAlmostEqual(payload["rows"][0]["popularity_recent_share_delta"], 1 / 6)
            self.assertEqual(payload["rows"][0]["popularity_momentum_label"], "rebounding to baseline")
            self.assertEqual(payload["rows"][0]["popularity_share_delta_rank_within_type"], 1)
            self.assertEqual(payload["rows"][0]["popularity_recent_share_delta_rank_within_type"], 1)
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_popularity_momentum_keeps_topic_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=json&refresh=1"
                f"&group_type=topic&group_key=policy&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
            self.assertEqual(payload["artifact"], "group_temporal_popularity_momentum")
            self.assertEqual(payload["filters"], {"group_type": "topic", "group_key": "policy"})
            self.assertEqual(payload["meta"]["source_mode"], "snapshot")
            self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
            self.assertEqual(len(payload["rows"]), 1)
            self.assertEqual(payload["rows"][0]["group_type"], "topic")
            self.assertEqual(payload["rows"][0]["group_key"], "policy")
            self.assertEqual(payload["rows"][0]["group"], "Policy")
            self.assertEqual(payload["rows"][0]["popularity_peak_bucket"], "2026-03-30")
            self.assertEqual(payload["rows"][0]["popularity_peak_articles"], 2)
            self.assertEqual(payload["rows"][0]["popularity_first_share"], 0.5)
            self.assertEqual(payload["rows"][0]["popularity_latest_share"], 0.5)
            self.assertEqual(payload["rows"][0]["popularity_share_delta"], 0.0)
            self.assertAlmostEqual(payload["rows"][0]["popularity_recent_share_delta"], 1 / 6)
            self.assertEqual(payload["rows"][0]["popularity_momentum_label"], "rebounding to baseline")
            self.assertEqual(payload["rows"][0]["popularity_share_delta_rank_within_type"], 1)
            self.assertEqual(payload["rows"][0]["popularity_recent_share_delta_rank_within_type"], 1)
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_current_group_temporal_popularity_momentum_keeps_group_filters(self):
        self._assert_export_current_group_temporal_popularity_momentum_filter(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_export_current_group_temporal_popularity_momentum_keeps_source_filters(self):
        self._assert_export_current_group_temporal_popularity_momentum_filter(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_current_group_temporal_popularity_momentum_normalizes_source_group_key_variants(self):
        self._assert_export_current_group_temporal_popularity_momentum_filter(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_current_group_temporal_popularity_momentum_keeps_topic_filters(self):
        self._assert_export_current_group_temporal_popularity_momentum_filter(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_export_snapshot_group_temporal_popularity_momentum_csv_keeps_group_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=csv&refresh=1"
                f"&group_type=tag&group_key=risk&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_popularity_momentum.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
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
            self.assertEqual(rows[0]["popularity_momentum_label"], "rebounding to baseline")
            self.assertEqual(rows[0]["popularity_share_delta"], "0.0")
            self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
            self.assertEqual(rows[0]["popularity_share_delta_rank_within_type"], "1")
            self.assertEqual(rows[0]["popularity_recent_share_delta_rank_within_type"], "1")
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_popularity_momentum_csv_keeps_source_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=csv&refresh=1"
                f"&group_type=source&group_key=source-a&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_popularity_momentum.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
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
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_popularity_momentum_csv_normalizes_source_group_key_variants(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=csv&refresh=1"
                f"&group_type=source&group_key=Source_A&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_popularity_momentum.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
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
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_popularity_momentum_csv_keeps_topic_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_popularity_momentum&format=csv&refresh=1"
                f"&group_type=topic&group_key=policy&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_popularity_momentum.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
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
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_current_group_temporal_popularity_momentum_csv_keeps_group_filters(self):
        self._assert_export_current_group_temporal_popularity_momentum_csv_filter(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_export_current_group_temporal_popularity_momentum_csv_keeps_source_filters(self):
        self._assert_export_current_group_temporal_popularity_momentum_csv_filter(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_current_group_temporal_popularity_momentum_csv_normalizes_source_group_key_variants(self):
        self._assert_export_current_group_temporal_popularity_momentum_csv_filter(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_current_group_temporal_popularity_momentum_csv_keeps_topic_filters(self):
        self._assert_export_current_group_temporal_popularity_momentum_csv_filter(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_export_current_group_temporal_buckets_keeps_group_filters(self):
        self._assert_export_current_group_temporal_buckets_filter(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_export_current_group_temporal_buckets_keeps_source_filters(self):
        self._assert_export_current_group_temporal_buckets_filter(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_current_group_temporal_buckets_normalizes_source_group_key_variants(self):
        self._assert_export_current_group_temporal_buckets_filter(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_current_group_temporal_buckets_supports_bucket_label_filters(self):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                "/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                "&group_type=source&group_key=Source_A&bucket_label=2026-04-06"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
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
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get("/api/news/stats?refresh=1")

    def test_export_current_group_temporal_buckets_supports_bucket_label_filters_without_group_key(self):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                "/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                "&bucket_label=2026-04-06"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
            self.assertEqual(payload["artifact"], "group_temporal_buckets")
            self.assertEqual(payload["filters"], {"bucket_label": "2026-04-06"})
            self.assertEqual(payload["meta"]["source_mode"], "current")
            self.assertIsNone(payload["meta"]["snapshot_date"])
            self.assertEqual(len(payload["rows"]), 6)
            self.assertEqual({row["group_type"] for row in payload["rows"]}, {"source", "topic", "tag"})
            self.assertEqual({row["group_key"] for row in payload["rows"]}, {"source a", "source b", "policy", "markets", "risk", "growth"})
            self.assertTrue(all(row["bucket_label"] == "2026-04-06" for row in payload["rows"]))
            self.assertEqual(sum(1 for row in payload["rows"] if row["bucket_status"] == "sparse"), 3)
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get("/api/news/stats?refresh=1")

    def test_export_current_group_temporal_buckets_csv_supports_bucket_label_filters_without_group_key(self):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                "/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                "&bucket_label=2026-04-06"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")

            rows = list(csv.DictReader(io.StringIO(exported.get_data(as_text=True))))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["group_type"] for row in rows}, {"source", "topic", "tag"})
            self.assertEqual(
                {row["group_key"] for row in rows},
                {"source a", "source b", "policy", "markets", "risk", "growth"},
            )
            self.assertTrue(all(row["bucket_label"] == "2026-04-06" for row in rows))
            self.assertEqual(sum(1 for row in rows if row["bucket_status"] == "sparse"), 3)
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get("/api/news/stats?refresh=1")

    def test_export_current_group_temporal_buckets_supports_bucket_label_filters_for_multi_word_topic_keys(self):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(
                json.dumps(_sparse_bucket_temporal_payload_with_multi_word_topic()),
                encoding="utf-8",
            )
            exported = self.client.get(
                "/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                "&group_type=topic&group_key=Climate_Risk&bucket_label=2026-03-30"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
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
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get("/api/news/stats?refresh=1")

    def test_export_current_group_temporal_buckets_csv_supports_bucket_label_filters_for_multi_word_topic_keys(self):
        original_payload = self.current_payload_path.read_text(encoding="utf-8")
        try:
            self.current_payload_path.write_text(
                json.dumps(_sparse_bucket_temporal_payload_with_multi_word_topic()),
                encoding="utf-8",
            )
            exported = self.client.get(
                "/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                "&group_type=topic&group_key=Climate_Risk&bucket_label=2026-03-30"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")

            rows = list(csv.DictReader(io.StringIO(exported.get_data(as_text=True))))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["group_type"], "topic")
            self.assertEqual(rows[0]["group_key"], "climate risk")
            self.assertEqual(rows[0]["group"], "Climate Risk")
            self.assertEqual(rows[0]["bucket_label"], "2026-03-30")
            self.assertEqual(rows[0]["bucket_status"], "ok")
            self.assertEqual(rows[0]["bucket_n_articles"], "2")
        finally:
            self.current_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get("/api/news/stats?refresh=1")

    def test_export_current_group_temporal_buckets_keeps_topic_filters(self):
        self._assert_export_current_group_temporal_buckets_filter(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_export_current_group_temporal_buckets_csv_keeps_group_filters(self):
        self._assert_export_current_group_temporal_buckets_csv_filter(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_export_current_group_temporal_buckets_csv_keeps_source_filters(self):
        self._assert_export_current_group_temporal_buckets_csv_filter(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_current_group_temporal_buckets_csv_normalizes_source_group_key_variants(self):
        self._assert_export_current_group_temporal_buckets_csv_filter(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_current_group_temporal_buckets_csv_keeps_topic_filters(self):
        self._assert_export_current_group_temporal_buckets_csv_filter(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_export_current_group_temporal_path_summary_keeps_group_filters(self):
        self._assert_export_current_group_temporal_path_summary_filter(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_export_current_group_temporal_path_summary_keeps_source_filters(self):
        self._assert_export_current_group_temporal_path_summary_filter(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_current_group_temporal_path_summary_normalizes_source_group_key_variants(self):
        self._assert_export_current_group_temporal_path_summary_filter(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_current_group_temporal_path_summary_keeps_topic_filters(self):
        self._assert_export_current_group_temporal_path_summary_filter(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_export_current_group_temporal_path_summary_csv_keeps_group_filters(self):
        self._assert_export_current_group_temporal_path_summary_csv_filter(
            group_type="tag",
            group_key="risk",
            expected_row_key="risk",
            expected_group="Risk",
        )

    def test_export_current_group_temporal_path_summary_csv_keeps_source_filters(self):
        self._assert_export_current_group_temporal_path_summary_csv_filter(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_current_group_temporal_path_summary_csv_normalizes_source_group_key_variants(self):
        self._assert_export_current_group_temporal_path_summary_csv_filter(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_current_group_temporal_path_summary_csv_keeps_topic_filters(self):
        self._assert_export_current_group_temporal_path_summary_csv_filter(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_export_snapshot_group_temporal_path_summary_keeps_group_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_path_summary&format=json&refresh=1"
                f"&group_type=tag&group_key=risk&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
            self.assertEqual(payload["artifact"], "group_temporal_path_summary")
            self.assertEqual(payload["filters"], {"group_type": "tag", "group_key": "risk"})
            self.assertEqual(payload["meta"]["source_mode"], "snapshot")
            self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
            self.assertEqual(len(payload["rows"]), 1)
            self.assertEqual(payload["rows"][0]["group"], "Risk")
            self.assertEqual(payload["rows"][0]["coverage_gap_count"], 1)
            self.assertEqual(payload["rows"][0]["coverage_gap_labels"], "2026-04-13")
            self.assertIn('"label": "2026-04-13"', payload["rows"][0]["coverage_gap_ranges_json"])
            self.assertEqual(payload["rows"][0]["popularity_peak_bucket"], "2026-03-30")
            self.assertAlmostEqual(payload["rows"][0]["popularity_recent_share_delta"], 1 / 6)
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_path_summary_keeps_source_filters(self):
        self._assert_export_snapshot_group_temporal_path_summary_filter(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_snapshot_group_temporal_path_summary_normalizes_source_group_key_variants(self):
        self._assert_export_snapshot_group_temporal_path_summary_filter(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_snapshot_group_temporal_path_summary_keeps_topic_filters(self):
        self._assert_export_snapshot_group_temporal_path_summary_filter(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_export_snapshot_group_temporal_buckets_keeps_group_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                f"&group_type=tag&group_key=risk&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
            self.assertEqual(payload["artifact"], "group_temporal_buckets")
            self.assertEqual(payload["filters"], {"group_type": "tag", "group_key": "risk"})
            self.assertEqual(payload["meta"]["source_mode"], "snapshot")
            self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
            self.assertEqual(len(payload["rows"]), 3)
            self.assertTrue(all(row["group"] == "Risk" for row in payload["rows"]))

            sparse_row = next(row for row in payload["rows"] if row["bucket_status"] == "sparse")
            self.assertEqual(sparse_row["bucket_start"], "2026-04-06")
            self.assertEqual(sparse_row["bucket_n_articles"], 1)
            self.assertEqual(sparse_row["group_sparse_bucket_count"], 1)
            self.assertEqual(sparse_row["group_coverage_gap_count"], 1)
            self.assertEqual(sparse_row["group_coverage_gap_labels"], "2026-04-13")
            self.assertIn('"label": "2026-04-13"', sparse_row["group_coverage_gap_ranges_json"])
            self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
            self.assertAlmostEqual(sparse_row["group_popularity_recent_share_delta"], 1 / 6)
            self.assertIn('"Source A": 1', sparse_row["bucket_source_counts_json"])
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def _assert_export_snapshot_group_temporal_buckets_filter(
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
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                f"&group_type={group_type}&group_key={group_key}&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
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
            self.assertIn('"label": "2026-04-13"', sparse_row["group_coverage_gap_ranges_json"])
            self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
            self.assertAlmostEqual(sparse_row["group_popularity_recent_share_delta"], 1 / 6)
            self.assertIn('"Source A": 1', sparse_row["bucket_source_counts_json"])
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_buckets_keeps_source_filters(self):
        self._assert_export_snapshot_group_temporal_buckets_filter(
            group_type="source",
            group_key="source-a",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_snapshot_group_temporal_buckets_normalizes_source_group_key_variants(self):
        self._assert_export_snapshot_group_temporal_buckets_filter(
            group_type="source",
            group_key="Source_A",
            expected_row_key="source a",
            expected_group="Source A",
        )

    def test_export_snapshot_group_temporal_buckets_keeps_topic_filters(self):
        self._assert_export_snapshot_group_temporal_buckets_filter(
            group_type="topic",
            group_key="policy",
            expected_row_key="policy",
            expected_group="Policy",
        )

    def test_export_snapshot_group_temporal_buckets_supports_bucket_label_filters_for_multi_word_topic_keys(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(
                json.dumps(_sparse_bucket_temporal_payload_with_multi_word_topic()),
                encoding="utf-8",
            )
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                f"&group_type=topic&group_key=Climate_Risk&bucket_label=2026-03-30&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
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
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_buckets_supports_bucket_label_filters_without_group_key(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_buckets&format=json&refresh=1"
                f"&bucket_label=2026-04-06&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)

            payload = exported.get_json()
            self.assertEqual(payload["artifact"], "group_temporal_buckets")
            self.assertEqual(payload["filters"], {"bucket_label": "2026-04-06"})
            self.assertEqual(payload["meta"]["source_mode"], "snapshot")
            self.assertEqual(payload["meta"]["snapshot_date"], self.snapshot_date)
            self.assertEqual(len(payload["rows"]), 6)
            self.assertEqual({row["group_type"] for row in payload["rows"]}, {"source", "topic", "tag"})
            self.assertEqual(
                {row["group_key"] for row in payload["rows"]},
                {"source a", "source b", "policy", "markets", "risk", "growth"},
            )
            self.assertTrue(all(row["bucket_label"] == "2026-04-06" for row in payload["rows"]))
            self.assertEqual(sum(1 for row in payload["rows"] if row["bucket_status"] == "sparse"), 3)
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_buckets_csv_keeps_group_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                f"&group_type=tag&group_key=risk&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_buckets.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
            self.assertIsNotNone(reader.fieldnames)
            self.assertIn("group_type", reader.fieldnames)
            self.assertIn("group_key", reader.fieldnames)
            self.assertIn("group_sparse_bucket_count", reader.fieldnames)
            self.assertIn("group_coverage_gap_labels", reader.fieldnames)
            self.assertIn("group_popularity_recent_share_delta", reader.fieldnames)
            self.assertIn("bucket_source_counts_json", reader.fieldnames)

            rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["group_type"] == "tag" for row in rows))
            self.assertTrue(all(row["group_key"] == "risk" for row in rows))
            self.assertTrue(all(row["group"] == "Risk" for row in rows))

            sparse_row = next(row for row in rows if row["bucket_status"] == "sparse")
            self.assertEqual(sparse_row["bucket_start"], "2026-04-06")
            self.assertEqual(sparse_row["bucket_n_articles"], "1")
            self.assertEqual(sparse_row["group_sparse_bucket_count"], "1")
            self.assertEqual(sparse_row["group_coverage_gap_count"], "1")
            self.assertEqual(sparse_row["group_coverage_gap_labels"], "2026-04-13")
            self.assertIn('"label": "2026-04-13"', sparse_row["group_coverage_gap_ranges_json"])
            self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
            self.assertAlmostEqual(float(sparse_row["group_popularity_recent_share_delta"]), 1 / 6)
            self.assertIn('"Source A": 1', sparse_row["bucket_source_counts_json"])
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_buckets_csv_keeps_source_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                f"&group_type=source&group_key=source-a&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_buckets.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
            self.assertIsNotNone(reader.fieldnames)
            self.assertIn("group_type", reader.fieldnames)
            self.assertIn("group_key", reader.fieldnames)
            self.assertIn("group_sparse_bucket_count", reader.fieldnames)
            self.assertIn("group_coverage_gap_labels", reader.fieldnames)
            self.assertIn("group_popularity_recent_share_delta", reader.fieldnames)
            self.assertIn("bucket_source_counts_json", reader.fieldnames)

            rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["group_type"] == "source" for row in rows))
            self.assertTrue(all(row["group_key"] == "source a" for row in rows))
            self.assertTrue(all(row["group"] == "Source A" for row in rows))

            sparse_row = next(row for row in rows if row["bucket_status"] == "sparse")
            self.assertEqual(sparse_row["bucket_start"], "2026-04-06")
            self.assertEqual(sparse_row["bucket_n_articles"], "1")
            self.assertEqual(sparse_row["group_sparse_bucket_count"], "1")
            self.assertEqual(sparse_row["group_coverage_gap_count"], "1")
            self.assertEqual(sparse_row["group_coverage_gap_labels"], "2026-04-13")
            self.assertIn('"label": "2026-04-13"', sparse_row["group_coverage_gap_ranges_json"])
            self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
            self.assertAlmostEqual(float(sparse_row["group_popularity_recent_share_delta"]), 1 / 6)
            self.assertIn('"Source A": 1', sparse_row["bucket_source_counts_json"])
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_buckets_csv_normalizes_source_group_key_variants(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                f"&group_type=source&group_key=Source_A&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_buckets.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
            self.assertIsNotNone(reader.fieldnames)
            self.assertIn("group_type", reader.fieldnames)
            self.assertIn("group_key", reader.fieldnames)
            self.assertIn("group_sparse_bucket_count", reader.fieldnames)
            self.assertIn("group_coverage_gap_labels", reader.fieldnames)
            self.assertIn("group_popularity_recent_share_delta", reader.fieldnames)
            self.assertIn("bucket_source_counts_json", reader.fieldnames)

            rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["group_type"] == "source" for row in rows))
            self.assertTrue(all(row["group_key"] == "source a" for row in rows))
            self.assertTrue(all(row["group"] == "Source A" for row in rows))

            sparse_row = next(row for row in rows if row["bucket_status"] == "sparse")
            self.assertEqual(sparse_row["bucket_start"], "2026-04-06")
            self.assertEqual(sparse_row["bucket_n_articles"], "1")
            self.assertEqual(sparse_row["group_sparse_bucket_count"], "1")
            self.assertEqual(sparse_row["group_coverage_gap_count"], "1")
            self.assertEqual(sparse_row["group_coverage_gap_labels"], "2026-04-13")
            self.assertIn('"label": "2026-04-13"', sparse_row["group_coverage_gap_ranges_json"])
            self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
            self.assertAlmostEqual(float(sparse_row["group_popularity_recent_share_delta"]), 1 / 6)
            self.assertIn('"Source A": 1', sparse_row["bucket_source_counts_json"])
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_buckets_csv_keeps_topic_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                f"&group_type=topic&group_key=policy&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_buckets.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
            self.assertIsNotNone(reader.fieldnames)
            self.assertIn("group_type", reader.fieldnames)
            self.assertIn("group_key", reader.fieldnames)
            self.assertIn("group_sparse_bucket_count", reader.fieldnames)
            self.assertIn("group_coverage_gap_labels", reader.fieldnames)
            self.assertIn("group_popularity_recent_share_delta", reader.fieldnames)
            self.assertIn("bucket_source_counts_json", reader.fieldnames)

            rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["group_type"] == "topic" for row in rows))
            self.assertTrue(all(row["group_key"] == "policy" for row in rows))
            self.assertTrue(all(row["group"] == "Policy" for row in rows))

            sparse_row = next(row for row in rows if row["bucket_status"] == "sparse")
            self.assertEqual(sparse_row["bucket_start"], "2026-04-06")
            self.assertEqual(sparse_row["bucket_n_articles"], "1")
            self.assertEqual(sparse_row["group_sparse_bucket_count"], "1")
            self.assertEqual(sparse_row["group_coverage_gap_count"], "1")
            self.assertEqual(sparse_row["group_coverage_gap_labels"], "2026-04-13")
            self.assertIn('"label": "2026-04-13"', sparse_row["group_coverage_gap_ranges_json"])
            self.assertEqual(sparse_row["group_popularity_peak_bucket"], "2026-03-30")
            self.assertAlmostEqual(float(sparse_row["group_popularity_recent_share_delta"]), 1 / 6)
            self.assertIn('"Source A": 1', sparse_row["bucket_source_counts_json"])
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_buckets_csv_supports_bucket_label_filters_for_multi_word_topic_keys(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(
                json.dumps(_sparse_bucket_temporal_payload_with_multi_word_topic()),
                encoding="utf-8",
            )
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                f"&group_type=topic&group_key=Climate_Risk&bucket_label=2026-03-30&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")

            rows = list(csv.DictReader(io.StringIO(exported.get_data(as_text=True))))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["group_type"], "topic")
            self.assertEqual(rows[0]["group_key"], "climate risk")
            self.assertEqual(rows[0]["group"], "Climate Risk")
            self.assertEqual(rows[0]["bucket_label"], "2026-03-30")
            self.assertEqual(rows[0]["bucket_status"], "ok")
            self.assertEqual(rows[0]["bucket_n_articles"], "2")
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_buckets_csv_supports_bucket_label_filters_without_group_key(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_buckets&format=csv&refresh=1"
                f"&bucket_label=2026-04-06&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")

            rows = list(csv.DictReader(io.StringIO(exported.get_data(as_text=True))))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["group_type"] for row in rows}, {"source", "topic", "tag"})
            self.assertEqual({row["group_key"] for row in rows}, {"source a", "source b", "policy", "markets", "risk", "growth"})
            self.assertTrue(all(row["bucket_label"] == "2026-04-06" for row in rows))
            self.assertEqual(sum(1 for row in rows if row["bucket_status"] == "sparse"), 3)
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_path_summary_csv_keeps_group_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_path_summary&format=csv&refresh=1"
                f"&group_type=tag&group_key=risk&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_path_summary.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
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
            self.assertEqual(rows[0]["coverage_gap_count"], "1")
            self.assertEqual(rows[0]["coverage_gap_labels"], "2026-04-13")
            self.assertIn('"label": "2026-04-13"', rows[0]["coverage_gap_ranges_json"])
            self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
            self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_path_summary_csv_keeps_source_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_path_summary&format=csv&refresh=1"
                f"&group_type=source&group_key=source-a&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_path_summary.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
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
            self.assertIn('"label": "2026-04-13"', rows[0]["coverage_gap_ranges_json"])
            self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
            self.assertEqual(rows[0]["popularity_peak_articles"], "2")
            self.assertEqual(rows[0]["popularity_first_share"], "0.5")
            self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
            self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_path_summary_csv_normalizes_source_group_key_variants(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_path_summary&format=csv&refresh=1"
                f"&group_type=source&group_key=Source_A&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_path_summary.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
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
            self.assertIn('"label": "2026-04-13"', rows[0]["coverage_gap_ranges_json"])
            self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
            self.assertEqual(rows[0]["popularity_peak_articles"], "2")
            self.assertEqual(rows[0]["popularity_first_share"], "0.5")
            self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
            self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_snapshot_group_temporal_path_summary_csv_keeps_topic_filters(self):
        original_payload = self.snapshot_payload_path.read_text(encoding="utf-8")
        try:
            self.snapshot_payload_path.write_text(json.dumps(_sparse_bucket_temporal_payload()), encoding="utf-8")
            exported = self.client.get(
                f"/api/news/export?artifact=group_temporal_path_summary&format=csv&refresh=1"
                f"&group_type=topic&group_key=policy&snapshot_date={self.snapshot_date}"
            )
            self.assertEqual(exported.status_code, 200)
            self.assertIn("text/csv", exported.content_type)
            self.assertEqual(exported.headers.get("Cache-Control"), "no-store")
            self.assertIn(
                "group_temporal_path_summary.csv",
                exported.headers.get("Content-Disposition", ""),
            )

            reader = csv.DictReader(io.StringIO(exported.get_data(as_text=True)))
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
            self.assertEqual(rows[0]["coverage_gap_count"], "1")
            self.assertEqual(rows[0]["coverage_gap_labels"], "2026-04-13")
            self.assertIn('"label": "2026-04-13"', rows[0]["coverage_gap_ranges_json"])
            self.assertEqual(rows[0]["popularity_peak_bucket"], "2026-03-30")
            self.assertEqual(rows[0]["popularity_peak_articles"], "2")
            self.assertEqual(rows[0]["popularity_first_share"], "0.5")
            self.assertEqual(rows[0]["popularity_latest_share"], "0.5")
            self.assertAlmostEqual(float(rows[0]["popularity_recent_share_delta"]), 1 / 6)
        finally:
            self.snapshot_payload_path.write_text(original_payload, encoding="utf-8")
            self.client.get(f"/api/news/stats?refresh=1&snapshot_date={self.snapshot_date}")

    def test_export_rejects_group_key_without_group_type(self):
        response = self.client.get(
            "/api/news/export?artifact=group_temporal_buckets&format=json&group_key=source-a"
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["status"], "bad_request")
        self.assertIn("group_key requires group_type", response.get_json()["error"])

    def test_stats_and_freshness(self):
        stats = self.client.get("/api/news/stats")
        self.assertEqual(stats.status_code, 200)
        self.assertIn("public, max-age=", stats.headers.get("Cache-Control", ""))
        stats_payload = stats.get_json()
        self.assertEqual(stats_payload["status"], "ok")
        self.assertIn("derived", stats_payload["data"])
        self.assertEqual(stats_payload["data"]["derived"]["input_articles"], 3)
        self.assertEqual(stats_payload["data"]["derived"]["excluded_unscraped_articles"], 1)
        self.assertEqual(stats_payload["data"]["derived"]["total_articles"], 2)
        self.assertEqual(stats_payload["data"]["derived"]["scored_articles"], 1)
        self.assertEqual(stats_payload["data"]["derived"]["zero_score_articles"], 0)
        self.assertEqual(stats_payload["data"]["derived"]["unscorable_articles"], 1)
        self.assertIn("score_status", stats_payload["data"]["derived"])
        self.assertEqual(stats_payload["data"]["derived"]["score_status"]["scored"], 1)
        self.assertEqual(stats_payload["data"]["derived"]["score_status"]["unscorable"], 1)
        self.assertIn("lens_correlations", stats_payload["data"]["derived"])
        self.assertIn("source_differentiation", stats_payload["data"]["derived"])
        self.assertIn("source_lens_effects", stats_payload["data"]["derived"])
        self.assertIn("lens_views", stats_payload["data"]["derived"])
        self.assertIn("lens_inventory", stats_payload["data"]["derived"])
        self.assertIn("lens_pca", stats_payload["data"]["derived"])
        self.assertIn("lens_mds", stats_payload["data"]["derived"])
        self.assertIn("lens_separation", stats_payload["data"]["derived"])
        self.assertIn("lens_time_series", stats_payload["data"]["derived"])
        self.assertIn("lens_temporal_embedding", stats_payload["data"]["derived"])
        self.assertIn("lens_temporal_embedding_mds", stats_payload["data"]["derived"])
        lens_correlations = stats_payload["data"]["derived"]["lens_correlations"]
        self.assertIn("lenses", lens_correlations)
        self.assertIn("correlation", lens_correlations)
        self.assertIn("covariance", lens_correlations)
        self.assertIn("pairwise_counts", lens_correlations)
        self.assertIn("pair_rankings", lens_correlations)
        self.assertIn("summary_by_matrix", lens_correlations)
        self.assertEqual(sorted(lens_correlations["pair_rankings"].keys()), ["corr_norm", "corr_raw", "cov_norm", "cov_raw", "pairwise"])
        self.assertEqual(sorted(lens_correlations["summary_by_matrix"].keys()), ["corr_norm", "corr_raw", "cov_norm", "cov_raw", "pairwise"])
        self.assertEqual(lens_correlations["summary_by_matrix"]["corr_raw"]["pair_count"], 0)
        self.assertIsNone(lens_correlations["summary_by_matrix"]["corr_raw"]["strongest_pair"])
        source_differentiation = stats_payload["data"]["derived"]["source_differentiation"]
        self.assertIn("status", source_differentiation)
        self.assertIn("source_counts", source_differentiation)
        self.assertIn("multivariate", source_differentiation)
        self.assertIn("classification", source_differentiation)
        source_lens_effects = stats_payload["data"]["derived"]["source_lens_effects"]
        self.assertIn("status", source_lens_effects)
        self.assertIn("permutations", source_lens_effects)
        self.assertIn("rows", source_lens_effects)
        source_topic_control = stats_payload["data"]["derived"]["source_topic_control"]
        self.assertEqual(source_topic_control["topic_basis"], "topic_tags")
        self.assertEqual(source_topic_control["multi_topic_policy"], "duplicate_per_topic")
        self.assertEqual(source_topic_control["pooled_label"], "topic-confounded")
        self.assertIn("topics", source_topic_control)
        self.assertIn("summary", source_topic_control)
        self.assertEqual(
            source_topic_control["pooled"]["source_differentiation"],
            source_differentiation,
        )
        self.assertEqual(
            source_topic_control["pooled"]["source_lens_effects"],
            source_lens_effects,
        )
        tag_sliced_analysis = stats_payload["data"]["derived"]["tag_sliced_analysis"]
        self.assertEqual(tag_sliced_analysis["tag_basis"], "topic_tags")
        self.assertEqual(tag_sliced_analysis["multi_tag_policy"], "duplicate_per_tag")
        self.assertEqual(tag_sliced_analysis["pooled_label"], "tag-confounded")
        self.assertIn("tags", tag_sliced_analysis)
        self.assertIn("summary", tag_sliced_analysis)
        self.assertEqual(
            tag_sliced_analysis["pooled"]["source_differentiation"],
            source_differentiation,
        )
        self.assertEqual(
            tag_sliced_analysis["pooled"]["source_lens_effects"],
            source_lens_effects,
        )
        event_control = stats_payload["data"]["derived"]["event_control"]
        self.assertIn(event_control["status"], {"ok", "unavailable"})
        self.assertIn("config", event_control)
        self.assertIn("summary", event_control)
        self.assertIn("events", event_control)
        self.assertIn("same_event_source_differentiation", event_control)
        self.assertIn("same_event_source_lens_effects", event_control)
        self.assertIn("same_event_pairwise_source_lens_deltas", event_control)
        self.assertIn("event_coverage", event_control)
        self.assertIn("same_event_variance_decomposition", event_control)
        self.assertIn("source_reliability", stats_payload["data"]["derived"])
        source_reliability = stats_payload["data"]["derived"]["source_reliability"]
        self.assertEqual(source_reliability["method"], "heuristic-v1")
        self.assertEqual(source_reliability["pooled_label"], "topic-confounded")
        self.assertIn("pooled", source_reliability)
        self.assertIn("event_controlled", source_reliability)
        self.assertIn("topics", source_reliability)
        self.assertIn("tags", source_reliability)
        self.assertIn("summary", source_reliability)
        self.assertIn(source_reliability["event_controlled"].get("status"), {"ok", "unavailable"})
        self.assertIn("event_controlled_status", source_reliability["summary"])
        self.assertIn("event_controlled_tier", source_reliability["summary"])
        self.assertIn("tag_count", source_reliability["summary"])
        self.assertIn("ok_tag_count", source_reliability["summary"])
        self.assertIn(source_reliability["pooled"].get("status"), {"ok", "unavailable"})
        lens_pca = stats_payload["data"]["derived"]["lens_pca"]
        self.assertIn("status", lens_pca)
        self.assertIn("reason", lens_pca)
        self.assertIn("components", lens_pca)
        self.assertIn("explained_variance", lens_pca)
        self.assertIn("variance_drivers", lens_pca)
        self.assertIn("article_points", lens_pca)
        self.assertIn("source_centroids", lens_pca)
        lens_mds = stats_payload["data"]["derived"]["lens_mds"]
        self.assertIn("status", lens_mds)
        self.assertIn("reason", lens_mds)
        self.assertIn("dimensions", lens_mds)
        self.assertIn("dimension_strength", lens_mds)
        self.assertIn("stress", lens_mds)
        self.assertIn("article_points", lens_mds)
        self.assertIn("source_centroids", lens_mds)
        lens_separation = stats_payload["data"]["derived"]["lens_separation"]
        self.assertIn("status", lens_separation)
        self.assertIn("reason", lens_separation)
        self.assertIn("n_sources", lens_separation)
        self.assertIn("separation_ratio", lens_separation)
        self.assertIn("silhouette_like_mean", lens_separation)
        lens_time_series = stats_payload["data"]["derived"]["lens_time_series"]
        self.assertIn("status", lens_time_series)
        self.assertIn("reason", lens_time_series)
        self.assertIn("series", lens_time_series)
        self.assertIn("summary", lens_time_series)
        lens_temporal_embedding = stats_payload["data"]["derived"]["lens_temporal_embedding"]
        self.assertIn("status", lens_temporal_embedding)
        self.assertIn("reason", lens_temporal_embedding)
        self.assertIn("points", lens_temporal_embedding)
        self.assertIn("summary", lens_temporal_embedding)
        lens_temporal_embedding_mds = stats_payload["data"]["derived"]["lens_temporal_embedding_mds"]
        self.assertIn("status", lens_temporal_embedding_mds)
        self.assertIn("reason", lens_temporal_embedding_mds)
        self.assertIn("points", lens_temporal_embedding_mds)
        self.assertIn("summary", lens_temporal_embedding_mds)
        lens_views = stats_payload["data"]["derived"]["lens_views"]
        self.assertIn("coverage_mode", lens_views)
        self.assertIn("lens_names", lens_views)
        self.assertIn("article_rows", lens_views)
        self.assertIn("source_rows", lens_views)
        self.assertIn("stability_rows", lens_views)
        self.assertIn("summary", lens_views)
        self.assertEqual(lens_views["summary"]["article_count"], 1)
        self.assertIsInstance(lens_views["summary"]["dominant_lens_counts"], list)
        self.assertIsInstance(lens_views["summary"]["lens_average_rows"], list)
        self.assertEqual(lens_views["summary"]["source_count"], 1)
        self.assertEqual(lens_views["summary"]["covered_articles"], 1)
        self.assertIsInstance(lens_views["summary"]["source_lens_average_rows"], list)
        self.assertEqual(lens_views["summary"]["stability_lens_count"], 1)
        self.assertEqual(lens_views["summary"]["stability_avg_stddev"], 0.0)
        self.assertEqual(lens_views["summary"]["stability_top_lens"], "L1")
        self.assertEqual(lens_views["summary"]["stability_total_samples"], 1)
        lens_inventory = stats_payload["data"]["derived"]["lens_inventory"]
        self.assertIn("coverage_mode", lens_inventory)
        self.assertIn("items_total", lens_inventory)
        self.assertIn("aggregation", lens_inventory)
        self.assertIn("lenses", lens_inventory)
        self.assertIn("data_quality", stats_payload["data"]["derived"])
        data_quality = stats_payload["data"]["derived"]["data_quality"]
        self.assertIn("summary", data_quality)
        self.assertIn("field_coverage", data_quality)
        self.assertEqual(data_quality["summary"]["total"], 2)
        self.assertEqual(data_quality["summary"]["scored"], 1)
        self.assertEqual(data_quality["summary"]["missing_ai_summary"], 0)
        self.assertEqual(data_quality["summary"]["missing_published"], 0)
        self.assertEqual(data_quality["summary"]["missing_source"], 0)
        coverage_by_field = {row["field"]: row for row in data_quality["field_coverage"]}
        self.assertEqual(coverage_by_field["Title"]["present"], 2)
        self.assertEqual(coverage_by_field["Lens Scores"]["present"], 1)
        chart_aggregates = stats_payload["data"]["derived"]["chart_aggregates"]
        self.assertEqual(len(chart_aggregates["tag_count_distribution"]), 6)
        self.assertEqual(len(chart_aggregates["publish_hour_counts_utc"]), 24)
        self.assertIn("source_tag_totals", chart_aggregates)
        self.assertIn("tag_totals", chart_aggregates)
        self.assertIn("score_status_by_source", chart_aggregates)
        self.assertEqual(chart_aggregates["source_tag_totals"][0]["source"], "PBS NewsHour")
        self.assertEqual(chart_aggregates["source_tag_totals"][0]["count"], 2)
        self.assertEqual(chart_aggregates["tag_totals"][0]["tag"], "OpenAI")
        self.assertEqual(chart_aggregates["tag_totals"][0]["count"], 2)
        source_tag_views = stats_payload["data"]["derived"]["source_tag_views"]
        self.assertIn("source_labels", source_tag_views)
        self.assertIn("tag_labels", source_tag_views)
        self.assertIn("source_rows", source_tag_views)
        self.assertIn("summary", source_tag_views)
        self.assertEqual(source_tag_views["source_labels"][0], "PBS NewsHour")
        self.assertEqual(source_tag_views["tag_labels"][0], "OpenAI")
        self.assertEqual(source_tag_views["source_rows"][0]["source"], "PBS NewsHour")
        self.assertEqual(source_tag_views["summary"]["source_count"], 2)
        self.assertEqual(source_tag_views["summary"]["tag_count"], 3)
        self.assertEqual(source_tag_views["summary"]["matrix_rows"], 4)
        self.assertEqual(source_tag_views["summary"]["non_zero_cells"], 4)
        self.assertEqual(source_tag_views["summary"]["total_assignments"], 4)

        health = self.client.get("/health/news-freshness")
        self.assertEqual(health.status_code, 200)
        health_payload = health.get_json()
        self.assertTrue(health_payload["is_fresh"])

    def test_upstream_endpoint(self):
        response = self.client.get("/api/news/upstream")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("meta", payload)
        self.assertIn("data", payload)
        upstream = payload["data"]["upstream"]
        self.assertIsInstance(upstream, dict)
        self.assertIn("articles", upstream)
        self.assertEqual(len(upstream["articles"]), 3)
        self.assertEqual(upstream["articles"][0]["id"], "a-1")
        self.assertNotIn("General", upstream["articles"][0].get("topic_tags", []))

        snapshot_response = self.client.get(f"/api/news/upstream?snapshot_date={self.snapshot_date}")
        self.assertEqual(snapshot_response.status_code, 200)
        snapshot_payload = snapshot_response.get_json()
        self.assertEqual(snapshot_payload["meta"]["source_mode"], "snapshot")
        self.assertEqual(snapshot_payload["meta"]["snapshot_date"], self.snapshot_date)

    def test_export_endpoints(self):
        export_json = self.client.get("/api/news/export?artifact=source_score_status&format=json")
        self.assertEqual(export_json.status_code, 200)
        payload = export_json.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["artifact"], "source_score_status")
        self.assertIsInstance(payload["rows"], list)
        self.assertGreaterEqual(len(payload["rows"]), 1)

        export_csv = self.client.get("/api/news/export?artifact=source_score_status&format=csv")
        self.assertEqual(export_csv.status_code, 200)
        self.assertIn("text/csv", export_csv.content_type)
        csv_body = export_csv.get_data(as_text=True)
        self.assertIn("source", csv_body)
        self.assertIn("scored", csv_body)

        export_source_effects = self.client.get("/api/news/export?artifact=source_lens_effects&format=json")
        self.assertEqual(export_source_effects.status_code, 200)
        effects_payload = export_source_effects.get_json()
        self.assertEqual(effects_payload["artifact"], "source_lens_effects")
        self.assertIsInstance(effects_payload["rows"], list)

        export_source_summary = self.client.get("/api/news/export?artifact=source_differentiation_summary&format=json")
        self.assertEqual(export_source_summary.status_code, 200)
        source_summary_payload = export_source_summary.get_json()
        self.assertEqual(source_summary_payload["artifact"], "source_differentiation_summary")
        self.assertIsInstance(source_summary_payload["rows"], list)
        self.assertEqual(len(source_summary_payload["rows"]), 1)

        export_event_summary = self.client.get("/api/news/export?artifact=event_control_summary&format=json")
        self.assertEqual(export_event_summary.status_code, 200)
        event_summary_payload = export_event_summary.get_json()
        self.assertEqual(event_summary_payload["artifact"], "event_control_summary")
        self.assertIsInstance(event_summary_payload["rows"], list)
        self.assertEqual(len(event_summary_payload["rows"]), 1)
        self.assertIn("event_count", event_summary_payload["rows"][0])

        export_events = self.client.get("/api/news/export?artifact=event_clusters&format=json")
        self.assertEqual(export_events.status_code, 200)
        events_payload = export_events.get_json()
        self.assertEqual(events_payload["artifact"], "event_clusters")
        self.assertIsInstance(events_payload["rows"], list)

        export_source_coverage = self.client.get("/api/news/export?artifact=event_source_coverage&format=json")
        self.assertEqual(export_source_coverage.status_code, 200)
        source_coverage_payload = export_source_coverage.get_json()
        self.assertEqual(source_coverage_payload["artifact"], "event_source_coverage")
        self.assertIsInstance(source_coverage_payload["rows"], list)

        export_pair_coverage = self.client.get("/api/news/export?artifact=event_source_pair_coverage&format=json")
        self.assertEqual(export_pair_coverage.status_code, 200)
        pair_coverage_payload = export_pair_coverage.get_json()
        self.assertEqual(pair_coverage_payload["artifact"], "event_source_pair_coverage")
        self.assertIsInstance(pair_coverage_payload["rows"], list)

        export_same_event_effects = self.client.get("/api/news/export?artifact=same_event_source_lens_effects&format=json")
        self.assertEqual(export_same_event_effects.status_code, 200)
        same_event_effects_payload = export_same_event_effects.get_json()
        self.assertEqual(same_event_effects_payload["artifact"], "same_event_source_lens_effects")
        self.assertIsInstance(same_event_effects_payload["rows"], list)

        export_pairwise_deltas = self.client.get("/api/news/export?artifact=same_event_pairwise_source_lens_deltas&format=json")
        self.assertEqual(export_pairwise_deltas.status_code, 200)
        pairwise_deltas_payload = export_pairwise_deltas.get_json()
        self.assertEqual(pairwise_deltas_payload["artifact"], "same_event_pairwise_source_lens_deltas")
        self.assertIsInstance(pairwise_deltas_payload["rows"], list)

        export_variance = self.client.get("/api/news/export?artifact=same_event_variance_decomposition&format=json")
        self.assertEqual(export_variance.status_code, 200)
        variance_payload = export_variance.get_json()
        self.assertEqual(variance_payload["artifact"], "same_event_variance_decomposition")
        self.assertIsInstance(variance_payload["rows"], list)

        export_same_event_summary = self.client.get(
            "/api/news/export?artifact=same_event_source_differentiation_summary&format=json"
        )
        self.assertEqual(export_same_event_summary.status_code, 200)
        same_event_summary_payload = export_same_event_summary.get_json()
        self.assertEqual(same_event_summary_payload["artifact"], "same_event_source_differentiation_summary")
        self.assertIsInstance(same_event_summary_payload["rows"], list)
        self.assertEqual(len(same_event_summary_payload["rows"]), 1)

        bad_artifact = self.client.get("/api/news/export?artifact=unknown")
        self.assertEqual(bad_artifact.status_code, 400)

        bad_format = self.client.get("/api/news/export?artifact=source_score_status&format=xml")
        self.assertEqual(bad_format.status_code, 400)

    def test_freshness_is_stale_when_generated_at_missing(self):
        payload = dict(SAMPLE_PAYLOAD)
        payload.pop("generated_at", None)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp:
            json.dump(payload, temp)
            temp.flush()
            temp_path = Path(temp.name)

        previous_url = os.environ.get("RSS_DAILY_JSON_URL")
        try:
            os.environ["RSS_DAILY_JSON_URL"] = f"file://{temp_path}"
            app = Flask(__name__)
            register_news_endpoints(app)
            client = app.test_client()

            response = client.get("/health/news-freshness?refresh=true")
            self.assertEqual(response.status_code, 503)
            body = response.get_json()
            self.assertEqual(body["status"], "stale")
            self.assertFalse(body["is_fresh"])
            self.assertEqual(body["reason"], "generated_at is missing from payload")
        finally:
            if previous_url is None:
                os.environ.pop("RSS_DAILY_JSON_URL", None)
            else:
                os.environ["RSS_DAILY_JSON_URL"] = previous_url
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
