import unittest
from types import SimpleNamespace
from unittest.mock import patch

import dash_bootstrap_components as dbc
from dash import dcc, html

import src.app  # noqa: F401
import src.pages.news_group_latent_space as news_group_latent_space
from src.pages.news_group_latent_space import (
    ALL_GROUPS_CLUSTER_VALUE,
    _centroid_figure,
    _bucket_coordinate_summary,
    _cluster_filter_hint,
    _cluster_popularity_comparison,
    _cluster_share_drift_comparison,
    _cluster_membership_summary,
    _cluster_options,
    _cluster_overview_table,
    _cluster_peer_movement_callout,
    _cluster_scope_summary,
    _group_movement_figure,
    _group_popularity_timeline,
    _movement_leaderboard,
    _movement_pattern_callout,
    _group_movement_summary,
    _group_scope_summary,
    _group_temporal_export_href,
    _group_temporal_export_panel,
    _group_temporal_scope_summary,
    _group_table,
    _group_options,
    _group_rows_for_cluster,
    _selected_temporal_group_row,
    _selected_cluster_row,
    _selected_group_row,
    _selected_group_summary,
    _temporal_bucket_diagnostics_table,
)


def _iter_children(node):
    if node is None:
        return
    if isinstance(node, (str, int, float, bool)):
        yield node
        return
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _iter_children(child)
    elif children is not None:
        yield from _iter_children(children)


def _find_components(node, component_type):
    return [child for child in _iter_children(node) if isinstance(child, component_type)]


def _sample_group_latent_payload(include_extra_cluster_groups: bool = False) -> dict:
    source_groups = [
        {
            "group": "Source C",
            "group_key": "source-c",
            "status": "ok",
            "n_articles": 9,
            "n_sources": 3,
            "date_start": "2026-03-02",
            "date_end": "2026-03-22",
            "cluster_id": "source-cluster-2",
            "cluster_label": "Source C Cluster",
            "pc1": 0.31,
            "pc2": 0.18,
            "mds1": -0.12,
            "mds2": 0.27,
            "dispersion_pca": 0.233,
            "dispersion_mds": 0.187,
            "nearest_groups": [
                {"group": "Source A", "distance_pca": 0.111},
                {"group": "Source B", "distance_pca": 0.245},
            ],
            "top_lens_deviations": [
                {"lens": "Evidence", "delta": 9.2},
            ],
        }
    ]
    source_clusters = [
        {
            "cluster_id": "source-cluster-2",
            "label": "Source C Cluster",
            "n_groups": 1,
            "n_articles": 9,
            "n_sources": 3,
            "clustering_threshold_pca": 0.187,
            "representative_groups": [{"group": "Source C", "n_articles": 9}],
            "defining_lens_deviations": [{"lens": "Evidence", "delta": 9.2}],
        }
    ]
    temporal_source_groups = [
        {
            "group": "Source C",
            "group_key": "source-c",
            "status": "ok",
            "n_articles": 9,
            "n_buckets": 3,
            "date_start": "2026-03-02",
            "date_end": "2026-03-22",
            "buckets": [
                {
                    "bucket_start": "2026-03-02",
                    "bucket_label": "2026-03-02",
                    "n_articles": 4,
                    "status": "ok",
                    "n_sources": 2,
                    "corpus_share": 0.4,
                    "pc1": 0.1,
                    "pc2": 0.2,
                    "mds1": -0.2,
                    "mds2": 0.3,
                    "dispersion_pca": 0.13,
                    "dispersion_mds": 0.11,
                    "source_counts": {"Source C": 3, "Source A": 1},
                    "top_lens_deviations": [
                        {"lens": "Evidence", "delta": 8.2},
                        {"lens": "Impact", "delta": -4.1},
                    ],
                },
                {
                    "bucket_start": "2026-03-09",
                    "bucket_label": "2026-03-09",
                    "n_articles": 3,
                    "status": "sparse",
                    "n_sources": 2,
                    "corpus_share": 0.3,
                    "pc1": 0.4,
                    "pc2": 0.5,
                    "dispersion_pca": 0.19,
                    "source_counts": {"Source C": 2, "Source B": 1},
                    "top_lens_deviations": [
                        {"lens": "Novelty", "delta": 5.4},
                    ],
                },
                {
                    "bucket_start": "2026-03-16",
                    "bucket_label": "2026-03-16",
                    "n_articles": 2,
                    "status": "ok",
                    "n_sources": 1,
                    "corpus_share": 0.2,
                    "pc1": 0.6,
                    "pc2": 0.7,
                    "mds1": 0.1,
                    "mds2": 0.6,
                    "dispersion_pca": 0.17,
                    "dispersion_mds": 0.16,
                    "source_counts": {"Source C": 2},
                    "top_lens_deviations": [
                        {"lens": "Impact", "delta": -6.0},
                        {"lens": "Evidence", "delta": 4.0},
                    ],
                },
            ],
            "path_summary": {
                "bucket_count": 3,
                "valid_pca_bucket_count": 3,
                "valid_mds_bucket_count": 2,
                "sparse_bucket_count": 1,
                "coverage_gap_count": 1,
                "coverage_gap_ranges": [
                    {
                        "start_bucket": "2026-03-09",
                        "end_bucket": "2026-03-09",
                        "missing_bucket_count": 1,
                        "label": "2026-03-09",
                    }
                ],
                "total_movement_pca": 0.91,
                "largest_jump_pca": 0.38,
                "direction_pca": {"pc1_delta": 0.5, "pc2_delta": 0.5},
                "total_movement_mds": 0.41,
                "largest_jump_mds": 0.24,
                "direction_mds": {"mds1_delta": 0.3, "mds2_delta": 0.3},
            },
        }
    ]
    tag_groups = [
        {
            "group": "AI Safety",
            "group_key": "ai-safety",
            "status": "ok",
            "n_articles": 7,
            "n_sources": 3,
            "date_start": "2026-03-02",
            "date_end": "2026-03-22",
            "cluster_id": "tag-cluster-1",
            "cluster_label": "Risk and Governance",
            "pc1": 0.18,
            "pc2": 0.29,
            "mds1": -0.16,
            "mds2": 0.24,
            "dispersion_pca": 0.164,
            "dispersion_mds": 0.132,
            "nearest_groups": [
                {"group": "Policy", "distance_pca": 0.082},
                {"group": "Risk", "distance_pca": 0.104},
            ],
            "top_lens_deviations": [
                {"lens": "Evidence", "delta": 5.8},
            ],
        },
        {
            "group": "Policy",
            "group_key": "policy",
            "status": "ok",
            "n_articles": 6,
            "n_sources": 2,
            "date_start": "2026-03-02",
            "date_end": "2026-03-22",
            "cluster_id": "tag-cluster-1",
            "cluster_label": "Risk and Governance",
            "pc1": 0.11,
            "pc2": 0.21,
            "mds1": -0.11,
            "mds2": 0.18,
            "dispersion_pca": 0.141,
            "dispersion_mds": 0.109,
            "nearest_groups": [
                {"group": "AI Safety", "distance_pca": 0.082},
                {"group": "Risk", "distance_pca": 0.121},
            ],
            "top_lens_deviations": [
                {"lens": "Impact", "delta": 3.7},
            ],
        },
    ]
    tag_clusters = [
        {
            "cluster_id": "tag-cluster-1",
            "label": "Risk and Governance",
            "n_groups": 2,
            "n_articles": 13,
            "n_sources": 3,
            "clustering_threshold_pca": 0.129,
            "representative_groups": [
                {"group": "AI Safety", "n_articles": 7},
                {"group": "Policy", "n_articles": 6},
            ],
            "defining_lens_deviations": [{"lens": "Evidence", "delta": 5.8}],
        }
    ]
    temporal_tag_groups = [
        {
            "group": "AI Safety",
            "group_key": "ai-safety",
            "status": "ok",
            "n_articles": 7,
            "n_buckets": 3,
            "date_start": "2026-03-02",
            "date_end": "2026-03-22",
            "buckets": [
                {
                    "bucket_start": "2026-03-02",
                    "bucket_label": "2026-03-02",
                    "n_articles": 2,
                    "status": "ok",
                    "n_sources": 2,
                    "corpus_share": 0.16,
                    "pc1": 0.08,
                    "pc2": 0.18,
                    "mds1": -0.18,
                    "mds2": 0.14,
                    "dispersion_pca": 0.12,
                    "dispersion_mds": 0.09,
                    "source_counts": {"Source A": 1, "Source C": 1},
                    "top_lens_deviations": [{"lens": "Evidence", "delta": 3.9}],
                },
                {
                    "bucket_start": "2026-03-09",
                    "bucket_label": "2026-03-09",
                    "n_articles": 3,
                    "status": "ok",
                    "n_sources": 2,
                    "corpus_share": 0.22,
                    "pc1": 0.16,
                    "pc2": 0.28,
                    "mds1": -0.13,
                    "mds2": 0.2,
                    "dispersion_pca": 0.11,
                    "dispersion_mds": 0.08,
                    "source_counts": {"Source A": 1, "Source B": 2},
                    "top_lens_deviations": [{"lens": "Impact", "delta": 4.1}],
                },
                {
                    "bucket_start": "2026-03-16",
                    "bucket_label": "2026-03-16",
                    "n_articles": 2,
                    "status": "ok",
                    "n_sources": 1,
                    "corpus_share": 0.14,
                    "pc1": 0.2,
                    "pc2": 0.34,
                    "mds1": -0.08,
                    "mds2": 0.27,
                    "dispersion_pca": 0.1,
                    "dispersion_mds": 0.07,
                    "source_counts": {"Source C": 2},
                    "top_lens_deviations": [{"lens": "Novelty", "delta": 2.7}],
                },
            ],
            "path_summary": {
                "bucket_count": 3,
                "valid_pca_bucket_count": 3,
                "valid_mds_bucket_count": 3,
                "sparse_bucket_count": 0,
                "coverage_gap_count": 0,
                "total_movement_pca": 0.22,
                "largest_jump_pca": 0.12,
                "direction_pca": {"pc1_delta": 0.12, "pc2_delta": 0.16},
                "total_movement_mds": 0.17,
                "largest_jump_mds": 0.09,
                "direction_mds": {"mds1_delta": 0.10, "mds2_delta": 0.13},
            },
        },
        {
            "group": "Policy",
            "group_key": "policy",
            "status": "ok",
            "n_articles": 6,
            "n_buckets": 3,
            "date_start": "2026-03-02",
            "date_end": "2026-03-22",
            "buckets": [
                {
                    "bucket_start": "2026-03-02",
                    "bucket_label": "2026-03-02",
                    "n_articles": 2,
                    "status": "ok",
                    "n_sources": 1,
                    "corpus_share": 0.15,
                    "pc1": 0.09,
                    "pc2": 0.19,
                    "mds1": -0.12,
                    "mds2": 0.16,
                    "dispersion_pca": 0.09,
                    "dispersion_mds": 0.07,
                    "source_counts": {"Source B": 2},
                    "top_lens_deviations": [{"lens": "Impact", "delta": 2.0}],
                },
                {
                    "bucket_start": "2026-03-09",
                    "bucket_label": "2026-03-09",
                    "n_articles": 2,
                    "status": "ok",
                    "n_sources": 1,
                    "corpus_share": 0.14,
                    "pc1": 0.1,
                    "pc2": 0.2,
                    "mds1": -0.11,
                    "mds2": 0.17,
                    "dispersion_pca": 0.08,
                    "dispersion_mds": 0.07,
                    "source_counts": {"Source B": 2},
                    "top_lens_deviations": [{"lens": "Evidence", "delta": 1.4}],
                },
                {
                    "bucket_start": "2026-03-16",
                    "bucket_label": "2026-03-16",
                    "n_articles": 2,
                    "status": "ok",
                    "n_sources": 2,
                    "corpus_share": 0.13,
                    "pc1": 0.11,
                    "pc2": 0.21,
                    "mds1": -0.1,
                    "mds2": 0.18,
                    "dispersion_pca": 0.09,
                    "dispersion_mds": 0.08,
                    "source_counts": {"Source A": 1, "Source C": 1},
                    "top_lens_deviations": [{"lens": "Risk", "delta": 1.1}],
                },
            ],
            "path_summary": {
                "bucket_count": 3,
                "valid_pca_bucket_count": 3,
                "valid_mds_bucket_count": 3,
                "sparse_bucket_count": 0,
                "coverage_gap_count": 0,
                "total_movement_pca": 0.03,
                "largest_jump_pca": 0.02,
                "direction_pca": {"pc1_delta": 0.02, "pc2_delta": 0.02},
                "total_movement_mds": 0.03,
                "largest_jump_mds": 0.02,
                "direction_mds": {"mds1_delta": 0.02, "mds2_delta": 0.02},
            },
        },
    ]
    if include_extra_cluster_groups:
        source_groups = [
            {
                "group": "Source A",
                "group_key": "source-a",
                "status": "ok",
                "n_articles": 6,
                "n_sources": 2,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "cluster_id": "source-cluster-1",
                "cluster_label": "Source A, Source B",
                "pc1": 0.22,
                "pc2": 0.15,
                "mds1": -0.21,
                "mds2": 0.12,
                "dispersion_pca": 0.141,
                "dispersion_mds": 0.103,
                "nearest_groups": [
                    {"group": "Source B", "distance_pca": 0.066},
                    {"group": "Source C", "distance_pca": 0.111},
                ],
                "top_lens_deviations": [
                    {"lens": "Impact", "delta": 4.6},
                ],
            },
            {
                "group": "Source B",
                "group_key": "source-b",
                "status": "ok",
                "n_articles": 5,
                "n_sources": 2,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "cluster_id": "source-cluster-1",
                "cluster_label": "Source A, Source B",
                "pc1": 0.27,
                "pc2": 0.11,
                "mds1": -0.14,
                "mds2": 0.19,
                "dispersion_pca": 0.127,
                "dispersion_mds": 0.098,
                "nearest_groups": [
                    {"group": "Source A", "distance_pca": 0.066},
                    {"group": "Source C", "distance_pca": 0.178},
                ],
                "top_lens_deviations": [
                    {"lens": "Evidence", "delta": -3.4},
                ],
            },
            *source_groups,
        ]
        source_clusters = [
            {
                "cluster_id": "source-cluster-1",
                "label": "Source A, Source B",
                "n_groups": 2,
                "n_articles": 11,
                "n_sources": 2,
                "clustering_threshold_pca": 0.143,
                "representative_groups": [
                    {"group": "Source A", "n_articles": 6},
                    {"group": "Source B", "n_articles": 5},
                ],
                "defining_lens_deviations": [{"lens": "Impact", "delta": 4.6}],
            },
            *source_clusters,
        ]
        temporal_source_groups = [
            {
                "group": "Source A",
                "group_key": "source-a",
                "status": "ok",
                "n_articles": 6,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "buckets": [
                    {
                        "bucket_start": "2026-03-02",
                        "bucket_label": "2026-03-02",
                        "n_articles": 2,
                        "status": "ok",
                        "n_sources": 1,
                        "corpus_share": 0.2,
                        "pc1": 0.15,
                        "pc2": 0.1,
                        "mds1": -0.22,
                        "mds2": 0.08,
                        "dispersion_pca": 0.12,
                        "dispersion_mds": 0.09,
                        "source_counts": {"Source A": 2},
                        "top_lens_deviations": [{"lens": "Impact", "delta": 3.2}],
                    },
                    {
                        "bucket_start": "2026-03-09",
                        "bucket_label": "2026-03-09",
                        "n_articles": 2,
                        "status": "ok",
                        "n_sources": 1,
                        "corpus_share": 0.2,
                        "pc1": 0.23,
                        "pc2": 0.14,
                        "mds1": -0.2,
                        "mds2": 0.12,
                        "dispersion_pca": 0.11,
                        "dispersion_mds": 0.08,
                        "source_counts": {"Source A": 2},
                        "top_lens_deviations": [{"lens": "Evidence", "delta": 2.6}],
                    },
                    {
                        "bucket_start": "2026-03-16",
                        "bucket_label": "2026-03-16",
                        "n_articles": 2,
                        "status": "ok",
                        "n_sources": 1,
                        "corpus_share": 0.2,
                        "pc1": 0.28,
                        "pc2": 0.19,
                        "mds1": -0.17,
                        "mds2": 0.16,
                        "dispersion_pca": 0.1,
                        "dispersion_mds": 0.07,
                        "source_counts": {"Source A": 2},
                        "top_lens_deviations": [{"lens": "Novelty", "delta": 1.9}],
                    },
                ],
                "path_summary": {
                    "bucket_count": 3,
                    "valid_pca_bucket_count": 3,
                    "valid_mds_bucket_count": 3,
                    "sparse_bucket_count": 0,
                    "coverage_gap_count": 0,
                    "total_movement_pca": 0.16,
                    "largest_jump_pca": 0.09,
                    "direction_pca": {"pc1_delta": 0.13, "pc2_delta": 0.09},
                    "total_movement_mds": 0.09,
                    "largest_jump_mds": 0.05,
                    "direction_mds": {"mds1_delta": 0.05, "mds2_delta": 0.08},
                },
            },
            {
                "group": "Source B",
                "group_key": "source-b",
                "status": "ok",
                "n_articles": 5,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "buckets": [
                    {
                        "bucket_start": "2026-03-02",
                        "bucket_label": "2026-03-02",
                        "n_articles": 2,
                        "status": "ok",
                        "n_sources": 1,
                        "corpus_share": 0.18,
                        "pc1": 0.21,
                        "pc2": 0.08,
                        "mds1": -0.17,
                        "mds2": 0.16,
                        "dispersion_pca": 0.11,
                        "dispersion_mds": 0.09,
                        "source_counts": {"Source B": 2},
                        "top_lens_deviations": [{"lens": "Evidence", "delta": -2.1}],
                    },
                    {
                        "bucket_start": "2026-03-09",
                        "bucket_label": "2026-03-09",
                        "n_articles": 2,
                        "status": "ok",
                        "n_sources": 1,
                        "corpus_share": 0.19,
                        "pc1": 0.26,
                        "pc2": 0.12,
                        "mds1": -0.13,
                        "mds2": 0.2,
                        "dispersion_pca": 0.1,
                        "dispersion_mds": 0.08,
                        "source_counts": {"Source B": 2},
                        "top_lens_deviations": [{"lens": "Impact", "delta": 2.4}],
                    },
                    {
                        "bucket_start": "2026-03-16",
                        "bucket_label": "2026-03-16",
                        "n_articles": 1,
                        "status": "sparse",
                        "n_sources": 1,
                        "corpus_share": 0.11,
                        "pc1": 0.31,
                        "pc2": 0.15,
                        "mds1": -0.09,
                        "mds2": 0.24,
                        "dispersion_pca": 0.12,
                        "dispersion_mds": 0.1,
                        "source_counts": {"Source B": 1},
                        "top_lens_deviations": [{"lens": "Novelty", "delta": 3.5}],
                    },
                ],
                "path_summary": {
                    "bucket_count": 3,
                    "valid_pca_bucket_count": 3,
                    "valid_mds_bucket_count": 3,
                    "sparse_bucket_count": 1,
                    "coverage_gap_count": 0,
                    "total_movement_pca": 0.13,
                    "largest_jump_pca": 0.06,
                    "direction_pca": {"pc1_delta": 0.1, "pc2_delta": 0.07},
                    "total_movement_mds": 0.11,
                    "largest_jump_mds": 0.06,
                    "direction_mds": {"mds1_delta": 0.08, "mds2_delta": 0.08},
                },
            },
            *temporal_source_groups,
        ]
    return {
        "meta": {"generated_at": "2026-05-07T23:20:00Z"},
        "data": {
            "derived": {
                "group_latent_space": {
                    "status": "ok",
                    "reason": "",
                    "config": {"min_articles_per_group": 3},
                    "summary": {
                        "group_counts": {"source": len(source_groups), "tag": len(tag_groups)},
                        "analyzed_group_counts": {"source": len(source_groups), "tag": len(tag_groups)},
                        "low_sample_group_counts": {"source": 0, "tag": 0},
                        "cluster_counts": {"source": len(source_clusters), "tag": len(tag_clusters)},
                    },
                    "groups": {"source": source_groups, "tag": tag_groups},
                    "clusters": {"source": source_clusters, "tag": tag_clusters},
                },
                "group_temporal_latent_space": {
                    "config": {"bucket_granularity": "week"},
                    "groups": {"source": temporal_source_groups, "tag": temporal_tag_groups},
                },
            }
        },
    }


class NewsGroupLatentSpaceTests(unittest.TestCase):
    def test_group_options_include_counts_and_low_sample_marker(self):
        options = _group_options(
            [
                {"group": "Source A", "group_key": "source-a", "n_articles": 5, "status": "ok"},
                {"group": "Source B", "group_key": "source-b", "n_articles": 2, "status": "low_sample"},
            ]
        )

        self.assertEqual(options[0], {"label": "Source A (5)", "value": "source-a"})
        self.assertEqual(options[1], {"label": "Source B (2) low sample", "value": "source-b"})

    def test_cluster_options_include_group_and_article_counts(self):
        options = _cluster_options(
            [
                {"cluster_id": "source-cluster-1", "label": "Source A, Source B", "n_groups": 2, "n_articles": 11},
                {"cluster_id": "source-cluster-2", "label": "Source C, Source D", "n_groups": 2, "n_articles": 9},
            ]
        )

        self.assertEqual(
            options[0],
            {
                "label": "All groups",
                "value": ALL_GROUPS_CLUSTER_VALUE,
            },
        )
        self.assertEqual(
            options[1],
            {
                "label": "Source A, Source B (2 groups, 11 articles)",
                "value": "source-cluster-1",
            },
        )
        self.assertEqual(
            options[2],
            {
                "label": "Source C, Source D (2 groups, 9 articles)",
                "value": "source-cluster-2",
            },
        )

    def test_selected_group_row_falls_back_to_first_row(self):
        rows = [
            {"group": "Source A", "group_key": "source-a"},
            {"group": "Source B", "group_key": "source-b"},
        ]

        self.assertEqual(_selected_group_row(rows, "source-b"), rows[1])
        self.assertEqual(_selected_group_row(rows, "missing"), rows[0])
        self.assertIsNone(_selected_group_row([], "missing"))

    def test_selected_cluster_row_matches_selected_group(self):
        selected_group = {"group": "Source A", "cluster_id": "source-cluster-2"}
        clusters = [
            {"cluster_id": "source-cluster-1", "label": "Cluster 1"},
            {"cluster_id": "source-cluster-2", "label": "Cluster 2"},
        ]

        self.assertEqual(_selected_cluster_row(clusters, selected_group), clusters[1])
        self.assertIsNone(_selected_cluster_row(clusters, {"group": "Source B"}))

    def test_selected_cluster_row_prefers_explicit_cluster_id(self):
        selected_group = {"group": "Source A", "cluster_id": "source-cluster-1"}
        clusters = [
            {"cluster_id": "source-cluster-1", "label": "Cluster 1"},
            {"cluster_id": "source-cluster-2", "label": "Cluster 2"},
        ]

        self.assertEqual(_selected_cluster_row(clusters, selected_group, "source-cluster-2"), clusters[1])

    def test_selected_group_row_falls_back_to_first_group_in_selected_cluster(self):
        rows = [
            {"group": "Source A", "group_key": "source-a", "cluster_id": "source-cluster-1"},
            {"group": "Source B", "group_key": "source-b", "cluster_id": "source-cluster-1"},
            {"group": "Source C", "group_key": "source-c", "cluster_id": "source-cluster-2"},
        ]

        selected_cluster = {"cluster_id": "source-cluster-2"}

        self.assertEqual(_selected_group_row(rows, "source-a", selected_cluster), rows[2])
        self.assertEqual(_selected_group_row(rows, None, selected_cluster), rows[2])

    def test_selected_group_row_normalizes_source_key_variants(self):
        rows = [
            {"group": "Source A", "group_key": "source-a", "cluster_id": "source-cluster-1"},
            {"group": "Source B", "group_key": "source-b", "cluster_id": "source-cluster-1"},
        ]

        self.assertEqual(_selected_group_row(rows, "Source_A"), rows[0])
        self.assertEqual(_selected_group_row(rows, "source a"), rows[0])

    def test_group_rows_for_cluster_supports_filtering_and_all_groups_escape_hatch(self):
        rows = [
            {"group": "Source A", "group_key": "source-a", "cluster_id": "source-cluster-1"},
            {"group": "Source B", "group_key": "source-b", "cluster_id": "source-cluster-1"},
            {"group": "Source C", "group_key": "source-c", "cluster_id": "source-cluster-2"},
        ]

        self.assertEqual(
            _group_rows_for_cluster(rows, "source-cluster-2"),
            [rows[2]],
        )
        self.assertEqual(
            _group_rows_for_cluster(rows, ALL_GROUPS_CLUSTER_VALUE),
            rows,
        )

    def test_cluster_filter_hint_describes_all_groups_and_selected_cluster_states(self):
        rows = [
            {"group": "Source A", "group_key": "source-a", "cluster_id": "source-cluster-1"},
            {"group": "Source B", "group_key": "source-b", "cluster_id": "source-cluster-1"},
            {"group": "Source C", "group_key": "source-c", "cluster_id": "source-cluster-2"},
        ]

        all_groups_hint = _cluster_filter_hint(rows, rows, None)
        filtered_hint = _cluster_filter_hint(
            rows,
            rows[:2],
            {"cluster_id": "source-cluster-1", "label": "Source A, Source B"},
        )

        all_groups_text = " ".join(str(node) for node in _iter_children(all_groups_hint) if isinstance(node, str))
        filtered_text = " ".join(str(node) for node in _iter_children(filtered_hint) if isinstance(node, str))

        self.assertIn("Cluster filter: All groups.", all_groups_text)
        self.assertIn("shows all 3 available options.", all_groups_text)
        self.assertIn("Cluster filter: Source A, Source B.", filtered_text)
        self.assertIn("limited to 2 options in this cluster.", filtered_text)

    def test_group_temporal_export_href_tracks_snapshot_mode(self):
        self.assertEqual(
            _group_temporal_export_href({"source_mode": "current"}, group_type="source"),
            "/api/news/export?artifact=group_temporal_buckets&format=csv&group_type=source",
        )
        self.assertEqual(
            _group_temporal_export_href(
                {"source_mode": "snapshot", "snapshot_date": "2026-05-01"},
                export_format="json",
                group_type="source",
                group_key="source-c",
            ),
            "/api/news/export?artifact=group_temporal_buckets&format=json&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
        )

    def test_group_temporal_export_panel_renders_bucket_and_momentum_export_buttons(self):
        component = _group_temporal_export_panel(
            {"source_mode": "snapshot", "snapshot_date": "2026-05-01"},
            "week",
            "source",
            "source-c",
        )

        self.assertIsInstance(component, dbc.Card)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Export Temporal Movement Data", rendered_text)
        self.assertIn("selected source temporal movement rows", rendered_text)
        self.assertIn("Exports follow the current dataset mode: snapshot (2026-05-01).", rendered_text)
        self.assertIn("Bucket rows preserve per-week centroid coordinates, counts, and sparse-bucket diagnostics.", rendered_text)
        self.assertIn("Path summary rows preserve movement totals, direction deltas, and coverage-gap ranges for each group.", rendered_text)
        self.assertIn("Popularity momentum rows preserve peak bucket volume", rendered_text)
        self.assertIn("interpretable momentum labels for each group.", rendered_text)

        buttons = _find_components(component, dbc.Button)
        hrefs = [getattr(button, "href", None) for button in buttons]
        self.assertEqual(
            hrefs,
            [
                "/api/news/export?artifact=group_temporal_buckets&format=csv&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_buckets&format=json&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_path_summary&format=csv&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_path_summary&format=json&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_popularity_momentum&format=csv&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_popularity_momentum&format=json&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
            ],
        )

    def test_centroid_figure_uses_only_rows_with_coordinates(self):
        figure = _centroid_figure(
            [
                {"group": "Source A", "group_key": "source-a", "n_articles": 5, "status": "ok", "pc1": 0.2, "pc2": 0.4, "dispersion_pca": 0.1},
                {"group": "Source B", "group_key": "source-b", "n_articles": 1, "status": "low_sample", "pc1": None, "pc2": 0.6, "dispersion_pca": 0.2},
            ],
            "source-a",
            "pc1",
            "pc2",
            "PCA Group Centroids",
        )

        self.assertEqual(figure.layout.title.text, "PCA Group Centroids")
        self.assertEqual(len(figure.data), 1)
        self.assertEqual(list(figure.data[0].x), [0.2])
        self.assertEqual(list(figure.data[0].y), [0.4])

    def test_centroid_figure_highlights_label_only_rows_with_normalized_selection(self):
        figure = _centroid_figure(
            [
                {"group": "Source A", "n_articles": 5, "status": "ok", "pc1": 0.2, "pc2": 0.4, "dispersion_pca": 0.1},
                {"group": "Source B", "group_key": "source-b", "n_articles": 4, "status": "ok", "pc1": 0.5, "pc2": 0.1, "dispersion_pca": 0.2},
            ],
            "Source_A",
            "pc1",
            "pc2",
            "PCA Group Centroids",
        )

        marker = figure.data[0].marker
        self.assertEqual(list(marker.color), ["#dc3545", "#0d6efd"])
        self.assertEqual(list(marker.opacity), [1.0, 0.78])
        self.assertEqual(list(marker.line.width), [2, 1])

    def test_cluster_membership_summary_renders_representatives_and_lenses(self):
        component = _cluster_membership_summary(
            {
                "cluster_id": "source-cluster-1",
                "label": "Source A, Source B",
                "n_groups": 2,
                "n_articles": 11,
                "n_sources": 2,
                "clustering_threshold_pca": 0.187,
                "representative_groups": [
                    {"group": "Source A", "n_articles": 6},
                    {"group": "Source B", "n_articles": 5},
                ],
                "defining_lens_deviations": [
                    {"lens": "Evidence", "delta": 9.2},
                    {"lens": "Impact", "delta": -5.6},
                ],
            }
        )

        self.assertIsInstance(component, dbc.Card)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Cluster Membership", rendered_text)
        self.assertIn("Source A", rendered_text)
        self.assertIn("Source B", rendered_text)
        self.assertIn("Evidence (9.20)", rendered_text)

    def test_cluster_overview_table_renders_cluster_comparison_rows(self):
        selected_cluster = {"cluster_id": "source-cluster-2"}
        component = _cluster_overview_table(
            [
                {
                    "cluster_id": "source-cluster-1",
                    "label": "Source A, Source B",
                    "n_groups": 2,
                    "n_articles": 11,
                    "representative_groups": [{"group": "Source A"}, {"group": "Source B"}],
                    "defining_lens_deviations": [{"lens": "Evidence", "delta": 9.2}],
                },
                {
                    "cluster_id": "source-cluster-2",
                    "label": "Source C, Source D",
                    "n_groups": 2,
                    "n_articles": 9,
                    "representative_groups": [{"group": "Source C"}, {"group": "Source D"}],
                    "defining_lens_deviations": [{"lens": "Impact", "delta": -5.6}],
                },
            ],
            selected_cluster,
        )

        self.assertIsInstance(component, dbc.Card)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Cluster Overview", rendered_text)
        self.assertIn("Source A, Source B", rendered_text)
        self.assertIn("Source C, Source D", rendered_text)
        self.assertIn("Impact (-5.60)", rendered_text)

        highlighted_rows = [
            row for row in _find_components(component, html.Tr) if getattr(row, "className", None) == "table-primary"
        ]
        self.assertEqual(len(highlighted_rows), 1)

    def test_cluster_scope_summary_uses_selected_cluster_counts(self):
        summary = _cluster_scope_summary(
            [
                {"cluster_id": "source-cluster-1", "label": "Source A, Source B", "n_groups": 2, "n_articles": 11},
                {"cluster_id": "source-cluster-2", "label": "Source C, Source D", "n_groups": 2, "n_articles": 9},
            ],
            {"cluster_id": "source-cluster-2", "label": "Source C, Source D", "n_groups": 2, "n_articles": 9},
        )

        rendered_text = " ".join(str(node) for node in _iter_children(summary) if isinstance(node, str))
        self.assertIn("Scope: Source C, Source D.", rendered_text)
        self.assertIn("2 groups, 9 articles.", rendered_text)

    def test_group_table_renders_only_cluster_filtered_rows(self):
        rows = [
            {"group": "Source A", "group_key": "source-a", "status": "ok", "n_articles": 6, "cluster_id": "source-cluster-1", "cluster_label": "Cluster 1", "pc1": 0.31, "pc2": 0.18},
            {"group": "Source B", "group_key": "source-b", "status": "ok", "n_articles": 5, "cluster_id": "source-cluster-1", "cluster_label": "Cluster 1", "pc1": 0.27, "pc2": 0.12},
            {"group": "Source C", "group_key": "source-c", "status": "low_sample", "n_articles": 2, "cluster_id": "source-cluster-2", "cluster_label": "Cluster 2", "pc1": -0.22, "pc2": 0.44},
        ]

        filtered_rows = _group_rows_for_cluster(rows, "source-cluster-2")
        component = _group_table(filtered_rows, "source-c")

        self.assertIsInstance(component, dbc.Card)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Top Groups", rendered_text)
        self.assertIn("Source C", rendered_text)
        self.assertNotIn("Source A", rendered_text)
        self.assertNotIn("Source B", rendered_text)

        highlighted_rows = [
            row for row in _find_components(component, html.Tr) if getattr(row, "className", None) == "table-primary"
        ]
        self.assertEqual(len(highlighted_rows), 1)

    def test_group_scope_summary_uses_selected_cluster_counts(self):
        rows = [
            {"group": "Source A", "group_key": "source-a", "n_articles": 6},
            {"group": "Source B", "group_key": "source-b", "n_articles": 5},
        ]

        summary = _group_scope_summary(
            rows,
            {"cluster_id": "source-cluster-1", "label": "Source A, Source B", "n_groups": 2, "n_articles": 11},
        )

        rendered_text = " ".join(str(node) for node in _iter_children(summary) if isinstance(node, str))
        self.assertIn("Scope: Source A, Source B.", rendered_text)
        self.assertIn("2 groups, 11 articles.", rendered_text)

    def test_selected_temporal_group_row_matches_group_key_and_falls_back(self):
        temporal_payload = {
            "groups": {
                "source": [
                    {"group": "Source A", "group_key": "source-a"},
                    {"group": "Source B", "group_key": "source-b"},
                ]
            }
        }

        self.assertEqual(_selected_temporal_group_row(temporal_payload, "source", "source-b"), temporal_payload["groups"]["source"][1])
        self.assertEqual(_selected_temporal_group_row(temporal_payload, "source", "missing"), temporal_payload["groups"]["source"][0])
        self.assertIsNone(_selected_temporal_group_row({}, "source", "missing"))

    def test_selected_temporal_group_row_normalizes_source_key_variants(self):
        temporal_payload = {
            "groups": {
                "source": [
                    {"group": "Source A", "group_key": "source-a"},
                    {"group": "Source B", "group_key": "source-b"},
                ]
            }
        }

        self.assertEqual(_selected_temporal_group_row(temporal_payload, "source", "Source_A"), temporal_payload["groups"]["source"][0])
        self.assertEqual(_selected_temporal_group_row(temporal_payload, "source", "source a"), temporal_payload["groups"]["source"][0])

    def test_group_movement_figure_renders_weekly_pca_path(self):
        figure = _group_movement_figure(
            {
                "group": "Source A",
                "buckets": [
                    {"bucket_start": "2026-03-02", "n_articles": 4, "status": "ok", "corpus_share": 0.4, "dispersion_pca": 0.13, "pc1": 0.1, "pc2": 0.2},
                    {"bucket_start": "2026-03-09", "n_articles": 3, "status": "sparse", "corpus_share": 0.3, "dispersion_pca": 0.19, "pc1": 0.4, "pc2": 0.5},
                ],
            },
            "week",
        )

        self.assertEqual(figure.layout.title.text, "Week PCA Centroid Path")
        self.assertEqual(len(figure.data), 3)
        self.assertEqual(list(figure.data[0].x), [0.1, 0.4])
        self.assertEqual(list(figure.data[0].y), [0.2, 0.5])
        self.assertEqual(list(figure.data[1].text), ["Start"])
        self.assertEqual(list(figure.data[2].text), ["Latest"])

    def test_group_movement_figure_renders_weekly_mds_path(self):
        figure = _group_movement_figure(
            {
                "group": "Source A",
                "buckets": [
                    {"bucket_start": "2026-03-02", "n_articles": 4, "status": "ok", "corpus_share": 0.4, "dispersion_mds": 0.11, "mds1": -0.2, "mds2": 0.3},
                    {"bucket_start": "2026-03-09", "n_articles": 3, "status": "ok", "corpus_share": 0.3, "dispersion_mds": 0.17, "mds1": 0.1, "mds2": 0.6},
                ],
            },
            "week",
            "mds",
        )

        self.assertEqual(figure.layout.title.text, "Week MDS Centroid Path")
        self.assertEqual(figure.layout.xaxis.title.text, "MDS1")
        self.assertEqual(figure.layout.yaxis.title.text, "MDS2")
        self.assertEqual(list(figure.data[0].x), [-0.2, 0.1])
        self.assertEqual(list(figure.data[0].y), [0.3, 0.6])

    def test_group_movement_figure_returns_empty_chart_when_selected_basis_has_no_path(self):
        figure = _group_movement_figure(
            {
                "group": "Source A",
                "buckets": [
                    {"bucket_start": "2026-03-02", "n_articles": 4, "status": "ok", "pc1": 0.1, "pc2": 0.2},
                ],
            },
            "week",
            "mds",
        )

        self.assertEqual(figure.layout.title.text, "Week MDS Centroid Path")
        self.assertEqual(len(figure.data), 0)
        self.assertEqual(figure.layout.annotations[0].text, "No MDS centroid path is available for the selected group.")

    def test_group_movement_summary_surfaces_sparse_group_reason(self):
        component = _group_movement_summary(
            {
                "status": "sparse",
                "reason": "Need at least 2 non-sparse buckets with PCA coordinates.",
                "date_start": "2026-03-02",
                "date_end": "2026-03-09",
                "path_summary": {
                    "bucket_count": 2,
                    "valid_pca_bucket_count": 1,
                    "sparse_bucket_count": 1,
                    "coverage_gap_count": 0,
                    "total_movement_pca": 0.0,
                    "largest_jump_pca": 0.0,
                    "direction_pca": None,
                },
            },
            "week",
            "pca",
            {"cluster_id": "source-cluster-2", "label": "Source C Cluster"},
        )

        self.assertIsInstance(component, dbc.Card)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Movement Summary (PCA)", rendered_text)
        self.assertIn("Cluster scope: Source C Cluster.", rendered_text)
        self.assertIn("Status", rendered_text)
        self.assertIn("sparse", rendered_text)
        self.assertIn("Need at least 2 non-sparse buckets", rendered_text)

    def test_group_movement_summary_surfaces_coverage_gap_ranges(self):
        component = _group_movement_summary(
            {
                "status": "ok",
                "date_start": "2026-03-02",
                "date_end": "2026-03-29",
                "path_summary": {
                    "bucket_count": 4,
                    "valid_pca_bucket_count": 2,
                    "sparse_bucket_count": 0,
                    "coverage_gap_count": 1,
                    "coverage_gap_ranges": [
                        {
                            "start_bucket": "2026-03-09",
                            "end_bucket": "2026-03-16",
                            "missing_bucket_count": 2,
                            "label": "2026-03-09 to 2026-03-16",
                        }
                    ],
                    "total_movement_pca": 0.52,
                    "largest_jump_pca": 0.31,
                    "direction_pca": {"pc1_delta": 0.2, "pc2_delta": -0.1},
                },
            },
            "week",
            "pca",
            {"cluster_id": "source-cluster-2", "label": "Source C Cluster"},
        )

        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Movement Summary (PCA)", rendered_text)
        self.assertIn("Coverage gaps", rendered_text)
        self.assertIn("1", rendered_text)
        self.assertIn("Missing week bucket range: 2026-03-09 to 2026-03-16.", rendered_text)

    def test_group_movement_summary_uses_mds_metrics_when_requested(self):
        component = _group_movement_summary(
            {
                "status": "ok",
                "date_start": "2026-03-02",
                "date_end": "2026-03-16",
                "path_summary": {
                    "bucket_count": 3,
                    "valid_mds_bucket_count": 3,
                    "sparse_bucket_count": 0,
                    "coverage_gap_count": 1,
                    "total_movement_mds": 0.41,
                    "largest_jump_mds": 0.24,
                    "direction_mds": {"mds1_delta": 0.2, "mds2_delta": -0.1},
                },
            },
            "week",
            "mds",
            {"cluster_id": "source-cluster-1", "label": "Source A, Source B"},
        )

        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Movement Summary (MDS)", rendered_text)
        self.assertIn("Cluster scope: Source A, Source B.", rendered_text)
        self.assertIn("Valid MDS buckets", rendered_text)
        self.assertIn("0.410", rendered_text)
        self.assertIn("MDS1 0.20, MDS2 -0.10", rendered_text)

    def test_group_popularity_timeline_renders_dual_axis_trend(self):
        temporal_row = _sample_group_latent_payload()["data"]["derived"]["group_temporal_latent_space"]["groups"]["tag"][0]

        component = _group_popularity_timeline(temporal_row, "week")

        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Popularity Timeline", rendered_text)
        self.assertIn("Peak volume lands in 2026-03-09 with 3 articles.", rendered_text)
        self.assertIn("Latest corpus share eases to 14.0% (2.0 pts below the first bucket).", rendered_text)

        graphs = _find_components(component, dcc.Graph)
        self.assertEqual(len(graphs), 1)
        figure = graphs[0].figure
        self.assertEqual(figure.layout.title.text, "Week Popularity Timeline")
        self.assertEqual(len(figure.data), 2)
        self.assertEqual(figure.data[0].type, "bar")
        self.assertEqual(figure.data[1].type, "scatter")

    def test_group_popularity_timeline_warns_without_bucket_rows(self):
        component = _group_popularity_timeline({"group": "Source A"}, "week")

        self.assertIsInstance(component, dbc.Alert)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("No popularity timeline is available for the current selection.", rendered_text)

    def test_group_temporal_scope_summary_surfaces_bucket_coverage_for_selected_group(self):
        component = _group_temporal_scope_summary(
            {
                "group": "Source C",
                "n_articles": 9,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "buckets": [
                    {
                        "bucket_start": "2026-03-02",
                        "bucket_label": "2026-03-02",
                        "n_articles": 4,
                        "n_sources": 2,
                        "status": "ok",
                        "corpus_share": 0.4,
                        "pc1": 0.1,
                        "pc2": 0.2,
                        "source_counts": {"Source C": 3, "Source A": 1},
                        "top_lens_deviations": [
                            {"lens": "Evidence", "delta": 8.2},
                            {"lens": "Impact", "delta": -4.1},
                        ],
                    }
                ],
                "path_summary": {
                    "bucket_count": 3,
                    "valid_pca_bucket_count": 2,
                    "sparse_bucket_count": 1,
                    "coverage_gap_count": 1,
                    "coverage_gap_ranges": [
                        {
                            "start_bucket": "2026-03-09",
                            "end_bucket": "2026-03-09",
                            "missing_bucket_count": 1,
                            "label": "2026-03-09",
                        }
                    ],
                },
            },
            "week",
            "pca",
            {"cluster_id": "source-cluster-2", "label": "Source C Cluster"},
        )

        self.assertIsInstance(component, dbc.Alert)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Temporal scope: Source C spans 3 week buckets", rendered_text)
        self.assertIn("from 2026-03-02 to 2026-03-22", rendered_text)
        self.assertIn("across 9 articles.", rendered_text)
        self.assertIn("Cluster scope: Source C Cluster.", rendered_text)
        self.assertIn("Movement basis: PCA centroid path.", rendered_text)
        self.assertIn("2 valid PCA buckets, 1 sparse bucket, 1 coverage gap.", rendered_text)
        self.assertIn("Missing week bucket range: 2026-03-09.", rendered_text)

    def test_group_temporal_scope_summary_formats_multi_bucket_gap_ranges(self):
        component = _group_temporal_scope_summary(
            {
                "group": "Source C",
                "n_articles": 9,
                "n_buckets": 4,
                "date_start": "2026-03-02",
                "date_end": "2026-03-29",
                "path_summary": {
                    "bucket_count": 4,
                    "valid_pca_bucket_count": 2,
                    "sparse_bucket_count": 0,
                    "coverage_gap_count": 1,
                    "coverage_gap_ranges": [
                        {
                            "start_bucket": "2026-03-09",
                            "end_bucket": "2026-03-16",
                            "missing_bucket_count": 2,
                            "label": "2026-03-09 to 2026-03-16",
                        }
                    ],
                },
            },
            "week",
            "pca",
            {"cluster_id": "source-cluster-2", "label": "Source C Cluster"},
        )

        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Temporal scope: Source C spans 4 week buckets", rendered_text)
        self.assertIn("2 valid PCA buckets, 0 sparse buckets, 1 coverage gap.", rendered_text)
        self.assertIn("Missing week bucket range: 2026-03-09 to 2026-03-16.", rendered_text)

    def test_bucket_coordinate_summary_switches_between_pca_and_mds_axes(self):
        bucket_row = {
            "pc1": 0.4,
            "pc2": 0.5,
            "mds1": -0.2,
            "mds2": 0.3,
        }

        self.assertEqual(_bucket_coordinate_summary(bucket_row, "pca"), "PC1 0.40, PC2 0.50")
        self.assertEqual(_bucket_coordinate_summary(bucket_row, "mds"), "MDS1 -0.20, MDS2 0.30")

    def test_temporal_bucket_diagnostics_table_surfaces_source_and_lens_summaries(self):
        component = _temporal_bucket_diagnostics_table(
            _sample_group_latent_payload()["data"]["derived"]["group_temporal_latent_space"]["groups"]["source"][0],
            "week",
            "pca",
        )

        self.assertIsInstance(component, html.Div)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Temporal Bucket Diagnostics", rendered_text)
        self.assertIn("Week buckets for the selected group. Coordinates follow the PCA movement basis.", rendered_text)
        self.assertIn("2026-03-02", rendered_text)
        self.assertIn("Source C (3), Source A (1)", rendered_text)
        self.assertIn("Evidence (8.2), Impact (-4.1)", rendered_text)
        self.assertIn("PC1 0.10, PC2 0.20", rendered_text)

    def test_selected_group_summary_reuses_temporal_scope_summary(self):
        component = _selected_group_summary(
            {
                "group": "Source C",
                "status": "ok",
                "n_articles": 9,
                "n_sources": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "cluster_label": "Source C Cluster",
                "dispersion_pca": 0.233,
                "dispersion_mds": 0.187,
            },
            {
                "group": "Source C",
                "n_articles": 9,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "buckets": [
                    {
                        "bucket_start": "2026-03-02",
                        "bucket_label": "2026-03-02",
                        "n_articles": 4,
                        "n_sources": 2,
                        "status": "ok",
                        "corpus_share": 0.4,
                        "pc1": 0.1,
                        "pc2": 0.2,
                        "source_counts": {"Source C": 3, "Source A": 1},
                        "top_lens_deviations": [
                            {"lens": "Evidence", "delta": 8.2},
                            {"lens": "Impact", "delta": -4.1},
                        ],
                    }
                ],
                "path_summary": {
                    "bucket_count": 3,
                    "valid_pca_bucket_count": 2,
                    "sparse_bucket_count": 1,
                    "coverage_gap_count": 1,
                    "coverage_gap_ranges": [
                        {
                            "start_bucket": "2026-03-09",
                            "end_bucket": "2026-03-09",
                            "missing_bucket_count": 1,
                            "label": "2026-03-09",
                        }
                    ],
                },
            },
            "week",
            "pca",
            {"cluster_id": "source-cluster-2", "label": "Source C Cluster"},
        )

        self.assertIsInstance(component, dbc.Card)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Temporal scope: Source C spans 3 week buckets", rendered_text)
        self.assertIn("Cluster scope: Source C Cluster.", rendered_text)
        self.assertIn("Movement basis: PCA centroid path.", rendered_text)
        self.assertIn("Missing week bucket range: 2026-03-09.", rendered_text)
        self.assertIn("Status", rendered_text)
        self.assertIn("Source C Cluster", rendered_text)
        self.assertIn("Popularity Timeline", rendered_text)
        self.assertIn("Peak volume lands in 2026-03-02 with 4 articles.", rendered_text)
        self.assertIn("Temporal Bucket Diagnostics", rendered_text)
        self.assertIn("Source C (3), Source A (1)", rendered_text)
        self.assertIn("Evidence (8.2), Impact (-4.1)", rendered_text)

    def test_movement_pattern_callout_distinguishes_steady_drift_and_jump_led_paths(self):
        steady_component = _movement_pattern_callout(
            _sample_group_latent_payload()["data"]["derived"]["group_temporal_latent_space"]["groups"]["source"][0],
            "pca",
        )
        jump_component = _movement_pattern_callout(
            {
                "path_summary": {
                    "valid_pca_bucket_count": 3,
                    "total_movement_pca": 0.40,
                    "largest_jump_pca": 0.34,
                    "direction_pca": {"pc1_delta": 0.18, "pc2_delta": -0.04},
                }
            },
            "pca",
        )

        steady_text = " ".join(str(node) for node in _iter_children(steady_component) if isinstance(node, str))
        jump_text = " ".join(str(node) for node in _iter_children(jump_component) if isinstance(node, str))

        self.assertIn("Movement pattern: steady drift.", steady_text)
        self.assertIn("Largest jump contributes 41.8% of total PCA movement", steady_text)
        self.assertIn("Net direction: PC1 0.50, PC2 0.50.", steady_text)
        self.assertIn("Movement pattern: jump-led.", jump_text)
        self.assertIn("Largest jump contributes 85.0% of total PCA movement", jump_text)
        self.assertIn("Net direction: PC1 0.18, PC2 -0.04.", jump_text)

    def test_selected_group_summary_surfaces_temporal_scope_fallback_when_no_temporal_row_exists(self):
        component = _selected_group_summary(
            {
                "group": "Source C",
                "status": "ok",
                "n_articles": 9,
                "n_sources": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "cluster_label": "Source C Cluster",
                "dispersion_pca": 0.233,
                "dispersion_mds": 0.187,
            },
            None,
            "week",
        )

        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("No temporal bucket coverage is available for the current selection.", rendered_text)
        self.assertIn("Source C Cluster", rendered_text)

    def test_selected_group_summary_surfaces_popularity_momentum_and_share_drift_rank(self):
        payload = _sample_group_latent_payload(include_extra_cluster_groups=True)
        rows = payload["data"]["derived"]["group_latent_space"]["groups"]["source"]
        temporal_payload = payload["data"]["derived"]["group_temporal_latent_space"]
        selected_row = next(row for row in rows if row["group_key"] == "source-b")
        temporal_row = _selected_temporal_group_row(temporal_payload, "source", "source-b")

        component = _selected_group_summary(
            selected_row,
            temporal_row,
            "week",
            "pca",
            {"cluster_id": "source-cluster-1", "label": "Source A, Source B"},
            rows,
            temporal_payload,
            "source",
        )

        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Popularity Momentum", rendered_text)
        self.assertIn("still falling", rendered_text)
        self.assertIn("Share Drift Rank", rendered_text)
        self.assertIn("2 of 2 in Source A, Source B", rendered_text)
        self.assertIn("Rank Trajectory", rendered_text)
        self.assertIn("2 -> 2 in Source A, Source B (held)", rendered_text)
        self.assertIn("Recent Step", rendered_text)
        self.assertIn("falling (-8.0 pts)", rendered_text)

    def test_selected_group_summary_uses_label_only_topic_popularity_rows(self):
        selected_row = {
            "group": "Climate Risk",
            "status": "ok",
            "n_articles": 7,
            "n_sources": 2,
            "cluster_id": "topic-cluster-1",
            "cluster_label": "Policy and Climate",
        }
        all_group_rows = [
            {
                "group": "AI Policy",
                "status": "ok",
                "n_articles": 8,
                "cluster_id": "topic-cluster-1",
                "cluster_label": "Policy and Climate",
            },
            selected_row,
        ]
        temporal_row = {
            "group": "Climate Risk",
            "status": "ok",
            "n_articles": 7,
            "n_buckets": 3,
            "date_start": "2026-03-02",
            "date_end": "2026-03-22",
            "path_summary": {
                "bucket_count": 3,
                "valid_pca_bucket_count": 3,
                "sparse_bucket_count": 0,
                "coverage_gap_count": 0,
            },
        }
        popularity_rows = [
            {
                "group": "AI Policy",
                "n_articles": 8,
                "peak_bucket": "2026-03-02",
                "peak_articles": 3,
                "first_share": 0.22,
                "latest_share": 0.16,
                "share_delta": -0.06,
                "recent_share_delta": -0.02,
                "momentum_label": "still falling",
            },
            {
                "group": "Climate Risk",
                "n_articles": 7,
                "peak_bucket": "2026-03-09",
                "peak_articles": 4,
                "first_share": 0.14,
                "latest_share": 0.20,
                "share_delta": 0.06,
                "recent_share_delta": 0.03,
                "momentum_label": "rising now",
            },
        ]

        with patch.object(news_group_latent_space, "_cluster_popularity_rows", return_value=popularity_rows):
            component = _selected_group_summary(
                selected_row,
                temporal_row,
                "week",
                "pca",
                {"cluster_id": "topic-cluster-1", "label": "Policy and Climate"},
                all_group_rows,
                {"groups": {"topic": [temporal_row]}},
                "topic",
            )

        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Popularity Momentum", rendered_text)
        self.assertIn("rising now", rendered_text)
        self.assertIn("Share Drift Rank", rendered_text)
        self.assertIn("1 of 2 in Policy and Climate", rendered_text)
        self.assertIn("Rank Trajectory", rendered_text)
        self.assertIn("2 -> 1 in Policy and Climate (up 1)", rendered_text)
        self.assertIn("Recent Step", rendered_text)
        self.assertIn("+3.0 pts", rendered_text)

    def test_cluster_peer_movement_callout_compares_selected_group_against_cluster_peers(self):
        payload = _sample_group_latent_payload(include_extra_cluster_groups=True)
        rows = payload["data"]["derived"]["group_latent_space"]["groups"]["source"]
        temporal_payload = payload["data"]["derived"]["group_temporal_latent_space"]
        selected_row = next(row for row in rows if row["group_key"] == "source-a")

        component = _cluster_peer_movement_callout(rows, selected_row, temporal_payload, "source", "pca")

        self.assertIsInstance(component, dbc.Alert)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Cluster movement context: Source A has PCA total movement 0.160", rendered_text)
        self.assertIn("0.030 above the Source A, Source B peer average of 0.130", rendered_text)
        self.assertIn("ranks 1 of 2 in its cluster.", rendered_text)

    def test_cluster_peer_movement_callout_handles_single_group_cluster(self):
        payload = _sample_group_latent_payload(include_extra_cluster_groups=True)
        rows = payload["data"]["derived"]["group_latent_space"]["groups"]["source"]
        temporal_payload = payload["data"]["derived"]["group_temporal_latent_space"]
        selected_row = next(row for row in rows if row["group_key"] == "source-c")

        component = _cluster_peer_movement_callout(rows, selected_row, temporal_payload, "source", "mds")

        self.assertIsInstance(component, dbc.Alert)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Source C Cluster contains only this selected group", rendered_text)

    def test_cluster_popularity_comparison_ranks_cluster_scope_and_highlights_selection(self):
        payload = _sample_group_latent_payload(include_extra_cluster_groups=True)
        rows = payload["data"]["derived"]["group_latent_space"]["groups"]["source"]
        temporal_payload = payload["data"]["derived"]["group_temporal_latent_space"]
        selected_row = next(row for row in rows if row["group_key"] == "source-b")

        component = _cluster_popularity_comparison(
            rows,
            selected_row,
            temporal_payload,
            "source",
            "week",
            {"cluster_id": "source-cluster-1", "label": "Source A, Source B"},
        )

        self.assertIsInstance(component, html.Div)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Popularity Comparison", rendered_text)
        self.assertIn(
            "Compares starting-to-latest week corpus-share rank, latest share, and peak bucket volume within Source A, Source B.",
            rendered_text,
        )
        self.assertIn("Showing 2 of 2 groups with bucket rows.", rendered_text)
        self.assertIn("2 -> 2 (held)", rendered_text)
        self.assertIn("Flat", rendered_text)
        self.assertIn("-7.0 pts", rendered_text)

        table_body = _find_components(component, html.Tbody)[0]
        row_components = [child for child in table_body.children if isinstance(child, html.Tr)]
        ordered_groups = [str(row.children[1].children) for row in row_components]
        self.assertEqual(ordered_groups, ["Source A", "Source B"])

        highlighted_rows = [row for row in row_components if getattr(row, "className", None) == "table-primary"]
        self.assertEqual(len(highlighted_rows), 1)
        self.assertEqual(str(highlighted_rows[0].children[1].children), "Source B")

    def test_cluster_popularity_comparison_surfaces_rank_trajectory_changes(self):
        rows = [
            {"group": "Alpha", "group_key": "alpha", "cluster_id": "source-cluster-1", "n_articles": 7},
            {"group": "Beta", "group_key": "beta", "cluster_id": "source-cluster-1", "n_articles": 6},
            {"group": "Gamma", "group_key": "gamma", "cluster_id": "source-cluster-1", "n_articles": 5},
        ]
        temporal_payload = {
            "groups": {
                "source": [
                    {
                        "group": "Alpha",
                        "group_key": "alpha",
                        "buckets": [
                            {"bucket_label": "2026-03-02", "n_articles": 3, "corpus_share": 0.30},
                            {"bucket_label": "2026-03-09", "n_articles": 2, "corpus_share": 0.18},
                            {"bucket_label": "2026-03-16", "n_articles": 2, "corpus_share": 0.10},
                        ],
                    },
                    {
                        "group": "Beta",
                        "group_key": "beta",
                        "buckets": [
                            {"bucket_label": "2026-03-02", "n_articles": 2, "corpus_share": 0.20},
                            {"bucket_label": "2026-03-09", "n_articles": 2, "corpus_share": 0.22},
                            {"bucket_label": "2026-03-16", "n_articles": 2, "corpus_share": 0.25},
                        ],
                    },
                    {
                        "group": "Gamma",
                        "group_key": "gamma",
                        "buckets": [
                            {"bucket_label": "2026-03-02", "n_articles": 1, "corpus_share": 0.10},
                            {"bucket_label": "2026-03-09", "n_articles": 2, "corpus_share": 0.12},
                            {"bucket_label": "2026-03-16", "n_articles": 2, "corpus_share": 0.15},
                        ],
                    },
                ]
            }
        }

        component = _cluster_popularity_comparison(
            rows,
            rows[1],
            temporal_payload,
            "source",
            "week",
            {"cluster_id": "source-cluster-1", "label": "Alpha, Beta, Gamma"},
        )

        table_body = _find_components(component, html.Tbody)[0]
        row_components = [child for child in table_body.children if isinstance(child, html.Tr)]
        ordered_groups = [str(row.children[1].children) for row in row_components]
        rank_trajectories = [str(row.children[7].children) for row in row_components]

        self.assertEqual(ordered_groups, ["Beta", "Gamma", "Alpha"])
        self.assertEqual(rank_trajectories, ["2 -> 1 (up 1)", "3 -> 2 (up 1)", "1 -> 3 (down 2)"])

    def test_cluster_share_drift_comparison_ranks_positive_momentum_first(self):
        payload = _sample_group_latent_payload(include_extra_cluster_groups=True)
        rows = payload["data"]["derived"]["group_latent_space"]["groups"]["source"]
        temporal_payload = payload["data"]["derived"]["group_temporal_latent_space"]
        selected_row = next(row for row in rows if row["group_key"] == "source-b")

        component = _cluster_share_drift_comparison(
            rows,
            selected_row,
            temporal_payload,
            "source",
            "week",
            {"cluster_id": "source-cluster-1", "label": "Source A, Source B"},
        )

        self.assertIsInstance(component, html.Div)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Share Drift Comparison", rendered_text)
        self.assertIn(
            "Ranks first-to-latest week corpus-share change within Source A, Source B and shows the latest week step.",
            rendered_text,
        )
        self.assertIn("Showing 2 of 2 groups with bucket rows.", rendered_text)
        self.assertIn("Mini-bars scale relative magnitude within the visible rows for each column.", rendered_text)
        self.assertIn("Recent Step", rendered_text)
        self.assertIn("Rank Trajectory", rendered_text)
        self.assertIn("1 -> 1 (held)", rendered_text)
        self.assertIn("2 -> 2 (held)", rendered_text)
        self.assertIn("Flat", rendered_text)
        self.assertIn("-7.0 pts", rendered_text)
        self.assertIn("-8.0 pts", rendered_text)

        table_body = _find_components(component, html.Tbody)[0]
        row_components = [child for child in table_body.children if isinstance(child, html.Tr)]
        ordered_groups = [str(row.children[1].children) for row in row_components]
        self.assertEqual(ordered_groups, ["Source A", "Source B"])
        first_rank_trajectory = str(row_components[0].children[4].children)
        first_drift_metric = row_components[0].children[5].children
        first_recent_metric = row_components[0].children[6].children
        self.assertEqual(first_rank_trajectory, "1 -> 1 (held)")
        self.assertEqual(getattr(first_drift_metric, "className", ""), "share-drift-metric")
        self.assertEqual(getattr(first_recent_metric, "className", ""), "share-drift-metric")

        drift_bars = [
            child
            for child in _iter_children(first_drift_metric)
            if isinstance(child, html.Div) and getattr(child, "className", "") == "share-drift-mini-bar"
        ]
        recent_bars = [
            child
            for child in _iter_children(first_recent_metric)
            if isinstance(child, html.Div) and getattr(child, "className", "") == "share-drift-mini-bar"
        ]
        self.assertEqual(len(drift_bars), 1)
        self.assertEqual(len(recent_bars), 1)
        self.assertEqual(drift_bars[0].title, "Scaled within the visible rows for this column.")
        self.assertEqual(recent_bars[0].title, "Scaled within the visible rows for this column.")

        highlighted_rows = [row for row in row_components if getattr(row, "className", None) == "table-primary"]
        self.assertEqual(len(highlighted_rows), 1)
        self.assertEqual(str(highlighted_rows[0].children[1].children), "Source B")

    def test_cluster_share_drift_comparison_surfaces_rank_trajectory_changes(self):
        rows = [
            {"group": "Alpha", "group_key": "alpha", "cluster_id": "source-cluster-1", "n_articles": 7},
            {"group": "Beta", "group_key": "beta", "cluster_id": "source-cluster-1", "n_articles": 6},
            {"group": "Gamma", "group_key": "gamma", "cluster_id": "source-cluster-1", "n_articles": 5},
        ]
        temporal_payload = {
            "groups": {
                "source": [
                    {
                        "group": "Alpha",
                        "group_key": "alpha",
                        "buckets": [
                            {"bucket_label": "2026-03-02", "n_articles": 3, "corpus_share": 0.30},
                            {"bucket_label": "2026-03-09", "n_articles": 2, "corpus_share": 0.18},
                            {"bucket_label": "2026-03-16", "n_articles": 2, "corpus_share": 0.10},
                        ],
                    },
                    {
                        "group": "Beta",
                        "group_key": "beta",
                        "buckets": [
                            {"bucket_label": "2026-03-02", "n_articles": 2, "corpus_share": 0.20},
                            {"bucket_label": "2026-03-09", "n_articles": 2, "corpus_share": 0.22},
                            {"bucket_label": "2026-03-16", "n_articles": 2, "corpus_share": 0.25},
                        ],
                    },
                    {
                        "group": "Gamma",
                        "group_key": "gamma",
                        "buckets": [
                            {"bucket_label": "2026-03-02", "n_articles": 1, "corpus_share": 0.10},
                            {"bucket_label": "2026-03-09", "n_articles": 2, "corpus_share": 0.12},
                            {"bucket_label": "2026-03-16", "n_articles": 2, "corpus_share": 0.15},
                        ],
                    },
                ]
            }
        }

        component = _cluster_share_drift_comparison(
            rows,
            rows[1],
            temporal_payload,
            "source",
            "week",
            {"cluster_id": "source-cluster-1", "label": "Alpha, Beta, Gamma"},
        )

        table_body = _find_components(component, html.Tbody)[0]
        row_components = [child for child in table_body.children if isinstance(child, html.Tr)]
        ordered_groups = [str(row.children[1].children) for row in row_components]
        rank_trajectories = [str(row.children[4].children) for row in row_components]

        self.assertEqual(ordered_groups, ["Beta", "Gamma", "Alpha"])
        self.assertEqual(rank_trajectories, ["2 -> 1 (up 1)", "3 -> 2 (up 1)", "1 -> 3 (down 2)"])

    def test_cluster_share_drift_comparison_breaks_drift_ties_by_recent_step(self):
        rows = [
            {"group": "Alpha", "group_key": "alpha", "cluster_id": "source-cluster-1", "n_articles": 6},
            {"group": "Beta", "group_key": "beta", "cluster_id": "source-cluster-1", "n_articles": 6},
        ]
        temporal_payload = {
            "groups": {
                "source": [
                    {
                        "group": "Alpha",
                        "group_key": "alpha",
                        "buckets": [
                            {"bucket_label": "2026-03-02", "n_articles": 2, "corpus_share": 0.10},
                            {"bucket_label": "2026-03-09", "n_articles": 1, "corpus_share": 0.05},
                            {"bucket_label": "2026-03-16", "n_articles": 3, "corpus_share": 0.15},
                        ],
                    },
                    {
                        "group": "Beta",
                        "group_key": "beta",
                        "buckets": [
                            {"bucket_label": "2026-03-02", "n_articles": 2, "corpus_share": 0.10},
                            {"bucket_label": "2026-03-09", "n_articles": 3, "corpus_share": 0.20},
                            {"bucket_label": "2026-03-16", "n_articles": 1, "corpus_share": 0.15},
                        ],
                    },
                ]
            }
        }

        component = _cluster_share_drift_comparison(
            rows,
            rows[1],
            temporal_payload,
            "source",
            "week",
            {"cluster_id": "source-cluster-1", "label": "Alpha, Beta"},
        )

        table_body = _find_components(component, html.Tbody)[0]
        row_components = [child for child in table_body.children if isinstance(child, html.Tr)]
        ordered_groups = [str(row.children[1].children) for row in row_components]
        recent_steps = [
            next((str(node) for node in _iter_children(row.children[6].children) if isinstance(node, str) and "pts" in str(node)), "")
            for row in row_components
        ]

        self.assertEqual(ordered_groups, ["Alpha", "Beta"])
        self.assertEqual(recent_steps, ["+10.0 pts", "-5.0 pts"])

    def test_cluster_share_drift_comparison_uses_backend_popularity_summary_without_bucket_rows(self):
        rows = [
            {"group": "Alpha", "group_key": "alpha", "cluster_id": "source-cluster-1", "n_articles": 6},
            {"group": "Beta", "group_key": "beta", "cluster_id": "source-cluster-1", "n_articles": 6},
        ]
        temporal_payload = {
            "groups": {
                "source": [
                    {
                        "group": "Alpha",
                        "group_key": "alpha",
                        "n_articles": 6,
                        "popularity_summary": {
                            "peak_bucket": "2026-03-09",
                            "peak_articles": 3,
                            "first_share": 0.10,
                            "latest_share": 0.15,
                            "share_delta": 0.05,
                            "recent_share_delta": 0.10,
                        },
                    },
                    {
                        "group": "Beta",
                        "group_key": "beta",
                        "n_articles": 6,
                        "popularity_summary": {
                            "peak_bucket": "2026-03-02",
                            "peak_articles": 2,
                            "first_share": 0.12,
                            "latest_share": 0.08,
                            "share_delta": -0.04,
                            "recent_share_delta": -0.02,
                        },
                    },
                ]
            }
        }

        component = _cluster_share_drift_comparison(
            rows,
            rows[0],
            temporal_payload,
            "source",
            "week",
            {"cluster_id": "source-cluster-1", "label": "Alpha, Beta"},
        )

        table_body = _find_components(component, html.Tbody)[0]
        row_components = [child for child in table_body.children if isinstance(child, html.Tr)]
        ordered_groups = [str(row.children[1].children) for row in row_components]

        self.assertEqual(ordered_groups, ["Alpha", "Beta"])
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("+5.0 pts", rendered_text)
        self.assertIn("-4.0 pts", rendered_text)

    def test_cluster_share_drift_comparison_highlights_label_only_tag_popularity_rows(self):
        rows = [
            {"group": "AI Safety", "cluster_id": "tag-cluster-1", "n_articles": 7},
            {"group": "Policy", "cluster_id": "tag-cluster-1", "n_articles": 6},
        ]
        popularity_rows = [
            {
                "group": "Policy",
                "n_articles": 6,
                "peak_bucket": "2026-03-16",
                "peak_articles": 3,
                "first_share": 0.12,
                "latest_share": 0.17,
                "share_delta": 0.05,
                "recent_share_delta": 0.03,
            },
            {
                "group": "AI Safety",
                "n_articles": 7,
                "peak_bucket": "2026-03-09",
                "peak_articles": 3,
                "first_share": 0.16,
                "latest_share": 0.12,
                "share_delta": -0.04,
                "recent_share_delta": -0.02,
            },
        ]

        with patch.object(news_group_latent_space, "_cluster_popularity_rows", return_value=popularity_rows):
            component = _cluster_share_drift_comparison(
                rows,
                rows[1],
                {"groups": {"tag": []}},
                "tag",
                "week",
                {"cluster_id": "tag-cluster-1", "label": "Risk and Governance"},
            )

        table_body = _find_components(component, html.Tbody)[0]
        row_components = [child for child in table_body.children if isinstance(child, html.Tr)]
        ordered_groups = [str(row.children[1].children) for row in row_components]
        self.assertEqual(ordered_groups, ["Policy", "AI Safety"])

        highlighted_rows = [row for row in row_components if getattr(row, "className", None) == "table-primary"]
        self.assertEqual(len(highlighted_rows), 1)
        self.assertEqual(str(highlighted_rows[0].children[1].children), "Policy")

    def test_movement_leaderboard_ranks_cluster_filtered_source_groups(self):
        payload = _sample_group_latent_payload(include_extra_cluster_groups=True)
        rows = payload["data"]["derived"]["group_latent_space"]["groups"]["source"][:2]
        temporal_payload = payload["data"]["derived"]["group_temporal_latent_space"]

        component = _movement_leaderboard(
            rows,
            temporal_payload,
            "source",
            "pca",
            "source-b",
            {"cluster_id": "source-cluster-1", "label": "Source A, Source B"},
        )

        self.assertIsInstance(component, dbc.Card)
        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Movement Leaderboard (PCA)", rendered_text)
        self.assertIn("Ranks the current source groups by PCA total movement within Source A, Source B.", rendered_text)
        self.assertIn("Showing 2 of 2 comparable groups.", rendered_text)

        table_body = _find_components(component, html.Tbody)[0]
        row_components = [child for child in table_body.children if isinstance(child, html.Tr)]
        ordered_groups = [str(row.children[1].children) for row in row_components]
        self.assertEqual(ordered_groups, ["Source A", "Source B"])

        highlighted_rows = [row for row in row_components if getattr(row, "className", None) == "table-primary"]
        self.assertEqual(len(highlighted_rows), 1)
        self.assertEqual(str(highlighted_rows[0].children[1].children), "Source B")

    def test_movement_leaderboard_supports_topic_groups_and_mds_basis(self):
        component = _movement_leaderboard(
            [
                {"group": "AI Policy", "group_key": "ai-policy", "n_articles": 7},
                {"group": "Climate", "group_key": "climate", "n_articles": 5},
            ],
            {
                "groups": {
                    "topic": [
                        {
                            "group": "AI Policy",
                            "group_key": "ai-policy",
                            "path_summary": {
                                "total_movement_mds": 0.22,
                                "largest_jump_mds": 0.14,
                                "valid_mds_bucket_count": 3,
                                "coverage_gap_count": 1,
                            },
                        },
                        {
                            "group": "Climate",
                            "group_key": "climate",
                            "path_summary": {
                                "total_movement_mds": 0.51,
                                "largest_jump_mds": 0.21,
                                "valid_mds_bucket_count": 4,
                                "coverage_gap_count": 0,
                            },
                        },
                    ]
                }
            },
            "topic",
            "mds",
            "climate",
            None,
        )

        rendered_text = " ".join(str(node) for node in _iter_children(component) if isinstance(node, str))
        self.assertIn("Movement Leaderboard (MDS)", rendered_text)
        self.assertIn("Ranks the current topic groups by MDS total movement within All groups.", rendered_text)

        table_body = _find_components(component, html.Tbody)[0]
        row_components = [child for child in table_body.children if isinstance(child, html.Tr)]
        ordered_groups = [str(row.children[1].children) for row in row_components]
        self.assertEqual(ordered_groups, ["Climate", "AI Policy"])

    def test_load_news_group_latent_space_swaps_basis_specific_scope_counts_and_nearest_note(self):
        payload = _sample_group_latent_payload()

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-load")),
        ):
            pca_outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "source",
                ALL_GROUPS_CLUSTER_VALUE,
                "source-c",
                "pca",
                "current",
                None,
            )
            mds_outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "source",
                ALL_GROUPS_CLUSTER_VALUE,
                "source-c",
                "mds",
                "current",
                None,
            )

        pca_scope_text = " ".join(str(node) for node in _iter_children(pca_outputs[9]) if isinstance(node, str))
        mds_scope_text = " ".join(str(node) for node in _iter_children(mds_outputs[9]) if isinstance(node, str))
        pca_summary_text = " ".join(str(node) for node in _iter_children(pca_outputs[10]) if isinstance(node, str))
        mds_summary_text = " ".join(str(node) for node in _iter_children(mds_outputs[10]) if isinstance(node, str))
        pca_selected_text = " ".join(str(node) for node in _iter_children(pca_outputs[13]) if isinstance(node, str))
        mds_selected_text = " ".join(str(node) for node in _iter_children(mds_outputs[13]) if isinstance(node, str))
        pca_nearest_text = " ".join(str(node) for node in _iter_children(pca_outputs[14]) if isinstance(node, str))
        mds_nearest_text = " ".join(str(node) for node in _iter_children(mds_outputs[14]) if isinstance(node, str))

        self.assertIn("3 valid PCA buckets, 1 sparse bucket, 1 coverage gap.", pca_scope_text)
        self.assertIn("2 valid MDS buckets, 1 sparse bucket, 1 coverage gap.", mds_scope_text)
        self.assertIn("Cluster scope: All groups.", pca_scope_text)
        self.assertIn("Cluster scope: All groups.", pca_summary_text)
        self.assertIn("Cluster scope: All groups.", mds_summary_text)
        self.assertIn("Movement basis: PCA centroid path.", pca_scope_text)
        self.assertIn("Movement basis: MDS centroid path.", mds_scope_text)
        self.assertIn("Missing week bucket range: 2026-03-09.", pca_summary_text)
        self.assertIn("Missing week bucket range: 2026-03-09.", mds_summary_text)
        self.assertIn("Movement Leaderboard (PCA)", pca_summary_text)
        self.assertIn("Movement Leaderboard (MDS)", mds_summary_text)
        self.assertIn("Ranks the current source groups by PCA total movement within All groups.", pca_summary_text)
        self.assertIn("Ranks the current source groups by MDS total movement within All groups.", mds_summary_text)
        self.assertIn("Popularity Timeline", pca_selected_text)
        self.assertIn("Peak volume lands in 2026-03-02 with 4 articles.", pca_selected_text)
        self.assertIn("Popularity Comparison", pca_selected_text)
        self.assertIn("Share Drift Comparison", pca_selected_text)
        self.assertIn(
            "Ranks first-to-latest week corpus-share change within All groups and shows the latest week step.",
            pca_selected_text,
        )
        self.assertIn("Mini-bars scale relative magnitude within the visible rows for each column.", pca_selected_text)
        self.assertIn("3 valid PCA buckets, 1 sparse bucket, 1 coverage gap.", pca_selected_text)
        self.assertIn("2 valid MDS buckets, 1 sparse bucket, 1 coverage gap.", mds_selected_text)
        self.assertIn("Temporal Bucket Diagnostics", pca_selected_text)
        self.assertIn("PC1 0.10, PC2 0.20", pca_selected_text)
        self.assertIn("MDS1 -0.20, MDS2 0.30", mds_selected_text)
        self.assertIn("Source C (3), Source A (1)", pca_selected_text)
        self.assertIn("Evidence (8.2), Impact (-4.1)", pca_selected_text)
        self.assertIn("Source C Cluster contains only this selected group", pca_selected_text)
        self.assertIn("Source C Cluster contains only this selected group", mds_selected_text)
        self.assertNotIn("Nearest-neighbor distances remain PCA-based", pca_nearest_text)
        self.assertIn("Nearest-neighbor distances remain PCA-based while the movement path uses MDS coordinates.", mds_nearest_text)

    def test_load_news_group_latent_space_realigns_selected_group_when_cluster_changes_under_mds(self):
        payload = _sample_group_latent_payload(include_extra_cluster_groups=True)

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-cluster")),
        ):
            outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "source",
                "source-cluster-1",
                "source-c",
                "mds",
                "current",
                None,
            )

        cluster_options = outputs[2]
        group_options = outputs[4]
        scope_text = " ".join(str(node) for node in _iter_children(outputs[9]) if isinstance(node, str))
        summary_text = " ".join(str(node) for node in _iter_children(outputs[10]) if isinstance(node, str))
        selected_text = " ".join(str(node) for node in _iter_children(outputs[13]) if isinstance(node, str))
        nearest_text = " ".join(str(node) for node in _iter_children(outputs[14]) if isinstance(node, str))

        self.assertEqual(outputs[3], "source-cluster-1")
        self.assertEqual(outputs[5], "source-a")
        self.assertEqual([option["value"] for option in cluster_options], [ALL_GROUPS_CLUSTER_VALUE, "source-cluster-1", "source-cluster-2"])
        self.assertEqual([option["value"] for option in group_options], ["source-a", "source-b"])
        self.assertIn("Temporal scope: Source A spans 3 week buckets", scope_text)
        self.assertIn("Cluster scope: Source A, Source B.", scope_text)
        self.assertIn("Movement basis: MDS centroid path.", scope_text)
        self.assertIn("3 valid MDS buckets, 0 sparse buckets, 0 coverage gaps.", scope_text)
        self.assertIn("Cluster scope: Source A, Source B.", summary_text)
        self.assertIn("Movement Leaderboard (MDS)", summary_text)
        self.assertIn("Ranks the current source groups by MDS total movement within Source A, Source B.", summary_text)
        self.assertIn("Source A", selected_text)
        self.assertIn("Movement basis: MDS centroid path.", selected_text)
        self.assertIn("Source A, Source B", selected_text)
        self.assertIn("Cluster movement context: Source A has MDS total movement 0.090", selected_text)
        self.assertIn("0.020 below the Source A, Source B peer average of 0.110", selected_text)
        self.assertIn("ranks 2 of 2 in its cluster.", selected_text)
        self.assertIn("Popularity Momentum", selected_text)
        self.assertIn("flat", selected_text)
        self.assertIn("Share Drift Rank", selected_text)
        self.assertIn("1 of 2 in Source A, Source B", selected_text)
        self.assertIn("Rank Trajectory", selected_text)
        self.assertIn("1 -> 1 in Source A, Source B (held)", selected_text)
        self.assertIn("Recent Step", selected_text)
        self.assertIn("flat (Flat)", selected_text)
        self.assertNotIn("Source C Cluster", selected_text)
        self.assertIn("Nearest Groups", nearest_text)
        self.assertIn("Source B", nearest_text)
        self.assertIn("Nearest-neighbor distances remain PCA-based while the movement path uses MDS coordinates.", nearest_text)

    def test_load_news_group_latent_space_falls_back_to_group_label_when_group_key_missing(self):
        payload = _sample_group_latent_payload(include_extra_cluster_groups=True)
        source_rows = payload["data"]["derived"]["group_latent_space"]["groups"]["source"]
        temporal_rows = payload["data"]["derived"]["group_temporal_latent_space"]["groups"]["source"]
        next(row for row in source_rows if row.get("group") == "Source C").pop("group_key", None)
        next(row for row in temporal_rows if row.get("group") == "Source C").pop("group_key", None)

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-load")),
        ):
            outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "source",
                ALL_GROUPS_CLUSTER_VALUE,
                "Source_C",
                "pca",
                "current",
                None,
            )

        self.assertEqual(outputs[5], "Source C")
        pca_figure = outputs[6]
        plotted_labels = list(pca_figure.data[0].text)
        plotted_colors = list(pca_figure.data[0].marker.color)
        self.assertEqual(plotted_colors[plotted_labels.index("Source C")], "#dc3545")
        status_text = " ".join(str(node) for node in _iter_children(outputs[0]) if isinstance(node, str))
        self.assertIn("selected source temporal movement rows", status_text)

    def test_load_news_group_latent_space_highlights_source_popularity_tables_with_label_only_temporal_rows(self):
        payload = _sample_group_latent_payload(include_extra_cluster_groups=True)
        temporal_sources = payload["data"]["derived"]["group_temporal_latent_space"]["groups"]["source"]
        for temporal_row in temporal_sources:
            temporal_row.pop("group_key", None)

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-load")),
        ):
            outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "source",
                "source-cluster-1",
                "Source_A",
                "pca",
                "current",
                None,
            )

        selected_component = outputs[13]
        selected_text = " ".join(str(node) for node in _iter_children(selected_component) if isinstance(node, str))
        buttons = _find_components(outputs[0], dbc.Button)

        self.assertEqual(outputs[3], "source-cluster-1")
        self.assertEqual(outputs[5], "source-a")
        self.assertIn("Temporal scope: Source A spans 3 week buckets", selected_text)
        self.assertIn("Popularity Comparison", selected_text)
        self.assertIn("Share Drift Comparison", selected_text)
        self.assertIn("Share Drift Rank", selected_text)
        self.assertIn("1 of 2 in Source A, Source B", selected_text)
        self.assertIn("Rank Trajectory", selected_text)
        self.assertIn("1 -> 1 in Source A, Source B (held)", selected_text)
        self.assertIn("Recent Step", selected_text)
        self.assertIn("flat (Flat)", selected_text)
        self.assertTrue(
            all("group_type=source&group_key=source-a" in str(getattr(button, "href", "")) for button in buttons)
        )

        highlighted_rows = [
            row
            for row in _find_components(selected_component, html.Tr)
            if getattr(row, "className", None) == "table-primary"
            and len(getattr(row, "children", [])) > 1
            and str(row.children[1].children) == "Source A"
        ]
        self.assertEqual(len(highlighted_rows), 2)

    def test_load_news_group_latent_space_supports_tag_leaderboard_and_pattern_callout(self):
        payload = _sample_group_latent_payload()

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-load")),
        ):
            outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "tag",
                ALL_GROUPS_CLUSTER_VALUE,
                "ai-safety",
                "pca",
                "current",
                None,
            )

        summary_text = " ".join(str(node) for node in _iter_children(outputs[10]) if isinstance(node, str))
        selected_text = " ".join(str(node) for node in _iter_children(outputs[13]) if isinstance(node, str))
        nearest_text = " ".join(str(node) for node in _iter_children(outputs[14]) if isinstance(node, str))

        self.assertEqual(outputs[3], ALL_GROUPS_CLUSTER_VALUE)
        self.assertEqual(outputs[5], "ai-safety")
        self.assertIn("Movement Leaderboard (PCA)", summary_text)
        self.assertIn("Ranks the current tag groups by PCA total movement within All groups.", summary_text)
        self.assertIn("Movement pattern: steady drift.", selected_text)
        self.assertIn("Largest jump contributes 54.5% of total PCA movement", selected_text)
        self.assertIn("Net direction: PC1 0.12, PC2 0.16.", selected_text)
        self.assertIn("AI Safety", selected_text)
        self.assertIn("Policy", nearest_text)

    def test_load_news_group_latent_space_surfaces_tag_share_drift_rank_trajectory_with_cluster_scope(self):
        payload = _sample_group_latent_payload()
        temporal_tags = payload["data"]["derived"]["group_temporal_latent_space"]["groups"]["tag"]
        ai_safety_row = next(row for row in temporal_tags if row["group_key"] == "ai-safety")
        policy_row = next(row for row in temporal_tags if row["group_key"] == "policy")
        ai_safety_row["buckets"][1]["corpus_share"] = 0.18
        ai_safety_row["buckets"][2]["corpus_share"] = 0.12
        policy_row["buckets"][1]["corpus_share"] = 0.14
        policy_row["buckets"][2]["corpus_share"] = 0.17

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-cluster")),
        ):
            outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "tag",
                "tag-cluster-1",
                "policy",
                "pca",
                "current",
                None,
            )

        summary_text = " ".join(str(node) for node in _iter_children(outputs[10]) if isinstance(node, str))
        selected_text = " ".join(str(node) for node in _iter_children(outputs[13]) if isinstance(node, str))

        self.assertEqual(outputs[3], "tag-cluster-1")
        self.assertEqual(outputs[5], "policy")
        self.assertIn("Ranks the current tag groups by PCA total movement within Risk and Governance.", summary_text)
        self.assertIn("Popularity Comparison", selected_text)
        self.assertIn("Share Drift Comparison", selected_text)
        self.assertIn(
            "Ranks first-to-latest week corpus-share change within Risk and Governance and shows the latest week step.",
            selected_text,
        )
        self.assertIn("Rank Trajectory", selected_text)
        self.assertIn("2 -> 1 in Risk and Governance (up 1)", selected_text)
        self.assertIn("2 -> 1 (up 1)", selected_text)
        self.assertIn("Policy", selected_text)

    def test_load_news_group_latent_space_highlights_tag_popularity_tables_with_label_only_temporal_rows(self):
        payload = _sample_group_latent_payload()
        temporal_tags = payload["data"]["derived"]["group_temporal_latent_space"]["groups"]["tag"]
        ai_safety_row = next(row for row in temporal_tags if row["group_key"] == "ai-safety")
        policy_row = next(row for row in temporal_tags if row["group_key"] == "policy")
        ai_safety_row["buckets"][1]["corpus_share"] = 0.18
        ai_safety_row["buckets"][2]["corpus_share"] = 0.12
        policy_row["buckets"][1]["corpus_share"] = 0.14
        policy_row["buckets"][2]["corpus_share"] = 0.17
        for temporal_row in temporal_tags:
            temporal_row.pop("group_key", None)

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-load")),
        ):
            outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "tag",
                "tag-cluster-1",
                "policy",
                "pca",
                "current",
                None,
            )

        selected_component = outputs[13]
        selected_text = " ".join(str(node) for node in _iter_children(selected_component) if isinstance(node, str))
        self.assertEqual(outputs[3], "tag-cluster-1")
        self.assertEqual(outputs[5], "policy")
        self.assertIn("Temporal scope: Policy spans 3 week buckets", selected_text)
        self.assertIn("Popularity Comparison", selected_text)
        self.assertIn("Share Drift Comparison", selected_text)
        self.assertIn("2 -> 1 in Risk and Governance (up 1)", selected_text)
        self.assertIn("2 -> 1 (up 1)", selected_text)

        highlighted_rows = [
            row
            for row in _find_components(selected_component, html.Tr)
            if getattr(row, "className", None) == "table-primary"
            and len(getattr(row, "children", [])) > 1
            and str(row.children[1].children) == "Policy"
        ]
        self.assertEqual(len(highlighted_rows), 2)

    def test_load_news_group_latent_space_matches_label_only_tag_group_and_temporal_rows(self):
        payload = _sample_group_latent_payload()
        tag_rows = payload["data"]["derived"]["group_latent_space"]["groups"]["tag"]
        temporal_tags = payload["data"]["derived"]["group_temporal_latent_space"]["groups"]["tag"]

        for row in tag_rows:
            row.pop("group_key", None)
        for row in temporal_tags:
            row.pop("group_key", None)

        ai_safety_row = next(row for row in temporal_tags if row["group"] == "AI Safety")
        policy_row = next(row for row in temporal_tags if row["group"] == "Policy")
        ai_safety_row["buckets"][1]["corpus_share"] = 0.18
        ai_safety_row["buckets"][2]["corpus_share"] = 0.12
        policy_row["buckets"][1]["corpus_share"] = 0.14
        policy_row["buckets"][2]["corpus_share"] = 0.17

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-cluster")),
        ):
            outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "tag",
                "tag-cluster-1",
                "Policy",
                "pca",
                "current",
                None,
            )

        selected_component = outputs[13]
        selected_text = " ".join(str(node) for node in _iter_children(selected_component) if isinstance(node, str))
        self.assertEqual(outputs[3], "tag-cluster-1")
        self.assertEqual(outputs[5], "Policy")
        self.assertIn("Popularity Momentum rising now", selected_text)
        self.assertIn("Rank Trajectory 2 -> 1 in Risk and Governance (up 1)", selected_text)

        highlighted_groups = [
            str(row.children[1].children)
            for row in _find_components(selected_component, html.Tr)
            if getattr(row, "className", None) == "table-primary" and len(getattr(row, "children", [])) > 1
        ]
        self.assertEqual(highlighted_groups, ["Policy", "Policy"])

    def test_load_news_group_latent_space_surfaces_topic_share_drift_rank_trajectory_with_cluster_scope(self):
        payload = _sample_group_latent_payload()
        topic_groups = [
            {
                "group": "AI Policy",
                "group_key": "ai-policy",
                "status": "ok",
                "n_articles": 8,
                "n_sources": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "cluster_id": "topic-cluster-1",
                "cluster_label": "Policy and Climate",
                "pc1": 0.24,
                "pc2": 0.17,
                "mds1": -0.18,
                "mds2": 0.11,
                "dispersion_pca": 0.152,
                "dispersion_mds": 0.118,
                "nearest_groups": [
                    {"group": "Climate Risk", "distance_pca": 0.074},
                ],
                "top_lens_deviations": [
                    {"lens": "Evidence", "delta": 4.2},
                ],
            },
            {
                "group": "Climate Risk",
                "group_key": "climate-risk",
                "status": "ok",
                "n_articles": 7,
                "n_sources": 2,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "cluster_id": "topic-cluster-1",
                "cluster_label": "Policy and Climate",
                "pc1": 0.19,
                "pc2": 0.13,
                "mds1": -0.14,
                "mds2": 0.09,
                "dispersion_pca": 0.143,
                "dispersion_mds": 0.109,
                "nearest_groups": [
                    {"group": "AI Policy", "distance_pca": 0.074},
                ],
                "top_lens_deviations": [
                    {"lens": "Impact", "delta": 3.1},
                ],
            },
        ]
        topic_clusters = [
            {
                "cluster_id": "topic-cluster-1",
                "label": "Policy and Climate",
                "n_groups": 2,
                "n_articles": 15,
                "n_sources": 3,
                "clustering_threshold_pca": 0.121,
                "representative_groups": [
                    {"group": "AI Policy", "n_articles": 8},
                    {"group": "Climate Risk", "n_articles": 7},
                ],
                "defining_lens_deviations": [{"lens": "Evidence", "delta": 4.2}],
            }
        ]
        temporal_topic_groups = [
            {
                "group": "AI Policy",
                "group_key": "ai-policy",
                "status": "ok",
                "n_articles": 8,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "buckets": [
                    {
                        "bucket_start": "2026-03-02",
                        "bucket_label": "2026-03-02",
                        "n_articles": 3,
                        "status": "ok",
                        "n_sources": 2,
                        "corpus_share": 0.24,
                        "pc1": 0.18,
                        "pc2": 0.14,
                        "mds1": -0.19,
                        "mds2": 0.08,
                        "dispersion_pca": 0.11,
                        "dispersion_mds": 0.08,
                        "source_counts": {"Source A": 1, "Source B": 2},
                        "top_lens_deviations": [{"lens": "Evidence", "delta": 3.4}],
                    },
                    {
                        "bucket_start": "2026-03-09",
                        "bucket_label": "2026-03-09",
                        "n_articles": 3,
                        "status": "ok",
                        "n_sources": 2,
                        "corpus_share": 0.2,
                        "pc1": 0.23,
                        "pc2": 0.18,
                        "mds1": -0.16,
                        "mds2": 0.12,
                        "dispersion_pca": 0.1,
                        "dispersion_mds": 0.08,
                        "source_counts": {"Source A": 2, "Source C": 1},
                        "top_lens_deviations": [{"lens": "Impact", "delta": 2.7}],
                    },
                    {
                        "bucket_start": "2026-03-16",
                        "bucket_label": "2026-03-16",
                        "n_articles": 2,
                        "status": "ok",
                        "n_sources": 1,
                        "corpus_share": 0.15,
                        "pc1": 0.27,
                        "pc2": 0.21,
                        "mds1": -0.11,
                        "mds2": 0.16,
                        "dispersion_pca": 0.09,
                        "dispersion_mds": 0.07,
                        "source_counts": {"Source C": 2},
                        "top_lens_deviations": [{"lens": "Novelty", "delta": 2.1}],
                    },
                ],
                "path_summary": {
                    "bucket_count": 3,
                    "valid_pca_bucket_count": 3,
                    "valid_mds_bucket_count": 3,
                    "sparse_bucket_count": 0,
                    "coverage_gap_count": 0,
                    "total_movement_pca": 0.15,
                    "largest_jump_pca": 0.07,
                    "direction_pca": {"pc1_delta": 0.09, "pc2_delta": 0.07},
                    "total_movement_mds": 0.12,
                    "largest_jump_mds": 0.06,
                    "direction_mds": {"mds1_delta": 0.08, "mds2_delta": 0.08},
                },
            },
            {
                "group": "Climate Risk",
                "group_key": "climate-risk",
                "status": "ok",
                "n_articles": 7,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "buckets": [
                    {
                        "bucket_start": "2026-03-02",
                        "bucket_label": "2026-03-02",
                        "n_articles": 2,
                        "status": "ok",
                        "n_sources": 1,
                        "corpus_share": 0.19,
                        "pc1": 0.12,
                        "pc2": 0.09,
                        "mds1": -0.15,
                        "mds2": 0.06,
                        "dispersion_pca": 0.1,
                        "dispersion_mds": 0.07,
                        "source_counts": {"Source B": 2},
                        "top_lens_deviations": [{"lens": "Impact", "delta": 2.1}],
                    },
                    {
                        "bucket_start": "2026-03-09",
                        "bucket_label": "2026-03-09",
                        "n_articles": 2,
                        "status": "ok",
                        "n_sources": 1,
                        "corpus_share": 0.18,
                        "pc1": 0.16,
                        "pc2": 0.11,
                        "mds1": -0.13,
                        "mds2": 0.08,
                        "dispersion_pca": 0.09,
                        "dispersion_mds": 0.07,
                        "source_counts": {"Source B": 2},
                        "top_lens_deviations": [{"lens": "Evidence", "delta": 1.8}],
                    },
                    {
                        "bucket_start": "2026-03-16",
                        "bucket_label": "2026-03-16",
                        "n_articles": 3,
                        "status": "ok",
                        "n_sources": 2,
                        "corpus_share": 0.17,
                        "pc1": 0.22,
                        "pc2": 0.15,
                        "mds1": -0.09,
                        "mds2": 0.13,
                        "dispersion_pca": 0.09,
                        "dispersion_mds": 0.08,
                        "source_counts": {"Source A": 1, "Source C": 2},
                        "top_lens_deviations": [{"lens": "Risk", "delta": 1.6}],
                    },
                ],
                "path_summary": {
                    "bucket_count": 3,
                    "valid_pca_bucket_count": 3,
                    "valid_mds_bucket_count": 3,
                    "sparse_bucket_count": 0,
                    "coverage_gap_count": 0,
                    "total_movement_pca": 0.14,
                    "largest_jump_pca": 0.06,
                    "direction_pca": {"pc1_delta": 0.1, "pc2_delta": 0.06},
                    "total_movement_mds": 0.11,
                    "largest_jump_mds": 0.05,
                    "direction_mds": {"mds1_delta": 0.06, "mds2_delta": 0.07},
                },
            },
        ]
        payload["data"]["derived"]["group_latent_space"]["groups"]["topic"] = topic_groups
        payload["data"]["derived"]["group_latent_space"]["clusters"]["topic"] = topic_clusters
        payload["data"]["derived"]["group_latent_space"]["summary"]["group_counts"]["topic"] = len(topic_groups)
        payload["data"]["derived"]["group_latent_space"]["summary"]["analyzed_group_counts"]["topic"] = len(topic_groups)
        payload["data"]["derived"]["group_latent_space"]["summary"]["low_sample_group_counts"]["topic"] = 0
        payload["data"]["derived"]["group_latent_space"]["summary"]["cluster_counts"]["topic"] = len(topic_clusters)
        payload["data"]["derived"]["group_temporal_latent_space"]["groups"]["topic"] = temporal_topic_groups
        ai_policy_row = next(row for row in temporal_topic_groups if row["group_key"] == "ai-policy")
        climate_risk_row = next(row for row in temporal_topic_groups if row["group_key"] == "climate-risk")
        ai_policy_row["buckets"][1]["corpus_share"] = 0.18
        ai_policy_row["buckets"][2]["corpus_share"] = 0.14
        climate_risk_row["buckets"][1]["corpus_share"] = 0.2
        climate_risk_row["buckets"][2]["corpus_share"] = 0.22

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-cluster")),
        ):
            outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "topic",
                "topic-cluster-1",
                "climate-risk",
                "pca",
                "current",
                None,
            )

        summary_text = " ".join(str(node) for node in _iter_children(outputs[10]) if isinstance(node, str))
        selected_text = " ".join(str(node) for node in _iter_children(outputs[13]) if isinstance(node, str))

        self.assertEqual(outputs[3], "topic-cluster-1")
        self.assertEqual(outputs[5], "climate-risk")
        self.assertIn("Ranks the current topic groups by PCA total movement within Policy and Climate.", summary_text)
        self.assertIn("Popularity Comparison", selected_text)
        self.assertIn("Share Drift Comparison", selected_text)
        self.assertIn(
            "Ranks first-to-latest week corpus-share change within Policy and Climate and shows the latest week step.",
            selected_text,
        )
        self.assertIn("Rank Trajectory", selected_text)
        self.assertIn("2 -> 1 in Policy and Climate (up 1)", selected_text)
        self.assertIn("2 -> 1 (up 1)", selected_text)
        self.assertIn("Climate Risk", selected_text)

    def test_load_news_group_latent_space_matches_label_only_topic_temporal_rows(self):
        payload = _sample_group_latent_payload()
        topic_groups = [
            {
                "group": "AI Policy",
                "group_key": "ai-policy",
                "status": "ok",
                "n_articles": 8,
                "n_sources": 3,
                "cluster_id": "topic-cluster-1",
                "cluster_label": "Policy and Climate",
                "pc1": 0.24,
                "pc2": 0.17,
            },
            {
                "group": "Climate Risk",
                "group_key": "climate-risk",
                "status": "ok",
                "n_articles": 7,
                "n_sources": 2,
                "cluster_id": "topic-cluster-1",
                "cluster_label": "Policy and Climate",
                "pc1": 0.19,
                "pc2": 0.13,
            },
        ]
        topic_clusters = [
            {
                "cluster_id": "topic-cluster-1",
                "label": "Policy and Climate",
                "n_groups": 2,
                "n_articles": 15,
                "n_sources": 3,
            }
        ]
        temporal_topic_groups = [
            {
                "group": "AI Policy",
                "status": "ok",
                "n_articles": 8,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "path_summary": {
                    "bucket_count": 3,
                    "valid_pca_bucket_count": 3,
                    "valid_mds_bucket_count": 0,
                    "sparse_bucket_count": 0,
                    "coverage_gap_count": 0,
                    "total_movement_pca": 0.15,
                    "largest_jump_pca": 0.07,
                    "direction_pca": {"pc1_delta": 0.09, "pc2_delta": 0.07},
                },
                "popularity_summary": {
                    "peak_bucket": "2026-03-02",
                    "peak_articles": 3,
                    "first_share": 0.24,
                    "latest_share": 0.14,
                    "share_delta": -0.10,
                    "recent_share_delta": -0.04,
                    "recent_share_direction": "falling",
                    "momentum_label": "still falling",
                },
            },
            {
                "group": "Climate Risk",
                "status": "ok",
                "n_articles": 7,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "path_summary": {
                    "bucket_count": 3,
                    "valid_pca_bucket_count": 3,
                    "valid_mds_bucket_count": 0,
                    "sparse_bucket_count": 0,
                    "coverage_gap_count": 0,
                    "total_movement_pca": 0.14,
                    "largest_jump_pca": 0.06,
                    "direction_pca": {"pc1_delta": 0.10, "pc2_delta": 0.06},
                },
                "popularity_summary": {
                    "peak_bucket": "2026-03-16",
                    "peak_articles": 3,
                    "first_share": 0.19,
                    "latest_share": 0.22,
                    "share_delta": 0.03,
                    "recent_share_delta": 0.02,
                    "recent_share_direction": "rising",
                    "momentum_label": "rising now",
                },
            },
        ]
        payload["data"]["derived"]["group_latent_space"]["groups"]["topic"] = topic_groups
        payload["data"]["derived"]["group_latent_space"]["clusters"]["topic"] = topic_clusters
        payload["data"]["derived"]["group_latent_space"]["summary"]["group_counts"]["topic"] = len(topic_groups)
        payload["data"]["derived"]["group_latent_space"]["summary"]["analyzed_group_counts"]["topic"] = len(topic_groups)
        payload["data"]["derived"]["group_latent_space"]["summary"]["low_sample_group_counts"]["topic"] = 0
        payload["data"]["derived"]["group_latent_space"]["summary"]["cluster_counts"]["topic"] = len(topic_clusters)
        payload["data"]["derived"]["group_temporal_latent_space"]["groups"]["topic"] = temporal_topic_groups

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-load")),
        ):
            outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "topic",
                "topic-cluster-1",
                "Climate_Risk",
                "pca",
                "current",
                None,
            )

        selected_text = " ".join(str(node) for node in _iter_children(outputs[13]) if isinstance(node, str))
        highlighted_rows = [
            row
            for row in _find_components(outputs[13], html.Tr)
            if getattr(row, "className", None) == "table-primary"
        ]

        self.assertEqual(outputs[3], "topic-cluster-1")
        self.assertEqual(outputs[5], "climate-risk")
        self.assertIn("Popularity Momentum", selected_text)
        self.assertIn("rising now", selected_text)
        self.assertIn("Share Drift Rank", selected_text)
        self.assertIn("1 of 2 in Policy and Climate", selected_text)
        self.assertIn("Rank Trajectory", selected_text)
        self.assertIn("2 -> 1 in Policy and Climate (up 1)", selected_text)
        self.assertEqual(len(highlighted_rows), 2)
        self.assertEqual(
            [str(row.children[1].children) for row in highlighted_rows],
            ["Climate Risk", "Climate Risk"],
        )

    def test_load_news_group_latent_space_highlights_topic_popularity_tables_with_label_only_temporal_rows(self):
        payload = _sample_group_latent_payload()
        topic_groups = [
            {
                "group": "AI Policy",
                "group_key": "ai-policy",
                "status": "ok",
                "n_articles": 8,
                "n_sources": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "cluster_id": "topic-cluster-1",
                "cluster_label": "Policy and Climate",
                "pc1": 0.24,
                "pc2": 0.17,
                "mds1": -0.18,
                "mds2": 0.11,
                "dispersion_pca": 0.152,
                "dispersion_mds": 0.118,
                "nearest_groups": [{"group": "Climate Risk", "distance_pca": 0.074}],
                "top_lens_deviations": [{"lens": "Evidence", "delta": 4.2}],
            },
            {
                "group": "Climate Risk",
                "group_key": "climate-risk",
                "status": "ok",
                "n_articles": 7,
                "n_sources": 2,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "cluster_id": "topic-cluster-1",
                "cluster_label": "Policy and Climate",
                "pc1": 0.19,
                "pc2": 0.13,
                "mds1": -0.14,
                "mds2": 0.09,
                "dispersion_pca": 0.143,
                "dispersion_mds": 0.109,
                "nearest_groups": [{"group": "AI Policy", "distance_pca": 0.074}],
                "top_lens_deviations": [{"lens": "Impact", "delta": 3.1}],
            },
        ]
        topic_clusters = [
            {
                "cluster_id": "topic-cluster-1",
                "label": "Policy and Climate",
                "n_groups": 2,
                "n_articles": 15,
                "n_sources": 3,
                "clustering_threshold_pca": 0.121,
                "representative_groups": [
                    {"group": "AI Policy", "n_articles": 8},
                    {"group": "Climate Risk", "n_articles": 7},
                ],
                "defining_lens_deviations": [{"lens": "Evidence", "delta": 4.2}],
            }
        ]
        temporal_topic_groups = [
            {
                "group": "AI Policy",
                "status": "ok",
                "n_articles": 8,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "buckets": [
                    {
                        "bucket_start": "2026-03-02",
                        "bucket_label": "2026-03-02",
                        "n_articles": 3,
                        "status": "ok",
                        "n_sources": 2,
                        "corpus_share": 0.24,
                        "pc1": 0.18,
                        "pc2": 0.14,
                    },
                    {
                        "bucket_start": "2026-03-09",
                        "bucket_label": "2026-03-09",
                        "n_articles": 3,
                        "status": "ok",
                        "n_sources": 2,
                        "corpus_share": 0.18,
                        "pc1": 0.23,
                        "pc2": 0.18,
                    },
                    {
                        "bucket_start": "2026-03-16",
                        "bucket_label": "2026-03-16",
                        "n_articles": 2,
                        "status": "ok",
                        "n_sources": 1,
                        "corpus_share": 0.14,
                        "pc1": 0.27,
                        "pc2": 0.21,
                    },
                ],
                "path_summary": {
                    "bucket_count": 3,
                    "valid_pca_bucket_count": 3,
                    "valid_mds_bucket_count": 3,
                    "sparse_bucket_count": 0,
                    "coverage_gap_count": 0,
                    "total_movement_pca": 0.15,
                    "largest_jump_pca": 0.07,
                    "direction_pca": {"pc1_delta": 0.09, "pc2_delta": 0.07},
                    "total_movement_mds": 0.12,
                    "largest_jump_mds": 0.06,
                    "direction_mds": {"mds1_delta": 0.08, "mds2_delta": 0.08},
                },
            },
            {
                "group": "Climate Risk",
                "status": "ok",
                "n_articles": 7,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "buckets": [
                    {
                        "bucket_start": "2026-03-02",
                        "bucket_label": "2026-03-02",
                        "n_articles": 2,
                        "status": "ok",
                        "n_sources": 1,
                        "corpus_share": 0.19,
                        "pc1": 0.12,
                        "pc2": 0.09,
                    },
                    {
                        "bucket_start": "2026-03-09",
                        "bucket_label": "2026-03-09",
                        "n_articles": 2,
                        "status": "ok",
                        "n_sources": 1,
                        "corpus_share": 0.20,
                        "pc1": 0.16,
                        "pc2": 0.11,
                    },
                    {
                        "bucket_start": "2026-03-16",
                        "bucket_label": "2026-03-16",
                        "n_articles": 3,
                        "status": "ok",
                        "n_sources": 2,
                        "corpus_share": 0.22,
                        "pc1": 0.22,
                        "pc2": 0.15,
                    },
                ],
                "path_summary": {
                    "bucket_count": 3,
                    "valid_pca_bucket_count": 3,
                    "valid_mds_bucket_count": 3,
                    "sparse_bucket_count": 0,
                    "coverage_gap_count": 0,
                    "total_movement_pca": 0.14,
                    "largest_jump_pca": 0.06,
                    "direction_pca": {"pc1_delta": 0.10, "pc2_delta": 0.06},
                    "total_movement_mds": 0.11,
                    "largest_jump_mds": 0.05,
                    "direction_mds": {"mds1_delta": 0.06, "mds2_delta": 0.07},
                },
            },
        ]
        payload["data"]["derived"]["group_latent_space"]["groups"]["topic"] = topic_groups
        payload["data"]["derived"]["group_latent_space"]["clusters"]["topic"] = topic_clusters
        payload["data"]["derived"]["group_latent_space"]["summary"]["group_counts"]["topic"] = len(topic_groups)
        payload["data"]["derived"]["group_latent_space"]["summary"]["analyzed_group_counts"]["topic"] = len(topic_groups)
        payload["data"]["derived"]["group_latent_space"]["summary"]["low_sample_group_counts"]["topic"] = 0
        payload["data"]["derived"]["group_latent_space"]["summary"]["cluster_counts"]["topic"] = len(topic_clusters)
        payload["data"]["derived"]["group_temporal_latent_space"]["groups"]["topic"] = temporal_topic_groups

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-load")),
        ):
            outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "topic",
                "topic-cluster-1",
                "climate-risk",
                "pca",
                "current",
                None,
            )

        selected_component = outputs[13]
        selected_text = " ".join(str(node) for node in _iter_children(selected_component) if isinstance(node, str))

        self.assertEqual(outputs[3], "topic-cluster-1")
        self.assertEqual(outputs[5], "climate-risk")
        self.assertIn("Temporal scope: Climate Risk spans 3 week buckets", selected_text)
        self.assertIn("Popularity Comparison", selected_text)
        self.assertIn("Share Drift Comparison", selected_text)
        self.assertIn("2 -> 1 in Policy and Climate (up 1)", selected_text)
        self.assertIn("2 -> 1 (up 1)", selected_text)

        highlighted_rows = [
            row
            for row in _find_components(selected_component, html.Tr)
            if getattr(row, "className", None) == "table-primary"
            and len(getattr(row, "children", [])) > 1
            and str(row.children[1].children) == "Climate Risk"
        ]
        self.assertEqual(len(highlighted_rows), 2)

    def test_load_news_group_latent_space_status_panel_includes_mode_aware_temporal_exports(self):
        current_payload = _sample_group_latent_payload()
        current_payload["meta"]["source_mode"] = "current"
        snapshot_payload = _sample_group_latent_payload()
        snapshot_payload["meta"]["source_mode"] = "snapshot"
        snapshot_payload["meta"]["snapshot_date"] = "2026-05-01"

        with (
            patch.object(news_group_latent_space, "api_get", side_effect=[(200, current_payload), (200, snapshot_payload)]),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-load")),
        ):
            current_outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "source",
                ALL_GROUPS_CLUSTER_VALUE,
                "source-c",
                "pca",
                "current",
                None,
            )
            snapshot_outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "source",
                ALL_GROUPS_CLUSTER_VALUE,
                "source-c",
                "pca",
                "snapshot",
                "2026-05-01",
            )

        current_buttons = _find_components(current_outputs[0], dbc.Button)
        snapshot_buttons = _find_components(snapshot_outputs[0], dbc.Button)
        current_hrefs = [getattr(button, "href", None) for button in current_buttons if getattr(button, "href", None)]
        snapshot_hrefs = [getattr(button, "href", None) for button in snapshot_buttons if getattr(button, "href", None)]

        self.assertEqual(
            current_hrefs,
            [
                "/api/news/export?artifact=group_temporal_buckets&format=csv&group_type=source&group_key=source-c",
                "/api/news/export?artifact=group_temporal_buckets&format=json&group_type=source&group_key=source-c",
                "/api/news/export?artifact=group_temporal_path_summary&format=csv&group_type=source&group_key=source-c",
                "/api/news/export?artifact=group_temporal_path_summary&format=json&group_type=source&group_key=source-c",
                "/api/news/export?artifact=group_temporal_popularity_momentum&format=csv&group_type=source&group_key=source-c",
                "/api/news/export?artifact=group_temporal_popularity_momentum&format=json&group_type=source&group_key=source-c",
            ],
        )
        self.assertEqual(
            snapshot_hrefs,
            [
                "/api/news/export?artifact=group_temporal_buckets&format=csv&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_buckets&format=json&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_path_summary&format=csv&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_path_summary&format=json&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_popularity_momentum&format=csv&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_popularity_momentum&format=json&group_type=source&group_key=source-c&snapshot_date=2026-05-01",
            ],
        )

    def test_load_news_group_latent_space_status_panel_keeps_tag_scope_in_temporal_exports(self):
        payload = _sample_group_latent_payload()
        payload["meta"]["source_mode"] = "snapshot"
        payload["meta"]["snapshot_date"] = "2026-05-01"

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-load")),
        ):
            outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "tag",
                "tag-cluster-1",
                "policy",
                "pca",
                "snapshot",
                "2026-05-01",
            )

        status_text = " ".join(str(node) for node in _iter_children(outputs[0]) if isinstance(node, str))
        buttons = _find_components(outputs[0], dbc.Button)
        hrefs = [getattr(button, "href", None) for button in buttons if getattr(button, "href", None)]

        self.assertEqual(outputs[3], "tag-cluster-1")
        self.assertEqual(outputs[5], "policy")
        self.assertIn("selected tag temporal movement rows", status_text)
        self.assertEqual(
            hrefs,
            [
                "/api/news/export?artifact=group_temporal_buckets&format=csv&group_type=tag&group_key=policy&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_buckets&format=json&group_type=tag&group_key=policy&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_path_summary&format=csv&group_type=tag&group_key=policy&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_path_summary&format=json&group_type=tag&group_key=policy&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_popularity_momentum&format=csv&group_type=tag&group_key=policy&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_popularity_momentum&format=json&group_type=tag&group_key=policy&snapshot_date=2026-05-01",
            ],
        )

    def test_load_news_group_latent_space_status_panel_keeps_topic_scope_in_temporal_exports(self):
        payload = _sample_group_latent_payload()
        payload["meta"]["source_mode"] = "snapshot"
        payload["meta"]["snapshot_date"] = "2026-05-01"
        topic_groups = [
            {
                "group": "AI Policy",
                "group_key": "ai-policy",
                "status": "ok",
                "n_articles": 8,
                "n_sources": 3,
                "cluster_id": "topic-cluster-1",
                "cluster_label": "Policy and Climate",
                "pc1": 0.24,
                "pc2": 0.17,
            },
            {
                "group": "Climate Risk",
                "group_key": "climate-risk",
                "status": "ok",
                "n_articles": 7,
                "n_sources": 2,
                "cluster_id": "topic-cluster-1",
                "cluster_label": "Policy and Climate",
                "pc1": 0.19,
                "pc2": 0.13,
            },
        ]
        topic_clusters = [
            {
                "cluster_id": "topic-cluster-1",
                "label": "Policy and Climate",
                "n_groups": 2,
                "n_articles": 15,
                "n_sources": 3,
            }
        ]
        temporal_topic_groups = [
            {
                "group": "AI Policy",
                "status": "ok",
                "n_articles": 8,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "path_summary": {
                    "bucket_count": 3,
                    "valid_pca_bucket_count": 3,
                    "valid_mds_bucket_count": 0,
                    "sparse_bucket_count": 0,
                    "coverage_gap_count": 0,
                },
                "popularity_summary": {
                    "peak_bucket": "2026-03-02",
                    "peak_articles": 3,
                    "first_share": 0.24,
                    "latest_share": 0.14,
                    "share_delta": -0.10,
                    "recent_share_delta": -0.04,
                    "recent_share_direction": "falling",
                    "momentum_label": "still falling",
                },
            },
            {
                "group": "Climate Risk",
                "status": "ok",
                "n_articles": 7,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
                "path_summary": {
                    "bucket_count": 3,
                    "valid_pca_bucket_count": 3,
                    "valid_mds_bucket_count": 0,
                    "sparse_bucket_count": 0,
                    "coverage_gap_count": 0,
                },
                "popularity_summary": {
                    "peak_bucket": "2026-03-16",
                    "peak_articles": 3,
                    "first_share": 0.19,
                    "latest_share": 0.22,
                    "share_delta": 0.03,
                    "recent_share_delta": 0.02,
                    "recent_share_direction": "rising",
                    "momentum_label": "rising now",
                },
            },
        ]
        payload["data"]["derived"]["group_latent_space"]["groups"]["topic"] = topic_groups
        payload["data"]["derived"]["group_latent_space"]["clusters"]["topic"] = topic_clusters
        payload["data"]["derived"]["group_latent_space"]["summary"]["group_counts"]["topic"] = len(topic_groups)
        payload["data"]["derived"]["group_latent_space"]["summary"]["analyzed_group_counts"]["topic"] = len(topic_groups)
        payload["data"]["derived"]["group_latent_space"]["summary"]["low_sample_group_counts"]["topic"] = 0
        payload["data"]["derived"]["group_latent_space"]["summary"]["cluster_counts"]["topic"] = len(topic_clusters)
        payload["data"]["derived"]["group_temporal_latent_space"]["groups"]["topic"] = temporal_topic_groups

        with (
            patch.object(news_group_latent_space, "api_get", return_value=(200, payload)),
            patch.object(news_group_latent_space, "ctx", SimpleNamespace(triggered_id="news-group-latent-load")),
        ):
            outputs = news_group_latent_space.load_news_group_latent_space(
                0,
                0,
                "topic",
                "topic-cluster-1",
                "Climate_Risk",
                "pca",
                "snapshot",
                "2026-05-01",
            )

        status_text = " ".join(str(node) for node in _iter_children(outputs[0]) if isinstance(node, str))
        buttons = _find_components(outputs[0], dbc.Button)
        hrefs = [getattr(button, "href", None) for button in buttons if getattr(button, "href", None)]

        self.assertEqual(outputs[3], "topic-cluster-1")
        self.assertEqual(outputs[5], "climate-risk")
        self.assertIn("selected topic temporal movement rows", status_text)
        self.assertEqual(
            hrefs,
            [
                "/api/news/export?artifact=group_temporal_buckets&format=csv&group_type=topic&group_key=climate-risk&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_buckets&format=json&group_type=topic&group_key=climate-risk&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_path_summary&format=csv&group_type=topic&group_key=climate-risk&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_path_summary&format=json&group_type=topic&group_key=climate-risk&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_popularity_momentum&format=csv&group_type=topic&group_key=climate-risk&snapshot_date=2026-05-01",
                "/api/news/export?artifact=group_temporal_popularity_momentum&format=json&group_type=topic&group_key=climate-risk&snapshot_date=2026-05-01",
            ],
        )


if __name__ == "__main__":
    unittest.main()
