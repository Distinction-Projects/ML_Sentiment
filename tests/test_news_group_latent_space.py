import unittest
from types import SimpleNamespace
from unittest.mock import patch

import dash_bootstrap_components as dbc
from dash import html

import src.app  # noqa: F401
import src.pages.news_group_latent_space as news_group_latent_space
from src.pages.news_group_latent_space import (
    ALL_GROUPS_CLUSTER_VALUE,
    _centroid_figure,
    _cluster_filter_hint,
    _cluster_membership_summary,
    _cluster_options,
    _cluster_overview_table,
    _cluster_peer_movement_callout,
    _cluster_scope_summary,
    _group_movement_figure,
    _group_movement_summary,
    _group_scope_summary,
    _group_temporal_scope_summary,
    _group_table,
    _group_options,
    _group_rows_for_cluster,
    _selected_temporal_group_row,
    _selected_cluster_row,
    _selected_group_row,
    _selected_group_summary,
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
                    "n_articles": 4,
                    "status": "ok",
                    "corpus_share": 0.4,
                    "pc1": 0.1,
                    "pc2": 0.2,
                    "mds1": -0.2,
                    "mds2": 0.3,
                    "dispersion_pca": 0.13,
                    "dispersion_mds": 0.11,
                },
                {
                    "bucket_start": "2026-03-09",
                    "n_articles": 3,
                    "status": "sparse",
                    "corpus_share": 0.3,
                    "pc1": 0.4,
                    "pc2": 0.5,
                    "dispersion_pca": 0.19,
                },
                {
                    "bucket_start": "2026-03-16",
                    "n_articles": 2,
                    "status": "ok",
                    "corpus_share": 0.2,
                    "pc1": 0.6,
                    "pc2": 0.7,
                    "mds1": 0.1,
                    "mds2": 0.6,
                    "dispersion_pca": 0.17,
                    "dispersion_mds": 0.16,
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
                        "n_articles": 2,
                        "status": "ok",
                        "corpus_share": 0.2,
                        "pc1": 0.15,
                        "pc2": 0.1,
                        "mds1": -0.22,
                        "mds2": 0.08,
                        "dispersion_pca": 0.12,
                        "dispersion_mds": 0.09,
                    },
                    {
                        "bucket_start": "2026-03-09",
                        "n_articles": 2,
                        "status": "ok",
                        "corpus_share": 0.2,
                        "pc1": 0.23,
                        "pc2": 0.14,
                        "mds1": -0.2,
                        "mds2": 0.12,
                        "dispersion_pca": 0.11,
                        "dispersion_mds": 0.08,
                    },
                    {
                        "bucket_start": "2026-03-16",
                        "n_articles": 2,
                        "status": "ok",
                        "corpus_share": 0.2,
                        "pc1": 0.28,
                        "pc2": 0.19,
                        "mds1": -0.17,
                        "mds2": 0.16,
                        "dispersion_pca": 0.1,
                        "dispersion_mds": 0.07,
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
                        "n_articles": 2,
                        "status": "ok",
                        "corpus_share": 0.18,
                        "pc1": 0.21,
                        "pc2": 0.08,
                        "mds1": -0.17,
                        "mds2": 0.16,
                        "dispersion_pca": 0.11,
                        "dispersion_mds": 0.09,
                    },
                    {
                        "bucket_start": "2026-03-09",
                        "n_articles": 2,
                        "status": "ok",
                        "corpus_share": 0.19,
                        "pc1": 0.26,
                        "pc2": 0.12,
                        "mds1": -0.13,
                        "mds2": 0.2,
                        "dispersion_pca": 0.1,
                        "dispersion_mds": 0.08,
                    },
                    {
                        "bucket_start": "2026-03-16",
                        "n_articles": 1,
                        "status": "sparse",
                        "corpus_share": 0.11,
                        "pc1": 0.31,
                        "pc2": 0.15,
                        "mds1": -0.09,
                        "mds2": 0.24,
                        "dispersion_pca": 0.12,
                        "dispersion_mds": 0.1,
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
                        "group_counts": {"source": len(source_groups)},
                        "analyzed_group_counts": {"source": len(source_groups)},
                        "low_sample_group_counts": {"source": 0},
                        "cluster_counts": {"source": len(source_clusters)},
                    },
                    "groups": {"source": source_groups},
                    "clusters": {"source": source_clusters},
                },
                "group_temporal_latent_space": {
                    "config": {"bucket_granularity": "week"},
                    "groups": {"source": temporal_source_groups},
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

    def test_group_temporal_scope_summary_surfaces_bucket_coverage_for_selected_group(self):
        component = _group_temporal_scope_summary(
            {
                "group": "Source C",
                "n_articles": 9,
                "n_buckets": 3,
                "date_start": "2026-03-02",
                "date_end": "2026-03-22",
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
        self.assertIn("3 valid PCA buckets, 1 sparse bucket, 1 coverage gap.", pca_selected_text)
        self.assertIn("2 valid MDS buckets, 1 sparse bucket, 1 coverage gap.", mds_selected_text)
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
        self.assertIn("Source A", selected_text)
        self.assertIn("Movement basis: MDS centroid path.", selected_text)
        self.assertIn("Source A, Source B", selected_text)
        self.assertIn("Cluster movement context: Source A has MDS total movement 0.090", selected_text)
        self.assertIn("0.020 below the Source A, Source B peer average of 0.110", selected_text)
        self.assertIn("ranks 2 of 2 in its cluster.", selected_text)
        self.assertNotIn("Source C Cluster", selected_text)
        self.assertIn("Nearest Groups", nearest_text)
        self.assertIn("Source B", nearest_text)
        self.assertIn("Nearest-neighbor distances remain PCA-based while the movement path uses MDS coordinates.", nearest_text)


if __name__ == "__main__":
    unittest.main()
