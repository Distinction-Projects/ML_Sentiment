import unittest

import dash_bootstrap_components as dbc
from dash import html

import src.app  # noqa: F401
from src.pages.news_group_latent_space import (
    ALL_GROUPS_CLUSTER_VALUE,
    _centroid_figure,
    _cluster_filter_hint,
    _cluster_membership_summary,
    _cluster_options,
    _cluster_overview_table,
    _cluster_scope_summary,
    _group_scope_summary,
    _group_table,
    _group_options,
    _group_rows_for_cluster,
    _selected_cluster_row,
    _selected_group_row,
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


if __name__ == "__main__":
    unittest.main()
