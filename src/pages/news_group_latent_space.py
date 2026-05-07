from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html

from src.pages.news_page_utils import api_get, build_news_intro, build_status_alert, snapshot_param


dash.register_page(
    __name__,
    path="/news/group-latent-space",
    name="News Group Latent Space",
    title="NewsLens | News Group Latent Space",
)

GROUP_TYPE_OPTIONS = [
    {"label": "Sources", "value": "source"},
    {"label": "Topics", "value": "topic"},
    {"label": "Tags", "value": "tag"},
]

STATUS_COLORS = {
    "ok": "#0d6efd",
    "low_sample": "#ffc107",
    "unavailable": "#6c757d",
}
ALL_GROUPS_CLUSTER_VALUE = "__all_groups__"


def _empty_figure(title: str, message: str | None = None) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 30, "r": 20, "t": 60, "b": 40},
    )
    if message:
        figure.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13},
        )
    return figure


def _format_decimal(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "n/a"


def _group_options(rows: list[dict]) -> list[dict]:
    options: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        group_key = str(row.get("group_key") or row.get("group") or "").strip()
        group_label = str(row.get("group") or "").strip()
        if not group_key or not group_label:
            continue
        suffix = f" ({int(row.get('n_articles') or 0)})"
        if str(row.get("status") or "") == "low_sample":
            suffix += " low sample"
        options.append({"label": f"{group_label}{suffix}", "value": group_key})
    return options


def _cluster_options(cluster_rows: list[dict]) -> list[dict]:
    options: list[dict] = [{"label": "All groups", "value": ALL_GROUPS_CLUSTER_VALUE}]
    for cluster_row in cluster_rows:
        if not isinstance(cluster_row, dict):
            continue
        cluster_id = str(cluster_row.get("cluster_id") or "").strip()
        cluster_label = str(cluster_row.get("label") or cluster_id).strip()
        if not cluster_id or not cluster_label:
            continue
        group_count = int(cluster_row.get("n_groups") or 0)
        article_count = int(cluster_row.get("n_articles") or 0)
        options.append(
            {
                "label": f"{cluster_label} ({group_count} groups, {article_count} articles)",
                "value": cluster_id,
            }
        )
    return options


def _group_rows_for_cluster(rows: list[dict], selected_cluster_id: str | None) -> list[dict]:
    normalized_cluster_id = str(selected_cluster_id or "").strip().lower()
    if not normalized_cluster_id or normalized_cluster_id == ALL_GROUPS_CLUSTER_VALUE:
        return [row for row in rows if isinstance(row, dict)]

    filtered_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_cluster_id = str(row.get("cluster_id") or "").strip().lower()
        if row_cluster_id == normalized_cluster_id:
            filtered_rows.append(row)
    return filtered_rows


def _selected_group_row(
    rows: list[dict],
    selected_group_key: str | None,
    selected_cluster_row: dict | None = None,
) -> dict | None:
    normalized_selected = str(selected_group_key or "").strip().lower()
    normalized_cluster_id = (
        str(selected_cluster_row.get("cluster_id") or "").strip().lower()
        if isinstance(selected_cluster_row, dict)
        else ""
    )
    if normalized_selected:
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_key = str(row.get("group_key") or row.get("group") or "").strip().lower()
            if row_key == normalized_selected:
                if not normalized_cluster_id:
                    return row
                row_cluster_id = str(row.get("cluster_id") or "").strip().lower()
                if row_cluster_id == normalized_cluster_id:
                    return row
                break
    if normalized_cluster_id:
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_cluster_id = str(row.get("cluster_id") or "").strip().lower()
            if row_cluster_id == normalized_cluster_id:
                return row
    for row in rows:
        if isinstance(row, dict):
            return row
    return None


def _selected_cluster_row(
    clusters: list[dict],
    selected_group_row: dict | None,
    selected_cluster_id: str | None = None,
) -> dict | None:
    normalized_selected_cluster_id = str(selected_cluster_id or "").strip().lower()
    if normalized_selected_cluster_id == ALL_GROUPS_CLUSTER_VALUE:
        return None
    if normalized_selected_cluster_id:
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            cluster_id = str(cluster.get("cluster_id") or "").strip().lower()
            if cluster_id == normalized_selected_cluster_id:
                return cluster
    if not isinstance(selected_group_row, dict):
        return None
    normalized_cluster_id = str(selected_group_row.get("cluster_id") or "").strip().lower()
    if not normalized_cluster_id:
        return None
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip().lower()
        if cluster_id == normalized_cluster_id:
            return cluster
    return None


def _cluster_filter_hint(
    all_rows: list[dict],
    filtered_rows: list[dict],
    selected_cluster_row: dict | None,
):
    total_options = len([row for row in all_rows if isinstance(row, dict)])
    filtered_options = len([row for row in filtered_rows if isinstance(row, dict)])
    if isinstance(selected_cluster_row, dict):
        cluster_label = str(selected_cluster_row.get("label") or selected_cluster_row.get("cluster_id") or "Selected cluster")
        message = (
            f"Cluster filter: {cluster_label}. "
            f"The Group menu is limited to {filtered_options} option"
            f"{'' if filtered_options == 1 else 's'} in this cluster."
        )
    else:
        message = (
            f"Cluster filter: All groups. "
            f"The Group menu shows all {total_options} available option"
            f"{'' if total_options == 1 else 's'}."
        )
    return dbc.Alert(message, color="light", className="mb-0 py-2")


def _summary_cards(group_type: str, group_latent: dict) -> list:
    summary = group_latent.get("summary") if isinstance(group_latent.get("summary"), dict) else {}
    config = group_latent.get("config") if isinstance(group_latent.get("config"), dict) else {}
    group_counts = summary.get("group_counts") if isinstance(summary.get("group_counts"), dict) else {}
    analyzed_counts = summary.get("analyzed_group_counts") if isinstance(summary.get("analyzed_group_counts"), dict) else {}
    low_sample_counts = summary.get("low_sample_group_counts") if isinstance(summary.get("low_sample_group_counts"), dict) else {}
    cluster_counts = summary.get("cluster_counts") if isinstance(summary.get("cluster_counts"), dict) else {}

    cards = [
        ("Groups", group_counts.get(group_type, 0)),
        ("Analyzed", analyzed_counts.get(group_type, 0)),
        ("Low Sample", low_sample_counts.get(group_type, 0)),
        ("Clusters", cluster_counts.get(group_type, 0)),
        ("Min Articles", config.get("min_articles_per_group", "n/a")),
    ]
    return [
        dbc.Col(
            dbc.Card(
                dbc.CardBody([html.P(label, className="text-muted mb-1"), html.H4(str(value), className="mb-0")]),
                className="shadow-sm",
            ),
            md=6,
            lg=2,
            className="mb-3",
        )
        for label, value in cards
    ]


def _centroid_figure(rows: list[dict], selected_group_key: str | None, x_key: str, y_key: str, title: str) -> go.Figure:
    chart_rows = []
    normalized_selected = str(selected_group_key or "").strip().lower()
    for row in rows:
        if not isinstance(row, dict):
            continue
        x_value = row.get(x_key)
        y_value = row.get(y_key)
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            continue
        chart_rows.append(row)

    if not chart_rows:
        return _empty_figure(title, "No centroid coordinates available for the selected group type.")

    dispersion_key = "dispersion_pca" if x_key.startswith("pc") else "dispersion_mds"
    return go.Figure(
        data=[
            go.Scatter(
                x=[row.get(x_key) for row in chart_rows],
                y=[row.get(y_key) for row in chart_rows],
                mode="markers+text",
                text=[row.get("group") for row in chart_rows],
                textposition="top center",
                customdata=[
                    [
                        row.get("n_articles"),
                        row.get("status"),
                        row.get("cluster_label") or "n/a",
                        row.get(dispersion_key),
                    ]
                    for row in chart_rows
                ],
                hovertemplate=(
                    "%{text}<br>Articles: %{customdata[0]}<br>Status: %{customdata[1]}"
                    "<br>Cluster: %{customdata[2]}<br>Dispersion: %{customdata[3]:.3f}"
                    "<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"
                ),
                marker={
                    "size": [max(10, min(28, int(row.get("n_articles") or 0) * 2)) for row in chart_rows],
                    "color": [
                        "#dc3545"
                        if str(row.get("group_key") or "").strip().lower() == normalized_selected
                        else STATUS_COLORS.get(str(row.get("status") or ""), STATUS_COLORS["unavailable"])
                        for row in chart_rows
                    ],
                    "opacity": [
                        1.0 if str(row.get("group_key") or "").strip().lower() == normalized_selected else 0.78
                        for row in chart_rows
                    ],
                    "line": {
                        "color": [
                            "#721c24" if str(row.get("group_key") or "").strip().lower() == normalized_selected else "#ffffff"
                            for row in chart_rows
                        ],
                        "width": [
                            2 if str(row.get("group_key") or "").strip().lower() == normalized_selected else 1
                            for row in chart_rows
                        ],
                    },
                },
            )
        ]
    ).update_layout(title=title, template="plotly_white", xaxis_title=x_key.upper(), yaxis_title=y_key.upper())


def _selected_group_summary(row: dict | None):
    if not isinstance(row, dict):
        return dbc.Alert("No group is available for the current selection.", color="warning", className="mb-0")

    date_start = row.get("date_start") or "n/a"
    date_end = row.get("date_end") or "n/a"
    metrics = [
        ("Status", str(row.get("status") or "n/a")),
        ("Articles", int(row.get("n_articles") or 0)),
        ("Sources", int(row.get("n_sources") or 0)),
        ("Date Range", f"{date_start} to {date_end}"),
        ("Cluster", row.get("cluster_label") or "n/a"),
        ("PCA Dispersion", _format_decimal(row.get("dispersion_pca"))),
        ("MDS Dispersion", _format_decimal(row.get("dispersion_mds"))),
    ]
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(str(row.get("group") or "Selected Group"), className="card-title"),
                dbc.Table(
                    [
                        html.Tbody([html.Tr([html.Th(label), html.Td(str(value))]) for label, value in metrics]),
                    ],
                    bordered=False,
                    size="sm",
                    class_name="mb-0",
                ),
            ]
        ),
        className="shadow-sm",
    )


def _nearest_groups_table(row: dict | None):
    rows = row.get("nearest_groups", []) if isinstance(row, dict) and isinstance(row.get("nearest_groups"), list) else []
    if not rows:
        return dbc.Alert("No nearest-neighbor rows are available for the selected group.", color="warning", className="mb-0")
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Nearest Groups", className="card-title"),
                dbc.Table(
                    [
                        html.Thead(html.Tr([html.Th("Group"), html.Th("Distance")])),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(str(neighbor.get("group") or "Unknown")),
                                        html.Td(_format_decimal(neighbor.get("distance_pca"))),
                                    ]
                                )
                                for neighbor in rows[:5]
                                if isinstance(neighbor, dict)
                            ]
                        ),
                    ],
                    bordered=True,
                    striped=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                    class_name="mb-0",
                ),
            ]
        ),
        className="shadow-sm",
    )


def _cluster_membership_summary(cluster_row: dict | None):
    if not isinstance(cluster_row, dict):
        return dbc.Alert(
            "No cluster-membership summary is available for the selected group.",
            color="warning",
            className="mb-0",
        )

    representative_groups = (
        cluster_row.get("representative_groups")
        if isinstance(cluster_row.get("representative_groups"), list)
        else []
    )
    lens_rows = (
        cluster_row.get("defining_lens_deviations")
        if isinstance(cluster_row.get("defining_lens_deviations"), list)
        else []
    )
    metrics = [
        ("Cluster", cluster_row.get("label") or cluster_row.get("cluster_id") or "n/a"),
        ("Groups", int(cluster_row.get("n_groups") or 0)),
        ("Articles", int(cluster_row.get("n_articles") or 0)),
        ("Sources", int(cluster_row.get("n_sources") or 0)),
        ("Threshold", _format_decimal(cluster_row.get("clustering_threshold_pca"))),
    ]
    lens_summary = ", ".join(
        f"{str(lens_row.get('lens') or 'Unknown')} ({_format_decimal(lens_row.get('delta'), 2)})"
        for lens_row in lens_rows[:3]
        if isinstance(lens_row, dict)
    ) or "n/a"

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Cluster Membership", className="card-title"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Table(
                                [
                                    html.Tbody(
                                        [html.Tr([html.Th(label), html.Td(str(value))]) for label, value in metrics]
                                    ),
                                ],
                                bordered=False,
                                size="sm",
                                class_name="mb-0",
                            ),
                            lg=4,
                            className="mb-3",
                        ),
                        dbc.Col(
                            [
                                html.P("Representative groups", className="text-muted mb-2"),
                                dbc.Table(
                                    [
                                        html.Thead(html.Tr([html.Th("Group"), html.Th("Articles")])),
                                        html.Tbody(
                                            [
                                                html.Tr(
                                                    [
                                                        html.Td(str(group_row.get("group") or "Unknown")),
                                                        html.Td(str(int(group_row.get("n_articles") or 0))),
                                                    ]
                                                )
                                                for group_row in representative_groups[:6]
                                                if isinstance(group_row, dict)
                                            ]
                                        ),
                                    ],
                                    bordered=True,
                                    striped=True,
                                    hover=True,
                                    responsive=True,
                                    size="sm",
                                    class_name="mb-0",
                                ),
                            ],
                            lg=4,
                            className="mb-3",
                        ),
                        dbc.Col(
                            [
                                html.P("Defining lens deviations", className="text-muted mb-2"),
                                html.P(lens_summary, className="mb-0"),
                            ],
                            lg=4,
                            className="mb-3",
                        ),
                    ]
                ),
            ]
        ),
        className="shadow-sm",
    )


def _cluster_scope_summary(cluster_rows: list[dict], selected_cluster_row: dict | None):
    normalized_rows = [row for row in cluster_rows if isinstance(row, dict)]
    if isinstance(selected_cluster_row, dict):
        cluster_label = str(selected_cluster_row.get("label") or selected_cluster_row.get("cluster_id") or "Selected cluster")
        group_count = int(selected_cluster_row.get("n_groups") or 0)
        article_count = int(selected_cluster_row.get("n_articles") or 0)
        summary = (
            f"Scope: {cluster_label}. "
            f"{group_count} group{'' if group_count == 1 else 's'}, "
            f"{article_count} article{'' if article_count == 1 else 's'}."
        )
    else:
        cluster_count = len(normalized_rows)
        group_count = sum(int(row.get("n_groups") or 0) for row in normalized_rows)
        article_count = sum(int(row.get("n_articles") or 0) for row in normalized_rows)
        summary = (
            f"Scope: All clusters. "
            f"{cluster_count} cluster{'' if cluster_count == 1 else 's'}, "
            f"{group_count} group{'' if group_count == 1 else 's'}, "
            f"{article_count} article{'' if article_count == 1 else 's'}."
        )
    return html.P(summary, className="text-muted mb-2")


def _cluster_overview_table(cluster_rows: list[dict], selected_cluster_row: dict | None):
    if not cluster_rows:
        return dbc.Alert("No cluster rows are available for the selected group type.", color="warning", className="mb-0")

    normalized_selected = str(selected_cluster_row.get("cluster_id") or "").strip().lower() if isinstance(selected_cluster_row, dict) else ""
    table_rows = []
    for cluster_row in cluster_rows[:10]:
        if not isinstance(cluster_row, dict):
            continue
        representative_groups = (
            cluster_row.get("representative_groups")
            if isinstance(cluster_row.get("representative_groups"), list)
            else []
        )
        lens_rows = (
            cluster_row.get("defining_lens_deviations")
            if isinstance(cluster_row.get("defining_lens_deviations"), list)
            else []
        )
        representative_summary = ", ".join(
            str(group_row.get("group") or "Unknown")
            for group_row in representative_groups[:3]
            if isinstance(group_row, dict)
        ) or "n/a"
        lens_summary = ", ".join(
            f"{str(lens_row.get('lens') or 'Unknown')} ({_format_decimal(lens_row.get('delta'), 2)})"
            for lens_row in lens_rows[:2]
            if isinstance(lens_row, dict)
        ) or "n/a"
        cluster_id = str(cluster_row.get("cluster_id") or "").strip().lower()
        table_rows.append(
            html.Tr(
                [
                    html.Td(str(cluster_row.get("label") or cluster_row.get("cluster_id") or "Unknown")),
                    html.Td(str(int(cluster_row.get("n_groups") or 0))),
                    html.Td(str(int(cluster_row.get("n_articles") or 0))),
                    html.Td(representative_summary),
                    html.Td(lens_summary),
                ],
                className="table-primary" if cluster_id and cluster_id == normalized_selected else None,
            )
        )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Cluster Overview", className="card-title"),
                _cluster_scope_summary(cluster_rows, selected_cluster_row),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Cluster"),
                                    html.Th("Groups"),
                                    html.Th("Articles"),
                                    html.Th("Representatives"),
                                    html.Th("Lens Deviations"),
                                ]
                            )
                        ),
                        html.Tbody(table_rows),
                    ],
                    bordered=True,
                    striped=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                    class_name="mb-0",
                ),
            ]
        ),
        className="shadow-sm",
    )


def _lens_deviation_table(row: dict | None):
    rows = row.get("top_lens_deviations", []) if isinstance(row, dict) and isinstance(row.get("top_lens_deviations"), list) else []
    if not rows:
        return dbc.Alert("No lens-deviation rows are available for the selected group.", color="warning", className="mb-0")
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Top Lens Deviations", className="card-title"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Lens"),
                                    html.Th("Group Mean"),
                                    html.Th("Corpus Mean"),
                                    html.Th("Delta"),
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(str(lens_row.get("lens") or "Unknown")),
                                        html.Td(_format_decimal(lens_row.get("mean_percent"), 2)),
                                        html.Td(_format_decimal(lens_row.get("corpus_mean_percent"), 2)),
                                        html.Td(_format_decimal(lens_row.get("delta"), 2)),
                                    ]
                                )
                                for lens_row in rows[:8]
                                if isinstance(lens_row, dict)
                            ]
                        ),
                    ],
                    bordered=True,
                    striped=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                    class_name="mb-0",
                ),
            ]
        ),
        className="shadow-sm",
    )


def _group_scope_summary(rows: list[dict], selected_cluster_row: dict | None = None):
    filtered_rows = [row for row in rows if isinstance(row, dict)]
    group_count = int(selected_cluster_row.get("n_groups") or len(filtered_rows)) if isinstance(selected_cluster_row, dict) else len(filtered_rows)
    article_count = (
        int(selected_cluster_row.get("n_articles") or 0)
        if isinstance(selected_cluster_row, dict)
        else sum(int(row.get("n_articles") or 0) for row in filtered_rows)
    )
    scope_label = (
        str(selected_cluster_row.get("label") or selected_cluster_row.get("cluster_id") or "Selected cluster")
        if isinstance(selected_cluster_row, dict)
        else "All groups"
    )
    summary = (
        f"Scope: {scope_label}. "
        f"{group_count} group{'' if group_count == 1 else 's'}, "
        f"{article_count} article{'' if article_count == 1 else 's'}."
    )
    if len(filtered_rows) > 15:
        summary += " Showing the first 15 groups."
    return html.P(summary, className="text-muted mb-2")


def _group_table(rows: list[dict], selected_group_key: str | None, selected_cluster_row: dict | None = None):
    if not rows:
        return dbc.Alert("No group rows are available for the selected group type.", color="warning", className="mb-0")

    normalized_selected = str(selected_group_key or "").strip().lower()
    table_rows = []
    for row in rows[:15]:
        if not isinstance(row, dict):
            continue
        selected = str(row.get("group_key") or "").strip().lower() == normalized_selected
        class_name = "table-primary" if selected else None
        table_rows.append(
            html.Tr(
                [
                    html.Td(str(row.get("group") or "Unknown")),
                    html.Td(str(row.get("status") or "n/a")),
                    html.Td(str(int(row.get("n_articles") or 0))),
                    html.Td(str(row.get("cluster_label") or "n/a")),
                    html.Td(_format_decimal(row.get("pc1"), 2)),
                    html.Td(_format_decimal(row.get("pc2"), 2)),
                ],
                className=class_name,
            )
        )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Top Groups", className="card-title"),
                _group_scope_summary(rows, selected_cluster_row),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Group"),
                                    html.Th("Status"),
                                    html.Th("Articles"),
                                    html.Th("Cluster"),
                                    html.Th("PC1"),
                                    html.Th("PC2"),
                                ]
                            )
                        ),
                        html.Tbody(table_rows),
                    ],
                    bordered=True,
                    striped=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                    class_name="mb-0",
                ),
            ]
        ),
        className="shadow-sm",
    )


layout = dbc.Container(
    [
        dcc.Interval(id="news-group-latent-load", interval=50, n_intervals=0, max_intervals=1),
        dbc.Row([dbc.Col(html.H3("News Group Latent Space", className="mb-2"), width=12)]),
        build_news_intro(
            "Map sources, topics, and tags into the shared lens PCA/MDS space to compare centroid position, dispersion, and nearest neighbors."
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Data mode"),
                        dcc.Dropdown(
                            id="news-group-latent-mode",
                            options=[
                                {"label": "Current", "value": "current"},
                                {"label": "Snapshot", "value": "snapshot"},
                            ],
                            value="current",
                            clearable=False,
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        dbc.Label("Snapshot date (UTC)"),
                        dcc.Input(
                            id="news-group-latent-snapshot-date",
                            type="date",
                            className="form-control",
                            disabled=True,
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        dbc.Label("Group type"),
                        dcc.Dropdown(
                            id="news-group-latent-type",
                            options=GROUP_TYPE_OPTIONS,
                            value="source",
                            clearable=False,
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        dbc.Label("Cluster"),
                        dcc.Dropdown(
                            id="news-group-latent-cluster",
                            options=[],
                            value=None,
                            placeholder="Select a cluster",
                            clearable=False,
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        dbc.Label("Group"),
                        dcc.Dropdown(
                            id="news-group-latent-group",
                            options=[],
                            value=None,
                            placeholder="Select a group",
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Actions"),
                        dbc.Button("Refresh", id="news-group-latent-refresh", color="secondary"),
                    ],
                    md=1,
                ),
                dbc.Col(html.Div(id="news-group-latent-status"), md=12, className="mt-3"),
            ],
            className="mb-2",
        ),
        dbc.Row(id="news-group-latent-cards"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="news-group-latent-pca"), lg=6, className="mb-3"),
                dbc.Col(dcc.Graph(id="news-group-latent-mds"), lg=6, className="mb-3"),
            ]
        ),
        dbc.Row([dbc.Col(html.Div(id="news-group-latent-table"), width=12, className="mb-3")]),
        dbc.Row([dbc.Col(html.Div(id="news-group-latent-cluster-summary"), width=12, className="mb-3")]),
        dbc.Row(
            [
                dbc.Col(html.Div(id="news-group-latent-selected"), lg=4, className="mb-3"),
                dbc.Col(html.Div(id="news-group-latent-nearest"), lg=4, className="mb-3"),
                dbc.Col(html.Div(id="news-group-latent-lenses"), lg=4, className="mb-3"),
            ]
        ),
    ],
    fluid=True,
    className="py-4",
)


@callback(
    Output("news-group-latent-status", "children"),
    Output("news-group-latent-cards", "children"),
    Output("news-group-latent-cluster", "options"),
    Output("news-group-latent-cluster", "value"),
    Output("news-group-latent-group", "options"),
    Output("news-group-latent-group", "value"),
    Output("news-group-latent-pca", "figure"),
    Output("news-group-latent-mds", "figure"),
    Output("news-group-latent-table", "children"),
    Output("news-group-latent-cluster-summary", "children"),
    Output("news-group-latent-selected", "children"),
    Output("news-group-latent-nearest", "children"),
    Output("news-group-latent-lenses", "children"),
    Input("news-group-latent-load", "n_intervals"),
    Input("news-group-latent-refresh", "n_clicks"),
    Input("news-group-latent-type", "value"),
    Input("news-group-latent-cluster", "value"),
    Input("news-group-latent-group", "value"),
    State("news-group-latent-mode", "value"),
    State("news-group-latent-snapshot-date", "value"),
)
def load_news_group_latent_space(
    _load_tick,
    _refresh_clicks,
    group_type,
    selected_cluster_id,
    selected_group_key,
    data_mode,
    snapshot_date,
):
    force_refresh = ctx.triggered_id == "news-group-latent-refresh"
    effective_group_type = group_type if group_type in {"source", "topic", "tag"} else "source"
    status_code, payload = api_get(
        "/api/news/stats",
        {
            "snapshot_date": snapshot_param(data_mode, snapshot_date),
            "refresh": "true" if force_refresh else None,
        },
    )

    empty_pca = _empty_figure("PCA Group Centroids")
    empty_mds = _empty_figure("MDS Group Centroids")
    if status_code != 200:
        error = payload.get("error", "Unknown error")
        alert = dbc.Alert(f"Stats error ({status_code}): {error}", color="danger")
        return alert, [], [], None, [], None, empty_pca, empty_mds, alert, alert, alert, alert, alert

    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    derived = payload.get("data", {}).get("derived", {}) if isinstance(payload.get("data"), dict) else {}
    group_latent = derived.get("group_latent_space", {}) if isinstance(derived.get("group_latent_space"), dict) else {}
    status = str(group_latent.get("status") or "unavailable")
    reason = str(group_latent.get("reason") or "").strip()
    groups = group_latent.get("groups", {}) if isinstance(group_latent.get("groups"), dict) else {}
    clusters = group_latent.get("clusters", {}) if isinstance(group_latent.get("clusters"), dict) else {}
    rows = groups.get(effective_group_type, []) if isinstance(groups.get(effective_group_type), list) else []
    cluster_rows = clusters.get(effective_group_type, []) if isinstance(clusters.get(effective_group_type), list) else []
    cluster_options = _cluster_options(cluster_rows)
    requested_group_row = _selected_group_row(rows, selected_group_key)
    if ctx.triggered_id == "news-group-latent-group":
        selected_cluster = _selected_cluster_row(cluster_rows, requested_group_row)
    else:
        selected_cluster = _selected_cluster_row(cluster_rows, requested_group_row, selected_cluster_id)
    selected_cluster_value = (
        str(selected_cluster.get("cluster_id"))
        if isinstance(selected_cluster, dict)
        else ALL_GROUPS_CLUSTER_VALUE
    )
    filtered_rows = _group_rows_for_cluster(rows, selected_cluster_value)
    group_options = _group_options(filtered_rows)
    selected_row = _selected_group_row(filtered_rows, selected_group_key, selected_cluster)
    selected_value = str(selected_row.get("group_key")) if isinstance(selected_row, dict) else None

    if status != "ok":
        message = reason or "Group latent space is unavailable for the current dataset."
        warning = dbc.Alert(message, color="warning", className="mb-0")
        return (
            build_status_alert(meta, leading_parts=[f"Group latent status: {status}"], color="warning"),
            _summary_cards(effective_group_type, group_latent),
            cluster_options,
            selected_cluster_value,
            group_options,
            selected_value,
            _empty_figure("PCA Group Centroids", message),
            _empty_figure("MDS Group Centroids", message),
            warning,
            warning,
            warning,
            warning,
            warning,
        )

    status_alert = build_status_alert(
        meta,
        leading_parts=[
            f"Group latent status: {status}",
            f"{effective_group_type.title()} groups: {len(rows)}",
        ],
        trailing_parts=[
            f"Min articles per group: {group_latent.get('config', {}).get('min_articles_per_group', 'n/a')}",
        ],
        color="info",
    )
    status_panel = html.Div(
        [
            status_alert,
            _cluster_filter_hint(rows, filtered_rows, selected_cluster),
        ],
        className="d-grid gap-2",
    )

    return (
        status_panel,
        _summary_cards(effective_group_type, group_latent),
        cluster_options,
        selected_cluster_value,
        group_options,
        selected_value,
        _centroid_figure(rows, selected_value, "pc1", "pc2", "PCA Group Centroids"),
        _centroid_figure(rows, selected_value, "mds1", "mds2", "MDS Group Centroids"),
        _group_table(filtered_rows, selected_value, selected_cluster),
        html.Div(
            [
                _cluster_overview_table(cluster_rows, selected_cluster),
                html.Div(className="mb-3"),
                _cluster_membership_summary(selected_cluster),
            ]
        ),
        _selected_group_summary(selected_row),
        _nearest_groups_table(selected_row),
        _lens_deviation_table(selected_row),
    )


@callback(
    Output("news-group-latent-snapshot-date", "disabled"),
    Input("news-group-latent-mode", "value"),
)
def toggle_group_latent_snapshot_input(data_mode):
    return data_mode != "snapshot"
