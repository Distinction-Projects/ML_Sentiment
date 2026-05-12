from __future__ import annotations

import re
from urllib.parse import urlencode

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html

from src.pages.news_page_utils import (
    api_get,
    build_news_intro,
    build_status_alert,
    mode_label,
    snapshot_param,
)


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
MOVEMENT_BASIS_OPTIONS = [
    {"label": "PCA", "value": "pca"},
    {"label": "MDS", "value": "mds"},
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


def _format_percent(value: object, digits: int = 1) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}%}"
    return "n/a"


def _normalize_group_lookup_key(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[\s_-]+", " ", text)


def _row_group_value(row: object) -> str | None:
    if not isinstance(row, dict):
        return None
    value = str(row.get("group_key") or row.get("group") or "").strip()
    return value or None


def _row_group_lookup_key(row: object) -> str:
    return _normalize_group_lookup_key(_row_group_value(row))


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
    normalized_selected = _normalize_group_lookup_key(selected_group_key)
    normalized_cluster_id = (
        str(selected_cluster_row.get("cluster_id") or "").strip().lower()
        if isinstance(selected_cluster_row, dict)
        else ""
    )
    if normalized_selected:
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_key = _row_group_lookup_key(row)
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


def _group_temporal_export_href(
    meta: dict | None,
    artifact: str = "group_temporal_buckets",
    export_format: str = "csv",
    group_type: str | None = None,
    group_key: str | None = None,
) -> str:
    params: dict[str, str] = {
        "artifact": artifact,
        "format": export_format,
    }
    normalized_group_type = str(group_type or "").strip().lower()
    normalized_group_key = str(group_key or "").strip()
    if normalized_group_type in {"source", "topic", "tag"}:
        params["group_type"] = normalized_group_type
    if normalized_group_key and "group_type" in params:
        params["group_key"] = normalized_group_key
    meta = meta if isinstance(meta, dict) else {}
    if meta.get("source_mode") == "snapshot" and meta.get("snapshot_date"):
        params["snapshot_date"] = str(meta["snapshot_date"])
    return f"/api/news/export?{urlencode(params)}"


def _group_temporal_export_panel(
    meta: dict | None,
    bucket_granularity: str = "week",
    group_type: str | None = None,
    group_key: str | None = None,
):
    granularity_label = str(bucket_granularity or "week").strip().title() or "Week"
    normalized_group_type = str(group_type or "").strip().lower()
    scope_description = (
        f"the current {normalized_group_type} view"
        if normalized_group_type in {"source", "topic", "tag"}
        else "the loaded dataset"
    )
    if str(group_key or "").strip() and normalized_group_type in {"source", "topic", "tag"}:
        scope_description = f"the selected {normalized_group_type}"
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6("Export Temporal Movement Data", className="mb-2"),
                html.P(
                    (
                        f"Download {scope_description} temporal movement rows for offline filtering. "
                        f"Exports follow the current dataset mode: {mode_label(meta if isinstance(meta, dict) else {})}."
                    ),
                    className="text-muted mb-3",
                ),
                html.P(
                    f"Bucket rows preserve per-{granularity_label.lower()} centroid coordinates, counts, and sparse-bucket diagnostics.",
                    className="small text-muted mb-2",
                ),
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            "Bucket Rows CSV",
                            href=_group_temporal_export_href(
                                meta,
                                export_format="csv",
                                group_type=group_type,
                                group_key=group_key,
                            ),
                            color="primary",
                            size="sm",
                            target="_blank",
                        ),
                        dbc.Button(
                            "Bucket Rows JSON",
                            href=_group_temporal_export_href(
                                meta,
                                export_format="json",
                                group_type=group_type,
                                group_key=group_key,
                            ),
                            color="outline-primary",
                            size="sm",
                            target="_blank",
                        ),
                    ],
                    size="sm",
                ),
                html.P(
                    "Path summary rows preserve movement totals, direction deltas, and coverage-gap ranges for each group.",
                    className="small text-muted mt-3 mb-2",
                ),
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            "Path Summary CSV",
                            href=_group_temporal_export_href(
                                meta,
                                artifact="group_temporal_path_summary",
                                export_format="csv",
                                group_type=group_type,
                                group_key=group_key,
                            ),
                            color="primary",
                            size="sm",
                            target="_blank",
                        ),
                        dbc.Button(
                            "Path Summary JSON",
                            href=_group_temporal_export_href(
                                meta,
                                artifact="group_temporal_path_summary",
                                export_format="json",
                                group_type=group_type,
                                group_key=group_key,
                            ),
                            color="outline-primary",
                            size="sm",
                            target="_blank",
                        ),
                    ],
                    size="sm",
                ),
                html.P(
                    (
                        "Popularity momentum rows preserve peak bucket volume, first-to-latest share drift, "
                        "latest-step share change, and interpretable momentum labels for each group."
                    ),
                    className="small text-muted mt-3 mb-2",
                ),
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            "Momentum CSV",
                            href=_group_temporal_export_href(
                                meta,
                                artifact="group_temporal_popularity_momentum",
                                export_format="csv",
                                group_type=group_type,
                                group_key=group_key,
                            ),
                            color="primary",
                            size="sm",
                            target="_blank",
                        ),
                        dbc.Button(
                            "Momentum JSON",
                            href=_group_temporal_export_href(
                                meta,
                                artifact="group_temporal_popularity_momentum",
                                export_format="json",
                                group_type=group_type,
                                group_key=group_key,
                            ),
                            color="outline-primary",
                            size="sm",
                            target="_blank",
                        ),
                    ],
                    size="sm",
                ),
            ]
        ),
        className="shadow-sm",
    )


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
    normalized_selected = _normalize_group_lookup_key(selected_group_key)
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
                        if _row_group_lookup_key(row) == normalized_selected
                        else STATUS_COLORS.get(str(row.get("status") or ""), STATUS_COLORS["unavailable"])
                        for row in chart_rows
                    ],
                    "opacity": [
                        1.0 if _row_group_lookup_key(row) == normalized_selected else 0.78
                        for row in chart_rows
                    ],
                    "line": {
                        "color": [
                            "#721c24" if _row_group_lookup_key(row) == normalized_selected else "#ffffff"
                            for row in chart_rows
                        ],
                        "width": [
                            2 if _row_group_lookup_key(row) == normalized_selected else 1
                            for row in chart_rows
                        ],
                    },
                },
            )
        ]
    ).update_layout(title=title, template="plotly_white", xaxis_title=x_key.upper(), yaxis_title=y_key.upper())


def _selected_group_summary(
    row: dict | None,
    temporal_row: dict | None = None,
    bucket_granularity: str = "week",
    movement_basis: str | None = "pca",
    selected_cluster_row: dict | None = None,
    all_group_rows: list[dict] | None = None,
    temporal_payload: dict | None = None,
    group_type: str = "source",
):
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
    popularity_comparison = None
    share_drift_comparison = None
    if all_group_rows and isinstance(temporal_payload, dict) and temporal_payload:
        metrics.extend(
            _selected_group_popularity_metrics(
                row,
                all_group_rows,
                temporal_payload,
                group_type,
                selected_cluster_row,
            )
        )
        popularity_comparison = _cluster_popularity_comparison(
            all_group_rows,
            row,
            temporal_payload,
            group_type,
            bucket_granularity,
            selected_cluster_row,
        )
        share_drift_comparison = _cluster_share_drift_comparison(
            all_group_rows,
            row,
            temporal_payload,
            group_type,
            bucket_granularity,
            selected_cluster_row,
        )
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(str(row.get("group") or "Selected Group"), className="card-title"),
                _group_temporal_scope_summary(temporal_row, bucket_granularity, movement_basis, selected_cluster_row),
                _movement_pattern_callout(temporal_row, movement_basis),
                _cluster_peer_movement_callout(
                    all_group_rows or [],
                    row,
                    temporal_payload or {},
                    group_type,
                    movement_basis,
                ),
                dbc.Table(
                    [
                        html.Tbody([html.Tr([html.Th(label), html.Td(str(value))]) for label, value in metrics]),
                    ],
                    bordered=False,
                    size="sm",
                    class_name="mb-0",
                ),
                html.Div(
                    _group_popularity_timeline(temporal_row, bucket_granularity),
                    className="mt-3",
                ),
                html.Div(popularity_comparison, className="mt-3") if popularity_comparison is not None else None,
                html.Div(share_drift_comparison, className="mt-3") if share_drift_comparison is not None else None,
                html.Div(
                    _temporal_bucket_diagnostics_table(temporal_row, bucket_granularity, movement_basis),
                    className="mt-3",
                ),
            ]
        ),
        className="shadow-sm",
    )


def _nearest_groups_table(row: dict | None, movement_basis: str | None = "pca"):
    rows = row.get("nearest_groups", []) if isinstance(row, dict) and isinstance(row.get("nearest_groups"), list) else []
    if not rows:
        return dbc.Alert("No nearest-neighbor rows are available for the selected group.", color="warning", className="mb-0")
    notes = []
    if str(movement_basis or "").strip().lower() == "mds":
        notes.append(
            html.P(
                "Nearest-neighbor distances remain PCA-based while the movement path uses MDS coordinates.",
                className="text-muted mb-3",
            )
        )
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Nearest Groups", className="card-title"),
                *notes,
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

    normalized_selected = _normalize_group_lookup_key(selected_group_key)
    table_rows = []
    for row in rows[:15]:
        if not isinstance(row, dict):
            continue
        selected = _row_group_lookup_key(row) == normalized_selected
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


def _selected_temporal_group_row(temporal_payload: dict, group_type: str, selected_group_key: str | None) -> dict | None:
    if not isinstance(temporal_payload, dict):
        return None
    groups = temporal_payload.get("groups") if isinstance(temporal_payload.get("groups"), dict) else {}
    rows = groups.get(group_type) if isinstance(groups.get(group_type), list) else []
    normalized_selected = _normalize_group_lookup_key(selected_group_key)
    if normalized_selected:
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_key = _row_group_lookup_key(row)
            if row_key == normalized_selected:
                return row
    for row in rows:
        if isinstance(row, dict):
            return row
    return None


def _movement_basis_config(movement_basis: str | None) -> dict[str, object]:
    basis = "mds" if str(movement_basis or "").strip().lower() == "mds" else "pca"
    if basis == "mds":
        return {
            "basis": "mds",
            "title": "MDS Centroid Path",
            "x_key": "mds1",
            "y_key": "mds2",
            "x_label": "MDS1",
            "y_label": "MDS2",
            "dispersion_key": "dispersion_mds",
            "dispersion_label": "MDS dispersion",
            "valid_bucket_label": "Valid MDS buckets",
            "valid_bucket_count_key": "valid_mds_bucket_count",
            "total_movement_key": "total_movement_mds",
            "largest_jump_key": "largest_jump_mds",
            "direction_key": "direction_mds",
            "direction_dimensions": (("mds1", "MDS1"), ("mds2", "MDS2")),
        }
    return {
        "basis": "pca",
        "title": "PCA Centroid Path",
        "x_key": "pc1",
        "y_key": "pc2",
        "x_label": "PC1",
        "y_label": "PC2",
        "dispersion_key": "dispersion_pca",
        "dispersion_label": "PCA dispersion",
        "valid_bucket_label": "Valid PCA buckets",
        "valid_bucket_count_key": "valid_pca_bucket_count",
        "total_movement_key": "total_movement_pca",
        "largest_jump_key": "largest_jump_pca",
        "direction_key": "direction_pca",
        "direction_dimensions": (("pc1", "PC1"), ("pc2", "PC2")),
    }


def _group_movement_figure(
    temporal_row: dict | None,
    bucket_granularity: str = "week",
    movement_basis: str | None = "pca",
) -> go.Figure:
    granularity_label = str(bucket_granularity or "week").strip().title() or "Week"
    basis_config = _movement_basis_config(movement_basis)
    title = f"{granularity_label} {basis_config['title']}"
    x_key = str(basis_config["x_key"])
    y_key = str(basis_config["y_key"])
    x_label = str(basis_config["x_label"])
    y_label = str(basis_config["y_label"])
    dispersion_key = str(basis_config["dispersion_key"])
    dispersion_label = str(basis_config["dispersion_label"])
    if not isinstance(temporal_row, dict):
        return _empty_figure(
            title,
            "No temporal group path is available for the current selection.",
        )

    bucket_rows = temporal_row.get("buckets") if isinstance(temporal_row.get("buckets"), list) else []
    plotted_rows = [
        row
        for row in bucket_rows
        if isinstance(row, dict) and isinstance(row.get(x_key), (int, float)) and isinstance(row.get(y_key), (int, float))
    ]
    if not plotted_rows:
        return _empty_figure(
            title,
            f"No {str(basis_config['basis']).upper()} centroid path is available for the selected group.",
        )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[row.get(x_key) for row in plotted_rows],
            y=[row.get(y_key) for row in plotted_rows],
            mode="lines+markers",
            line={"color": "#0d6efd", "width": 2},
            marker={
                "size": [max(10, min(24, int(row.get("n_articles") or 0) * 3)) for row in plotted_rows],
                "color": [
                    STATUS_COLORS.get("low_sample", "#ffc107")
                    if str(row.get("status") or "") != "ok"
                    else "#0d6efd"
                    for row in plotted_rows
                ],
                "line": {"color": "#ffffff", "width": 1},
            },
            customdata=[
                [
                    row.get("bucket_start"),
                    row.get("n_articles"),
                    row.get("status"),
                    row.get("corpus_share"),
                    row.get(dispersion_key),
                ]
                for row in plotted_rows
            ],
            hovertemplate=(
                "Bucket: %{customdata[0]}<br>Articles: %{customdata[1]}<br>Status: %{customdata[2]}"
                f"<br>Corpus share: %{{customdata[3]:.2%}}<br>{dispersion_label}: %{{customdata[4]:.3f}}"
                f"<br>{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>"
            ),
            name="Path",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[plotted_rows[0].get(x_key)],
            y=[plotted_rows[0].get(y_key)],
            mode="markers+text",
            text=["Start"],
            textposition="top left",
            marker={"size": 12, "color": "#198754", "symbol": "diamond"},
            hovertemplate="Start<extra></extra>",
            name="Start",
        )
    )
    if len(plotted_rows) > 1:
        figure.add_trace(
            go.Scatter(
                x=[plotted_rows[-1].get(x_key)],
                y=[plotted_rows[-1].get(y_key)],
                mode="markers+text",
                text=["Latest"],
                textposition="bottom right",
                marker={"size": 12, "color": "#dc3545", "symbol": "diamond"},
                hovertemplate="Latest<extra></extra>",
                name="Latest",
            )
        )
    figure.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )
    return figure


def _format_direction_summary(direction: object, direction_dimensions: tuple[tuple[str, str], ...]) -> str:
    if not isinstance(direction, dict):
        return "n/a"
    parts = []
    for dimension_key, dimension_label in direction_dimensions:
        delta = direction.get(f"{dimension_key}_delta")
        if isinstance(delta, (int, float)):
            parts.append(f"{dimension_label} {_format_decimal(delta, 2)}")
    return ", ".join(parts) if parts else "n/a"


def _movement_cluster_scope_text(selected_cluster_row: dict | None) -> str:
    if isinstance(selected_cluster_row, dict):
        cluster_label = str(selected_cluster_row.get("label") or selected_cluster_row.get("cluster_id") or "Selected cluster")
        return f"Cluster scope: {cluster_label}."
    return "Cluster scope: All groups."


def _coverage_gap_ranges_text(coverage_gap_ranges: object, bucket_granularity: str = "week") -> str:
    if not isinstance(coverage_gap_ranges, list):
        return ""

    labels = []
    for gap in coverage_gap_ranges:
        if not isinstance(gap, dict):
            continue
        label = str(gap.get("label") or "").strip()
        if not label:
            start_bucket = str(gap.get("start_bucket") or "").strip()
            end_bucket = str(gap.get("end_bucket") or "").strip()
            if start_bucket and end_bucket:
                label = start_bucket if start_bucket == end_bucket else f"{start_bucket} to {end_bucket}"
            else:
                label = start_bucket or end_bucket
        if label:
            labels.append(label)

    if not labels:
        return ""

    granularity_label = str(bucket_granularity or "week").strip().lower() or "week"
    noun = "range" if len(labels) == 1 else "ranges"
    return f" Missing {granularity_label} bucket {noun}: {', '.join(labels)}."


def _group_movement_summary(
    temporal_row: dict | None,
    bucket_granularity: str = "week",
    movement_basis: str | None = "pca",
    selected_cluster_row: dict | None = None,
):
    granularity_label = str(bucket_granularity or "week").strip().title() or "Week"
    if not isinstance(temporal_row, dict):
        return dbc.Alert("No temporal movement summary is available for the current selection.", color="warning", className="mb-0")

    basis_config = _movement_basis_config(movement_basis)
    path_summary = temporal_row.get("path_summary") if isinstance(temporal_row.get("path_summary"), dict) else {}
    coverage_gap_ranges_text = _coverage_gap_ranges_text(path_summary.get("coverage_gap_ranges"), bucket_granularity).strip()
    metrics = [
        ("Status", str(temporal_row.get("status") or "n/a")),
        (f"{granularity_label} buckets", int(path_summary.get("bucket_count") or 0)),
        (str(basis_config["valid_bucket_label"]), int(path_summary.get(str(basis_config["valid_bucket_count_key"])) or 0)),
        ("Sparse buckets", int(path_summary.get("sparse_bucket_count") or 0)),
        ("Coverage gaps", int(path_summary.get("coverage_gap_count") or 0)),
        ("Total movement", _format_decimal(path_summary.get(str(basis_config["total_movement_key"])), 3)),
        ("Largest jump", _format_decimal(path_summary.get(str(basis_config["largest_jump_key"])), 3)),
        ("Direction", _format_direction_summary(path_summary.get(str(basis_config["direction_key"])), basis_config["direction_dimensions"])),
        ("Date range", f"{temporal_row.get('date_start') or 'n/a'} to {temporal_row.get('date_end') or 'n/a'}"),
    ]
    notes = []
    if coverage_gap_ranges_text:
        notes.append(dbc.Alert(coverage_gap_ranges_text, color="light", className="py-2 mb-0"))
    if str(temporal_row.get("status") or "") != "ok":
        reason = str(temporal_row.get("reason") or "").strip()
        if reason:
            notes.append(dbc.Alert(reason, color="warning", className="py-2 mb-0"))

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(f"Movement Summary ({str(basis_config['basis']).upper()})", className="card-title"),
                html.P(_movement_cluster_scope_text(selected_cluster_row), className="text-muted mb-3"),
                dbc.Table(
                    [html.Tbody([html.Tr([html.Th(label), html.Td(str(value))]) for label, value in metrics])],
                    bordered=False,
                    size="sm",
                    class_name="mb-0",
                ),
                html.Div(notes, className="mt-3") if notes else None,
            ]
        ),
        className="shadow-sm",
    )


def _movement_leaderboard(
    rows: list[dict],
    temporal_payload: dict,
    group_type: str,
    movement_basis: str | None = "pca",
    selected_group_key: str | None = None,
    selected_cluster_row: dict | None = None,
):
    comparable_rows = [row for row in rows if isinstance(row, dict)]
    if not comparable_rows:
        return dbc.Alert("No comparable group movement rows are available for the current selection.", color="warning", className="mb-0")

    groups = temporal_payload.get("groups") if isinstance(temporal_payload.get("groups"), dict) else {}
    temporal_rows = groups.get(group_type) if isinstance(groups.get(group_type), list) else []
    temporal_rows_by_key: dict[str, dict] = {}
    for temporal_row in temporal_rows:
        if not isinstance(temporal_row, dict):
            continue
        temporal_group_key = _row_group_lookup_key(temporal_row)
        if temporal_group_key:
            temporal_rows_by_key[temporal_group_key] = temporal_row

    basis_config = _movement_basis_config(movement_basis)
    total_movement_key = str(basis_config["total_movement_key"])
    largest_jump_key = str(basis_config["largest_jump_key"])
    valid_bucket_count_key = str(basis_config["valid_bucket_count_key"])
    normalized_selected = _normalize_group_lookup_key(selected_group_key)
    leaderboard_rows: list[dict[str, object]] = []

    for row in comparable_rows:
        group_key = _normalize_group_lookup_key(row.get("group_key") or row.get("group"))
        if not group_key:
            continue
        temporal_row = temporal_rows_by_key.get(group_key)
        path_summary = (
            temporal_row.get("path_summary")
            if isinstance(temporal_row, dict) and isinstance(temporal_row.get("path_summary"), dict)
            else {}
        )
        total_movement = path_summary.get(total_movement_key)
        if not isinstance(total_movement, (int, float)):
            continue
        leaderboard_rows.append(
            {
                "group": str(row.get("group") or "Unknown"),
                "group_key": group_key,
                "n_articles": int(row.get("n_articles") or 0),
                "total_movement": float(total_movement),
                "largest_jump": (
                    float(path_summary.get(largest_jump_key))
                    if isinstance(path_summary.get(largest_jump_key), (int, float))
                    else None
                ),
                "valid_bucket_count": int(path_summary.get(valid_bucket_count_key) or 0),
                "coverage_gap_count": int(path_summary.get("coverage_gap_count") or 0),
            }
        )

    if not leaderboard_rows:
        return dbc.Alert(
            "No comparable total-movement values are available for the current selection.",
            color="warning",
            className="mb-0",
        )

    leaderboard_rows.sort(
        key=lambda item: (
            -float(item["total_movement"]),
            -float(item["largest_jump"] or 0.0),
            str(item["group"]).lower(),
        )
    )
    basis_label = str(basis_config["basis"]).upper()
    scope_label = (
        str(selected_cluster_row.get("label") or selected_cluster_row.get("cluster_id") or "Selected cluster")
        if isinstance(selected_cluster_row, dict)
        else "All groups"
    )
    summary = (
        f"Ranks the current {group_type} groups by {basis_label} total movement within {scope_label}. "
        f"Showing {min(len(leaderboard_rows), 8)} of {len(leaderboard_rows)} comparable groups."
    )
    table_rows = []
    for index, leaderboard_row in enumerate(leaderboard_rows[:8], start=1):
        class_name = "table-primary" if str(leaderboard_row["group_key"]) == normalized_selected else None
        table_rows.append(
            html.Tr(
                [
                    html.Td(str(index)),
                    html.Td(str(leaderboard_row["group"])),
                    html.Td(str(int(leaderboard_row["n_articles"]))),
                    html.Td(_format_decimal(leaderboard_row["total_movement"])),
                    html.Td(_format_decimal(leaderboard_row["largest_jump"])),
                    html.Td(str(int(leaderboard_row["valid_bucket_count"]))),
                    html.Td(str(int(leaderboard_row["coverage_gap_count"]))),
                ],
                className=class_name,
            )
        )

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(f"Movement Leaderboard ({basis_label})", className="card-title"),
                html.P(summary, className="text-muted mb-3"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Rank"),
                                    html.Th("Group"),
                                    html.Th("Articles"),
                                    html.Th("Total"),
                                    html.Th("Largest Jump"),
                                    html.Th("Valid Buckets"),
                                    html.Th("Gap Count"),
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


def _movement_pattern_callout(
    temporal_row: dict | None,
    movement_basis: str | None = "pca",
):
    if not isinstance(temporal_row, dict):
        return None

    basis_config = _movement_basis_config(movement_basis)
    path_summary = temporal_row.get("path_summary") if isinstance(temporal_row.get("path_summary"), dict) else {}
    total_movement = path_summary.get(str(basis_config["total_movement_key"]))
    largest_jump = path_summary.get(str(basis_config["largest_jump_key"]))
    valid_bucket_count = int(path_summary.get(str(basis_config["valid_bucket_count_key"])) or 0)
    if not isinstance(total_movement, (int, float)) or not isinstance(largest_jump, (int, float)):
        return None

    basis_label = str(basis_config["basis"]).upper()
    direction_text = _format_direction_summary(path_summary.get(str(basis_config["direction_key"])), basis_config["direction_dimensions"])
    bucket_suffix = "" if valid_bucket_count == 1 else "s"
    if total_movement <= 0:
        message = (
            f"Movement pattern: no measurable {basis_label} movement across "
            f"{valid_bucket_count} valid bucket{bucket_suffix}."
        )
        return dbc.Alert(message, color="light", className="py-2 mb-3")

    jump_share = min(max(float(largest_jump) / float(total_movement), 0.0), 1.0)
    if jump_share >= 0.75:
        label = "jump-led"
        color = "warning"
        explanation = (
            f"Largest jump contributes {_format_percent(jump_share)} of total {basis_label} movement, "
            "so the path is dominated by one sharp shift."
        )
    elif valid_bucket_count >= 3 and jump_share <= 0.6:
        label = "steady drift"
        color = "info"
        explanation = (
            f"Largest jump contributes {_format_percent(jump_share)} of total {basis_label} movement, "
            "so movement is spread across multiple buckets."
        )
    else:
        label = "mixed"
        color = "light"
        explanation = (
            f"Largest jump contributes {_format_percent(jump_share)} of total {basis_label} movement, "
            "so the path blends gradual drift with one notable step."
        )

    message = (
        f"Movement pattern: {label}. {explanation} "
        f"{valid_bucket_count} valid bucket{bucket_suffix} contribute to this summary."
    )
    if direction_text != "n/a":
        message += f" Net direction: {direction_text}."
    return dbc.Alert(message, color=color, className="py-2 mb-3")


def _group_temporal_scope_summary(
    temporal_row: dict | None,
    bucket_granularity: str = "week",
    movement_basis: str | None = "pca",
    selected_cluster_row: dict | None = None,
):
    granularity_label = str(bucket_granularity or "week").strip().lower() or "week"
    if not isinstance(temporal_row, dict):
        return dbc.Alert(
            "No temporal bucket coverage is available for the current selection.",
            color="warning",
            className="mb-2 py-2",
        )

    basis_config = _movement_basis_config(movement_basis)
    path_summary = temporal_row.get("path_summary") if isinstance(temporal_row.get("path_summary"), dict) else {}
    bucket_count = int(path_summary.get("bucket_count") or temporal_row.get("n_buckets") or 0)
    valid_bucket_count = int(path_summary.get(str(basis_config["valid_bucket_count_key"])) or 0)
    sparse_bucket_count = int(path_summary.get("sparse_bucket_count") or 0)
    coverage_gap_count = int(path_summary.get("coverage_gap_count") or 0)
    coverage_gap_ranges_text = _coverage_gap_ranges_text(path_summary.get("coverage_gap_ranges"), bucket_granularity)
    article_count = int(temporal_row.get("n_articles") or 0)
    group_label = str(temporal_row.get("group") or "Selected group")
    date_start = str(temporal_row.get("date_start") or "n/a")
    date_end = str(temporal_row.get("date_end") or "n/a")
    summary = (
        f"Temporal scope: {group_label} spans {bucket_count} {granularity_label} bucket"
        f"{'' if bucket_count == 1 else 's'} from {date_start} to {date_end} across "
        f"{article_count} article{'' if article_count == 1 else 's'}. "
        f"{_movement_cluster_scope_text(selected_cluster_row)} "
        f"Movement basis: {str(basis_config['basis']).upper()} centroid path. "
        f"{valid_bucket_count} valid {str(basis_config['basis']).upper()} bucket"
        f"{'' if valid_bucket_count == 1 else 's'}, "
        f"{sparse_bucket_count} sparse bucket{'' if sparse_bucket_count == 1 else 's'}, "
        f"{coverage_gap_count} coverage gap{'' if coverage_gap_count == 1 else 's'}."
        f"{coverage_gap_ranges_text}"
    )
    return dbc.Alert(summary, color="light", className="mb-2 py-2")


def _source_count_summary(source_counts: object, limit: int = 2) -> str:
    if not isinstance(source_counts, dict) or not source_counts:
        return "n/a"
    items = []
    for source, count in list(source_counts.items())[:limit]:
        items.append(f"{str(source)} ({int(count or 0)})")
    return ", ".join(items) if items else "n/a"


def _lens_deviation_summary(lens_rows: object, limit: int = 2) -> str:
    if not isinstance(lens_rows, list) or not lens_rows:
        return "n/a"
    parts = []
    for lens_row in lens_rows[:limit]:
        if not isinstance(lens_row, dict):
            continue
        lens_label = str(lens_row.get("lens") or "").strip()
        if not lens_label:
            continue
        parts.append(f"{lens_label} ({_format_decimal(lens_row.get('delta'), 1)})")
    return ", ".join(parts) if parts else "n/a"


def _bucket_coordinate_summary(bucket_row: dict, movement_basis: str | None = "pca") -> str:
    basis_config = _movement_basis_config(movement_basis)
    x_key = str(basis_config["x_key"])
    y_key = str(basis_config["y_key"])
    x_label = str(basis_config["x_label"])
    y_label = str(basis_config["y_label"])
    x_value = bucket_row.get(x_key)
    y_value = bucket_row.get(y_key)
    if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
        return "n/a"
    return f"{x_label} {_format_decimal(x_value, 2)}, {y_label} {_format_decimal(y_value, 2)}"


def _group_popularity_timeline(
    temporal_row: dict | None,
    bucket_granularity: str = "week",
):
    if not isinstance(temporal_row, dict):
        return dbc.Alert(
            "No popularity timeline is available for the current selection.",
            color="warning",
            className="mb-0",
        )

    bucket_rows = temporal_row.get("buckets") if isinstance(temporal_row.get("buckets"), list) else []
    plotted_rows = []
    for bucket_row in bucket_rows:
        if not isinstance(bucket_row, dict):
            continue
        bucket_label = str(bucket_row.get("bucket_label") or bucket_row.get("bucket_start") or "").strip()
        if not bucket_label:
            continue
        n_articles = bucket_row.get("n_articles")
        corpus_share = bucket_row.get("corpus_share")
        if not isinstance(n_articles, (int, float)) and not isinstance(corpus_share, (int, float)):
            continue
        plotted_rows.append(
            {
                "label": bucket_label,
                "articles": int(n_articles or 0),
                "corpus_share": float(corpus_share) if isinstance(corpus_share, (int, float)) else None,
                "status": str(bucket_row.get("status") or "n/a"),
            }
        )

    if not plotted_rows:
        return dbc.Alert(
            "No popularity timeline is available for the current selection.",
            color="warning",
            className="mb-0",
        )

    first_share = next((row["corpus_share"] for row in plotted_rows if isinstance(row["corpus_share"], float)), None)
    latest_share = next(
        (row["corpus_share"] for row in reversed(plotted_rows) if isinstance(row["corpus_share"], float)),
        None,
    )
    peak_articles_row = max(plotted_rows, key=lambda row: row["articles"])

    trend_note = (
        f"Peak volume lands in {peak_articles_row['label']} with {peak_articles_row['articles']} article"
        f"{'' if peak_articles_row['articles'] == 1 else 's'}."
    )
    if isinstance(first_share, float) and isinstance(latest_share, float):
        share_delta = latest_share - first_share
        if abs(share_delta) < 0.0005:
            trend_note += f" Latest corpus share holds flat at {_format_percent(latest_share)} versus the first bucket."
        elif share_delta > 0:
            trend_note += (
                f" Latest corpus share rises to {_format_percent(latest_share)} "
                f"({share_delta * 100:.1f} pts above the first bucket)."
            )
        else:
            trend_note += (
                f" Latest corpus share eases to {_format_percent(latest_share)} "
                f"({abs(share_delta) * 100:.1f} pts below the first bucket)."
            )

    granularity_label = str(bucket_granularity or "week").strip().title() or "Week"
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[row["label"] for row in plotted_rows],
            y=[row["articles"] for row in plotted_rows],
            name="Articles",
            marker={
                "color": [
                    "#ffc107" if row["status"] != "ok" else "#0d6efd"
                    for row in plotted_rows
                ]
            },
            customdata=[
                [row["corpus_share"], row["status"]]
                for row in plotted_rows
            ],
            hovertemplate=(
                "Bucket: %{x}<br>Articles: %{y}<br>Corpus share: %{customdata[0]:.1%}"
                "<br>Status: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[row["label"] for row in plotted_rows],
            y=[row["corpus_share"] for row in plotted_rows],
            mode="lines+markers",
            name="Corpus Share",
            line={"color": "#198754", "width": 2},
            marker={"size": 9},
            yaxis="y2",
            hovertemplate="Bucket: %{x}<br>Corpus share: %{y:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"{granularity_label} Popularity Timeline",
        template="plotly_white",
        margin={"l": 30, "r": 30, "t": 60, "b": 40},
        xaxis_title=granularity_label,
        yaxis_title="Articles",
        yaxis2={
            "title": "Corpus Share",
            "overlaying": "y",
            "side": "right",
            "tickformat": ".0%",
            "rangemode": "tozero",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
    )

    return html.Div(
        [
            html.H6("Popularity Timeline", className="mt-0 mb-2"),
            html.P(
                f"{granularity_label} bucket counts and corpus share for the selected group. {trend_note}",
                className="text-muted mb-2",
            ),
            dcc.Graph(figure=figure, config={"displayModeBar": False}, style={"height": "280px"}),
        ]
    )


def _cluster_popularity_comparison(
    all_group_rows: list[dict],
    selected_row: dict | None,
    temporal_payload: dict,
    group_type: str,
    bucket_granularity: str = "week",
    selected_cluster_row: dict | None = None,
):
    if not isinstance(selected_row, dict):
        return dbc.Alert(
            "No popularity comparison is available for the current selection.",
            color="warning",
            className="mb-0",
        )

    selected_cluster_value = (
        str(selected_cluster_row.get("cluster_id"))
        if isinstance(selected_cluster_row, dict)
        else ALL_GROUPS_CLUSTER_VALUE
    )
    scoped_rows = _group_rows_for_cluster(all_group_rows, selected_cluster_value)
    if not scoped_rows:
        return dbc.Alert(
            "No popularity comparison rows are available for the current selection.",
            color="warning",
            className="mb-0",
        )

    popularity_rows = _cluster_popularity_rows(
        scoped_rows,
        temporal_payload,
        group_type,
    )
    if not popularity_rows:
        return dbc.Alert(
            "No popularity comparison rows are available for the current selection.",
            color="warning",
            className="mb-0",
        )

    selected_group_key = _normalize_group_lookup_key(selected_row.get("group_key") or selected_row.get("group"))
    first_rank_rows = _rank_share_position_rows(
        popularity_rows,
        value_field="first_share",
        rank_field="first_share_rank",
    )
    popularity_rows = _rank_share_position_rows(
        popularity_rows,
        value_field="latest_share",
        rank_field="latest_share_rank",
    )
    first_rank_by_key = {
        _row_group_lookup_key(popularity_row): popularity_row.get("first_share_rank")
        for popularity_row in first_rank_rows
    }

    granularity_label = str(bucket_granularity or "week").strip().title() or "Week"
    scope_label = (
        str(selected_cluster_row.get("label") or selected_cluster_row.get("cluster_id") or "Selected cluster")
        if isinstance(selected_cluster_row, dict)
        else "All groups"
    )
    summary = (
        f"Compares starting-to-latest {granularity_label.lower()} corpus-share rank, latest share, "
        f"and peak bucket volume within {scope_label}. "
        f"Showing {min(len(popularity_rows), 6)} of {len(popularity_rows)} groups with bucket rows."
    )

    table_rows = []
    for index, popularity_row in enumerate(popularity_rows[:6], start=1):
        group_key = _row_group_lookup_key(popularity_row)
        class_name = "table-primary" if group_key == selected_group_key else None
        first_rank = first_rank_by_key.get(group_key)
        latest_rank = popularity_row.get("latest_share_rank")
        table_rows.append(
            html.Tr(
                [
                    html.Td(str(int(latest_rank) if isinstance(latest_rank, int) else index)),
                    html.Td(str(popularity_row["group"])),
                    html.Td(str(int(popularity_row["n_articles"]))),
                    html.Td(str(popularity_row["peak_bucket"])),
                    html.Td(str(int(popularity_row["peak_articles"]))),
                    html.Td(_format_percent(popularity_row["first_share"])),
                    html.Td(_format_percent(popularity_row["latest_share"])),
                    html.Td(_rank_trajectory_compact_label(first_rank, latest_rank)),
                    html.Td(_share_delta_label(popularity_row["share_delta"])),
                ],
                className=class_name,
            )
        )

    return html.Div(
        [
            html.H6("Popularity Comparison", className="mt-0 mb-2"),
            html.P(summary, className="text-muted mb-2"),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Latest Rank"),
                                html.Th("Group"),
                                html.Th("Articles"),
                                html.Th("Peak Bucket"),
                                html.Th("Peak Articles"),
                                html.Th("First Share"),
                                html.Th("Latest Share"),
                                html.Th("Rank Trajectory"),
                                html.Th("Share Drift"),
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
    )


def _cluster_popularity_rows(
    scoped_rows: list[dict],
    temporal_payload: dict,
    group_type: str,
) -> list[dict[str, object]]:
    groups = temporal_payload.get("groups") if isinstance(temporal_payload.get("groups"), dict) else {}
    temporal_rows = groups.get(group_type) if isinstance(groups.get(group_type), list) else []
    temporal_rows_by_key: dict[str, dict] = {}
    for temporal_row in temporal_rows:
        if not isinstance(temporal_row, dict):
            continue
        temporal_group_key = _normalize_group_lookup_key(temporal_row.get("group_key") or temporal_row.get("group"))
        if temporal_group_key:
            temporal_rows_by_key[temporal_group_key] = temporal_row

    popularity_rows: list[dict[str, object]] = []
    for group_row in scoped_rows:
        if not isinstance(group_row, dict):
            continue
        group_key = _row_group_lookup_key(group_row)
        if not group_key:
            continue
        temporal_row = temporal_rows_by_key.get(group_key)
        popularity_summary = (
            temporal_row.get("popularity_summary")
            if isinstance(temporal_row, dict) and isinstance(temporal_row.get("popularity_summary"), dict)
            else {}
        )
        if popularity_summary:
            share_delta = popularity_summary.get("share_delta")
            recent_share_delta = popularity_summary.get("recent_share_delta")
            popularity_rows.append(
                {
                    "group": str(group_row.get("group") or temporal_row.get("group") or "Unknown"),
                    "group_key": group_key,
                    "n_articles": int(group_row.get("n_articles") or temporal_row.get("n_articles") or 0),
                    "peak_bucket": popularity_summary.get("peak_bucket"),
                    "peak_articles": int(popularity_summary.get("peak_articles") or 0),
                    "first_share": popularity_summary.get("first_share"),
                    "latest_share": popularity_summary.get("latest_share"),
                    "share_delta": share_delta,
                    "recent_share_delta": recent_share_delta,
                    "share_direction": popularity_summary.get("share_direction") or _share_delta_direction(share_delta),
                    "recent_share_direction": popularity_summary.get("recent_share_direction")
                    or _share_delta_direction(recent_share_delta),
                    "momentum_label": popularity_summary.get("momentum_label")
                    or _popularity_momentum_label(share_delta, recent_share_delta),
                }
            )
            continue
        bucket_rows = temporal_row.get("buckets") if isinstance(temporal_row, dict) and isinstance(temporal_row.get("buckets"), list) else []
        plotted_rows = []
        for bucket_row in bucket_rows:
            if not isinstance(bucket_row, dict):
                continue
            bucket_label = str(bucket_row.get("bucket_label") or bucket_row.get("bucket_start") or "").strip()
            n_articles = bucket_row.get("n_articles")
            corpus_share = bucket_row.get("corpus_share")
            if not bucket_label or (not isinstance(n_articles, (int, float)) and not isinstance(corpus_share, (int, float))):
                continue
            plotted_rows.append(
                {
                    "label": bucket_label,
                    "articles": int(n_articles or 0),
                    "corpus_share": float(corpus_share) if isinstance(corpus_share, (int, float)) else None,
                }
            )
        if not plotted_rows:
            continue

        peak_articles_row = max(plotted_rows, key=lambda row: int(row["articles"]))
        first_share = next((row["corpus_share"] for row in plotted_rows if isinstance(row["corpus_share"], float)), None)
        latest_share = next(
            (row["corpus_share"] for row in reversed(plotted_rows) if isinstance(row["corpus_share"], float)),
            None,
        )
        previous_share = next(
            (row["corpus_share"] for row in reversed(plotted_rows[:-1]) if isinstance(row["corpus_share"], float)),
            None,
        )
        share_delta = (
            float(latest_share) - float(first_share)
            if isinstance(first_share, float) and isinstance(latest_share, float)
            else None
        )
        recent_share_delta = (
            float(latest_share) - float(previous_share)
            if isinstance(previous_share, float) and isinstance(latest_share, float)
            else None
        )
        popularity_rows.append(
            {
                "group": str(group_row.get("group") or "Unknown"),
                "group_key": group_key,
                "n_articles": int(group_row.get("n_articles") or 0),
                "peak_bucket": str(peak_articles_row["label"]),
                "peak_articles": int(peak_articles_row["articles"]),
                "first_share": first_share,
                "latest_share": latest_share,
                "share_delta": share_delta,
                "recent_share_delta": recent_share_delta,
                "share_direction": _share_delta_direction(share_delta),
                "recent_share_direction": _share_delta_direction(recent_share_delta),
                "momentum_label": _popularity_momentum_label(share_delta, recent_share_delta),
            }
        )

    return popularity_rows


def _share_delta_direction(value: object) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    if abs(float(value)) < 0.0005:
        return "flat"
    return "rising" if float(value) > 0 else "falling"


def _popularity_momentum_label(share_delta: object, recent_share_delta: object) -> str | None:
    overall_direction = _share_delta_direction(share_delta)
    recent_direction = _share_delta_direction(recent_share_delta)

    if overall_direction == "rising":
        if recent_direction == "rising":
            return "rising now"
        if recent_direction == "falling":
            return "cooling after gains"
        return "holding gains"
    if overall_direction == "falling":
        if recent_direction == "rising":
            return "rebounding"
        if recent_direction == "falling":
            return "still falling"
        return "holding losses"
    if overall_direction == "flat":
        if recent_direction == "rising":
            return "rebounding to baseline"
        if recent_direction == "falling":
            return "slipping from baseline"
        return "flat"
    if recent_direction == "rising":
        return "recently rising"
    if recent_direction == "falling":
        return "recently falling"
    return recent_direction


def _share_delta_label(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    if abs(float(value)) < 0.0005:
        return "Flat"
    return f"{float(value) * 100:+.1f} pts"


def _recent_step_summary(direction: object, recent_share_delta: object) -> str:
    normalized_direction = str(direction or "").strip().lower()
    delta_label = _share_delta_label(recent_share_delta)
    if normalized_direction in {"rising", "falling", "flat"}:
        if delta_label == "n/a":
            return normalized_direction
        return f"{normalized_direction} ({delta_label})"
    return delta_label


def _rank_share_drift_rows(popularity_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked_rows = [dict(row) for row in popularity_rows if isinstance(row, dict)]
    ranked_rows.sort(
        key=lambda row: (
            -(float(row["share_delta"]) if isinstance(row["share_delta"], float) else -999.0),
            -(float(row["recent_share_delta"]) if isinstance(row["recent_share_delta"], float) else -999.0),
            -(float(row["latest_share"]) if isinstance(row["latest_share"], float) else -1.0),
            str(row["group"]).lower(),
        )
    )
    for index, popularity_row in enumerate(ranked_rows, start=1):
        popularity_row["share_drift_rank"] = index
    return ranked_rows


def _rank_share_position_rows(
    popularity_rows: list[dict[str, object]],
    *,
    value_field: str,
    rank_field: str,
) -> list[dict[str, object]]:
    ranked_rows = [dict(row) for row in popularity_rows if isinstance(row, dict)]
    ranked_rows.sort(
        key=lambda row: (
            -(float(row[value_field]) if isinstance(row.get(value_field), (int, float)) else -999.0),
            -(float(row["recent_share_delta"]) if isinstance(row.get("recent_share_delta"), (int, float)) else -999.0),
            -(int(row["peak_articles"]) if isinstance(row.get("peak_articles"), (int, float)) else -1),
            str(row.get("group") or "").lower(),
        )
    )
    for index, popularity_row in enumerate(ranked_rows, start=1):
        popularity_row[rank_field] = index
    return ranked_rows


def _rank_trajectory_summary(first_rank: object, latest_rank: object, scope_label: str) -> str:
    if not isinstance(first_rank, int) or not isinstance(latest_rank, int):
        return "n/a"
    if latest_rank < first_rank:
        return f"{first_rank} -> {latest_rank} in {scope_label} (up {first_rank - latest_rank})"
    if latest_rank > first_rank:
        return f"{first_rank} -> {latest_rank} in {scope_label} (down {latest_rank - first_rank})"
    return f"{first_rank} -> {latest_rank} in {scope_label} (held)"


def _rank_trajectory_compact_label(first_rank: object, latest_rank: object) -> str:
    if not isinstance(first_rank, int) or not isinstance(latest_rank, int):
        return "n/a"
    if latest_rank < first_rank:
        return f"{first_rank} -> {latest_rank} (up {first_rank - latest_rank})"
    if latest_rank > first_rank:
        return f"{first_rank} -> {latest_rank} (down {latest_rank - first_rank})"
    return f"{first_rank} -> {latest_rank} (held)"


def _selected_group_popularity_metrics(
    row: dict,
    all_group_rows: list[dict],
    temporal_payload: dict,
    group_type: str,
    selected_cluster_row: dict | None = None,
) -> list[tuple[str, str]]:
    selected_cluster_value = (
        str(selected_cluster_row.get("cluster_id"))
        if isinstance(selected_cluster_row, dict)
        else ALL_GROUPS_CLUSTER_VALUE
    )
    scoped_rows = _group_rows_for_cluster(all_group_rows, selected_cluster_value)
    popularity_rows = _rank_share_drift_rows(_cluster_popularity_rows(scoped_rows, temporal_payload, group_type))
    if not popularity_rows:
        return []

    first_rank_rows = _rank_share_position_rows(
        popularity_rows,
        value_field="first_share",
        rank_field="first_share_rank",
    )
    latest_rank_rows = _rank_share_position_rows(
        popularity_rows,
        value_field="latest_share",
        rank_field="latest_share_rank",
    )
    first_rank_by_key = {
        _row_group_lookup_key(popularity_row): popularity_row.get("first_share_rank")
        for popularity_row in first_rank_rows
    }
    latest_rank_by_key = {
        _row_group_lookup_key(popularity_row): popularity_row.get("latest_share_rank")
        for popularity_row in latest_rank_rows
    }

    selected_group_key = _normalize_group_lookup_key(row.get("group_key") or row.get("group"))
    scope_label = (
        str(selected_cluster_row.get("label") or selected_cluster_row.get("cluster_id") or "Selected cluster")
        if isinstance(selected_cluster_row, dict)
        else "All groups"
    )
    for popularity_row in popularity_rows:
        if _row_group_lookup_key(popularity_row) != selected_group_key:
            continue

        return [
            ("Popularity Momentum", str(popularity_row.get("momentum_label") or "n/a")),
            (
                "Share Drift Rank",
                f"{int(popularity_row.get('share_drift_rank') or 0)} of {len(popularity_rows)} in {scope_label}",
            ),
            (
                "Rank Trajectory",
                _rank_trajectory_summary(
                    first_rank_by_key.get(selected_group_key),
                    latest_rank_by_key.get(selected_group_key),
                    scope_label,
                ),
            ),
            (
                "Recent Step",
                _recent_step_summary(
                    popularity_row.get("recent_share_direction"),
                    popularity_row.get("recent_share_delta"),
                ),
            ),
        ]
    return []


def _share_delta_metric(value: object, max_abs_value: object) -> html.Div:
    label = _share_delta_label(value)
    if not isinstance(value, (int, float)):
        return html.Div(label, className="share-drift-metric text-muted")

    numeric_value = float(value)
    abs_value = abs(numeric_value)
    scale_max = float(max_abs_value) if isinstance(max_abs_value, (int, float)) and float(max_abs_value) > 0 else abs_value
    relative_width = min(abs_value / scale_max, 1.0) if scale_max > 0 else 0.0

    if abs_value < 0.0005:
        label_class = "share-drift-label text-muted"
        fill_style = {
            "left": "calc(50% - 3px)",
            "width": "6px",
            "backgroundColor": "#6c757d",
        }
    elif numeric_value > 0:
        label_class = "share-drift-label text-success"
        fill_style = {
            "left": "50%",
            "width": f"{relative_width * 50:.1f}%",
            "backgroundColor": "#198754",
        }
    else:
        label_class = "share-drift-label text-danger"
        fill_style = {
            "left": f"{50 - (relative_width * 50):.1f}%",
            "width": f"{relative_width * 50:.1f}%",
            "backgroundColor": "#dc3545",
        }

    return html.Div(
        [
            html.Div(label, className=label_class),
            html.Div(
                [
                    html.Div(
                        className="share-drift-mini-bar-midline",
                        style={
                            "position": "absolute",
                            "left": "50%",
                            "top": "2px",
                            "bottom": "2px",
                            "width": "1px",
                            "backgroundColor": "#adb5bd",
                        },
                    ),
                    html.Div(
                        className="share-drift-mini-bar-fill",
                        style={
                            "position": "absolute",
                            "top": "6px",
                            "height": "8px",
                            "borderRadius": "999px",
                            **fill_style,
                        },
                    ),
                ],
                className="share-drift-mini-bar",
                title="Scaled within the visible rows for this column.",
                style={
                    "position": "relative",
                    "width": "72px",
                    "height": "20px",
                    "borderRadius": "999px",
                    "backgroundColor": "#f8f9fa",
                    "overflow": "hidden",
                },
            ),
        ],
        className="share-drift-metric",
    )


def _cluster_share_drift_comparison(
    all_group_rows: list[dict],
    selected_row: dict | None,
    temporal_payload: dict,
    group_type: str,
    bucket_granularity: str = "week",
    selected_cluster_row: dict | None = None,
):
    if not isinstance(selected_row, dict):
        return dbc.Alert(
            "No share-drift comparison is available for the current selection.",
            color="warning",
            className="mb-0",
        )

    selected_cluster_value = (
        str(selected_cluster_row.get("cluster_id"))
        if isinstance(selected_cluster_row, dict)
        else ALL_GROUPS_CLUSTER_VALUE
    )
    scoped_rows = _group_rows_for_cluster(all_group_rows, selected_cluster_value)
    if not scoped_rows:
        return dbc.Alert(
            "No share-drift comparison rows are available for the current selection.",
            color="warning",
            className="mb-0",
        )

    popularity_rows = _cluster_popularity_rows(scoped_rows, temporal_payload, group_type)
    if not popularity_rows:
        return dbc.Alert(
            "No share-drift comparison rows are available for the current selection.",
            color="warning",
            className="mb-0",
        )

    selected_group_key = _normalize_group_lookup_key(selected_row.get("group_key") or selected_row.get("group"))
    popularity_rows = _rank_share_drift_rows(popularity_rows)
    first_rank_rows = _rank_share_position_rows(
        popularity_rows,
        value_field="first_share",
        rank_field="first_share_rank",
    )
    latest_rank_rows = _rank_share_position_rows(
        popularity_rows,
        value_field="latest_share",
        rank_field="latest_share_rank",
    )
    first_rank_by_key = {
        _row_group_lookup_key(popularity_row): popularity_row.get("first_share_rank")
        for popularity_row in first_rank_rows
    }
    latest_rank_by_key = {
        _row_group_lookup_key(popularity_row): popularity_row.get("latest_share_rank")
        for popularity_row in latest_rank_rows
    }

    granularity_label = str(bucket_granularity or "week").strip().title() or "Week"
    scope_label = (
        str(selected_cluster_row.get("label") or selected_cluster_row.get("cluster_id") or "Selected cluster")
        if isinstance(selected_cluster_row, dict)
        else "All groups"
    )
    summary = (
        f"Ranks first-to-latest {granularity_label.lower()} corpus-share change within {scope_label} "
        f"and shows the latest {granularity_label.lower()} step. "
        f"Showing {min(len(popularity_rows), 6)} of {len(popularity_rows)} groups with bucket rows."
    )
    visible_rows = popularity_rows[:6]
    drift_scale_max = max(
        (abs(float(row["share_delta"])) for row in visible_rows if isinstance(row.get("share_delta"), float)),
        default=0.0,
    )
    recent_scale_max = max(
        (abs(float(row["recent_share_delta"])) for row in visible_rows if isinstance(row.get("recent_share_delta"), float)),
        default=0.0,
    )

    table_rows = []
    for index, popularity_row in enumerate(visible_rows, start=1):
        group_key = _row_group_lookup_key(popularity_row)
        class_name = "table-primary" if group_key == selected_group_key else None
        table_rows.append(
            html.Tr(
                [
                    html.Td(str(index)),
                    html.Td(str(popularity_row["group"])),
                    html.Td(_format_percent(popularity_row["first_share"])),
                    html.Td(_format_percent(popularity_row["latest_share"])),
                    html.Td(
                        _rank_trajectory_compact_label(
                            first_rank_by_key.get(group_key),
                            latest_rank_by_key.get(group_key),
                        )
                    ),
                    html.Td(_share_delta_metric(popularity_row["share_delta"], drift_scale_max)),
                    html.Td(_share_delta_metric(popularity_row["recent_share_delta"], recent_scale_max)),
                    html.Td(str(int(popularity_row["peak_articles"]))),
                    html.Td(str(popularity_row["peak_bucket"])),
                ],
                className=class_name,
            )
        )

    return html.Div(
        [
            html.H6("Share Drift Comparison", className="mt-0 mb-2"),
            html.P(summary, className="text-muted mb-2"),
            html.P("Mini-bars scale relative magnitude within the visible rows for each column.", className="text-muted small mb-2"),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Rank"),
                                html.Th("Group"),
                                html.Th("First Share"),
                                html.Th("Latest Share"),
                                html.Th("Rank Trajectory"),
                                html.Th("Drift"),
                                html.Th("Recent Step"),
                                html.Th("Peak Articles"),
                                html.Th("Peak Bucket"),
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
    )


def _temporal_bucket_diagnostics_table(
    temporal_row: dict | None,
    bucket_granularity: str = "week",
    movement_basis: str | None = "pca",
):
    if not isinstance(temporal_row, dict):
        return dbc.Alert(
            "No bucket-level diagnostics are available for the current selection.",
            color="warning",
            className="mb-0",
        )

    bucket_rows = temporal_row.get("buckets") if isinstance(temporal_row.get("buckets"), list) else []
    if not bucket_rows:
        return dbc.Alert(
            "No bucket-level diagnostics are available for the current selection.",
            color="warning",
            className="mb-0",
        )

    granularity_label = str(bucket_granularity or "week").strip().title() or "Week"
    basis_label = str(_movement_basis_config(movement_basis)["basis"]).upper()
    table_rows = []
    for bucket_row in bucket_rows[:8]:
        if not isinstance(bucket_row, dict):
            continue
        bucket_label = str(bucket_row.get("bucket_label") or bucket_row.get("bucket_start") or "Unknown")
        table_rows.append(
            html.Tr(
                [
                    html.Td(bucket_label),
                    html.Td(str(bucket_row.get("status") or "n/a")),
                    html.Td(str(int(bucket_row.get("n_articles") or 0))),
                    html.Td(str(int(bucket_row.get("n_sources") or 0))),
                    html.Td(_format_percent(bucket_row.get("corpus_share"))),
                    html.Td(_bucket_coordinate_summary(bucket_row, movement_basis)),
                    html.Td(_source_count_summary(bucket_row.get("source_counts"))),
                    html.Td(_lens_deviation_summary(bucket_row.get("top_lens_deviations"))),
                ]
            )
        )

    return html.Div(
        [
            html.H6("Temporal Bucket Diagnostics", className="mt-0 mb-2"),
            html.P(
                f"{granularity_label} buckets for the selected group. Coordinates follow the {basis_label} movement basis.",
                className="text-muted mb-2",
            ),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Bucket"),
                                html.Th("Status"),
                                html.Th("Articles"),
                                html.Th("Sources"),
                                html.Th("Corpus Share"),
                                html.Th("Coordinates"),
                                html.Th("Top Sources"),
                                html.Th("Top Lens Drift"),
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
    )


def _cluster_peer_movement_callout(
    all_group_rows: list[dict],
    selected_row: dict | None,
    temporal_payload: dict,
    group_type: str,
    movement_basis: str | None = "pca",
):
    if not isinstance(selected_row, dict):
        return None

    selected_group_label = str(selected_row.get("group") or "Selected group")
    selected_group_key = _normalize_group_lookup_key(selected_row.get("group_key") or selected_group_label)
    selected_cluster_id = str(selected_row.get("cluster_id") or "").strip().lower()
    selected_cluster_label = str(selected_row.get("cluster_label") or "this cluster")
    if not selected_cluster_id:
        return None

    groups = temporal_payload.get("groups") if isinstance(temporal_payload.get("groups"), dict) else {}
    temporal_rows = groups.get(group_type) if isinstance(groups.get(group_type), list) else []
    temporal_rows_by_key = {}
    for temporal_row in temporal_rows:
        if not isinstance(temporal_row, dict):
            continue
        temporal_group_key = _normalize_group_lookup_key(temporal_row.get("group_key") or temporal_row.get("group"))
        if temporal_group_key:
            temporal_rows_by_key[temporal_group_key] = temporal_row

    basis_config = _movement_basis_config(movement_basis)
    basis_label = str(basis_config["basis"]).upper()
    total_movement_key = str(basis_config["total_movement_key"])
    peer_values: list[tuple[str, float]] = []
    selected_value: float | None = None
    cluster_group_count = 0

    for group_row in all_group_rows:
        if not isinstance(group_row, dict):
            continue
        group_cluster_id = str(group_row.get("cluster_id") or "").strip().lower()
        if group_cluster_id != selected_cluster_id:
            continue
        cluster_group_count += 1
        group_key = _normalize_group_lookup_key(group_row.get("group_key") or group_row.get("group"))
        if not group_key:
            continue
        temporal_row = temporal_rows_by_key.get(group_key)
        path_summary = temporal_row.get("path_summary") if isinstance(temporal_row, dict) and isinstance(temporal_row.get("path_summary"), dict) else {}
        total_movement = path_summary.get(total_movement_key)
        if not isinstance(total_movement, (int, float)):
            continue
        if group_key == selected_group_key:
            selected_value = float(total_movement)
            continue
        peer_values.append((str(group_row.get("group") or "Unknown"), float(total_movement)))

    if cluster_group_count < 2:
        return dbc.Alert(
            f"Cluster movement context: {selected_cluster_label} contains only this selected group, so there are no cluster peers to compare.",
            color="light",
            className="mb-3 py-2",
        )

    if selected_value is None:
        return dbc.Alert(
            f"Cluster movement context: {selected_group_label} does not yet have a comparable {basis_label} total movement in {selected_cluster_label}.",
            color="light",
            className="mb-3 py-2",
        )

    if not peer_values:
        return dbc.Alert(
            f"Cluster movement context: {selected_group_label} has no peer groups with comparable {basis_label} movement totals in {selected_cluster_label}.",
            color="light",
            className="mb-3 py-2",
        )

    peer_average = sum(value for _, value in peer_values) / len(peer_values)
    difference = selected_value - peer_average
    if abs(difference) < 0.005:
        comparison_text = f"in line with the {selected_cluster_label} peer average of {_format_decimal(peer_average)}"
    elif difference > 0:
        comparison_text = (
            f"{_format_decimal(difference)} above the {selected_cluster_label} peer average of {_format_decimal(peer_average)}"
        )
    else:
        comparison_text = (
            f"{_format_decimal(abs(difference))} below the {selected_cluster_label} peer average of {_format_decimal(peer_average)}"
        )
    rank = 1 + sum(1 for _, value in peer_values if value > selected_value)
    summary = (
        f"Cluster movement context: {selected_group_label} has {basis_label} total movement {_format_decimal(selected_value)}, "
        f"{comparison_text}, and ranks {rank} of {cluster_group_count} in its cluster."
    )
    return dbc.Alert(summary, color="light", className="mb-3 py-2")


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
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(id="news-group-latent-movement-scope"),
                        html.Div(
                            [
                                dbc.Label("Movement basis", className="mb-1"),
                                dbc.RadioItems(
                                    id="news-group-latent-movement-basis",
                                    options=MOVEMENT_BASIS_OPTIONS,
                                    value="pca",
                                    inline=True,
                                    class_name="mb-2",
                                ),
                            ]
                        ),
                        dcc.Graph(id="news-group-latent-movement"),
                    ],
                    lg=8,
                    className="mb-3",
                ),
                dbc.Col(html.Div(id="news-group-latent-movement-summary"), lg=4, className="mb-3"),
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
    Output("news-group-latent-movement", "figure"),
    Output("news-group-latent-movement-scope", "children"),
    Output("news-group-latent-movement-summary", "children"),
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
    Input("news-group-latent-movement-basis", "value"),
    State("news-group-latent-mode", "value"),
    State("news-group-latent-snapshot-date", "value"),
)
def load_news_group_latent_space(
    _load_tick,
    _refresh_clicks,
    group_type,
    selected_cluster_id,
    selected_group_key,
    movement_basis,
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
    movement_title = f"Weekly {_movement_basis_config(movement_basis)['title']}"
    empty_movement = _empty_figure(movement_title)
    if status_code != 200:
        error = payload.get("error", "Unknown error")
        alert = dbc.Alert(f"Stats error ({status_code}): {error}", color="danger")
        return (
            alert,
            [],
            [],
            None,
            [],
            None,
            empty_pca,
            empty_mds,
            empty_movement,
            alert,
            alert,
            alert,
            alert,
            alert,
            alert,
            alert,
        )

    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    derived = payload.get("data", {}).get("derived", {}) if isinstance(payload.get("data"), dict) else {}
    group_latent = derived.get("group_latent_space", {}) if isinstance(derived.get("group_latent_space"), dict) else {}
    group_temporal = (
        derived.get("group_temporal_latent_space")
        if isinstance(derived.get("group_temporal_latent_space"), dict)
        else {}
    )
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
    selected_value = _row_group_value(selected_row)
    bucket_granularity = (
        str(group_temporal.get("config", {}).get("bucket_granularity") or "week")
        if isinstance(group_temporal.get("config"), dict)
        else "week"
    )
    temporal_row = _selected_temporal_group_row(group_temporal, effective_group_type, selected_value)

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
            _empty_figure(movement_title, message),
            warning,
            warning,
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
            _group_temporal_export_panel(
                meta,
                bucket_granularity,
                group_type=effective_group_type,
                group_key=selected_value,
            ),
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
        _group_movement_figure(temporal_row, bucket_granularity, movement_basis),
        _group_temporal_scope_summary(temporal_row, bucket_granularity, movement_basis, selected_cluster),
        html.Div(
            [
                _group_movement_summary(temporal_row, bucket_granularity, movement_basis, selected_cluster),
                _movement_leaderboard(
                    filtered_rows,
                    group_temporal,
                    effective_group_type,
                    movement_basis,
                    selected_value,
                    selected_cluster,
                ),
            ],
            className="d-grid gap-3",
        ),
        _group_table(filtered_rows, selected_value, selected_cluster),
        html.Div(
            [
                _cluster_overview_table(cluster_rows, selected_cluster),
                html.Div(className="mb-3"),
                _cluster_membership_summary(selected_cluster),
            ]
        ),
        _selected_group_summary(
            selected_row,
            temporal_row,
            bucket_granularity,
            movement_basis,
            selected_cluster,
            rows,
            group_temporal,
            effective_group_type,
        ),
        _nearest_groups_table(selected_row, movement_basis),
        _lens_deviation_table(selected_row),
    )


@callback(
    Output("news-group-latent-snapshot-date", "disabled"),
    Input("news-group-latent-mode", "value"),
)
def toggle_group_latent_snapshot_input(data_mode):
    return data_mode != "snapshot"
