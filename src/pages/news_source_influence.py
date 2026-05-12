from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import html

from src.pages.news_page_utils import build_news_intro


dash.register_page(
    __name__,
    path="/news/source-influence",
    name="News Source Influence",
    title="NewsLens | News Source Influence",
)


US_ADULT_POPULATION_MILLIONS = 268.3


SOURCE_REFERENCES = {
    "pew": {
        "label": "Pew Research Center News Media Tracker, 2025",
        "url": "https://www.pewresearch.org/journalism/feature/news-media-tracker/",
    },
    "pew_data": {
        "label": "Pew tracker dataset",
        "url": "https://docs.google.com/spreadsheets/d/10UKS5LiJi9qmPshiBAH0Hoelgvto1qYCoW2qNTo2hVg/edit?usp=sharing",
    },
    "census": {
        "label": "U.S. Census QuickFacts",
        "url": "https://www.census.gov/quickfacts/fact/table/US/PST045225",
    },
    "similarweb": {
        "label": "Similarweb U.S. News & Media rankings, April 2026",
        "url": "https://www.similarweb.com/top-websites/united-states/news-and-media/",
    },
    "abc_au": {
        "label": "ABC Australia international audience, 2025",
        "url": "https://www.abc.net.au/about/media-centre/press-releases/abc-grows-international-audience/105372698",
    },
    "al_jazeera": {
        "label": "Al Jazeera English network profile",
        "url": "https://network.aljazeera.net/al-jazeera-english",
    },
    "bbc": {
        "label": "Ofcom BBC Annual Report 2024-25",
        "url": "https://www.ofcom.org.uk/siteassets/resources/documents/tv-radio-and-on-demand/bbc/bbc-annual-report/2025/ofcoms-annual-report-on-the-bbc-2024-25.pdf",
    },
    "dw": {
        "label": "DW usage data 2025",
        "url": "https://corporate.dw.com/en/usage-data-2025-dw-reports-worldwide-reach-of-337-million-weekly-users/a-73359910",
    },
    "france24": {
        "label": "France 24 audience profile",
        "url": "https://www.francemediasmonde.com/en/france-24/",
    },
    "ft": {
        "label": "Financial Times 2024 audience reporting",
        "url": "https://pressgazette.co.uk/media_business/financial-times-reports-global-revenue-boost-to-540m-for-2024/",
    },
    "nbc": {
        "label": "NBC News audience release, 2025",
        "url": "https://nbcuniversalnewsgroup.com/nbcnews/2025/12/23/nbc-news-to-finish-2025-as-the-1-news-organization-in-the-u-s-reaching-126-million-americans-each-month-across-television-digital-streaming/",
    },
    "nyt": {
        "label": "New York Times Company Q2 2025 results",
        "url": "https://www.sec.gov/Archives/edgar/data/71691/000007169125000117/pressrelease06302025.htm",
    },
    "sky": {
        "label": "Sky News Distribution",
        "url": "https://skynewsdistribution.sky/",
    },
}


SOURCE_ROWS = [
    {
        "source": "ABC News",
        "rss": "https://abcnews.go.com/abcnews/topstories",
        "use_pct": 36,
        "awareness_pct": 92,
        "trust_pct": 44,
        "note": "Major U.S. broadcast network news brand.",
        "refs": ["pew", "pew_data"],
    },
    {
        "source": "ABC News (Australia)",
        "rss": "https://www.abc.net.au/news/feed/51120/rss.xml",
        "use_pct": None,
        "awareness_pct": None,
        "trust_pct": None,
        "note": "ABC reported more than 11M people outside Australia engaging with ABC content in early 2025.",
        "refs": ["abc_au"],
    },
    {
        "source": "Al Jazeera",
        "rss": "https://www.aljazeera.com/xml/rss/all.xml",
        "use_pct": None,
        "awareness_pct": None,
        "trust_pct": None,
        "note": "Al Jazeera English reports reach into 350M+ households across 150+ countries.",
        "refs": ["al_jazeera"],
    },
    {
        "source": "BBC News",
        "rss": "http://feeds.bbci.co.uk/news/rss.xml",
        "use_pct": 21,
        "awareness_pct": 79,
        "trust_pct": 35,
        "note": "Pew-measured U.S. use plus BBC World Service global audience context.",
        "refs": ["pew", "pew_data", "bbc"],
    },
    {
        "source": "CBS News",
        "rss": "https://www.cbsnews.com/latest/rss/main",
        "use_pct": 30,
        "awareness_pct": 90,
        "trust_pct": 39,
        "note": "Major U.S. broadcast network news brand.",
        "refs": ["pew", "pew_data"],
    },
    {
        "source": "DW",
        "rss": "https://rss.dw.com/rdf/rss-en-all",
        "use_pct": None,
        "awareness_pct": None,
        "trust_pct": None,
        "note": "DW reported 337M weekly users worldwide in 2025.",
        "refs": ["dw"],
    },
    {
        "source": "Financial Times",
        "rss": "https://www.ft.com/?format=rss",
        "use_pct": None,
        "awareness_pct": None,
        "trust_pct": None,
        "note": "FT reporting indicates roughly 1.5M paying readers and a 2.8M global paying audience in 2024.",
        "refs": ["ft"],
    },
    {
        "source": "Fox News",
        "rss": "http://feeds.foxnews.com/foxnews/latest",
        "use_pct": 38,
        "awareness_pct": 93,
        "trust_pct": 37,
        "note": "Pew's highest-use source among the NewsLens RSS outlets measured in its 2025 tracker.",
        "refs": ["pew", "pew_data", "similarweb"],
    },
    {
        "source": "France 24",
        "rss": "https://www.france24.com/en/rss",
        "use_pct": None,
        "awareness_pct": None,
        "trust_pct": None,
        "note": "France 24 reports 137M weekly audience in 2024 and 3.3M weekly viewers in North America.",
        "refs": ["france24"],
    },
    {
        "source": "NBC News",
        "rss": "http://feeds.nbcnews.com/nbcnews/public/news",
        "use_pct": 35,
        "awareness_pct": 92,
        "trust_pct": 42,
        "note": "NBC News reported reaching 126M Americans monthly across TV, digital, and streaming in 2025.",
        "refs": ["pew", "pew_data", "nbc"],
    },
    {
        "source": "NPR",
        "rss": "https://feeds.npr.org/1001/rss.xml",
        "use_pct": 20,
        "awareness_pct": 58,
        "trust_pct": 29,
        "note": "National public radio network and digital news brand.",
        "refs": ["pew", "pew_data"],
    },
    {
        "source": "PBS NewsHour",
        "rss": "https://www.pbs.org/newshour/feeds/rss/headlines",
        "use_pct": 21,
        "awareness_pct": 85,
        "trust_pct": 41,
        "note": "Mapped to Pew's PBS measure because NewsHour is PBS's flagship public affairs news program.",
        "refs": ["pew", "pew_data"],
    },
    {
        "source": "SkyNews",
        "rss": "http://feeds.skynews.com/feeds/rss/home.xml",
        "use_pct": None,
        "awareness_pct": None,
        "trust_pct": None,
        "note": "Sky News says its distribution reaches audiences in 100+ countries across broadcast and digital platforms.",
        "refs": ["sky"],
    },
    {
        "source": "The Guardian",
        "rss": "https://www.theguardian.com/world/rss",
        "use_pct": 8,
        "awareness_pct": 57,
        "trust_pct": 14,
        "note": "Pew-measured U.S. use for a global English-language publisher.",
        "refs": ["pew", "pew_data"],
    },
    {
        "source": "The New York Times",
        "rss": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "use_pct": 19,
        "awareness_pct": 84,
        "trust_pct": 32,
        "note": "NYTimes.com ranked #2 among U.S. News & Media sites in Similarweb's April 2026 ranking.",
        "refs": ["pew", "pew_data", "similarweb", "nyt"],
    },
    {
        "source": "The Washington Post",
        "rss": "https://feeds.washingtonpost.com/rss/national",
        "use_pct": 12,
        "awareness_pct": 81,
        "trust_pct": 25,
        "note": "National U.S. newspaper with Pew-measured readership and trust signals.",
        "refs": ["pew", "pew_data"],
    },
]


def _estimated_adults(use_pct: int | None) -> str:
    if not isinstance(use_pct, int):
        return "not measured"
    return f"~{(US_ADULT_POPULATION_MILLIONS * use_pct / 100):.1f}M"


def _pct(value: int | None) -> str:
    return f"{value}%" if isinstance(value, int) else "not in Pew tracker"


def _reference_links(keys: list[str]):
    links = []
    for index, key in enumerate(keys):
        ref = SOURCE_REFERENCES.get(key)
        if not ref:
            continue
        if index:
            links.append(" | ")
        links.append(html.A(ref["label"], href=ref["url"], target="_blank", rel="noreferrer"))
    return links


def _summary_cards():
    measured_rows = [row for row in SOURCE_ROWS if isinstance(row.get("use_pct"), int)]
    average_use = sum(int(row["use_pct"]) for row in measured_rows) / len(measured_rows)
    average_awareness = sum(int(row["awareness_pct"]) for row in measured_rows) / len(measured_rows)
    high_reach_count = sum(1 for row in measured_rows if int(row["use_pct"]) >= 30)
    return dbc.Row(
        [
            dbc.Col(dbc.Card(dbc.CardBody([html.Small("RSS Sources"), html.H4(str(len(SOURCE_ROWS)), className="mb-0")])), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Small("Pew-Measured Sources"), html.H4(str(len(measured_rows)), className="mb-0")])), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Small("Avg. Regular U.S. Use"), html.H4(f"{average_use:.0f}%", className="mb-0")])), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([html.Small("Avg. U.S. Awareness"), html.H4(f"{average_awareness:.0f}%", className="mb-0")])), md=3),
            dbc.Col(dbc.Alert(f"{high_reach_count} NewsLens sources are regular news sources for at least 30% of U.S. adults in Pew's 2025 tracker.", color="info"), md=12),
        ],
        className="g-3 mb-3",
    )


def _influence_table():
    header = html.Thead(
        html.Tr(
            [
                html.Th("Source"),
                html.Th("RSS feed"),
                html.Th("U.S. adults who regularly get news there"),
                html.Th("Est. U.S. adults"),
                html.Th("Heard of"),
                html.Th("Trust"),
                html.Th("Influence note"),
                html.Th("Evidence"),
            ]
        )
    )
    body = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(row["source"]),
                    html.Td(html.A("feed", href=row["rss"], target="_blank", rel="noreferrer")),
                    html.Td(_pct(row["use_pct"])),
                    html.Td(_estimated_adults(row["use_pct"])),
                    html.Td(_pct(row["awareness_pct"])),
                    html.Td(_pct(row["trust_pct"])),
                    html.Td(row["note"]),
                    html.Td(_reference_links(row["refs"])),
                ]
            )
            for row in SOURCE_ROWS
        ]
    )
    return dbc.Table([header, body], bordered=True, striped=True, hover=True, responsive=True, size="sm")


def _argument_panel():
    return dbc.Alert(
        [
            html.H5("How to frame the influence claim", className="alert-heading"),
            html.P(
                "This source set is defensibly influential because it combines high-reach U.S. broadcast brands, "
                "major public media, national newspapers, and international broadcasters with large global distribution. "
                "For the Pew-measured outlets, the regular-use figures are percentages of U.S. adults, not just web visitors.",
                className="mb-2",
            ),
            html.P(
                "The estimates are not unique unduplicated audience totals. A person can read several of these outlets, "
                "and global broadcaster reach is measured differently from U.S. survey use. Treat the table as evidence of "
                "reach and agenda-setting capacity, not as a quality or endorsement ranking.",
                className="mb-0",
            ),
        ],
        color="light",
        className="border",
    )


layout = dbc.Container(
    [
        dbc.Row([dbc.Col(html.H3("News Source Influence", className="mb-3"), width=12)]),
        build_news_intro(
            "Document which RSS sources NewsLens pulls from and why this source set is influential in the U.S. and globally."
        ),
        _summary_cards(),
        _argument_panel(),
        dbc.Row([dbc.Col(html.H4("Source Reach Evidence", className="mt-3 mb-3"), width=12)]),
        dbc.Row([dbc.Col(_influence_table(), width=12)]),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Alert(
                        [
                            html.Strong("Method: "),
                            "U.S. regular-use estimates use Pew's 2025 percentages and an approximate 268.3M U.S. adult base derived from Census population and under-18 share. International notes use each publisher or measurement source's own audience language.",
                        ],
                        color="secondary",
                        className="small",
                    ),
                    width=12,
                )
            ]
        ),
    ],
    fluid=True,
    className="py-4",
)
