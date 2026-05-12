const US_ADULT_POPULATION_MILLIONS = 268.3;

const REFERENCES = {
  pew: {
    label: "Pew Research Center News Media Tracker, 2025",
    url: "https://www.pewresearch.org/journalism/feature/news-media-tracker/"
  },
  pewData: {
    label: "Pew tracker dataset",
    url: "https://docs.google.com/spreadsheets/d/10UKS5LiJi9qmPshiBAH0Hoelgvto1qYCoW2qNTo2hVg/edit?usp=sharing"
  },
  census: {
    label: "U.S. Census QuickFacts",
    url: "https://www.census.gov/quickfacts/fact/table/US/PST045225"
  },
  similarweb: {
    label: "Similarweb U.S. News & Media rankings, April 2026",
    url: "https://www.similarweb.com/top-websites/united-states/news-and-media/"
  },
  abcAu: {
    label: "ABC Australia international audience, 2025",
    url: "https://www.abc.net.au/about/media-centre/press-releases/abc-grows-international-audience/105372698"
  },
  alJazeera: {
    label: "Al Jazeera English network profile",
    url: "https://network.aljazeera.net/al-jazeera-english"
  },
  bbc: {
    label: "Ofcom BBC Annual Report 2024-25",
    url: "https://www.ofcom.org.uk/siteassets/resources/documents/tv-radio-and-on-demand/bbc/bbc-annual-report/2025/ofcoms-annual-report-on-the-bbc-2024-25.pdf"
  },
  dw: {
    label: "DW usage data 2025",
    url: "https://corporate.dw.com/en/usage-data-2025-dw-reports-worldwide-reach-of-337-million-weekly-users/a-73359910"
  },
  france24: {
    label: "France 24 audience profile",
    url: "https://www.francemediasmonde.com/en/france-24/"
  },
  ft: {
    label: "Financial Times 2024 audience reporting",
    url: "https://pressgazette.co.uk/media_business/financial-times-reports-global-revenue-boost-to-540m-for-2024/"
  },
  nbc: {
    label: "NBC News audience release, 2025",
    url: "https://nbcuniversalnewsgroup.com/nbcnews/2025/12/23/nbc-news-to-finish-2025-as-the-1-news-organization-in-the-u-s-reaching-126-million-americans-each-month-across-television-digital-streaming/"
  },
  nyt: {
    label: "New York Times Company Q2 2025 results",
    url: "https://www.sec.gov/Archives/edgar/data/71691/000007169125000117/pressrelease06302025.htm"
  },
  sky: {
    label: "Sky News Distribution",
    url: "https://skynewsdistribution.sky/"
  }
};

const SOURCE_ROWS = [
  {
    source: "ABC News",
    rss: "https://abcnews.go.com/abcnews/topstories",
    usePct: 36,
    awarenessPct: 92,
    trustPct: 44,
    note: "Major U.S. broadcast network news brand.",
    refs: ["pew", "pewData"]
  },
  {
    source: "ABC News (Australia)",
    rss: "https://www.abc.net.au/news/feed/51120/rss.xml",
    note: "ABC reported more than 11M people outside Australia engaging with ABC content in early 2025.",
    refs: ["abcAu"]
  },
  {
    source: "Al Jazeera",
    rss: "https://www.aljazeera.com/xml/rss/all.xml",
    note: "Al Jazeera English reports reach into 350M+ households across 150+ countries.",
    refs: ["alJazeera"]
  },
  {
    source: "BBC News",
    rss: "http://feeds.bbci.co.uk/news/rss.xml",
    usePct: 21,
    awarenessPct: 79,
    trustPct: 35,
    note: "Pew-measured U.S. use plus BBC World Service global audience context.",
    refs: ["pew", "pewData", "bbc"]
  },
  {
    source: "CBS News",
    rss: "https://www.cbsnews.com/latest/rss/main",
    usePct: 30,
    awarenessPct: 90,
    trustPct: 39,
    note: "Major U.S. broadcast network news brand.",
    refs: ["pew", "pewData"]
  },
  {
    source: "DW",
    rss: "https://rss.dw.com/rdf/rss-en-all",
    note: "DW reported 337M weekly users worldwide in 2025.",
    refs: ["dw"]
  },
  {
    source: "Financial Times",
    rss: "https://www.ft.com/?format=rss",
    note: "FT reporting indicates roughly 1.5M paying readers and a 2.8M global paying audience in 2024.",
    refs: ["ft"]
  },
  {
    source: "Fox News",
    rss: "http://feeds.foxnews.com/foxnews/latest",
    usePct: 38,
    awarenessPct: 93,
    trustPct: 37,
    note: "Pew's highest-use source among the NewsLens RSS outlets measured in its 2025 tracker.",
    refs: ["pew", "pewData", "similarweb"]
  },
  {
    source: "France 24",
    rss: "https://www.france24.com/en/rss",
    note: "France 24 reports 137M weekly audience in 2024 and 3.3M weekly viewers in North America.",
    refs: ["france24"]
  },
  {
    source: "NBC News",
    rss: "http://feeds.nbcnews.com/nbcnews/public/news",
    usePct: 35,
    awarenessPct: 92,
    trustPct: 42,
    note: "NBC News reported reaching 126M Americans monthly across TV, digital, and streaming in 2025.",
    refs: ["pew", "pewData", "nbc"]
  },
  {
    source: "NPR",
    rss: "https://feeds.npr.org/1001/rss.xml",
    usePct: 20,
    awarenessPct: 58,
    trustPct: 29,
    note: "National public radio network and digital news brand.",
    refs: ["pew", "pewData"]
  },
  {
    source: "PBS NewsHour",
    rss: "https://www.pbs.org/newshour/feeds/rss/headlines",
    usePct: 21,
    awarenessPct: 85,
    trustPct: 41,
    note: "Mapped to Pew's PBS measure because NewsHour is PBS's flagship public affairs news program.",
    refs: ["pew", "pewData"]
  },
  {
    source: "SkyNews",
    rss: "http://feeds.skynews.com/feeds/rss/home.xml",
    note: "Sky News says its distribution reaches audiences in 100+ countries across broadcast and digital platforms.",
    refs: ["sky"]
  },
  {
    source: "The Guardian",
    rss: "https://www.theguardian.com/world/rss",
    usePct: 8,
    awarenessPct: 57,
    trustPct: 14,
    note: "Pew-measured U.S. use for a global English-language publisher.",
    refs: ["pew", "pewData"]
  },
  {
    source: "The New York Times",
    rss: "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    usePct: 19,
    awarenessPct: 84,
    trustPct: 32,
    note: "NYTimes.com ranked #2 among U.S. News & Media sites in Similarweb's April 2026 ranking.",
    refs: ["pew", "pewData", "similarweb", "nyt"]
  },
  {
    source: "The Washington Post",
    rss: "https://feeds.washingtonpost.com/rss/national",
    usePct: 12,
    awarenessPct: 81,
    trustPct: 25,
    note: "National U.S. newspaper with Pew-measured readership and trust signals.",
    refs: ["pew", "pewData"]
  }
];

function formatPct(value) {
  return Number.isFinite(value) ? `${value}%` : "not in Pew tracker";
}

function estimatedAdults(value) {
  if (!Number.isFinite(value)) {
    return "not measured";
  }
  return `~${((US_ADULT_POPULATION_MILLIONS * value) / 100).toFixed(1)}M`;
}

function ReferenceLinks({ keys }) {
  return keys
    .map((key) => REFERENCES[key])
    .filter(Boolean)
    .map((ref, index) => (
      <span key={ref.url}>
        {index > 0 ? " | " : ""}
        <a href={ref.url} target="_blank" rel="noreferrer">
          {ref.label}
        </a>
      </span>
    ));
}

export async function render(searchParams) {
  const measuredRows = SOURCE_ROWS.filter((row) => Number.isFinite(row.usePct));
  const averageUse = measuredRows.reduce((total, row) => total + row.usePct, 0) / measuredRows.length;
  const averageAwareness = measuredRows.reduce((total, row) => total + row.awarenessPct, 0) / measuredRows.length;
  const highReachCount = measuredRows.filter((row) => row.usePct >= 30).length;

  return (
    <>
      <div className="stats-grid">
        <div className="stat-card">
          <small>RSS Sources</small>
          <strong>{SOURCE_ROWS.length}</strong>
        </div>
        <div className="stat-card">
          <small>Pew-Measured Sources</small>
          <strong>{measuredRows.length}</strong>
        </div>
        <div className="stat-card">
          <small>Avg. Regular U.S. Use</small>
          <strong>{averageUse.toFixed(0)}%</strong>
        </div>
        <div className="stat-card">
          <small>Avg. U.S. Awareness</small>
          <strong>{averageAwareness.toFixed(0)}%</strong>
        </div>
      </div>

      <div className="panel">
        <p className="section-kicker">Influence Claim</p>
        <h3>Mass reach, not a quality ranking</h3>
        <p className="muted">
          This source set is defensibly influential because it combines high-reach U.S. broadcast brands, major public
          media, national newspapers, and international broadcasters with large global distribution. For the Pew-measured
          outlets, the regular-use figures are percentages of U.S. adults, not just web visitors.
        </p>
        <p className="muted">
          {highReachCount} NewsLens sources are regular news sources for at least 30% of U.S. adults in Pew&apos;s 2025
          tracker. The estimates are not unique unduplicated audience totals; a person can read several of these outlets.
        </p>
      </div>

      <div className="panel">
        <div className="panel-heading">
          <div>
            <p className="section-kicker">Source Reach Evidence</p>
            <h3>RSS sources, U.S. usage, and external audience signals</h3>
          </div>
        </div>
        <table className="news-table">
          <thead>
            <tr>
              <th>Source</th>
              <th>RSS feed</th>
              <th>U.S. adults who regularly get news there</th>
              <th>Est. U.S. adults</th>
              <th>Heard of</th>
              <th>Trust</th>
              <th>Influence note</th>
              <th>Evidence</th>
            </tr>
          </thead>
          <tbody>
            {SOURCE_ROWS.map((row) => (
              <tr key={row.source}>
                <td>{row.source}</td>
                <td>
                  <a href={row.rss} target="_blank" rel="noreferrer">
                    feed
                  </a>
                </td>
                <td>{formatPct(row.usePct)}</td>
                <td>{estimatedAdults(row.usePct)}</td>
                <td>{formatPct(row.awarenessPct)}</td>
                <td>{formatPct(row.trustPct)}</td>
                <td>{row.note}</td>
                <td>
                  <ReferenceLinks keys={row.refs} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <p className="section-kicker">Method</p>
        <p className="muted">
          U.S. regular-use estimates use Pew&apos;s 2025 percentages and an approximate 268.3M U.S. adult base derived
          from Census population and under-18 share. International notes use each publisher or measurement source&apos;s
          own audience language, so they should be read as context rather than directly comparable audience totals.
        </p>
        <p className="muted">
          Population reference:{" "}
          <a href={REFERENCES.census.url} target="_blank" rel="noreferrer">
            {REFERENCES.census.label}
          </a>
          .
        </p>
      </div>
    </>
  );
}
