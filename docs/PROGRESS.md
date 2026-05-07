# NewsLens Agent Progress

Use this file as persistent state for daily incremental agent runs.

## Operating Loop
- Pick one current TODO or backlog item.
- Do one small, meaningful chunk of progress.
- Keep changes as safe and reversible as practical.
- Do not try to finish an entire large project in one run.
- Skip unclear or high-risk work and leave a note.
- End by updating this file with the next concrete step.

## Current Focus
- Keep improving NewsLens as a professional research dashboard.
- Prioritize useful visual analysis, backend-derived analytics, tests, and deployment reliability.
- Backlog source: `docs/BACKLOG.md`.

## Last Run
- Date: 2026-05-06
- Status: added visible cluster-filter messaging so the shrinking `Group` option set is explicit on `/news/group-latent-space`.
- Changes: kept `/news/group-latent-space` read-only, added a small `Cluster filter` status hint for both `All groups` and specific-cluster states, composed that hint under the existing page status alert, and extended focused helper tests for the new copy.
- Validation:
  - `.venv/bin/python -m py_compile src/pages/news_group_latent_space.py tests/test_news_group_latent_space.py`
  - `.venv/bin/python -m unittest tests.test_news_group_latent_space -v`

## Next Concrete Step
- Mirror the active cluster filter in the `Top Groups` table on `/news/group-latent-space` so the table matches the `Group` dropdown scope.
- Keep the PCA/MDS charts unchanged for that step to limit behavior drift.
- Add one focused helper test covering the filtered table rows instead of broad callback coverage.

## Notes For Future Agents
- Prefer additive `derived.*` backend fields for reusable analysis.
- Prefer page-specific frontend improvements with smoke coverage.
- If deployment changes are made, verify GitHub Actions and live smoke checks when practical.
