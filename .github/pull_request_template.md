## Summary of Changes
<!-- One paragraph describing what this PR does and why. -->

## Logbook Entry
<!-- Link to the logbook entry for this PR's step(s). -->
- [ ] `docs/logbook/NNN-<slug>.md` — [link here](../docs/logbook/)

## What changed
| Type | Path | Change summary |
|---|---|---|
| code | src/... | |
| config | params.yaml / dvc.yaml | |
| data | data/... | |
| docs | docs/... | |

## Tests run
- [ ] `pytest -q` passes
- [ ] `ruff check` passes
- [ ] `black --check` passes
- [ ] `python scripts/check_logbook.py` passes

## Data version
- Git tag: <!-- e.g. data-v1 -->
- DVC md5 of changed tracked files: <!-- paste from dvc diff -->

## MLflow run IDs (if applicable)
<!-- Paste run IDs or N/A -->

## Screenshots / evidence
<!-- Attach or link screenshots for MLflow UI, DVC diff, etc. -->

## Reviewer checklist
- [ ] Logbook entry narrative is complete (no TODO sections)
- [ ] No hard-coded paths or magic numbers
- [ ] No data committed (only .dvc files)
- [ ] No secrets committed (.env not included)
- [ ] Commit messages follow Conventional Commits with step number
