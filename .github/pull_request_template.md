## Pull Request Checklist

**Required before merging:**

- [ ] **Logbook entry link:** `docs/logbook/<step>.md` — paste the link here:
- [ ] **What changed:** (brief summary of code/data/config changes)
- [ ] **Tests run:** `pytest -q` passing? ✅ / ❌
- [ ] **Lint/format:** `ruff check` and `black --check` passing? ✅ / ❌
- [ ] **Data version:** Git tag + DVC md5 (if data changed):
- [ ] **MLflow run IDs** (if models were trained):
- [ ] **DVC status:** `dvc status` output (paste here if data changed):
- [ ] **Step-Close Protocol:** `scripts/log_step.py` was run?

**Conventional Commit format:** `feat(step-NNN): <description>`

---
*Reviewed by:*
