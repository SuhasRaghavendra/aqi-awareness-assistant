# Step 013 — Byte-stable champion bundle

- **Date/time:** 2026-11-10T23:00:00+05:30
- **Author:** <student-A>      **Reviewer:** <student-B>
- **Phase:** 2           **Branch / PR:** `main` / #<n>
- **Git commit(s):** `a58763273a3d86e664994002809500b7d9389879` (fix applied on top)      **Data version:** `train.parquet` md5 `49bd4c88748327afa12057c2fe71b0e7`
- **MLflow run IDs:** `ecef19343ca84f17b42e4deeaa0c1559` (model version 7)

## 1. Objective

Make the committed champion bundle byte-identical in every checkout. The bundle
is hash-pinned (`MANIFEST.sha256` covers `MANIFEST.json`, which in turn records
the SHA-256 of `reference_stats.json` and `history_seed.parquet`), so CI and the
Docker image must be able to verify it from a clean clone on any operating
system.

## 2. What changed

| Type | Path | Change summary |
|---|---|---|
| code | `src/serving/bundle.py` | `save()` writes `MANIFEST.json`/`MANIFEST.sha256`/`reference_stats.json` with `newline="\n"`; `load()`/`save()` accept `str` or `Path` |
| code | `src/models/package_model.py` | `reference_stats.json` written with `newline="\n"` so its digest is platform-independent |
| config | `.gitattributes` | `artifacts/champion/** -text` — Git never rewrites line endings for the hash-pinned bundle |
| tests | `tests/test_bundle.py` | New checks: no CRLF in text artifacts, `.gitattributes` rule present, `AqiBundle.load(str)` supported |
| pipeline | `dvc.lock` | `register` stage re-run after the fix; new bundle digests recorded |
| docs | `docs/decisions.md` | ADR D-025 |

## 3. Commands run (verbatim)

```bash
dvc repro register
python -m src.models.register_model --promote-only
python -m src.models.register_model --verify
python -m pytest -q
ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/
```

## 4. Results / evidence

Fresh-clone probe with `core.autocrlf=true` in a scratch copy of
`.gitattributes` + `artifacts/champion/`:

| File | Before fix (clone) | After fix (clone) |
|---|---|---|
| `MANIFEST.json` | 3,444 B, 123 CRLF → sha `640d6933…` ≠ recorded `43786979…` | 3,321 B, 0 CRLF → sha `f765a61d…` = recorded |
| `MANIFEST.sha256` | 66 B (CRLF) | 65 B (LF) |
| `reference_stats.json` | 22,680 B (CRLF; LF on Linux CI) | 21,831 B (LF) |

- Clone bytes now match the repo bytes exactly; `reference_stats_sha` and
  `history_seed_sha` verify against the manifest inside the clone, and
  `AqiBundle.load(<clone>)` returns a 103-feature bundle.
- Bundle total: **811,362 bytes** (5 files).
- Current MANIFEST SHA-256:
  `f765a61da001da89b646d5bf7989827494ce99a4580e8bfd0b3f79898532fac1`.
- Registry: `Indian-AQI-Predictor` version **7** with `challenger=7` and
  `champion=7`; parity on the 423-row fixture is **max diff 0.000e+00**.
- Tests: **75 passed** (3 new bundle-contract tests); `ruff` and `black` clean;
  `dvc status` reports the pipeline up to date.

## 5. Decisions and rationale

- **D-025:** write bundle text artifacts with LF on every platform and mark the
  bundle directory `-text` in `.gitattributes`. The alternative — trusting the
  working tree's native line endings — makes the integrity check depend on the
  checkout OS and on `core.autocrlf`, which is exactly what broke here.
- The manifest is regenerated rather than patched, so every digest in it is
  computed from the bytes that are actually committed.
- Promotion stayed a separate, explicit step (`--promote-only`); the `register`
  DVC stage only ever moves `challenger`.

## 6. Problems encountered and fixes

- `MANIFEST.json` was written with LF on disk but Git's `core.autocrlf=true`
  converted it to CRLF on clone (3,321 → 3,444 bytes), so the recorded digest
  no longer matched and the registry gate would fail in CI/Docker.
- `reference_stats.json` was written in text mode, which produced CRLF on
  Windows and LF on Linux — the recorded `reference_stats_sha` could therefore
  only be valid on one platform.
- The regeneration step ran twice: version 6 was built for the byte fix, then
  `black` reformatted the same two source files, making the `register` stage
  stale. Version 7 is the final promoted artifact.
- `AqiBundle.load("path")` failed with a bare `TypeError` because the path was
  used without coercion; it now accepts `str | Path`.

## 7. Verification

```bash
dvc status
python -m src.models.register_model --verify
python -m pytest -q
ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/
python scripts/check_logbook.py
```

Expected: DVC up to date, registry parity max diff ≤ 1e-6, all tests pass, lint
and format clean, logbook check passes.

## 8. Next step

Phase 3: build the FastAPI inference service on top of the byte-stable
`artifacts/champion/` bundle.
