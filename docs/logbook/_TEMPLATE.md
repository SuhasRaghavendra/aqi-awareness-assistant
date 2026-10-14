# Step <NNN> — <short title>
- **Date/time:** <ISO timestamp, IST>
- **Author:** <student>      **Reviewer:** <other student>
- **Phase:** <1–9>           **Branch / PR:** <branch> / #<n>
- **Git commit(s):** <hash(es)>      **Data version:** <git tag + dvc md5 of tracked inputs>
- **MLflow run IDs (if any):** <ids>

## 1. Objective
What this step is meant to achieve and why (1–3 sentences).

## 2. What changed
| Type | Path / object | Change summary |
|---|---|---|
| code | src/... | added / modified: ... |
| params | params.yaml | key: old → new |
| data | data/... | rows: A → B; columns added/removed |
| docs/config | ... | ... |

(Paste the `git diff --stat` and, for data, the before/after stats table that the pipeline stage emitted.)

## 3. Commands run (verbatim)
```bash
<exact commands>
```

## 4. Results / evidence
Key numbers, small tables, figure paths, screenshot paths. Quote real output, never invented numbers.

## 5. Decisions and rationale
Each decision, alternatives considered, why chosen (also copy to docs/decisions.md if it is a design decision).

## 6. Problems encountered and fixes
Errors, surprises, data issues, and how they were resolved.

## 7. Verification
How to reproduce/check this step (commands + expected output). Tests added/passing.

## 8. Next step
One line.
