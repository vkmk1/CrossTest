# Thesis handoff package: learned market state for forecasting

Owner: Vikram Kakaria (Princeton ORFE senior thesis, advisor Prof. Robert Almgren)
Prepared: October 9, 2026

This folder is the complete brief for implementing the thesis codebase with Claude Code.
Start Claude Code in a fresh repo, copy these files into the repo root, and say:

> Read CLAUDE.md, SPEC.md, ARCHITECTURE.md, DATA.md and EXPERIMENTS.md. Then implement Phase 0 from EXPERIMENTS.md on daily CRSP data.

Files:

| File | What it is |
|---|---|
| `CLAUDE.md` | Standing instructions for Claude Code: rules, repo layout, conventions. Goes in repo root. |
| `SPEC.md` | The research design: question, hypotheses, the two-stage model, ablations, success criteria. |
| `ARCHITECTURE.md` | Tensor-level description of both networks, shapes, losses, defaults. |
| `DATA.md` | WRDS access, CRSP daily and TAQ intraday pipelines, bar and feature definitions, splits, leakage rules. |
| `EXPERIMENTS.md` | Phased experiment registry with go/no-go gates, metrics, and reporting format. |
| `configs/` | Starter YAML configs for the daily pilot. |

Order of work: Phase 0 (daily pilot, two weeks) decides whether the project continues as designed.
Do not build the intraday pipeline until Phase 0 passes its gate.
