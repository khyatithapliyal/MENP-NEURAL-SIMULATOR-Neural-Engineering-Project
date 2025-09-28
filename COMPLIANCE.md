# MENP Safety, Standards, and Compliance (RUO)

This project provides research-use-only (RUO) safety checks and reporting to aid clinical readiness. It does not constitute regulatory approval.

## Standards tables

- Place official B-field limit tables in `DATA/VALIDATED_PARAMETER_RANGES/`.
- The repo includes `limits_official.csv` (ICNIRP 2010 general public) with `limits_official.meta.json` documenting provenance.
- The optimizer/CLI accept standards via CSV/JSON; the notebook has a strict-mode section using the official table.

## Thermal safety

- Default thermal model uses ΔT = P·t / (ρ·c).
- Optionally enable a perfusion-corrected model and/or Monte Carlo uncertainty (p95) for conservative gating.

## Clinical strict mode

Strict mode requires:
- A standards table (e.g., ICNIRP 2010) to gate B-field by frequency.
- Calibration providing `alpha_target_V_per_Am`.

Under strict mode:
- The optimizer only considers candidates that satisfy field and thermal limits.
- Compliance report includes inputs, standards info (path+SHA256), safety summary, calibration presence, and provenance.

## Provenance and audit

- Runs can log to JSONL audit, include package versions, Python/Git commit.
- CLI: `--out-dir` enables saving safety, thermal, optimizer CSV, compliance.

## Disclaimer

This software and its reports are for research use only and do not replace formal regulatory evaluation.
