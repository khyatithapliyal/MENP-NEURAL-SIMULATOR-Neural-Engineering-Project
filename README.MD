title: MENP Neural Stimulation Simulator (RUO)

authors: The MENP Simulation Team

date: 2025-09-28

abstract:
  We present a reproducible, standards-gated simulation workflow for magneto‑electric nanoparticle (MENP) neurostimulation. The system links magnetic drive to electric fields, a safety gate (exposure limits + bioheat), and a neural response surrogate to explore safe/efficacious protocols. It integrates real neural firing data (Allen) with a 60/30/10 blend (real/synthetic/generated), produces audit‑ready artifacts with provenance and hashes, and runs end‑to‑end from a single notebook.

keywords: magnetoelectric nanoparticles; external stimulation; activating function; safety limits; thermal headroom; reproducibility; provenance

---

# 1. Introduction
MENPs transduce magnetic excitation into electric polarization, offering a potential mechanism for minimally invasive neuromodulation. This project delivers a research‑use‑only (RUO) simulator that couples a unit‑consistent physics model with safety constraints and a neural surrogate to identify safe candidate protocols. Emphasis is on transparency (hash‑pinned standards and calibration), reproducibility (manifest and environment capture), and practicality (single‑entry notebook and exportable bundle).

Motivation and contributions
- Single‑entry notebook orchestrates safety → thermal MC → optimization → export
- Real‑data integration (Allen) with 60/30/10 blend by default
- Safety gate combines exposure limits (ICNIRP‑style) and thermal headroom
- Audit‑ready outputs: CSV/JSON/HTML with a manifest (SHA256) and environment versions

Scope: RUO. This is not a medical device. No clinical claims are made.

# 2. Methods (high level)
Physics and neural proxy
- Electric field scales with magnetic drive via an effective coupling and decays dipole‑like; a leaky integrator surrogate approximates spiking response from field gradients (activating function heuristic).

Safety model
- Exposure limit margin: interpolated B‑field limit vs frequency (official table required from `DATA/VALIDATED_PARAMETER_RANGES/limits_official.csv`).
- Thermal constraint: short‑exposure heating with Monte Carlo uncertainty; we report p95 and thermal headroom (limit − ΔT).

Neural evidence (60/30/10)
- Real firing patterns: Allen spike trains/summary statistics are bootstrapped/segmented to 10k+ rates.
- Synthetic and generated components fill 40% for diversity; all sources are tagged.

Optimization
- Grid search over B, f, duty, duration with safety gating and efficacy objective (responders fraction). Pareto and “best” safe protocol exported.

# 3. How to run (Windows PowerShell)
Prereqs: Python 3.9+ recommended.

```powershell
# optional: create and activate a virtual environment
py -3.11 -m venv .venv
..\.venv\Scripts\Activate.ps1

# install dependencies
python -m pip install -r requirements.txt
```

Single-entry run
1) Open `01_physics_engine_7_9_10_39pm.ipynb` and Run All.
2) The notebook will:
   - Audit inputs and compute SHA256 for standards/calibration/real data
   - Generate `DATA/allen_bootstrap_firing_rates.csv` if needed
   - Run safety preview (with thermal MC p95)
   - Run safety‑constrained grid optimizer
   - Export a submission bundle under `MENP_Runs/MENP_Run_<timestamp>_notebook_full`

# 4. Outputs and how to read them
Bundle directory: `MENP_Runs/MENP_Run_<timestamp>_notebook_full`
- optimizer_results.csv — Per‑protocol table: responders_frac, mean_rate, field/thermal flags, field_margin, deltaT_C, thermal_headroom_C, safety_score, responders_frac_ci_low/high (distribution‑level bootstrap among safe rows).
- best_protocol.json — The best safe protocol with params, safety, and efficacy.
- pareto.json — Minimal Pareto set (params/efficacy/safety) for trade‑off inspection.
- safety_summary.json — Safety preview (single protocol) with standards metadata and thermal MC stats.
- compliance_report.json — RUO readiness boolean derived from safety + calibration presence.
- MENP_Summary.html — Human‑readable snapshot for quick review.
- manifest.json — Paths and SHA256 for standards/calibration/real CSV; environment versions; UTC timestamp.

Interpretation tips
- Field margin x>1 and ΔT well below the limit → safe region; thermal_headroom_C quantifies buffer.
- responders_frac: higher is better; consider CI to gauge stability across safe results.
- Microtesla fields are expected under strict standards; milli‑tesla may violate limits.

# 5. Results (typical strict run)
On a microtesla grid (B≈1e−5–2e−4 T, f=50–100 Hz, duty≈0.5): all evaluated points are safe; the best safe protocol typically sits at the lowest B with acceptable efficacy. The bundle contains 8 rows, 8 safe, with p95 thermal rise ≪ 0.1 °C and large field margins.

# 6. Limitations
- Effective coupling is lumped and should be calibrated against device‑specific measurements.
- Neural surrogate omits morphology/network effects.
- Bioheat model is simplified for short exposures.
- Safety conclusions depend on the provided standards table and calibration.

# 7. Future directions
- FEM 5‑layer head fields and 3D perfused thermal with benchmarks (±10%).
- External validation on held‑out datasets (coverage/ECE/R²) with reports.
- Expanded compliance gates and generated reports (IEC 60601 family, EMC/SAR).
- Risk and lifecycle documentation for regulated contexts (ISO 14971, IEC 62304).

# 8. Ethical and usage note
Research‑use only (RUO). Not for medical decision‑making.

# 9. References (selected)
- Rattay, F. IEEE TBME, 1986 — activating function.
- Pennes, H. J Appl Physiol, 1948 — bioheat equation.
- ICNIRP (2010); IEEE C95.1‑2019 — exposure limits.
- Nan et al., JAP 2008; Ma et al., Adv Mater 2011 — magnetoelectric composites.

---

Contact: Add contact info here.  |  License: Add your license here.
