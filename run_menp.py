#!/usr/bin/env python3
"""
MENP main runner: safety check + optional optimizer.

Example:
  python run_menp.py --B 0.05 --f 100 --duration 0.5 --duty 0.5
  python run_menp.py --optimize --grid-B 0.02 0.04 0.06 --grid-f 50 100 200 --grid-duty 0.25 0.5 1.0 --grid-duration 0.5
  python run_menp.py --standards-file DATA/VALIDATED_PARAMETER_RANGES/b_limits_icnirp.csv --optimize
"""
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import sys
import json
import time
import csv

# Local modules
from src.safety import (
    load_limit_table_csv,
    load_limit_table_json,
    safety_summary,
    file_sha256,
)
from src.safety import safety_score
from src.opt import OptimizerConfig, grid_search_protocols, evaluate_protocol
from src.provenance import collect_versions, AuditLogger, RUO_BANNER, get_pip_freeze
from src import provenance as prov
from src.compliance import generate_compliance_report
from src.predict import EfficacyPredictor, validate_on_grid, save_validation_plot
try:
    from src.fem import load_field_map, calibrate_scale, apply_scale
except Exception:
    load_field_map = None  # optional
from src.thermal import monte_carlo_deltaT
from src.calibration import load_calibration, extract_alpha_target
try:
    from src.mri import roi_field_stats
except Exception:
    roi_field_stats = None  # optional
try:
    from src.data_ingest import load_training_data
except Exception:
    load_training_data = None  # optional


def load_limit_table(path: Optional[str]) -> Optional[List[Tuple[float, float]]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Standards file not found: {p}", file=sys.stderr)
        return None
    if p.suffix.lower() == ".csv":
        return load_limit_table_csv(str(p))
    if p.suffix.lower() == ".json":
        return load_limit_table_json(str(p))
    print(f"[WARN] Unsupported standards file extension: {p.suffix}", file=sys.stderr)
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MENP safety check and optimizer")
    ap.add_argument("--B", type=float, default=0.05, help="Magnetic field [T]")
    ap.add_argument("--f", type=float, default=100.0, help="Frequency [Hz]")
    ap.add_argument("--duration", type=float, default=0.5, help="Exposure duration [s]")
    ap.add_argument("--duty", type=float, default=1.0, help="Duty cycle [0..1], thermal scaling")
    ap.add_argument("--standards-file", type=str, default=None, help="CSV/JSON standards table for B-limit")
    ap.add_argument("--optimize", action="store_true", help="Run safety-constrained optimizer")
    ap.add_argument("--grid-B", type=float, nargs="+", default=[0.02, 0.04, 0.06], help="Grid for B [T]")
    ap.add_argument("--grid-f", type=float, nargs="+", default=[50, 100, 200], help="Grid for f [Hz]")
    ap.add_argument("--grid-duty", type=float, nargs="+", default=[0.25, 0.5, 1.0], help="Grid for duty")
    ap.add_argument("--grid-duration", type=float, nargs="+", default=[0.5], help="Grid for duration [s]")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory to save results (JSON/CSV). If not set, no files are written.")
    ap.add_argument("--audit-log", type=str, default=None, help="Path to JSONL audit log file.")
    ap.add_argument("--calibration", type=str, default=None, help="Optional calibration JSON (see DATA/calibration.example.json)")
    # FEM / coil map
    ap.add_argument("--field-map", type=str, default=None, help="Path to field map (.npz/.csv/.nii.gz)")
    ap.add_argument("--calib-points", type=str, default=None, help="CSV with columns x,y,z,measured_B_T for calibration")
    # MRI ROI
    ap.add_argument("--roi-mask", type=str, default=None, help="NIfTI mask for ROI stats (requires nibabel)")
    ap.add_argument("--roi-threshold", type=float, default=0.5, help="Mask threshold")
    # Training data
    ap.add_argument("--training-data", type=str, nargs="+", default=None, help="Paths to CSV/JSON/NWB training datasets to summarize")
    # Real data override for simulator
    ap.add_argument("--real-data-csv", type=str, default=None, help="CSV of real firing rates to use for the 60/30/10 blend")
    # Device constraints
    ap.add_argument("--max-B", type=float, default=None, help="Max B [T] device can deliver")
    ap.add_argument("--max-f", type=float, default=None, help="Max frequency [Hz]")
    ap.add_argument("--max-duty", type=float, default=None, help="Max duty [0..1]")
    ap.add_argument("--min-duty", type=float, default=0.0, help="Min duty [0..1]")
    ap.add_argument("--min-duration", type=float, default=0.0, help="Min duration [s]")
    ap.add_argument("--max-duration", type=float, default=None, help="Max duration [s]")
    # Phase 3 options
    ap.add_argument("--use-predictor", action="store_true", help="Use ML predictor to accelerate grid search (if model available)")
    ap.add_argument("--model-path", type=str, default=None, help="Optional path to a .joblib model for predictor")
    ap.add_argument("--html-report", action="store_true", help="Emit a compact HTML summary report")
    ap.add_argument("--strict-clinical", action="store_true", help="Require standards file and calibration (alpha_target) and fail otherwise")
    # Uncertainty options
    ap.add_argument("--ci-bootstrap-best", type=int, default=0, help="If >0, bootstrap the best protocol's responders_frac with N resamples")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    versions = collect_versions({"pip_freeze": get_pip_freeze()})

    logger = AuditLogger(Path(args.audit_log)) if args.audit_log else None

    # Optional calibration load
    calibration = load_calibration(args.calibration)

    # Load optional standards table
    limit_table = load_limit_table(args.standards_file)

    # Strict clinical workflow prechecks (RUO guardrails)
    if args.strict_clinical:
        missing = []
        if limit_table is None:
            missing.append("--standards-file")
        cal_ok = False
        if isinstance(calibration, dict):
            try:
                cal_ok = extract_alpha_target(calibration) is not None
            except Exception:
                cal_ok = False
        if not cal_ok:
            missing.append("--calibration (alpha_target_V_per_Am)")
        if missing:
            print("[STRICT] Missing required inputs for clinical workflow:", ", ".join(missing), file=sys.stderr)
            return 2

    # Thermal Monte Carlo quicklook (used for safety gating via p95)
    # Scale Q by duty cycle for average heating
    Q_mean = 1e3 * max(0.0, min(1.0, args.duty))
    Q_sigma = 100.0 * max(0.0, min(1.0, args.duty))
    mc = monte_carlo_deltaT(Q_mean=Q_mean, Q_sigma=Q_sigma, duration_s=args.duration, n=2000)
    # Safety summary for the provided single protocol, override ΔT with MC p95
    standards_info = None
    if args.standards_file:
        try:
            standards_info = {
                "path": str(Path(args.standards_file).resolve()),
                "sha256": file_sha256(args.standards_file),
            }
        except Exception:
            standards_info = {"path": args.standards_file, "sha256": None}

    ss = safety_summary(
        B_T=args.B,
        freq_hz=args.f,
        duration_s=args.duration,
        limit_table=limit_table,
        deltaT_override_C=mc['p95'],
        duty=args.duty,
        standards_meta=standards_info,
    )
    status = "PASS" if ss["overall_safe"] else "FAIL"
    print(RUO_BANNER)
    print("=== Safety Check ===")
    if standards_info:
        print(f"Standards: {standards_info.get('path')} (SHA256 {standards_info.get('sha256')})")
    print(f"Status: {status}")
    # Use scientific notation for small limits to avoid printing as 0.0000 T
    try:
        fld_lim = float(ss['field_limit_T']) if ss.get('field_limit_T') is not None else float('nan')
    except Exception:
        fld_lim = float('nan')
    print(f"Field limit @ {args.f:.1f} Hz: {fld_lim:.3e} T; margin x{ss['field_margin']:.2f} -- {'OK' if ss['field_safe'] else 'Too high'}")
    print(f"DeltaT (MC p95): {ss['deltaT_C']:.3f} C (limit {ss['deltaT_limit_C']:.2f} C) -- {'OK' if ss['deltaT_safe'] else 'Too high'}")

    out_dir: Optional[Path] = None

    if args.out_dir:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_dir)
        # If given path exists and is a directory, create a timestamped subdir to avoid overwriting
        if out_dir.exists() and out_dir.is_dir():
            out_dir = out_dir / f"MENP_Run_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # audit
        if logger is None:
            logger = AuditLogger(out_dir / "audit.jsonl")
        else:
            logger.path = out_dir / "audit.jsonl"
        logger.log("start", {"versions": versions, "ruo": True})
        # Save safety summary
        with open(out_dir / "safety_summary.json", "w") as f:
            json.dump({
                "inputs": {"B_T": args.B, "f_Hz": args.f, "duration_s": args.duration, "duty": args.duty},
                "standards": standards_info,
                "provenance": versions,
                "ruo_banner": RUO_BANNER,
                "summary": ss,
                "thermal_mc": mc,
                "calibration": calibration,
            }, f, indent=2)
        print(f"Saved: {out_dir / 'safety_summary.json'}")
    if out_dir and logger:
        logger.log("safety_saved", {"path": str(out_dir / 'safety_summary.json')})
    # Thermal MC report
    print(f"Thermal MC: mean={mc['mean']:.3f}C, p95={mc['p95']:.3f}C, p99={mc['p99']:.3f}C")
    if out_dir:
        with open(out_dir / "thermal_mc.json", "w") as f:
            json.dump(mc, f, indent=2)
        if logger:
            logger.log("thermal_mc_saved", {"path": str(out_dir / 'thermal_mc.json')})

    # Compliance report
    compliance = generate_compliance_report(
        inputs={"B_T": args.B, "f_Hz": args.f, "duration_s": args.duration, "duty": args.duty},
        safety_summary=ss,
        thermal_mc=mc,
        calibration=calibration,
        standards_info=standards_info,
        versions=versions,
    )
    print(f"Compliance: clinical_pipeline_ready={compliance['clinical_pipeline_ready']}")
    if out_dir:
        with open(out_dir / "compliance_report.json", "w") as f:
            json.dump(compliance, f, indent=2)
        if logger:
            logger.log("compliance_saved", {"path": str(out_dir / 'compliance_report.json')})

    # FEM field map handling
    fm_info = None
    if args.field_map:
        if load_field_map is None:
            print("[WARN] FEM field-map support not available in this environment.")
        else:
            try:
                fm = load_field_map(args.field_map)
                fm_info = {
                    "path": str(Path(args.field_map).resolve()),
                    "nx": int(len(fm.x)), "ny": int(len(fm.y)), "nz": int(len(fm.z)),
                    "x_range_m": [float(fm.x.min()), float(fm.x.max())],
                    "y_range_m": [float(fm.y.min()), float(fm.y.max())],
                    "z_range_m": [float(fm.z.min()), float(fm.z.max())],
                }
                print(f"Loaded field map: {fm_info['nx']}x{fm_info['ny']}x{fm_info['nz']}")
                if out_dir:
                    with open(out_dir / "fieldmap_info.json", "w") as f:
                        json.dump(fm_info, f, indent=2)
                    logger.log("fieldmap_loaded", fm_info)
                # Calibration
                if args.calib_points:
                    import pandas as pd
                    cal = pd.read_csv(args.calib_points)
                    pts = cal[['x', 'y', 'z', 'measured_B_T']].to_dict(orient='records')
                    s, rmse = calibrate_scale(fm, pts)
                    print(f"Calibrated scale: {s:.4f}, RMSE={rmse:.6f} T")
                    if out_dir:
                        with open(out_dir / "fieldmap_calibration.json", "w") as f:
                            json.dump({"scale": s, "rmse_T": rmse, "calib_points": str(Path(args.calib_points).resolve())}, f, indent=2)
                        logger.log("fieldmap_calibrated", {"scale": s, "rmse_T": rmse})
            except Exception as e:
                print(f"[WARN] Failed to process field map: {e}")

    # MRI ROI stats
    if args.roi_mask and args.field_map and roi_field_stats is not None and fm_info is not None:
        try:
            fm = load_field_map(args.field_map)
            stats = roi_field_stats(fm, args.roi_mask, threshold=args.roi_threshold)
            print(f"ROI vox={stats['n_vox']} mean_B={stats['mean_B_T']:.4f} T, p95={stats['p95_B_T']:.4f} T, max={stats['max_B_T']:.4f} T")
            if out_dir:
                with open(out_dir / "roi_field_stats.json", "w") as f:
                    json.dump(stats, f, indent=2)
                logger.log("roi_stats_saved", stats)
        except Exception as e:
            print(f"[WARN] ROI stats failed: {e}")

    # Training data summary
    if args.training_data and load_training_data is not None:
        try:
            import pandas as pd
            frames = []
            for pth in args.training_data:
                df = load_training_data(pth)
                frames.append(df)
            all_df = pd.concat(frames, ignore_index=True)
            summary = {
                "n_rows": int(len(all_df)),
                "columns": list(all_df.columns),
                "missing_features": getattr(all_df, 'attrs', {}).get('missing_features', []),
            }
            print(f"Training data rows: {summary['n_rows']}")
            if out_dir:
                with open(out_dir / "training_data_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                logger.log("training_data_summary_saved", summary)
        except Exception as e:
            print(f"[WARN] Training data ingest failed: {e}")

    # If not optimizing, optionally emit a minimal HTML summary when requested
    if not args.optimize:
        if args.html_report and out_dir is not None:
            status_html = '<span class="ok">OK</span>' if ss['overall_safe'] else '<span class="warn">Check</span>'
            html = f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>MENP Summary</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}} .ok{{color:#0a0}} .warn{{color:#a60}}</style>
</head><body>
<h2>MENP Summary (RUO)</h2>
<p><b>Inputs</b>: B={float(args.B):.4f} T, f={float(args.f):.1f} Hz, duty={float(args.duty):.2f}, dur={float(args.duration):.3f} s</p>
<p><b>Safety</b>: Field margin x{float(ss['field_margin']):.2f}, ΔT={float(ss['deltaT_C']):.3f} C (MC p95={float(mc.get('p95', float('nan'))):.3f}) — {status_html}</p>
<p><b>Compliance</b>: Clinical pipeline ready = {str(compliance.get('clinical_pipeline_ready', False))}</p>
<h3>Calibration</h3>
<pre>{json.dumps(calibration, indent=2)}</pre>
<h3>Provenance</h3>
<pre>{json.dumps(versions, indent=2)}</pre>
</body></html>
"""
            html_path = out_dir / "MENP_Summary.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"Saved: {html_path}")
            if logger:
                logger.log("html_report_saved", {"path": str(html_path)})
        return 0

    print("\n=== Optimizer ===")
    # Only load a predictor when explicitly requested; otherwise keep disabled
    predictor = EfficacyPredictor(args.model_path, allow_defaults=True) if args.use_predictor else EfficacyPredictor(None, allow_defaults=False)
    cfg = OptimizerConfig(
        n_neurons=200,
        duration_s=args.grid_duration[0] if args.grid_duration else args.duration,
        r_m=2e-6,
        r_jitter_frac=0.2,
        power_density_W_per_m3=1e3,
        deltaT_limit_C=1.0,
        limit_table=limit_table,
        calibration=calibration,
        real_data_csv=args.real_data_csv,
        max_B_T=args.max_B,
        max_f_Hz=args.max_f,
        max_duty=args.max_duty,
        min_duty=args.min_duty,
        min_duration_s=args.min_duration,
        max_duration_s=args.max_duration,
    )
    # If predictor is available and requested, run a lightweight predictor-only pass to rank candidates,
    # then evaluate top-K with simulator+safety for correctness.
    obj_name = "responders_frac"
    if args.use_predictor and predictor.available():
        ranked = []
        for B in args.grid_B:
            for f in args.grid_f:
                for d in args.grid_duty:
                    for dur in args.grid_duration:
                        # Respect device constraints early
                        if cfg.max_B_T is not None and B > cfg.max_B_T:
                            continue
                        if cfg.max_f_Hz is not None and f > cfg.max_f_Hz:
                            continue
                        if cfg.max_duty is not None and d > cfg.max_duty:
                            continue
                        if cfg.min_duty is not None and d < cfg.min_duty:
                            continue
                        if cfg.min_duration_s is not None and dur < cfg.min_duration_s:
                            continue
                        if cfg.max_duration_s is not None and dur > cfg.max_duration_s:
                            continue
                        try:
                            pred = predictor.predict(B, f, d, dur, cfg.r_m)
                            ranked.append(((B,f,d,dur), pred.get("responders_frac", 0.0)))
                        except Exception:
                            pass
        # Take top K by predicted responders
        ranked.sort(key=lambda x: x[1], reverse=True)
        top = [r[0] for r in ranked[: min(50, len(ranked))]]  # cap to 50
        # Evaluate top with full pipeline grid (reuse API by narrowing lists)
        if top:
            B_list = sorted({b for (b,_,_,_) in top}); f_list = sorted({f for (_,f,_,_) in top})
            d_list = sorted({d for (_,_,d,_) in top}); dur_list = sorted({du for (_,_,_,du) in top})
            res = grid_search_protocols(B_list, f_list, d_list, dur_list, cfg=cfg, objective=obj_name)
        else:
            res = grid_search_protocols(args.grid_B, args.grid_f, args.grid_duty, args.grid_duration, cfg=cfg, objective=obj_name)
    else:
        res = grid_search_protocols(args.grid_B, args.grid_f, args.grid_duty, args.grid_duration, cfg=cfg, objective=obj_name)
    best = res.get("best")
    n = len(res.get("results", []))
    n_safe = sum(1 for e in res.get("results", []) if e["safety"]["overall_safe"]) if n else 0
    print(f"Grid evaluated: {n} combos; safe: {n_safe}")

    if best is None:
        print("No safe protocols found given current constraints.")
        return 0

    p = best["params"]; s = best["safety"]; eff = best["efficacy"]
    print("Best safe:")
    print(f"  B={p['B_T']:.4f} T, f={p['f_Hz']:.1f} Hz, duty={p['duty']:.2f}, dur={p['duration_s']:.3f} s")
    print(f"  responders_frac={eff.get('responders_frac', float('nan')):.3f}, mean_rate={eff.get('mean_rate', float('nan')):.3f}")
    print(f"  field_margin=x{s['field_margin']:.2f}, deltaT={s['deltaT_C']:.3f} C (OK={s['deltaT_safe']})")

    if out_dir:
        # Save full grid results to CSV
        csv_path = out_dir / "optimizer_results.csv"
        # Compute distribution-level bootstrap CI for responders_frac among safe results
        import numpy as _np
        safe_vals = [
            float(e["efficacy"].get("responders_frac", float('nan')))
            for e in res.get("results", []) if e.get("safety", {}).get("overall_safe", False)
        ]
        safe_vals = [v for v in safe_vals if _np.isfinite(v)]
        if safe_vals:
            _rng = _np.random.default_rng(123)
            _a = _np.array(safe_vals, dtype=float)
            _boots = []
            for _ in range(1000):
                _s = _rng.choice(_a, size=len(_a), replace=True)
                _boots.append(float(_np.nanmean(_s)))
            _ci_lo, _ci_hi = _np.nanpercentile(_boots, [2.5, 97.5]).tolist()
        else:
            _ci_lo, _ci_hi = float('nan'), float('nan')

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "B_T", "f_Hz", "duty", "duration_s",
                "responders_frac", "mean_rate",
                "field_safe", "dT_safe", "overall_safe",
                "field_margin", "deltaT_C", "thermal_headroom_C", "safety_score",
                "alpha_target_V_per_Am", "responder_threshold_Hz", "orientation_factor",
                "responders_frac_ci_low", "responders_frac_ci_high"
            ])
            # Calibration context (same for entire run)
            alpha_target = None
            try:
                alpha_target = extract_alpha_target(calibration)
            except Exception:
                alpha_target = None
            resp_thr = calibration.get("responder_threshold_Hz") if isinstance(calibration, dict) else None
            orient_fac = calibration.get("orientation_factor") if isinstance(calibration, dict) else None
            for e in res["results"]:
                pr = e["params"]; sf = e["safety"]; ef = e["efficacy"]
                sscore = safety_score(sf)
                # Thermal headroom = limit - deltaT
                _headroom = float(cfg.deltaT_limit_C) - float(sf.get("deltaT_C", float("nan")))
                writer.writerow([
                    pr["B_T"], pr["f_Hz"], pr["duty"], pr["duration_s"],
                    ef.get("responders_frac", float("nan")), ef.get("mean_rate", float("nan")),
                    sf["field_safe"], sf["deltaT_safe"], sf["overall_safe"],
                    sf["field_margin"], sf["deltaT_C"], _headroom, sscore,
                    alpha_target, resp_thr, orient_fac,
                    _ci_lo, _ci_hi
                ])
        print(f"Saved: {csv_path}")
        logger.log("optimizer_csv_saved", {"path": str(csv_path)})

        # Optional per-protocol bootstrap CI for the best protocol
        best_ci = None
        if int(args.ci_bootstrap_best) > 0 and best is not None:
            try:
                import copy, numpy as _np
                Bp = float(best["params"]["B_T"]) ; Fp = float(best["params"]["f_Hz"]) ; Dp = float(best["params"]["duty"]) ; Up = float(best["params"]["duration_s"])
                vals = []
                rng = _np.random.default_rng(2025)
                seeds = rng.integers(1, 2**31-1, size=int(args.ci_bootstrap_best))
                for sd in seeds:
                    cfg_b = copy.deepcopy(cfg)
                    cfg_b.rng_seed = int(sd)
                    # Use a lighter config for bootstrap to keep runtime reasonable
                    try:
                        cfg_b.n_neurons = int(max(40, min(80, getattr(cfg_b, 'n_neurons', 200))))
                        cfg_b.dt = max(getattr(cfg_b, 'dt', 1e-4), 2e-4)
                    except Exception:
                        pass
                    e = evaluate_protocol(Bp, Fp, Dp, Up, cfg_b)
                    vals.append(float(e["efficacy"].get("responders_frac", float('nan'))))
                arr = _np.array(vals, dtype=float)
                lo, hi = _np.nanpercentile(arr, [2.5, 97.5]).tolist()
                best_ci = {"responders_frac_ci_low": lo, "responders_frac_ci_high": hi, "n_bootstrap": int(args.ci_bootstrap_best)}
            except Exception as _ex:
                best_ci = {"error": str(_ex)}

        # Save best entry as JSON
        with open(out_dir / "best_protocol.json", "w") as f:
            payload = {"best": best, "device_constraints": {
                "max_B_T": args.max_B, "max_f_Hz": args.max_f, "max_duty": args.max_duty,
                "min_duty": args.min_duty, "min_duration_s": args.min_duration, "max_duration_s": args.max_duration,
            }, "standards": standards_info, "provenance": versions, "ruo_banner": RUO_BANNER, "calibration": calibration}
            if best_ci:
                payload["best_bootstrap_ci"] = best_ci
            json.dump(payload, f, indent=2)
        print(f"Saved: {out_dir / 'best_protocol.json'}")
        logger.log("best_saved", {"path": str(out_dir / 'best_protocol.json')})

        # Save Pareto front (subset of safe results)
        pareto = res.get("pareto", [])
        pareto_min = []
        for e in pareto:
            pareto_min.append({
                "params": e.get("params", {}),
                "efficacy": {
                    "responders_frac": e.get("efficacy", {}).get("responders_frac"),
                    "mean_rate": e.get("efficacy", {}).get("mean_rate"),
                },
                "safety": {
                    "field_margin": e.get("safety", {}).get("field_margin"),
                    "deltaT_C": e.get("safety", {}).get("deltaT_C"),
                    "overall_safe": e.get("safety", {}).get("overall_safe"),
                }
            })
        with open(out_dir / "pareto.json", "w") as f:
            json.dump({
                "pareto": pareto_min,
                "objective": obj_name,
                "note": "Pareto front among safe results: maximize responders_frac, minimize B_T"
            }, f, indent=2)
        print(f"Saved: {out_dir / 'pareto.json'}")
        logger.log("pareto_saved", {"count": len(pareto_min)})

        if args.html_report:
            # Minimal HTML summary
            best_score = safety_score(best.get('safety', {}))
            pareto_count = len(res.get('pareto', []))
            ci_line = ""
            if best_ci and isinstance(best_ci, dict) and "responders_frac_ci_low" in best_ci:
                ci_line = f"\n95% CI responders_frac: [{float(best_ci['responders_frac_ci_low']):.3f}, {float(best_ci['responders_frac_ci_high']):.3f}] (n={int(best_ci['n_bootstrap'])})"
            # Thermal headroom
            _head = float(ss.get('deltaT_limit_C', 1.0)) - float(ss.get('deltaT_C', float('nan')))
            html = f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>MENP Summary</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}} .ok{{color:#0a0}} .warn{{color:#a60}}</style>
</head><body>
<h2>MENP Summary (RUO)</h2>
<p><b>Safety</b>: Field margin x{float(ss['field_margin']):.2f}, ΔT={float(ss['deltaT_C']):.3f} C (MC p95={float(mc.get('p95', float('nan'))):.3f}) headroom={_head:.3f} C — {'<span class="ok">OK</span>' if ss['overall_safe'] else '<span class="warn">Check</span>'}</p>
<p><b>Best safety score</b>: {best_score:.3f} (Pareto count: {pareto_count})</p>
<p><b>Compliance</b>: Clinical pipeline ready = {str(compliance.get('clinical_pipeline_ready', False))}</p>
<h3>Best Safe Protocol</h3>
<pre>B={p['B_T']:.4f} T, f={p['f_Hz']:.1f} Hz, duty={p['duty']:.2f}, dur={p['duration_s']:.3f} s\nresponders_frac={best['efficacy'].get('responders_frac', float('nan')):.3f}  mean_rate={best['efficacy'].get('mean_rate', float('nan')):.3f}{ci_line}</pre>
<h3>Calibration</h3>
<pre>{json.dumps(calibration, indent=2)}</pre>
<h3>Provenance</h3>
<pre>{json.dumps(versions, indent=2)}</pre>
</body></html>
"""
            html_path = out_dir / "MENP_Summary.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"Saved: {html_path}")
            logger.log("html_report_saved", {"path": str(html_path)})

        # Provenance manifest with hashes
        try:
            manifest = {
                "inputs": {
                    "standards_file": standards_info,
                    "calibration_path": str(Path(args.calibration).resolve()) if args.calibration else None,
                    "real_data_csv": str(Path(args.real_data_csv).resolve()) if getattr(args, 'real_data_csv', None) else None,
                },
                "hashes": {
                    "standards_sha256": standards_info.get("sha256") if isinstance(standards_info, dict) else None,
                    "calibration_sha256": file_sha256(args.calibration) if args.calibration and Path(args.calibration).exists() else None,
                    "real_data_csv_sha256": file_sha256(args.real_data_csv) if getattr(args, 'real_data_csv', None) and Path(args.real_data_csv).exists() else None,
                },
                "provenance": versions,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with open(out_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            logger.log("manifest_saved", {"path": str(out_dir / 'manifest.json')})
        except Exception as ex:
            print(f"[WARN] Could not write manifest: {ex}")

        # Predictor validation vs simulator (if used and available)
        if args.use_predictor and predictor.available():
            val = validate_on_grid(predictor, args.grid_B, args.grid_f, args.grid_duty, args.grid_duration, r_m=cfg.r_m, calibration=calibration, n_neurons=80, dt=1e-4)
            with open(out_dir / "predictor_validation.json", "w") as f:
                json.dump(val, f, indent=2)
            plot_path = save_validation_plot(val, str(out_dir / "predictor_validation.png"))
            print(f"Predictor validation: N={val.get('count',0)} MAE={val.get('mae')} R2={val.get('r2')}  Plot={plot_path}")
            logger.log("predictor_validated", {"count": val.get('count',0), "mae": val.get('mae'), "r2": val.get('r2')})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
