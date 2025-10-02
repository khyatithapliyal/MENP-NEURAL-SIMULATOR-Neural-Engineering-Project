from __future__ import annotations

import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False
    plt = None

from pathlib import Path

# Core units and simulator
from src.units import T, Hz, m
from src.sim import simulate_population, LIFParams, AdaptiveLIFParams
try:
    from src.sim import _CLASS_MODEL_CONFIGS  # type: ignore
except Exception:
    _CLASS_MODEL_CONFIGS = {}
from src.physics import MaterialProps

# Optional safety (skips if not available)
try:
    from src.safety import load_limit_table_csv, safety_summary, file_sha256
except Exception:
    load_limit_table_csv = safety_summary = file_sha256 = None

# Demo parameters
B_T_SI = 1e-4
FREQ_HZ_SI = 50.0
R_M_SI = 2e-6
DURATION_S = 0.5
DT_S = 1e-4

# Neuron classes we’ll show
CLASSES = [
    ("Pyramidal", "pyramidal"),
    ("PV interneuron", "pv"),
    ("SST interneuron", "sst"),
]


def _class_specific_args(neuron_class: str) -> dict:
    cfg = _CLASS_MODEL_CONFIGS.get(neuron_class or "", {}) if isinstance(_CLASS_MODEL_CONFIGS, dict) else {}
    lif_kwargs = cfg.get("lif_params", {}) if isinstance(cfg, dict) else {}
    adaptive_kwargs = cfg.get("adaptive_params", {}) if isinstance(cfg, dict) else {}
    model = cfg.get("model", "lif") if isinstance(cfg, dict) else "lif"
    args = {"model": model}
    if lif_kwargs:
        try:
            args["lif"] = LIFParams(**lif_kwargs)
        except Exception:
            pass
    if adaptive_kwargs:
        try:
            args["adaptive"] = AdaptiveLIFParams(**adaptive_kwargs)
        except Exception:
            pass
    return args


def safety_quicklook() -> None:
    if load_limit_table_csv is None or safety_summary is None:
        print("[Safety] safety module not available; skipping")
        return

    limits_path = Path("DATA") / "VALIDATED_PARAMETER_RANGES" / "limits_official.csv"
    table = load_limit_table_csv(str(limits_path)) if limits_path.exists() else None

    ss = safety_summary(
        B_T=B_T_SI,
        freq_hz=FREQ_HZ_SI,
        duration_s=DURATION_S,
        duty=1.0,
        limit_table=table,
        deltaT_override_C=0.2,
        standards_meta={
            "path": str(limits_path.resolve()),
            "sha256": file_sha256(str(limits_path))
        } if limits_path.exists() else None,
    )
    print(f"[Safety] field_margin=x{ss.get('field_margin', float('nan')):.2f} ΔT={ss.get('deltaT_C', float('nan')):.3f} C overall={ss.get('overall_safe', False)}")


def population_quicklook() -> None:
    mp = MaterialProps()
    sim = simulate_population(
        B=B_T_SI * T,
        f=FREQ_HZ_SI * Hz,
        r=R_M_SI * m,
        n_neurons=100,
        duration_s=DURATION_S,
        dt=DT_S,
        theta_mode="isotropic",
        r_jitter_frac=0.1,
        mp=mp,
        rng=np.random.default_rng(123),
        calibration=None,
    )
    s = sim["summary"]
    print(f"[Population] n=100 responders_frac={s['responders_frac']:.3f} mean_rate={s['mean_rate']:.3f} Hz")


def _one_neuron_result(B_T: float, f_Hz: float, r_m: float, neuron_class: str, mp: MaterialProps):
    out = simulate_population(
        B=B_T * T,
        f=f_Hz * Hz,
        r=r_m * m,
        n_neurons=1,
        duration_s=DURATION_S,
        dt=DT_S,
        neuron_class=neuron_class,
        mp=mp,
        rng=np.random.default_rng(0),
        calibration=None,
        **_class_specific_args(neuron_class),
    )
    return int(out["spike_counts"][0]), float(out["firing_rates"][0])


def _find_threshold_params(neuron_key: str, mp: MaterialProps):
    """Return (f_Hz, r_m, below_B, above_B, found) near-threshold settings."""
    a = float(mp.total_radius())
    f_val = FREQ_HZ_SI
    r_list = [3.0 * a, 1.5 * a]
    B_grid = np.array([5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0])
    for r_m in r_list:
        last_B = None
        for B in B_grid:
            sc, _ = _one_neuron_result(B, f_val, r_m, neuron_key, mp)
            if sc > 0:
                below_B = (last_B if last_B is not None else B / 2.0)
                return f_val, r_m, below_B, B, True
            last_B = B
    return f_val, r_list[-1], B_grid[-2], B_grid[-1], False


def _fr_vs_B(B_vals: np.ndarray, neuron_key: str, f_Hz: float, r_m: float, mp: MaterialProps):
    return np.array([_one_neuron_result(B, f_Hz, r_m, neuron_key, mp)[1] for B in B_vals], dtype=float)


def print_mechanistic_differences() -> None:
    print("[Mechanistic differences]")
    for label, key in CLASSES:
        cfg = _CLASS_MODEL_CONFIGS.get(key, {}) if isinstance(_CLASS_MODEL_CONFIGS, dict) else {}
        lif = cfg.get("lif_params", {})
        adap = cfg.get("adaptive_params", {})
        model = cfg.get("model", "lif")
        parts = []
        C_m = lif.get("C_m"); g_L = lif.get("g_L")
        if C_m is not None: parts.append(f"C_m={C_m:.2e} F")
        if g_L is not None: parts.append(f"g_L={g_L:.2e} S")
        if C_m is not None and g_L is not None and g_L > 0:
            parts.append(f"tau_m={C_m/g_L*1e3:.1f} ms")
        if lif.get("V_threshold") is not None: parts.append(f"V_th={lif['V_threshold']:.3f} V")
        if lif.get("V_reset") is not None: parts.append(f"V_reset={lif['V_reset']:.3f} V")
        if lif.get("g_ext") is not None: parts.append(f"g_ext={lif['g_ext']:.2e} S")
        if adap:
            sub = []
            if adap.get('tau_w') is not None: sub.append(f"tau_w={adap['tau_w']:.3f} s")
            if adap.get('a') is not None: sub.append(f"a={adap['a']:.2e} A/V")
            if adap.get('b') is not None: sub.append(f"b={adap['b']:.2e} A")
            if sub: parts.append("adaptive: " + ", ".join(sub))
        print(f"  - {label}: model={model}; " + (", ".join(parts) if parts else "defaults"))


def generate_threshold_plots(save_dir: str = "figures") -> None:
    if not HAS_PLT:
        print("[Plots] matplotlib not available. Install with: pip install matplotlib")
        return
    os.makedirs(save_dir, exist_ok=True)
    mp = MaterialProps()
    for label, key in CLASSES:
        f_Hz, r_m, below_B, above_B, found = _find_threshold_params(key, mp)
        B_min, B_max = (max(below_B * 0.5, 1e-5), above_B * 2.0) if found else (5e-4, 1.0)
        B_sweep = np.geomspace(B_min, B_max, 60)
        rates = _fr_vs_B(B_sweep, key, f_Hz, r_m, mp)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(B_sweep, rates, marker="o", lw=1.4, ms=3, label="Firing rate")
        ax.set_xscale("log")
        ax.set_xlabel("B (Tesla)")
        ax.set_ylabel("Firing rate (Hz)")
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.set_title(f"Physics-driven MENP stimulation — {label}\nOne neuron; f={f_Hz:.1f} Hz, r={r_m*1e6:.2f} μm")

        if found:
            fr_at_thr = _one_neuron_result(above_B, f_Hz, r_m, key, mp)[1]
            ax.axvline(above_B, color="crimson", ls="--", lw=1.2, label=f"Threshold ≈ {above_B:.3g} T")
            ax.scatter([above_B], [fr_at_thr], color="crimson", zorder=5)
            print(f"[{label}] threshold ≈ {above_B:.4g} T at f={f_Hz:.1f} Hz, r={r_m*1e6:.2f} μm")
        else:
            print(f"[{label}] no spikes up to {B_max:.1f} T at f={f_Hz:.1f} Hz, r={r_m*1e6:.2f} μm")

        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(save_dir, f"threshold_{key}.png")
        fig.savefig(out_path, dpi=150)
        print(f"[Saved] {out_path}")
    plt.show()


def generate_voltage_trace_plots(save_dir: str = "figures") -> None:
    if not HAS_PLT:
        print("[Plots] matplotlib not available. Install with: pip install matplotlib")
        return
    os.makedirs(save_dir, exist_ok=True)
    mp = MaterialProps()
    for label, key in CLASSES:
        f_Hz, r_m, below_B, above_B, found = _find_threshold_params(key, mp)
        B_use = above_B if found else 1.0
        out = simulate_population(
            B=B_use * T,
            f=f_Hz * Hz,
            r=r_m * m,
            n_neurons=1,
            duration_s=DURATION_S,
            dt=DT_S,
            neuron_class=key,
            mp=mp,
            rng=np.random.default_rng(0),
            **_class_specific_args(key),
        )
        t = out["time"]; V = out["last_voltage"]; spikes = out.get("last_spike_times", np.array([]))
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(t, V, lw=1.3)
        if spikes is not None and np.size(spikes) > 0:
            ax.vlines(spikes, ymin=float(np.min(V)) - 0.002, ymax=float(np.max(V)) + 0.002,
                      color="crimson", lw=1.0, alpha=0.6, label="Spikes")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Membrane V (V)")
        ttl_B = f"B={B_use:.3g} T" + (" (near threshold)" if found else " (high-end trial)")
        ax.set_title(f"Physics-driven MENP stimulation — {label}\nV(t) at {ttl_B}, f={f_Hz:.1f} Hz, r={r_m*1e6:.2f} μm")
        ax.grid(True, ls="--", alpha=0.35); ax.legend(loc="upper right")
        fig.tight_layout()
        out_path = os.path.join(save_dir, f"voltage_{key}.png"); fig.savefig(out_path, dpi=150)
        print(f"[Saved] {out_path}")
    plt.show()


def generate_voltage_trace_pair_plots(save_dir: str = "figures") -> None:
    if not HAS_PLT:
        print("[Plots] matplotlib not available. Install with: pip install matplotlib")
        return
    os.makedirs(save_dir, exist_ok=True)
    mp = MaterialProps()
    for label, key in CLASSES:
        f_Hz, r_m, below_B, above_B, found = _find_threshold_params(key, mp)
        if not found:
            print(f"[Voltage pairs] {label}: threshold not found on grid; skipping overlay")
            continue
        out_lo = simulate_population(B=below_B * T, f=f_Hz * Hz, r=r_m * m,
                                     n_neurons=1, duration_s=DURATION_S, dt=DT_S,
                                     neuron_class=key, mp=mp, rng=np.random.default_rng(1), **_class_specific_args(key))
        out_hi = simulate_population(B=above_B * T, f=f_Hz * Hz, r=r_m * m,
                                     n_neurons=1, duration_s=DURATION_S, dt=DT_S,
                                     neuron_class=key, mp=mp, rng=np.random.default_rng(2), **_class_specific_args(key))
        t = out_hi["time"]
        V_lo, spikes_lo = out_lo["last_voltage"], out_lo.get("last_spike_times", np.array([]))
        V_hi, spikes_hi = out_hi["last_voltage"], out_hi.get("last_spike_times", np.array([]))
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        ax.plot(t, V_lo, lw=1.2, color="#1f77b4", label=f"Below threshold (B={below_B:.3g} T)")
        ax.plot(t, V_hi, lw=1.2, color="#d62728", label=f"Above threshold (B={above_B:.3g} T)")
        if np.size(spikes_lo) > 0:
            ax.vlines(spikes_lo, ymin=float(np.min(V_lo)) - 0.002, ymax=float(np.max(V_lo)) + 0.002,
                      color="#1f77b4", alpha=0.5, lw=0.8)
        if np.size(spikes_hi) > 0:
            ax.vlines(spikes_hi, ymin=float(np.min(V_hi)) - 0.002, ymax=float(np.max(V_hi)) + 0.002,
                      color="#d62728", alpha=0.5, lw=0.8)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Membrane V (V)")
        ax.set_title(f"Physics-driven MENP stimulation — {label}\nV(t) below vs above threshold at f={f_Hz:.1f} Hz, r={r_m*1e6:.2f} μm")
        ax.grid(True, ls="--", alpha=0.35); ax.legend(); fig.tight_layout()
        out_path = os.path.join(save_dir, f"voltage_pair_{key}.png"); fig.savefig(out_path, dpi=150)
        print(f"[Saved] {out_path}")
    plt.show()


def _load_allen_spike_trains() -> dict:
    
    cands = [Path("DATA") / "spike_trains.npz", Path("allen_data_extraction_files") / "spike_trains.npz"]
    for p in cands:
        if p.exists():
            data = np.load(str(p), allow_pickle=True)
            for key in ("spike_times_by_unit", "spike_trains", "spike_times_list", "spike_times"):
                if key in data.files and isinstance(data[key], np.ndarray) and data[key].dtype == object:
                    times_list = [np.asarray(x, dtype=float).ravel() if x is not None else np.array([], dtype=float) for x in data[key]]
                    dur = None
                    if "duration_s" in data.files:
                        dur = float(np.asarray(data["duration_s"], dtype=float).ravel()[0])
                    elif "duration" in data.files:
                        dur = float(np.asarray(data["duration"], dtype=float).ravel()[0])
                    else:
                        dur = float(max((float(x.max()) if x.size else 0.0) for x in times_list))
                    return {"spike_times_list": times_list, "duration_s": dur}
            # counts fallback
            dur_key = "duration_s" if "duration_s" in data.files else ("duration" if "duration" in data.files else None)
            if dur_key:
                dur = float(np.asarray(data[dur_key], dtype=float).ravel()[0])
                sc_keys = [k for k in data.files if k.lower().startswith("spike_count")]
                if sc_keys:
                    sc = np.asarray(data[sc_keys[0]], dtype=float).ravel()
                    return {"spike_counts": sc, "duration_s": dur}
    # CSV fallback: synthesize counts
    for p in [Path("DATA") / "firing_statistics.csv", Path("allen_data_extraction_files") / "firing_statistics.csv"]:
        if p.exists():
            try:
                import pandas as pd
                rates = pd.read_csv(p).filter(regex="(?i)mean_firing_rate|rate_hz").iloc[:,0].to_numpy(dtype=float)
                dur = 10.0
                sc = np.clip(np.round(rates * dur), 0, None)
                return {"spike_counts": sc, "duration_s": dur}
            except Exception:
                pass
    raise FileNotFoundError("No Allen spike trains found in DATA/ or allen_data_extraction_files/")


def _lif_response_to_spike_train(spike_times: np.ndarray, duration_s: float, dt: float, lif: LIFParams) -> dict:
    t = np.arange(0, duration_s, dt)
    tau = 0.01; amp = 30e-12
    I = np.zeros_like(t)
    for ts in spike_times:
        idx = int(ts / dt)
        if 0 <= idx < I.size:
            tail = np.arange(0, I.size - idx) * dt
            I[idx:] += (amp * (tail / tau) * np.exp(1 - tail / tau))
    V = lif.E_L; Vth, Vreset = lif.V_threshold, lif.V_reset
    C, gL, EL = lif.C_m, lif.g_L, lif.E_L
    spikes = []; V_trace = np.empty_like(t)
    for i, ti in enumerate(t):
        I_leak = gL * (EL - V); I_ext = I[i]
        V = V + ((I_leak + I_ext) / C) * dt
        V = min(max(V, -0.12), 0.05)
        if V >= Vth:
            spikes.append(ti); V = Vreset
        V_trace[i] = V
    fr = (len(spikes) / duration_s) if duration_s > 0 else 0.0
    return {"time": t, "voltage": V_trace, "spike_times": np.array(spikes), "firing_rate": fr}


def run_allen_driven_examples():
    if not HAS_PLT:
        print("[Allen-driven] matplotlib not available; skipping plots")
        return
    data = _load_allen_spike_trains()
    duration = float(data.get("duration_s", 10.0)); dt = DT_S; lif = LIFParams()
    if isinstance(data.get("spike_times_list"), list):
        times_list = data["spike_times_list"]; n_show = int(min(3, len(times_list)))
        for i in range(n_show):
            spike_times = np.asarray(times_list[i], dtype=float).ravel()
            res = _lif_response_to_spike_train(spike_times, duration, dt, lif)
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.plot(res["time"], res["voltage"], lw=1.2)
            if res["spike_times"].size:
                ax.vlines(res["spike_times"], ymin=min(res["voltage"]) - 0.002, ymax=max(res["voltage"]) + 0.002,
                          color="crimson", alpha=0.6, lw=1.0, label="Output spikes")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Membrane V (V)"); ax.grid(True, ls="--", alpha=0.3)
            ax.set_title(f"Allen-driven LIF (unit {i+1}) — input spikes={spike_times.size}, output rate={res['firing_rate']:.2f} Hz")
            fig.tight_layout()
    else:
        sc = data["spike_counts"].astype(int); n_show = int(min(3, sc.size)); rng = np.random.default_rng(0)
        for i in range(n_show):
            n_spk = int(max(0, sc[i])); spike_times = np.sort(rng.uniform(0.1 * duration, 0.9 * duration, size=n_spk)) if n_spk > 0 else np.array([])
            res = _lif_response_to_spike_train(spike_times, duration, dt, lif)
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.plot(res["time"], res["voltage"], lw=1.2)
            if res["spike_times"].size:
                ax.vlines(res["spike_times"], ymin=min(res["voltage"]) - 0.002, ymax=max(res["voltage"]) + 0.002,
                          color="crimson", alpha=0.6, lw=1.0, label="Output spikes")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Membrane V (V)"); ax.grid(True, ls="--", alpha=0.3)
            ax.set_title(f"Allen-driven LIF (unit {i+1}) — input spikes={n_spk}, output rate={res['firing_rate']:.2f} Hz")
            fig.tight_layout()
    plt.show()


def single_neuron_snapshots() -> None:
    mp = MaterialProps()
    for label, key in CLASSES:
        f_Hz, r_m, below_B, above_B, found = _find_threshold_params(key, mp)
        if not found:
            print(f"[{label}] No threshold found on grid; showing high-end trial.")
            sc_lo, fr_lo = _one_neuron_result(below_B, f_Hz, r_m, key, mp)
            sc_hi, fr_hi = _one_neuron_result(above_B, f_Hz, r_m, key, mp)
            print(f"  Try: B={below_B:.4f} T → spikes={sc_lo}, rate={fr_lo:.2f} Hz")
            print(f"       B={above_B:.4f} T → spikes={sc_hi}, rate={fr_hi:.2f} Hz")
            continue
        sc_lo, fr_lo = _one_neuron_result(below_B, f_Hz, r_m, key, mp)
        sc_hi, fr_hi = _one_neuron_result(above_B, f_Hz, r_m, key, mp)
        print(f"[{label}] f={f_Hz:.1f} Hz r={r_m*1e6:.2f} μm")
        print(f"  Below threshold:  B={below_B:.4f} T, spikes={sc_lo}, rate={fr_lo:.2f} Hz")
        print(f"  Above threshold:  B={above_B:.4f} T, spikes={sc_hi},        rate={fr_hi:.2f} Hz")


def main() -> None:
    safety_quicklook()
    population_quicklook()
    print_mechanistic_differences()
    single_neuron_snapshots()
    generate_threshold_plots()
    generate_voltage_trace_plots()
    generate_voltage_trace_pair_plots()
    try:
        run_allen_driven_examples()
    except Exception as e:
        print(f"[Allen-driven] skipped ({e})")


if __name__ == "__main__":
    main()