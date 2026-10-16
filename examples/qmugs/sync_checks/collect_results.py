###########################################################################################
# Sync checks: collect every check's measured numbers against its reference (pure parsing)
#
# Reads logs/sync-checks/jobs.tsv, sacct, the SLURM .out files in examples/qmugs, the run
# directories under logs/ and the helper JSONs, and writes
#   <out_dir>/results.json      per check: job state, measured values, references, deviations,
#                               and a mechanical verdict against the band of sync-checks.md
#   <out_dir>/results_table.md  the same as a table
# Idempotent and safe to run at any time; checks that have not finished are reported as such.
# The mechanical verdict is an input to sync-checks-posttest.md, not its conclusion.
###########################################################################################

import os
import re
import sys
import glob
import json
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--jobs", required=True, help="logs/sync-checks/jobs.tsv")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--example_dir", default="/work/xw262/HydraGNN/examples/qmugs")
parser.add_argument("--logs_dir", default="/work/xw262/HydraGNN/logs")
args = parser.parse_args()

EX, L, SC = args.example_dir, args.logs_dir, args.out_dir


def log_path(*parts):
    '''a logs/ path; sync-check runs other than T2 and T4 were later moved to logs/sync-checks/runs/'''
    moved = os.path.join(L, "sync-checks", "runs", *parts)
    return moved if os.path.exists(moved) else os.path.join(L, *parts)

# ----------------------------------------------------------------------------------------------------
# references (sync-checks.md; sources cited there)

T1_REF = [  # (train, val, test), logs/qmugs-train-55914623-NN1-PM-FSDP0-V2/run.log epochs 0-4
    (2.8432e-4, 2.5459e-4, 2.5891e-4), (2.5906e-4, 2.5431e-4, 2.5822e-4), (2.5901e-4, 2.5455e-4, 2.5853e-4),
    (2.5894e-4, 2.5428e-4, 2.5831e-4), (2.5895e-4, 2.5499e-4, 2.5913e-4),
]
T2_BAND = {  # epoch: (val_lo, val_hi, test_lo, test_hi) over 53709019 / 53692783 / 53631819 / 53686488
    0: (4.0436e-3, 4.1223e-3, 4.1170e-3, 4.1938e-3), 1: (3.9733e-3, 3.9845e-3, 4.0466e-3, 4.0574e-3),
    4: (3.9630e-3, 3.9706e-3, 4.0314e-3, 4.0388e-3), 9: (3.9526e-3, 3.9571e-3, 4.0276e-3, 4.0321e-3),
    14: (3.9557e-3, 3.9583e-3, 4.0273e-3, 4.0299e-3),
}
T4_REF = {0: (4.8514e-3, 4.9418e-3), 3: (4.5311e-3, 4.6217e-3), 6: (4.3476e-3, 4.4307e-3), 9: (4.2293e-3, 4.3142e-3)}
E_REF = {"E1": (2.4392e-4, 2.4862e-4), "E1b": (2.4392e-4, 2.4862e-4), "E2": (3.91011e-3, 3.98406e-3), "E3": (2.90826e-3, 2.97336e-3)}
DDP_DENSE_CKPT_BYTES = 6310166533  # logs/qmugs-train-53709019-NN2-DCC-FSDP0-V2/*_epoch_99.pk
U4_REF = {"minimum stored edge length": "0.954591", "isolated atoms": "0", "self loops": "0",
          "non-finite pos entries": "0", "non-finite target entries": "0",
          "max |target entry| quantiles": "p50=0.528093 p90=0.622816 p99=0.692347 max=1.39478",
          "n_basis": "min=128 p50=526 max=800", "conformers scanned": "18580"}


def rel(measured, reference):
    return None if measured is None or reference in (None, 0) else (measured - reference) / reference


def read(path):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------------------------------
# jobs and their SLURM state

jobs = {}
with open(args.jobs) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            jobs.setdefault(parts[0], []).append(parts[1])  # all submissions, latest last


def sacct(jobid):
    out = subprocess.run(
        ["sacct", "-j", jobid, "-X", "-n", "-P", "-o", "State,Elapsed,NodeList,ExitCode"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    if not out:
        return {"state": "UNKNOWN"}
    state, elapsed, nodes, exit_code = (out[0].split("|") + ["", "", "", ""])[:4]
    return {"state": state, "elapsed": elapsed, "nodes": nodes, "exit_code": exit_code}


def out_file(jobid):
    hits = glob.glob(os.path.join(EX, "slurm_output", "**", f"*-job-{jobid}.out"), recursive=True)
    return hits[0] if hits else None


EPOCH_RE = re.compile(r"Epoch: (\d+), Train Loss: (\S+), Val Loss: (\S+), Test Loss: (\S+)")


def epochs(text):
    '''rank-0 epoch lines -> {epoch: (train, val, test)}'''
    found = {}
    for line in text.splitlines():
        if line.startswith("0: ") or line.startswith("Epoch:"):
            m = EPOCH_RE.search(line)
            if m:
                found[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
    return found


def training_extras(text):
    mem = [float(m.group(1)) for m in re.finditer(r"Max memory allocated after optimizer step: ([0-9.]+) GB", text)]
    grad = [float(m.group(1)) for m in re.finditer(r"grad_norm=([0-9.eE+-]+|nan|inf)", text) if m.group(1) not in ("nan", "inf")]
    rt = [float(m.group(1)) for m in re.finditer(r"relative_total=([0-9.eE+-]+)", text)]
    run = re.search(r"--log=(\S+)", text)
    return {
        "peak_mem_alloc_gb": max(mem) if mem else None,
        "grad_norm_first": grad[0] if grad else None,
        "grad_norm_max": max(grad) if grad else None,
        "relative_total_first": rt[0] if rt else None,
        "relative_total_last": rt[-1] if rt else None,
        "nan_in_losses": "nan" in " ".join(l for l in text.splitlines() if "Epoch:" in l),
        "skipped_steps": text.count("skipping the optimizer step"),
        "negative_pivot_assertion": "negative pivot" in text,
        "load_existing_model": bool(re.search(r"Load existing model", text)),
        "run_name": run.group(1).strip('"') if run else None,
        "traceback": "Traceback" in text,
    }


# ----------------------------------------------------------------------------------------------------
# per-check evaluation

results = {}
for check in sorted(jobs):
    if check == "COLLECT":
        continue
    jobid = jobs[check][-1]
    r = {"jobid": jobid, "all_submissions": jobs[check], **sacct(jobid)}
    of = out_file(jobid)
    r["out_file"] = of
    text = read(of) if of else ""
    verdict = None
    finished = r["state"].startswith(("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"))

    if check.startswith("T"):
        ep = epochs(text)
        r["epochs"] = ep
        r.update(training_extras(text))
        if check == "T1":
            dev = {e: [rel(m, ref) for m, ref in zip(ep[e], T1_REF[e])] for e in range(5) if e in ep}
            r["rel_dev_vs_55914623"] = dev
            verdict = len(dev) == 5 and all(abs(x) <= 5e-3 for v in dev.values() for x in v)
        elif check == "T2":
            checks = {}
            for e, (vlo, vhi, tlo, thi) in T2_BAND.items():
                if e in ep:
                    tol = 0.02 if e == 0 else 0.01
                    _, v, t = ep[e]
                    checks[e] = vlo * (1 - tol) <= v <= vhi * (1 + tol) and tlo * (1 - tol) <= t <= thi * (1 + tol)
            r["band_checks"] = checks
            verdict = len(checks) == len(T2_BAND) and all(checks.values()) and not r["nan_in_losses"]
        elif check == "T3":
            run = r["run_name"]
            ckpts = sorted(glob.glob(os.path.join(log_path(run), f"{run}_epoch_*.pk"))) if run else []
            r["checkpoint_sizes"] = {os.path.basename(c): os.path.getsize(c) for c in ckpts}
            r["size_dev_bytes_vs_ddp"] = [s - DDP_DENSE_CKPT_BYTES for s in r["checkpoint_sizes"].values()]
            verdict = bool(ckpts) and all(abs(d) <= 1_000_000 for d in r["size_dev_bytes_vs_ddp"])
        elif check == "T4":
            dev = {e: [rel(ep[e][1], ref[0]), rel(ep[e][2], ref[1])] for e, ref in T4_REF.items() if e in ep}
            r["rel_dev_vs_53692804"] = dev
            mem_ok = r["peak_mem_alloc_gb"] is not None and abs(rel(r["peak_mem_alloc_gb"], 10.210353152)) <= 0.10
            rt_ok = r["relative_total_first"] is not None and 0.90 <= r["relative_total_first"] <= 1.00
            ep9_ok = 9 in dev and abs(dev[9][0]) <= 0.10
            verdict = ep9_ok and rt_ok and mem_ok and not r["negative_pivot_assertion"] and not r["nan_in_losses"]
        elif check in ("T5_dense", "T5_schur"):
            source = "T2" if check == "T5_dense" else "T4"
            src_text = read(out_file(jobs[source][-1]) or "")
            src_ep = epochs(src_text)
            if src_ep and ep:
                last_train = src_ep[max(src_ep)][0]
                r["first_resumed_train_vs_source_last"] = rel(ep[min(ep)][0], last_train)
            if check == "T5_schur":
                src_rt = training_extras(src_text)["relative_total_last"]
                r["first_relative_total_vs_source_last"] = rel(r["relative_total_first"], src_rt)
            verdict = (
                r["load_existing_model"] and not r["traceback"]
                and r.get("first_resumed_train_vs_source_last") is not None
                and abs(r["first_resumed_train_vs_source_last"]) <= 0.03
                and (check == "T5_dense" or (r.get("first_relative_total_vs_source_last") is not None and abs(r["first_relative_total_vs_source_last"]) <= 0.05))
            )
        elif check in ("T7", "T8"):
            trains = [ep[e][0] for e in sorted(ep)]
            verdict = len(trains) == 5 and all(x == x for x in trains) and trains[-1] < trains[0]

    elif check.startswith("E"):
        name = check if check in ("E1", "E1b", "E2", "E3") else check
        res = load_json(os.path.join(SC, f"{check}.json"))
        r["result"] = res
        r["traceback_tail"] = "\n".join(l for l in text.splitlines() if "Error" in l)[-2000:] or None
        if check in E_REF and res:
            vref, tref = E_REF[check]
            r["rel_dev_val"], r["rel_dev_test"] = rel(res["val_loss"], vref), rel(res["test_loss"], tref)
            verdict = abs(r["rel_dev_val"]) <= 1e-4 and abs(r["rel_dev_test"]) <= 1e-4
        elif check.startswith("E4a"):
            # expected failure: strict load refuses the #511 bias keys
            verdict = finished and res is None and "Unexpected key" in text
        elif check.startswith("E4b") and res:
            ref = E_REF["E2" if check.endswith("dense") else "E3"]
            r["rel_shift_val"], r["rel_shift_test"] = rel(res["val_loss"], ref[0]), rel(res["test_loss"], ref[1])
            verdict = bool(res["dropped_unexpected_keys"])  # a measurement; dropped keys must be the #511 biases

    elif check == "U1":
        verdict = "cuda available: True | device count: 4" in text and "2.8.0+cu128" in text
    elif check == "U2":
        new = [l for l in text.splitlines() if re.match(r"\[(PASS|FAIL)\]", l)]
        ref = [l for l in read(os.path.join(EX, "slurm_output", "schur_decoder_test", "schur-decoder-job-53631178.out")).splitlines() if re.match(r"\[(PASS|FAIL)\]", l)]
        r["n_pass"], r["n_fail"] = sum(l.startswith("[PASS]") for l in new), sum(l.startswith("[FAIL]") for l in new)
        r["differing_lines"] = [a for a, b in zip(new, ref) if a != b]
        verdict = r["n_pass"] == 33 and r["n_fail"] == 0 and new == ref
    elif check == "U3":
        a = load_json(log_path("qmugs-schur-diagnostics", "schur_decomp_summary.json"))
        b = load_json(log_path("qmugs-schur-diagnostics-synccheck", "schur_decomp_summary.json"))
        if a and b:
            def flat(d, p=""):
                for k, v in d.items():
                    yield from (flat(v, p + k + ".") if isinstance(v, dict) else [(p + k, v)])
            fa, fb = dict(flat(a)), dict(flat(b))
            nums = [k for k in fa if k in fb and isinstance(fa[k], (int, float)) and not isinstance(fa[k], bool)]
            devs = {k: abs(fa[k] - fb[k]) / max(abs(fa[k]), 1e-300) for k in nums}
            r["keys_only_in_one"] = sorted(set(fa) ^ set(fb))
            r["max_rel_dev"] = max(devs.values()) if devs else None
            r["max_rel_dev_key"] = max(devs, key=devs.get) if devs else None
            r["non_numeric_differences"] = [k for k in fa if k in fb and k not in nums and fa[k] != fb[k]]
            verdict = not r["keys_only_in_one"] and not r["non_numeric_differences"] and (r["max_rel_dev"] or 0) <= 1e-10
    elif check == "U4":
        blocks = re.split(r"^=== ", text, flags=re.M)[1:]
        stats = {}
        for b in blocks:
            name = os.path.basename(b.splitlines()[0].strip())
            stats[name] = {k: (re.search(re.escape(k) + r":?\s+(.*)", b).group(1).strip() if re.search(re.escape(k) + r":?\s+(.*)", b) else None) for k in U4_REF}
        r["stats"] = stats
        verdict = len(stats) == 2 and all(all(v.get(k) == U4_REF[k] for k in U4_REF) for v in stats.values())
    elif check == "U5":
        m = re.search(r"(\d+) passed", text)
        r["passed"] = int(m.group(1)) if m else None
        verdict = r["passed"] == 86 and " failed" not in text
    elif check.startswith("P1_compare"):
        res = load_json(os.path.join(SC, "P1_compare_pickles_ignore_natoms.json" if check.endswith("ignore_natoms") else "P1_compare_pickles.json"))
        r["n_mismatch_total"] = res["n_mismatch_total"] if res else None
        r["splits"] = {k: {kk: v[kk] for kk in ("ntotal_reference", "ntotal_candidate", "n_compared", "n_sample_mismatch", "meta_mismatch")} for k, v in res["splits"].items()} if res else None
        r["artifacts"] = res["artifacts"] if res else None
        verdict = res is not None and res["n_mismatch_total"] == 0
    elif check in ("P1", "P2", "P3"):
        r["split_sizes"] = {m.group(1): int(m.group(2)) for m in re.finditer(r"Streaming split=(\w+): global_graphs=(\d+)", text)}
        verdict = r["state"] == "COMPLETED" and len(r["split_sizes"]) == 3
        if check == "P3" and "P2" in results:
            r["equal_to_P2"] = r["split_sizes"] == results["P2"].get("split_sizes")
            verdict = verdict and r["equal_to_P2"]
    elif check.startswith("P4"):
        res = load_json(os.path.join(SC, {
            "P4_decodercmp": "P4_QMugs001_max800_delta_uptr_decodercmp.json", "P4_delta": "P4_QMugs001_max800_delta.json",
            "P4_raw": "P4_QMugs0001_max800_synccheck.json", "P4_raw_uptr": "P4_QMugs0001_max800_uptr_synccheck.json"}[check]))
        if res:
            r["summary"] = {k: v for k, v in res.items() if k != "records"}
            exact = res["all_targets_equal"] and res["all_artifacts_equal"] and res["all_promolecular_upper_triangle_equal"] and res["all_atomic_numbers_equal"]
            physics = res["max_symmetry_abs"] <= 1e-12 and res["min_eigenvalue_over_max"] >= -1e-10 and res["max_abs_trace_SD_error"] <= 1e-6 and res["max_geometry_abs_diff_angstrom"] <= 1e-4
            lo, hi = res["dipole_norm_ratio_if_debye_range"]
            alo, ahi = res["dipole_norm_ratio_if_au_range"]
            dipole = (0.999 <= lo and hi <= 1.001) or (0.999 <= alo and ahi <= 1.001)
            r["exact_ok"], r["physics_ok"], r["dipole_ok"] = exact, physics, dipole
            verdict = exact and physics and dipole
    elif check.startswith("I"):
        log_dir = log_path(f"qmugs-inference-{jobid}-NN1-DCC")
        g = load_json(os.path.join(log_dir, "global_evaluation_metrics.json"))
        r["log_dir"] = log_dir
        r["global_metrics"] = {k: v for k, v in g.items()} if g else None
        r["load_warnings"] = [l for l in text.splitlines() if "missing keys" in l or "unexpected keys" in l]
        r["fallback_warnings"] = [l for l in text.splitlines() if "falling back to" in l]
        verdict = g is not None and not r["load_warnings"]
        if check == "I1" and g:
            per = load_json(os.path.join(log_dir, "per_matrix_metrics.json")) or []
            n_pad = 800
            if per:
                padded_mse = sum(p["mse"] * p["density_matrix_dim"] ** 2 for p in per) / (len(per) * n_pad ** 2)
                r["padded_mse_from_unpadded"] = padded_mse
                r["rel_dev_vs_E1_test"] = rel(padded_mse, 2.4862e-4)
                verdict = verdict and abs(r["rel_dev_vs_E1_test"]) <= 0.01
    elif check == "F1":
        log_dir = log_path(f"qmugs-inference-{jobs['I2'][-1]}-NN1-DCC") if "I2" in jobs else None
        figs = glob.glob(os.path.join(log_dir, "figs", "**", "*.png"), recursive=True) if log_dir else []
        r["n_png"] = len(figs)
        verdict = r["state"] == "COMPLETED" and len(figs) > 0 and "Traceback" not in text

    r["finished"] = finished
    r["mechanical_verdict"] = None if not finished else ("PASS" if verdict else "FAIL")
    results[check] = r

with open(os.path.join(SC, "results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)

rows = ["| check | job | state | elapsed | nodes | mechanical verdict |", "|---|---|---|---|---|---|"]
for check, r in results.items():
    rows.append(f"| {check} | {r['jobid']} | {r['state']} | {r.get('elapsed', '')} | {r.get('nodes', '')} | {r['mechanical_verdict']} |")
with open(os.path.join(SC, "results_table.md"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("\n".join(rows))
