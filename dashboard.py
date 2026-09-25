"""Build the results dashboard and serve it on localhost.

Aggregates the tracked result CSVs (analysis/*.csv, logs/comparison.csv,
results/comparison_stage1.csv) plus the saved PNG figures into one
self-contained HTML page, writes it to results/dashboard/index.html, and
serves that folder over HTTP.

    python dashboard.py                  # build + serve on http://localhost:8000
    python dashboard.py --port 8080      # different port
    python dashboard.py --open           # also open it in the default browser
    python dashboard.py --no-serve       # build only

Re-run after compare.py / the analysis scripts to pick up new numbers. The page
template lives in dashboard/template.html.
"""
import argparse
import base64
import functools
import http.server
import json
import os
import webbrowser

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(ROOT, "analysis")
TEMPLATE = os.path.join(ROOT, "dashboard", "template.html")
OUT_DIR = os.path.join(ROOT, "results", "dashboard")

# The template is written as a page body; this shell makes it a full document.
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
html,body{margin:0}
img{max-width:100%}
[hidden]{display:none!important}
</style>
"""
TAIL = "\n</body>\n</html>\n"

FIGURES = [
    ("docs/figures/geometry_dose_response.png", "Geometry dose-response (report figure)"),
    ("docs/figures/lambda_curve.png", "λ efficiency–safety curve (report figure)"),
    ("results/lambda_ablation.png", "λ ablation (results/)"),
]


def csv(name):
    return pd.read_csv(os.path.join(A, name))


def agg(df, keys, val):
    """Mean / SD / count of `val` per `keys` group, as JSON-ready dicts."""
    g = df.groupby(keys)[val].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0)
    rows = []
    for _, r in g.iterrows():
        row = {k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in keys}
        row.update(mean=round(float(r["mean"]), 3), sd=round(float(r["std"]), 3), n=int(r["count"]))
        rows.append(row)
    return rows


def build_data():
    out = {}

    # Corridor: regular vs irregular nets
    d2 = csv("irregular_net_compare2.csv")
    out["nets"] = agg(pd.concat([csv("irregular_net_compare.csv"), d2[d2.net != "regular"]]),
                      ["net", "controller"], "delay_per_trip")

    # Geometry dose-response; span 400 joins the r<0.5 sweep onto the r>=0.5 one
    lowr = csv("geometry_sweep_lowr.csv")
    geo = {"400": agg(pd.concat([lowr[lowr.ratio != 0.5], csv("geometry_sweep.csv")]),
                      ["ratio", "controller"], "delay_per_trip")}
    for span in ["450", "550", "700"]:
        geo[span] = agg(csv(f"geometry_sweep_span{span}.csv"), ["ratio", "controller"], "delay_per_trip")
    out["geo"] = geo

    # Classical controllers vs min_green
    cs = csv("corridor_sweep.csv")
    cs = cs[cs.scenario.isin(["corridor_peak", "corridor_tidal"])]
    out["mingreen"] = agg(cs, ["scenario", "controller", "min_green"], "delay_per_trip")

    # All corridor controllers at min_green=10
    ippo = csv("ippo_sweep.csv")
    cols = ["controller", "delay_per_trip"]
    at10 = cs[cs.min_green == 10]
    out["peak"] = agg(pd.concat([
        at10[at10.scenario == "corridor_peak"][cols],
        csv("irregular_net_compare.csv").query("net=='regular' and controller=='idqn'")[cols],
        ippo[ippo.scenario == "corridor_peak"][cols],
    ]), ["controller"], "delay_per_trip")
    out["tidal"] = agg(pd.concat([
        at10[at10.scenario == "corridor_tidal"][cols],
        ippo[ippo.scenario == "corridor_tidal"][cols],
    ]), ["controller"], "delay_per_trip")

    # Safety weight ablation
    la = csv("lambda_ablation.csv")
    out["lam_delay"] = agg(la, ["geometry", "lam"], "delay_per_trip")
    out["lam_safety"] = agg(la, ["geometry", "lam"], "safety_total_per_trip")

    out["mappo"] = csv("mappo_retune_smoke.csv").round(3).to_dict("records")
    out["het"] = csv("heterogeneity_breakdown.csv").to_dict("records")

    # Incident
    iw = csv("incident_window_compare.csv")
    out["incident"] = agg(iw, ["controller", "incident"], "window_delay")
    out["incident_whole"] = agg(iw, ["controller", "incident"], "whole_episode_delay")
    out["incident10"] = agg(csv("incident_n10.csv"), ["controller"], "delta")

    # Isolated intersection (stage 1 at lambda=0.5, plus timing sweeps)
    st = pd.read_csv(os.path.join(ROOT, "results", "comparison_stage1.csv"))
    st = st[st.lam.astype(str).isin(["5", "05"])]
    out["stage1"] = [{
        "scenario": r.scenario, "algo": r.algo, "n": int(r.n_runs),
        "mean": round(r.system_mean_waiting_time_mean, 3),
        "sd": round(0 if pd.isna(r.system_mean_waiting_time_std) else r.system_mean_waiting_time_std, 3),
    } for r in st.itertuples()]
    out["static"] = agg(csv("static_sweep.csv"), ["green"], "delay")
    out["actuated"] = agg(csv("actuated_sweep.csv"), ["min_green"], "delay")

    out["stats"] = json.loads(csv("headline_stats.csv").to_json(orient="records"))

    cmp_cols = ["scenario", "algo", "n_runs", "trip_time_loss_mean_mean", "trip_time_loss_mean_std",
                "system_mean_waiting_time_mean", "system_mean_speed_mean", "system_safety_total_mean"]
    comparison = pd.read_csv(os.path.join(ROOT, "logs", "comparison.csv"))
    out["comparison"] = json.loads(comparison[cmp_cols].to_json(orient="records"))

    figs = []
    for path, cap in FIGURES:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            print(f"  skip missing figure {path}")
            continue
        with open(full, "rb") as f:
            src = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        figs.append({"src": src, "cap": cap, "path": path})
    out["figs"] = figs
    return out


def build():
    with open(TEMPLATE) as f:
        body = f.read()
    data = json.dumps(build_data(), separators=(",", ":"))
    html = HEAD + body.replace("/*__DATA__*/null", data) + TAIL
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "index.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"Wrote {os.path.relpath(path, ROOT)} ({len(html) / 1024:.0f} KB)")
    return path


def serve(port, open_browser):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=OUT_DIR)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://localhost:{port}/"
        print(f"Serving dashboard at {url}  (Ctrl+C to stop)", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    ap.add_argument("--no-serve", action="store_true", help="build the HTML and exit")
    args = ap.parse_args()
    build()
    if not args.no_serve:
        serve(args.port, args.open)


if __name__ == "__main__":
    main()
