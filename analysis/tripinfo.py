"""Reduce SUMO tripinfo output to completed-trip metrics.

Why not the step CSV
--------------------
`system_mean_waiting_time` (sumo-rl's per-step metric, and the Stage-1 headline)
averages `getWaitingTime()` over the vehicles still IN the network, then
averages that over steps. It is survivorship-biased in the wrong direction: a
controller that strands traffic is graded on the vehicles it did not strand,
and a fully deadlocked network accumulates one second of "waiting" per second
regardless of how many vehicles are stuck. That inverted the Stage-1 ranking.

tripinfo has one row per trip that actually FINISHED, so:

  * trips_completed      = throughput. A policy cannot improve it by holding
                           vehicles back; not finishing a trip costs the count.
  * trip_time_loss_mean  = delay vs a free-flow run of the same route, per
                           completed trip. SUMO's `timeLoss`; the headline.
  * trip_completion_rate = completed / departed, when the demand is known.
                           Guards the one way throughput still misleads --
                           "cleared everything" vs "inserted almost nothing".

Delay is also broken out per vehicle class (moto / auto / car), because a plan
that clears cars by stranding motorcycles must not score like one that does not.

    python -m analysis.tripinfo logs/eval_fixedtime_peak_seed42_g60_tripinfo.xml
"""
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

NAN = float("nan")

# per-trip attributes worth averaging -> output metric name
_TRIP_ATTRS = {
    "timeLoss": "trip_time_loss_mean",
    "waitingTime": "trip_waiting_time_mean",
    "duration": "trip_duration_mean",
    "departDelay": "trip_depart_delay_mean",
    "routeLength": "trip_route_length_mean",
}


def _vclass(vtype: str) -> str:
    """Base vehicle class for a vType id; "moto@1" (a distribution member)
    and "moto" are the same class."""
    return (vtype or "unknown").split("@")[0]


def _parse(path: str):
    """Trip elements from a tripinfo file, tolerating a missing closing tag.

    SUMO streams tripinfo as trips finish and only writes </tripinfos> on a
    clean shutdown, so a killed or timed-out run leaves the root element open.
    The trips already flushed are still valid measurements.
    """
    with open(path) as fh:
        text = fh.read()
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        root = ET.fromstring(text.rstrip() + "\n</tripinfos>")
    return root.findall("tripinfo")


def _mean(values) -> float:
    return sum(values) / len(values) if values else NAN


def reduce_tripinfo(path: str, departed: int = None) -> dict:
    """Completed-trip metrics for one run. `{}` if the file does not exist.

    `departed` is the number of vehicles the demand inserted; supply it to get
    a completion rate (tripinfo itself only records trips that finished).
    """
    if not os.path.exists(path):
        return {}

    trips = _parse(path)
    by_attr = defaultdict(list)
    by_class = defaultdict(list)
    for t in trips:
        for attr in _TRIP_ATTRS:
            raw = t.get(attr)
            if raw is not None:
                by_attr[attr].append(float(raw))
        loss = t.get("timeLoss")
        if loss is not None:
            by_class[_vclass(t.get("vType"))].append(float(loss))

    out = {"trips_completed": len(trips)}
    for attr, name in _TRIP_ATTRS.items():
        out[name] = _mean(by_attr[attr])
    for vclass, losses in sorted(by_class.items()):
        out[f"trip_time_loss_mean_{vclass}"] = _mean(losses)
        out[f"trips_completed_{vclass}"] = len(losses)
    if departed:
        out["trip_completion_rate"] = len(trips) / departed
    return out


def count_departures(route_file: str, horizon: float = None) -> float:
    """Vehicles the demand asks for -- the denominator for completion rate.

    Counts <vehicle> elements plus each <flow>, whether it is specified by
    `number` or by `vehsPerHour`/`period`/`probability` (make_scenarios.py
    writes vehsPerHour). `horizon` clips each flow to an episode shorter than
    the flow window; a 1200 s episode does not see 3600 s of demand.

    This is the demand REQUESTED, not the vehicles SUMO managed to insert --
    deliberately. A controller that jams the approaches so badly that vehicles
    cannot even enter should be charged for them, and a denominator taken from
    the run itself would quietly forgive that.
    """
    root = ET.parse(route_file).getroot()
    n = float(len(root.findall("vehicle")))
    for flow in root.findall("flow"):
        begin = float(flow.get("begin", 0.0))
        end = float(flow.get("end", horizon if horizon is not None else 3600.0))
        if horizon is not None:
            end = min(end, horizon)
        window = max(end - begin, 0.0)

        if flow.get("number") is not None:
            total = float(flow.get("number"))
            # `number` spreads over the flow's own window; a clipped episode
            # only sees the fraction of that window it covers
            full = max(float(flow.get("end", end)) - begin, 0.0)
            n += total * (window / full) if full else 0.0
        elif flow.get("vehsPerHour") is not None:
            n += float(flow.get("vehsPerHour")) * window / 3600.0
        elif flow.get("period") is not None:
            period = float(flow.get("period"))
            n += window / period if period else 0.0
        elif flow.get("probability") is not None:
            n += float(flow.get("probability")) * window
    return n


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("tripinfo")
    p.add_argument("--route-file", default=None,
                   help="demand file, to report a completion rate")
    a = p.parse_args()
    departed = count_departures(a.route_file) if a.route_file else None
    for k, v in reduce_tripinfo(a.tripinfo, departed).items():
        print(f"{k:32s} {v}")
