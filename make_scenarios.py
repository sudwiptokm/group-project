"""Generate peak / off-peak demand variants from traffic.rou.xml.

Only the flow rates are scaled; routes, vTypeDistribution and edge ids are left
untouched (edge ids must keep matching the .net.xml).

STOCHASTIC ARRIVALS, and why only the corridor gets them
--------------------------------------------------------
`vehsPerHour="X"` makes SUMO insert vehicles at evenly spaced, DETERMINISTIC
intervals. `sumo_seed` does not move those arrival times, so two "different
demand seeds" are the same arrival pattern. Measured on the corridor: ten seeds
gave delay per completed trip within 0.1 s and trip counts within 1 of each
other. A held-out seed that is not a different demand is not held out, and a
"+/- over seeds" computed from it is not seed variance -- which is audit defect
4 (docs/FINDINGS_2026-08-12.md) reappearing.

Corridor scenarios are therefore emitted as `period="exp(rate)"`, an exponential
headway (Poisson arrivals) that `sumo_seed` genuinely redraws.

The single-intersection files keep `vehsPerHour` deliberately. Every reported
peak and off-peak number was measured on that demand; regenerating it would
invalidate the published tables to no benefit, since those conclusions rest on
paired comparisons at a fixed demand rather than on seed variance. New work
gets the corrected form; existing results keep the demand they were measured
on. Do not "tidy" this into one rule without re-running the single-intersection
tables.

THE TIDAL SCENARIO
------------------
`corridor_peak` and `corridor_offpeak` are stationary and symmetric: eastbound
and westbound demand are equal and constant for the whole hour. One fixed
offset plan can serve that, which is the same property that made the single
intersection unwinnable -- a fixed plan is near-optimal by construction when
demand never changes shape.

`corridor_tidal` removes that property. Eastbound dominates the first half
hour, westbound the second, at the same total load. A green wave's offsets are
directional, so a single fixed plan is tuned for the wrong direction for half
the episode. Adapting to the reversal is something only a controller that
reacts (or one that has learned the reversal) can do.

Total arterial demand is held at corridor_peak's 2100 veh/h -- 1400 dominant
against 700 minor, reversing at half time -- so the two scenarios differ in
structure and not in how much traffic there is. The cross streets keep the peak
rate throughout, so the arterial reversal is the only thing that changes.

The 2:1 ratio is chosen, not arbitrary. An even green split serves roughly 1125
veh/h per approach here (2 lanes, ~1800 veh/h/lane saturation, ~5/16 duty at the
low floors), so a 1400 veh/h dominant direction is more than a fixed round-robin
can discharge while 700 wastes green on the other. That gap is the whole point:
below it, the first version of this scenario capped the dominant direction at
peak's 1050 veh/h and cut total demand by a third, and measured EASIER than peak
(green_wave 19.5 s vs 21.4 s) with the fixed-vs-reactive margin shrinking from
-2.0 s to -0.4 s. A tidal scenario that a round-robin handles comfortably tests
nothing a coordinating controller is for.

The cost of that choice: the fixed plan is expected to oversaturate in the
dominant direction, so trip counts will diverge between controllers and delay
per COMPLETED trip becomes survivorship-biased -- the metric defect this project
withdrew a headline over. analysis/corridor_sweep.py therefore reports
completion counts per cell and flags floors where the controllers complete
materially different numbers of trips. Read that flag before comparing delays.
"""
import re

SRC = "traffic.rou.xml"
FACTORS = {"traffic_peak.rou.xml": 1.5, "traffic_offpeak.rou.xml": 0.5}

CORRIDOR_SRC = "corridor.rou.xml"
CORRIDOR_FACTORS = {
    "corridor_peak.rou.xml": 1.5,
    "corridor_offpeak.rou.xml": 0.5,
}

CORRIDOR_TIDAL_DST = "corridor_tidal.rou.xml"
EPISODE_SECONDS = 3600.0
TIDAL_SWITCH_S = 1800.0

# Per-flow (first half, second half) multipliers on the corridor base rates.
# Arterial flows swap; cross streets stay at the peak rate throughout, so the
# only thing that changes across the switch is the arterial direction.
# Unlisted flow ids raise: a new flow in corridor.rou.xml must be assigned a
# side of the tide deliberately, not defaulted into one.
TIDAL_PROFILE = {
    "f_eb": (2.0, 1.0),
    "f_wb": (1.0, 2.0),
    "f_x1": (1.5, 1.5),
    "f_x2": (1.5, 1.5),
    "f_x3": (1.5, 1.5),
}

_FLOW = re.compile(r'vehsPerHour="([0-9.]+)"')
_FLOW_EL = re.compile(r"( *)<flow\b([^>]*?)/>")
_ATTR = re.compile(r'(\w+)="([^"]*)"')

SECONDS_PER_HOUR = 3600.0


def scale_file(src: str, dst: str, factor: float, stochastic: bool = False) -> None:
    with open(src) as fh:
        text = fh.read()

    def repl(m):
        rate_per_hour = float(m.group(1)) * factor
        if stochastic:
            # exp(lambda) with lambda in vehicles per SECOND -> Poisson arrivals
            return f'period="exp({rate_per_hour / SECONDS_PER_HOUR:.6f})"'
        return f'vehsPerHour="{max(1, round(rate_per_hour))}"'

    with open(dst, "w") as fh:
        fh.write(_FLOW.sub(repl, text))
    print(f"wrote {dst} (x{factor}{', stochastic arrivals' if stochastic else ''})")


def _flow_el(indent: str, attrs: dict, flow_id: str, rate_per_hour: float,
             begin: float, end: float) -> str:
    """One <flow/> element, keeping every attribute except the ones we set."""
    carried = " ".join(
        f'{k}="{v}"' for k, v in attrs.items()
        if k not in ("id", "begin", "end", "vehsPerHour", "period")
    )
    return (f'{indent}<flow id="{flow_id}" {carried} '
            f'begin="{begin:g}" end="{end:g}" '
            f'period="exp({rate_per_hour / SECONDS_PER_HOUR:.6f})"/>')


def write_tidal(src: str, dst: str) -> None:
    """Emit the tidal corridor scenario: the arterial reverses at half time.

    Each flow in `src` is looked up in TIDAL_PROFILE. A flow whose two
    multipliers differ is split into two windows (0..TIDAL_SWITCH_S and
    TIDAL_SWITCH_S..EPISODE_SECONDS); one whose multipliers match is emitted
    once, spanning the episode. Arrivals are exponential for the same reason
    the other corridor scenarios' are -- see the module docstring.

    Output is sorted by `begin`, and that is load-bearing, not cosmetic. SUMO
    reads a route file in order and silently discards any flow beginning
    earlier than one it has already read. Emitting the split flows in source
    order (eb_h1, eb_h2, wb_h1, ...) cost the whole westbound first half and
    all three cross streets -- 1219 vehicles inserted instead of 2300, no error
    raised. The flows are rewritten in place as one block, which assumes they
    are contiguous in `src`; they are in corridor.rou.xml.
    """
    with open(src) as fh:
        text = fh.read()

    matches = list(_FLOW_EL.finditer(text))
    if not matches:
        raise ValueError(f"{src} contains no <flow> elements")

    elements = []                                   # (begin, source order, xml)
    for m in matches:
        indent, raw = m.group(1), m.group(2)
        attrs = dict(_ATTR.findall(raw))
        flow_id = attrs["id"]
        first, second = TIDAL_PROFILE[flow_id]      # unlisted id -> KeyError
        base = float(attrs["vehsPerHour"])
        if first == second:
            windows = [(flow_id, base * first, 0.0, EPISODE_SECONDS)]
        else:
            windows = [
                (f"{flow_id}_h1", base * first, 0.0, TIDAL_SWITCH_S),
                (f"{flow_id}_h2", base * second, TIDAL_SWITCH_S, EPISODE_SECONDS),
            ]
        for name, rate, begin, end in windows:
            elements.append((begin, len(elements),
                             _flow_el(indent, attrs, name, rate, begin, end)))

    elements.sort(key=lambda e: (e[0], e[1]))       # stable within a window
    body = "\n".join(e[2] for e in elements)

    with open(dst, "w") as fh:
        fh.write(text[:matches[0].start()] + body + text[matches[-1].end():])
    print(f"wrote {dst} (tidal, switch at {TIDAL_SWITCH_S:g}s, stochastic arrivals)")


if __name__ == "__main__":
    for dst, factor in FACTORS.items():
        scale_file(SRC, dst, factor)
    for dst, factor in CORRIDOR_FACTORS.items():
        scale_file(CORRIDOR_SRC, dst, factor, stochastic=True)
    write_tidal(CORRIDOR_SRC, CORRIDOR_TIDAL_DST)
