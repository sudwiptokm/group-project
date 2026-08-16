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
"""
import re

SRC = "traffic.rou.xml"
FACTORS = {"traffic_peak.rou.xml": 1.5, "traffic_offpeak.rou.xml": 0.5}

CORRIDOR_SRC = "corridor.rou.xml"
CORRIDOR_FACTORS = {
    "corridor_peak.rou.xml": 1.5,
    "corridor_offpeak.rou.xml": 0.5,
}

_FLOW = re.compile(r'vehsPerHour="([0-9.]+)"')

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


if __name__ == "__main__":
    for dst, factor in FACTORS.items():
        scale_file(SRC, dst, factor)
    for dst, factor in CORRIDOR_FACTORS.items():
        scale_file(CORRIDOR_SRC, dst, factor, stochastic=True)
