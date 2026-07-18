"""Generate peak / off-peak demand variants from traffic.rou.xml.

Only the vehsPerHour flow rates are scaled; routes, vTypeDistribution and edge
ids are left untouched (edge ids must keep matching intersection.net.xml).
"""
import re

SRC = "traffic.rou.xml"
FACTORS = {"traffic_peak.rou.xml": 1.5, "traffic_offpeak.rou.xml": 0.5}

_FLOW = re.compile(r'vehsPerHour="([0-9.]+)"')


def scale_file(src: str, dst: str, factor: float) -> None:
    with open(src) as fh:
        text = fh.read()

    def repl(m):
        return f'vehsPerHour="{max(1, round(float(m.group(1)) * factor))}"'

    with open(dst, "w") as fh:
        fh.write(_FLOW.sub(repl, text))
    print(f"wrote {dst} (x{factor})")


if __name__ == "__main__":
    for dst, factor in FACTORS.items():
        scale_file(SRC, dst, factor)
