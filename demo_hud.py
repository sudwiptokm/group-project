"""Live metric overlay for the sumo-gui demo.

Draws a few text lines on the sumo-gui map (as POIs; sumo-gui draws a POI's ID as its label
(poiName_show in gui-settings.xml)) and refreshes them every decision step, so a side-by-side
run shows the controller name and its running metrics inside each window.

Metrics come from the info dict sumo-rl returns from env.step():
  system_mean_waiting_time  mean wait of vehicles currently in the network
  system_total_stopped      vehicles at speed < 0.1 m/s
  system_mean_speed         mean speed of vehicles in the network
TRIPS DONE and AVG DELAY/TRIP come from TripTracker: vehicles that actually reached
their destination, and their mean SUMO timeLoss. These are the fair scoreboard -- a
controller cannot improve them by stranding traffic. The in-network average wait is
kept on the last line but labelled biased (it resets whenever a car creeps forward).
The last line is the running mean of system_mean_waiting_time (what compare.py
time-averages). Only used under --gui.
"""
import os
import time

LINE_GAP = 22.0   # metres between text lines on the map


class TripTracker:
    """Completed-trip count and mean delay (SUMO timeLoss), tracked live.

    Hooks env._sumo_step so it runs once per simulation second. A vehicle's timeLoss can no
    longer be read once it arrives, so each second's value is cached and the cached one
    (at most 1 s stale) is banked when the vehicle shows up in the arrived list. This is the
    same quantity demo_summary.py reads from the tripinfo file, but available while running.
    """

    def __init__(self, env):
        self.sumo = env.sumo
        self.done, self.total_loss, self._last = 0, 0.0, {}
        orig = env._sumo_step

        def counting_step():
            orig()
            for vid in self.sumo.simulation.getArrivedIDList():
                self.total_loss += self._last.pop(vid, 0.0)
                self.done += 1
            veh = self.sumo.vehicle
            self._last = {v: veh.getTimeLoss(v) for v in veh.getIDList()}
        env._sumo_step = counting_step

    @property
    def mean_delay(self) -> float:
        return self.total_loss / self.done if self.done else 0.0


class Hud:
    def __init__(self, env, title: str, subtitle: str = "", x_off: float = 130.0):
        self.env, self.title, self.subtitle = env, title, subtitle
        self.sumo = env.sumo
        self.sum_wait, self.n = 0.0, 0
        self.trips = TripTracker(env)
        (x0, y0), (x1, y1) = self.sumo.simulation.getNetBoundary()
        self.n_lines = 8
        self.x = x0 + x_off   # clear of the legend bar; labels are centred on this x
        self.top = y1 - 10   # inside the net bounds, top-left corner (no road there)
        # sumo-gui labels a POI with its ID, so each line's text IS the ID of its POI
        # and a refresh means remove + re-add at the same spot
        self.ids = [None] * self.n_lines
        # frame the whole junction so the text is on screen
        self.sumo.gui.setSchema("View #0", "hetero-demo")   # settings file loads it but does not activate it
        self.sumo.gui.setBoundary("View #0", x0 - 20, y0 - 20, x1 + 20, y1 + 20)
        self._set(0, self.title)
        self._set(1, self.subtitle)

    def _set(self, i, text):
        text = text.strip() or f"-{i}"
        if text == self.ids[i]:
            return
        if self.ids[i] is not None:
            self.sumo.poi.remove(self.ids[i])
        self.sumo.poi.add(text, self.x, self.top - i * LINE_GAP,
                          (255, 220, 80, 255), width=1, height=1)
        self.ids[i] = text

    def update(self, info: dict, done: bool = False):
        w = float(info.get("system_mean_waiting_time", 0.0))
        self.sum_wait += w
        self.n += 1
        t = int(self.sumo.simulation.getTime())
        self._set(2, f"sim time {t // 60:02d}:{t % 60:02d}")
        self._set(3, f"waiting now {w:5.1f} s   stopped {int(info.get('system_total_stopped', 0))}")
        self._set(4, f"in network {self.sumo.vehicle.getIDCount()}   mean speed {float(info.get('system_mean_speed', 0.0)):4.1f} m/s")
        tag = "FINAL " if done else ""
        self._set(5, f"{tag}TRIPS DONE {self.trips.done}  (higher = better)")
        self._set(6, f"{tag}AVG DELAY/TRIP {self.trips.mean_delay:5.1f} s  (lower = better)")
        self._set(7, f"avg wait, in-network only {self.sum_wait / self.n:5.1f} s  (biased)")

    def hold(self):
        """Keep the final numbers on screen (quit-on-end closes the window at env.close())."""
        secs = int(os.environ.get("DEMO_HOLD", "600"))
        print(f"[{self.title}] done -- holding window {secs}s (Ctrl-C to close)")
        try:
            # sumo-gui (--quit-on-end) closes itself if TraCI goes quiet after the last step,
            # so keep pinging it rather than sleeping
            for _ in range(secs * 5):
                time.sleep(0.2)
                self.sumo.simulation.getTime()
        except KeyboardInterrupt:
            pass
        except Exception as e:   # sumo-gui was closed by hand: not an error
            print(f"[{self.title}] window closed ({type(e).__name__})")
