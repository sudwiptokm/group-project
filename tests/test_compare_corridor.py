"""compare.py must aggregate corridor controller CSVs into a comparison row."""
import pandas as pd

import compare


def test_run_means_reads_controller_csv(tmp_path):
    # a fake corridor eval CSV in the expected filename shape
    p = tmp_path / "eval_green_wave_corridor_peak_seed0_conn0_ep1.csv"
    pd.DataFrame({
        "system_mean_waiting_time": [1.0, 3.0],
        "system_total_stopped": [2.0, 4.0],
        "system_mean_speed": [5.0, 5.0],
        "system_total_waiting_time": [10.0, 10.0],
    }).to_csv(p, index=False)

    df = compare._run_means(str(tmp_path), "green_wave", "corridor_peak")
    assert len(df) == 1
    # metrics are time-averaged over the episode rows
    assert df["system_mean_waiting_time"].iloc[0] == 2.0
