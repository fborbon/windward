"""Sensor quality checks, the alert rule, and the committed sensor-watch payload and routes
(analysis/sensor_quality.py, analysis/care_sensor_watch.py); no archive needed."""
import numpy as np
from fastapi.testclient import TestClient

from analysis import sensor_quality as sq
from analysis.care_sensor_watch import alert_days, component, cusum, sensor_faults
from api.main import app
from data_sources.edp_scada import load_events, load_watch


def _series(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return 40 + 5 * np.sin(np.arange(n) / 144 * 2 * np.pi) + rng.normal(0, 0.3, n)


def test_quality_checks_find_each_fault():
    x = _series()
    ref = np.arange(len(x)) < 3000
    x[3500:3520] = x[3499]          # frozen for 20 rows
    x[3700] = x[3699] + 60          # spike
    x[3900] = 900.0                 # impossible temperature
    x[4100:4130] += np.random.default_rng(1).normal(0, 8, 30)  # erratic
    x[4300:4310] = np.nan           # gap
    c = sq.check_sensor(x, ref, "°C")
    assert c["stale"][3505:3515].all()
    assert c["step"][3700]
    assert c["out_of_range"][3900]
    assert c["erratic"][4100:4130].any()
    assert c["gap"][4300:4310].all()
    assert sq.bad_mask(c)[:3000].mean() < 0.01  # the clean reference period stays clean


def test_coarse_sensor_is_not_called_stale():
    """A whole-degree sensor repeats for hours on a calm day; that is resolution, not a fault."""
    x = np.round(_series(seed=2) / 3) * 3
    ref = np.arange(len(x)) < 3000
    assert sq.check_sensor(x, ref, "°C")["stale"].mean() < 0.01


def test_alert_rule():
    eps = np.array([[0, 3, 0, 0, 3, 3, 0, 5]])
    alert, warn = alert_days(eps, {"episodes_per_day": 3, "persist_days": 2})
    assert alert.tolist() == [[False, True, False, False, True, True, False, True]]
    # a second alert day within 2 days of another: days 5 (after 4) and 7 (after 5)
    assert warn.tolist() == [[False, False, False, False, False, True, False, True]]
    _, any_day = alert_days(eps, {"episodes_per_day": 3, "persist_days": 1})
    assert (any_day == alert).all()


def test_cusum_counts_a_steady_trickle():
    """Two episodes a day against a normal rate of 0.2 a day is invisible to a 3-a-day rule but
    adds up; the normal rate itself never warns."""
    rule = {"rate_factor": 4.0, "alpha": 0.001, "beta": 0.1}
    mon = np.full((2, 20), 144)
    base = np.array([0.2, 0.2]) / 144
    eps = np.vstack([np.full(20, 2), np.random.default_rng(0).poisson(0.2, 20)])
    stat, warn = cusum(eps, base, mon, rule)
    assert warn[0].any() and not warn[1].any()
    assert (eps[0] < 3).all()
    first = int(np.argmax(warn[0]))
    h = np.log(0.9 / 0.001)
    assert stat[0, first] >= h > stat[0, first - 1]


def test_sensor_fault_needs_a_quiet_twin():
    """A sensor leaving its peer model is a sensor fault only when a near-duplicate stays quiet;
    without one it is a 'check the sensor' note and still counts for the machine."""
    pw = np.array([[1, 1], [0, 0], [0, 0]], bool)
    no_quality = np.zeros((3, 2))
    fault, check = sensor_faults(pw, [[1], [0], [0]], [[0.99], [0.99], [0.99]], no_quality)
    assert fault[0].all() and not check[0].any()
    fault, check = sensor_faults(pw, [[2], [0], [0]], [[0.6], [0.6], [0.6]], no_quality)
    assert not fault[0].any() and check[0].all()
    fault, _ = sensor_faults(np.zeros((3, 2), bool), [[1], [0], [0]], [[0.99]] * 3, np.array([[0.5, 0], [0, 0], [0, 0]]))
    assert fault[0].tolist() == [True, False]


def test_components_from_descriptions():
    assert component("Temperature in gearbox bearing on high speed shaft") == "gearbox"
    assert component("Temperature in generator bearing 2 (Drive End)") == "generator bearing"
    assert component("Temperature in HV transformer phase L1") == "transformer"
    assert component("Temperature oil in hydraulic group") == "hydraulic"
    assert component("Windspeed") == "operating conditions"


def test_payload_covers_every_case():
    w = load_watch()
    assert w is not None
    assert len(w["cases"]) == 95
    s = w["summary"]["all"]
    assert s["faults"] == 45 and s["normal_cases"] == 50
    for c in w["cases"].values():
        if c["lead_days"] is not None:
            assert c["label"] == "anomaly" and c["lead_days"] >= 1
    leads = [t["lead_days"] for t in w["time_to_failure"]["alarmed_faults"]]
    assert len(leads) == s["faults_alarmed"]
    surv = [p["share_still_ahead"] for p in w["time_to_failure"]["survival"]]
    assert surv[0] == 1.0 and all(a >= b for a, b in zip(surv, surv[1:]))


def test_events_carry_watch_outcome():
    ev = load_events()
    assert ev["watch_alarm"].notna().all()
    assert (ev.loc[ev["watch_lead_days"].notna(), "event_label"] == "anomaly").all()


def test_routes():
    client = TestClient(app)
    r = client.get("/edp/sensor-watch")
    assert r.status_code == 200 and "cases" not in r.json()
    assert "time_to_failure" in r.json()
    first = next(iter(load_watch()["cases"]))
    r = client.get(f"/edp/sensor-watch/{first}")
    assert r.status_code == 200 and "daily" in r.json()
    assert client.get("/edp/sensor-watch/99999").status_code == 404
    e = client.get("/edp/events").json()[0]
    assert "watch_alarm" in e and "watch_lead_days" in e
