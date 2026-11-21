"""Check the dashboard and its forecast API using fixture or live weather."""

import argparse
import json
import math
import socket
import threading
import time
from datetime import timedelta
from unittest.mock import patch
from urllib.request import urlopen


def get(url):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=45) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2)


def check(url, *, require_all_cells):
    url = url.rstrip("/")
    assert b'<html' in get(url + '/').lower(), "Dashboard page was not served"
    assert get(url + '/assets/forecast.js'), "JavaScript was not served"
    assert get(url + '/assets/forecast.css'), "Stylesheet was not served"
    health = json.loads(get(url + '/health'))
    assert health["model_loaded"] and health["status"] == "healthy", "Model failed to start"
    grid = json.loads(get(url + '/assets/canada-grid.json'))
    forecast = json.loads(get(url + '/forecast/canada'))
    grid_ids = {cell["id"] for cell in grid["cells"]}
    cells = forecast["cells"]
    assert grid_ids and cells and cells.keys() <= grid_ids, "Forecast cells do not match the map"
    if require_all_cells:
        assert cells.keys() == grid_ids, "Fixture forecasts should cover the whole map"
    assert not forecast.get("stale"), "Forecast is stale"

    hours = forecast["hour_ends"]
    assert len(hours) == 24 and all(b - a == 3600 for a, b in zip(hours, hours[1:])), "Invalid time window"
    for cell in cells.values():
        power = cell["hourly_kw"]
        assert len(power) == 24 and all(math.isfinite(value) and value >= 0 for value in power), "Invalid hourly power"
        assert math.isclose(sum(power), cell["energy_kwh"], abs_tol=0.001), "Incorrect energy total"
    assert forecast["highest_cell"] == max(cells, key=lambda cell: cells[cell]["energy_kwh"]), "Incorrect cell ranking"
    print(f"Dashboard, model startup and forecast verified for {len(cells)}/{len(grid_ids)} cells.")


def fixture_weather(cells, window_start):
    hourly = {
        "time": [int((window_start + timedelta(hours=hour)).timestamp()) for hour in range(1, 25)],
        "shortwave_radiation": [0] * 7 + [120, 240, 460, 620, 720, 680, 540, 360, 200, 70] + [0] * 7,
        "temperature_2m": [24] * 24,
        "cloud_cover": [15] * 24,
    }
    return [(cell, {"hourly": hourly}) for cell in cells]


def check_container():
    import uvicorn

    from main import app

    with socket.socket() as listener, patch("hourly_forecast.fetch_weather", fixture_weather):
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            check(f"http://127.0.0.1:{port}", require_all_cells=True)
        finally:
            server.should_exit = True
            thread.join(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Check this deployed URL using live weather")
    args = parser.parse_args()
    if args.url:
        check(args.url, require_all_cells=False)
    else:
        check_container()
