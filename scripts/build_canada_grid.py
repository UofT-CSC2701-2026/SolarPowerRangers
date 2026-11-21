"""Build web/canada-grid.json from the Natural Earth 1:50m GeoJSON layers."""

import argparse
import json
from math import floor
from pathlib import Path

from pyproj import Transformer
from shapely import box, make_valid, transform, union_all
from shapely.geometry import Point, shape
from timezonefinder import TimezoneFinder

CELL_SIZE = 300_000
DESTINATION = Path(__file__).resolve().parents[1] / "web" / "canada-grid.json"
CITIES = {
    "Vancouver": (-123.12, 49.28),
    "Calgary": (-114.07, 51.04),
    "Winnipeg": (-97.14, 49.90),
    "Toronto": (-79.38, 43.65),
    "Montréal": (-73.57, 45.50),
    "Halifax": (-63.57, 44.65),
    "Yellowknife": (-114.37, 62.45),
    "Iqaluit": (-68.52, 63.75),
}


def load_features(path):
    return json.loads(path.read_text())["features"]


def polygons(geometry):
    if geometry.geom_type == "Polygon":
        if not geometry.is_empty:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from polygons(part)


def build(countries, provinces):
    project = Transformer.from_crs(4326, 3347, always_xy=True).transform
    unproject = Transformer.from_crs(3347, 4326, always_xy=True).transform
    feature = next(
        item for item in load_features(countries)
        if item["properties"]["ADM0_A3"] == "CAN"
    )
    canada = make_valid(transform(shape(feature["geometry"]), project, interleaved=False))
    regions = [
        (item["properties"]["name"], make_valid(transform(shape(item["geometry"]), project, interleaved=False)))
        for item in load_features(provinces)
        if item["properties"]["adm0_a3"] == "CAN"
    ]
    west, south, east, north = canada.bounds
    scale = 900 / (east - west)
    padding = 25

    def point(x, y):
        return [round((x - west) * scale + padding, 1), round((north - y) * scale + padding, 1)]

    def path(geometry):
        rings = []
        for polygon in polygons(geometry.simplify(2_600, preserve_topology=True)):
            for ring in (polygon.exterior, *polygon.interiors):
                coordinates = [point(x, y) for x, y in list(ring.coords)[:-1]]
                rings.append("M" + "L".join(f"{x:g},{y:g}" for x, y in coordinates) + "Z")
        return "".join(rings)

    timezones = TimezoneFinder()
    cells = []

    def add_cell(cell_id, land, center, region):
        longitude, latitude = unproject(center.x, center.y)
        cells.append({
            "id": cell_id,
            "path": path(land),
            "center": point(center.x, center.y),
            "latitude": round(latitude, 5),
            "longitude": round(longitude, 5),
            "region": region,
            "timezone": timezones.timezone_at(lat=latitude, lng=longitude),
        })

    # A small province can otherwise disappear as a sampling location when the
    # nearest regular cell center falls on another province's land.
    small_regions = [
        (name, canada.intersection(geometry))
        for name, geometry in regions
        if geometry.area < CELL_SIZE ** 2 / 10
    ]
    regular_land = canada.difference(union_all([geometry for _, geometry in small_regions]))
    for row in range(floor(south / CELL_SIZE), floor(north / CELL_SIZE) + 1):
        for column in range(floor(west / CELL_SIZE), floor(east / CELL_SIZE) + 1):
            x, y = column * CELL_SIZE, row * CELL_SIZE
            square = box(x, y, x + CELL_SIZE, y + CELL_SIZE)
            land = regular_land.intersection(square)
            if land.area < CELL_SIZE ** 2 * 0.015:
                continue
            center = Point(x + CELL_SIZE / 2, y + CELL_SIZE / 2)
            if not land.contains(center):
                center = max(polygons(land), key=lambda part: part.area).representative_point()
            region = min(regions, key=lambda item: item[1].distance(center))[0]
            add_cell(f"{column}-{row}", land, center, region)

    for region, land in small_regions:
        if not land.is_empty:
            center = max(polygons(land), key=lambda part: part.area).representative_point()
            add_cell(region.lower().replace(" ", "-"), land, center, region)

    toronto = point(*project(*CITIES["Toronto"]))
    default = min(cells, key=lambda cell: sum((a - b) ** 2 for a, b in zip(cell["center"], toronto)))
    return {
        "viewBox": [0, 0, 950, round((north - south) * scale + padding * 2)],
        "outline": path(canada),
        "provinces": "".join(path(geometry) for _, geometry in regions),
        "cities": [{"name": name, "point": point(*project(*coords))} for name, coords in CITIES.items()],
        "cells": cells,
        "default": default["id"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("countries", type=Path)
    parser.add_argument("provinces", type=Path)
    args = parser.parse_args()
    DESTINATION.write_text(json.dumps(build(args.countries, args.provinces), separators=(",", ":")) + "\n")
