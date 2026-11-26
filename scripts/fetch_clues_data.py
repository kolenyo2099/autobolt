#!/usr/bin/env python3
"""Fetch and compile static clue lookup tables from Oldcircle/geo-sleuth into public/data/clues_tables.json."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "public" / "data" / "clues_tables.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def gh_raw_json(path: str) -> dict:
    raw = subprocess.check_output(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/vnd.github.raw",
            f"repos/Oldcircle/geo-sleuth/contents/{path}",
        ]
    )
    return json.loads(raw.decode("utf-8"))


def strip_meta(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def simplify_prefix_entries(raw_entries: dict, limit_note: int = 120) -> dict:
    out: dict[str, list[dict[str, str]]] = {}
    for prefix, items in raw_entries.items():
        cleaned = []
        for item in items:
            entry: dict[str, str] = {}
            if item.get("admin1"):
                entry["admin1"] = str(item["admin1"])
            if item.get("admin2"):
                entry["admin2"] = str(item["admin2"])
            if item.get("city"):
                entry["city"] = str(item["city"])
            if item.get("note"):
                note = str(item["note"]).strip()
                entry["note"] = note[:limit_note] + ("…" if len(note) > limit_note else "")
            if entry:
                cleaned.append(entry)
        if cleaned:
            out[prefix] = cleaned
    return out


def main() -> None:
    print("Fetching global tables from Oldcircle/geo-sleuth...")
    calling_raw = strip_meta(gh_raw_json("skills/geo-sleuth/data/calling_codes.json"))
    driving_raw = strip_meta(gh_raw_json("skills/geo-sleuth/data/driving_side.json"))
    territories_raw = gh_raw_json("skills/geo-sleuth/data/territories.json")

    packs: dict[str, dict] = {}
    region_meta = {
        "cn": "China",
        "kr": "South Korea",
        "tr": "Turkey",
        "kz": "Kazakhstan",
        "uz": "Uzbekistan",
        "lk": "Sri Lanka",
        "kg": "Kyrgyzstan",
        "tj": "Tajikistan",
    }

    for cc, cname in region_meta.items():
        print(f"Fetching region pack: {cc.upper()} ({cname})...")
        plates_json = gh_raw_json(f"skills/geo-sleuth/regions/{cc}/data/plates.json")
        areas_json = gh_raw_json(f"skills/geo-sleuth/regions/{cc}/data/area_codes.json")
        packs[cc.upper()] = {
            "country": cname,
            "plates": simplify_prefix_entries(plates_json.get("entries", {})),
            "areaCodes": simplify_prefix_entries(areas_json.get("entries", {})),
        }

    territories_by_sovereign = {}
    for sov, items in territories_raw.get("by_sovereign", {}).items():
        territories_by_sovereign[sov] = [
            {
                "name": str(it.get("name", "")),
                "region": str(it.get("region", "")),
                "subregion": str(it.get("subregion", "")),
                "status": str(it.get("status", "")),
            }
            for it in items
        ]

    bundle = {
        "_meta": {
            "source": "Oldcircle/geo-sleuth (Wikipedia CC BY-SA 4.0 & MIT)",
            "generatedBy": "scripts/fetch_clues_data.py",
        },
        "callingCodes": calling_raw,
        "drivingSide": driving_raw,
        "territories": territories_by_sovereign,
        "regionPacks": packs,
    }

    OUT_PATH.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_kb = OUT_PATH.stat().st_size / 1024.0
    print(
        f"Wrote {OUT_PATH} ({size_kb:.1f} KB): "
        f"{len(calling_raw)} calling codes, {len(driving_raw)} driving sides, "
        f"{len(territories_by_sovereign)} sovereigns, {len(packs)} region packs."
    )


if __name__ == "__main__":
    main()
