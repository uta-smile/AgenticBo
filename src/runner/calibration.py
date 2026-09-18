"""Derive a radius decision from recorded development measurements."""
import math
import statistics


def summarize_radius(records, settings):
    if len(records) != settings["samples_per_radius"]:
        raise ValueError("Calibration sample count differs from its settings")
    variations = []
    for row in records:
        if type(row.get("valid")) is not bool or type(row.get("sensible_geometry")) is not bool:
            raise ValueError("Calibration validity and geometry must be boolean measurements")
        value = row.get("center_rmsd_angstrom")
        if row["valid"]:
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError("Valid calibration samples require finite center-relative RMSD")
            variations.append(value)
        elif value is not None or row["sensible_geometry"]:
            raise ValueError("Invalid calibration samples cannot have geometry or variation measurements")
        if not math.isfinite(row["objective"]) or not 0 <= row["objective"] <= 1:
            raise ValueError("Invalid calibration objective")
    summary = {
        "valid_rate": sum(row["valid"] for row in records) / len(records),
        "sensible_geometry_rate": sum(row["sensible_geometry"] for row in records) / len(records),
        "median_center_rmsd": statistics.median(variations) if variations else None,
        "median_objective": statistics.median(row["objective"] for row in records),
    }
    summary["qualifies"] = (
        summary["valid_rate"] >= settings["min_valid_rate"]
        and summary["sensible_geometry_rate"] >= settings["min_geometry_rate"]
        and summary["median_center_rmsd"] is not None
        and summary["median_center_rmsd"] >= settings["min_median_center_rmsd_angstrom"]
    )
    return summary
