#!/usr/bin/env python3
# Written by kaustubh dwivedi (nuts & bolts version)

import json
import yaml
import math
import random
import logging
import pandas as pd
from pathlib import Path

# ------------------------------
# Load config
# ------------------------------
BASE = Path(".")
CONFIG_PATH = BASE / "config_nuts_and_bolts.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

OUT_DIR = Path(cfg.get("output_dir", "./output_nuts_and_bolts"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_HEUR = cfg.get("use_heuristics_if_missing", True)
LOG_MISSING = cfg.get("log_missing_tables", True)
TABLES_INLINE = cfg.get("tables", {})
TABLE_FILES = cfg.get("table_files", {})
DEFAULTS = cfg.get("defaults", {})

logging.basicConfig(level=getattr(logging, cfg.get("log_level", "INFO")))
logger = logging.getLogger("boltgen")

table_source_report = {}

# ------------------------------
# CSV Loader
# ------------------------------
def try_load_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if df.shape[1] < 2:
            return None
        mapping = {}
        kcol, vcol = df.columns[:2]
        for _, r in df.iterrows():
            mapping[str(r[kcol]).strip()] = r[vcol]
        return mapping
    except:
        return None

# ------------------------------
# Table Resolver
# ------------------------------
def load_table(table_name):
    inline = TABLES_INLINE.get(table_name)
    if inline:
        table_source_report[table_name] = "inline_yaml"
        return inline

    csv_path = TABLE_FILES.get(table_name)
    if csv_path:
        m = try_load_csv(csv_path)
        if m:
            table_source_report[table_name] = f"csv:{csv_path}"
            return m
        else:
            table_source_report[table_name] = f"csv_not_found:{csv_path}"

    table_source_report[table_name] = "heuristic"
    return None

def get_value(table_name, key, fallback_fn=None):
    table = load_table(table_name)
    if table and key in table:
        return table[key], table_source_report[table_name]

    if table:
        # attempt fuzzy match
        keys = list(table.keys())
        closest = min(keys, key=lambda k: abs(len(str(k)) - len(str(key))))
        return table[closest], table_source_report[table_name]

    if fallback_fn and USE_HEUR:
        return fallback_fn(key), "heuristic"

    return None, "missing"

# ------------------------------
# Heuristic Functions
# ------------------------------
def pitch_fallback(size):
    return DEFAULTS["metric_pitch_default"].get(size, 1.5)

def stress_area_fallback(size):
    return DEFAULTS["tensile_stress_area_default"].get(size, 20.0)

def bolt_grade_fallback(grade):
    return DEFAULTS["bolt_grade_defaults"].get(grade, {"yield_MPa":600, "tensile_MPa":800})

def nut_grade_fallback(grade):
    return DEFAULTS["nut_grade_defaults"].get(grade, {"proof_load_MPa":600})

def torque_coeff_fallback(type_):
    return DEFAULTS["torque_coefficient_default"].get(type_, 0.18)

def engagement_fallback(material_pair):
    return DEFAULTS["thread_engagement_defaults"].get(material_pair, 1.5)

# ------------------------------
# Computation functions
# ------------------------------
def tensile_stress_area(size):
    val, _ = get_value("tensile_stress_area", size, stress_area_fallback)
    return float(val)

def thread_pitch(size):
    val, _ = get_value("thread_pitch", size, pitch_fallback)
    return float(val)

def proof_load(grade):
    gd = nut_grade_fallback(grade)
    return gd["proof_load_MPa"]

def bolt_strength(grade):
    gd = bolt_grade_fallback(grade)
    return gd["yield_MPa"], gd["tensile_MPa"]

def tightening_torque(F_preload, d_nominal, K=0.2):
    return K * F_preload * d_nominal / 1000.0  # Nm

def preload_from_grade(tensile_strength, stress_area):
    return 0.75 * tensile_strength * stress_area  # N

def shear_capacity(stress_area, yield_MPa):
    return stress_area * yield_MPa * 0.577  # Von Mises approx.

# ------------------------------
# Main Example Generator
# ------------------------------
def generate_single_example():
    size = random.choice(["M4", "M5", "M6", "M8", "M10", "M12", "M16", "M20"])
    grade = random.choice(["8.8", "10.9", "12.9"])

    pitch = thread_pitch(size)
    A_stress = tensile_stress_area(size)

    yield_MPa, tensile_MPa = bolt_strength(grade)

    preload = preload_from_grade(tensile_MPa, A_stress)
    torque = tightening_torque(preload, d_nominal=float(size[1:]), K=torque_coeff_fallback("uncoated"))

    shear_cap = shear_capacity(A_stress, yield_MPa)

    example = {
        "bolt_spec": {
            "size": size,
            "grade": grade,
            "pitch_mm": pitch,
            "stress_area_mm2": A_stress
        },
        "calculated_values": {
            "yield_strength_MPa": yield_MPa,
            "tensile_strength_MPa": tensile_MPa,
            "recommended_preload_N": preload,
            "estimated_tightening_torque_Nm": torque,
            "shear_capacity_N": shear_cap,
            "thread_engagement_min_diameters": engagement_fallback("steel_to_steel")
        }
    }
    return example

# ------------------------------
# Dataset Generation
# ------------------------------
def generate_dataset(n=1000):
    data = []
    for i in range(n):
        data.append(generate_single_example())

    out_json = OUT_DIR / "nuts_and_bolts_dataset_final.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    report_path = OUT_DIR / "nuts_and_bolts_table_sources.json"
    with open(report_path, "w") as f:
        json.dump(table_source_report, f, indent=2)

    logger.info(f"Dataset saved to {out_json}")
    return data

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    generate_dataset(500)
