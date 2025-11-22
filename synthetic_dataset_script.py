#!/usr/bin/env python3
# written by kaustubh Dwivedi (updated)
"""
generate_gear_dataset_with_more_params.py (config-driven)

Loads config.yaml (hybrid: inline tables + CSV paths). If a table value exists inline,
it will be used. Otherwise the script will try the CSV path. If neither present,
it falls back to heuristics (as before).

Run:
  python generate_gear_dataset_with_more_params.py
"""

import os
import math
import json
import random
from pathlib import Path
import pandas as pd
import yaml
import logging
from collections import defaultdict

# -----------------------------
# Configuration & load YAML
# -----------------------------
BASE_DIR = Path(".").resolve()
CONFIG_PATH = BASE_DIR / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}. Create it first.")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

USE_HEUR = cfg.get("use_heuristics_if_missing", True)
LOG_MISSING = cfg.get("log_missing_tables", True)
PDF_PATH = cfg.get("source_pdf", "/mnt/data/Roadmap_Gear Design.pdf")
OUT_DIR = Path(cfg.get("output_dir", "./output"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# setup logging
log_level = cfg.get("log_level", "INFO")
logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("geargen")

# report of table sources
table_source_report = {}

# convenience shortcuts to YAML sections
TABLES_INLINE = cfg.get("tables", {}) or {}
TABLE_FILE_PATHS = cfg.get("table_files", {}) or {}
DEFAULTS = cfg.get("defaults", {})

# random seed & dataset size
RANDOM_SEED = 1234
NUM_EXAMPLES = cfg.get("samples")
random.seed(RANDOM_SEED)

# -----------------------------
# Utility helpers
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))
def try_load_csv_to_mapping(csv_path: str):
    """Try to load a simple CSV into a mapping/dict if it has two columns (key,value)."""
    p = Path(csv_path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        logger.warning(f"Failed to read CSV {csv_path}: {e}")
        return None
    # attempt to find two numeric-like columns
    cols = list(df.columns)
    if len(cols) >= 2:
        kcol, vcol = cols[0], cols[1]
        mapping = {}
        for _, r in df.iterrows():
            k = r[kcol]
            v = r[vcol]
            try:
                ki = int(k) if float(k).is_integer() else float(k)
            except Exception:
                ki = str(k).strip()
            try:
                mapping[ki] = float(v)
            except Exception:
                mapping[ki] = v
        return mapping
    return None

def load_table_from_config(table_name):
    """Return either inline mapping, CSV mapping, or None."""
    # 1) inline table in YAML
    inline = TABLES_INLINE.get(table_name)
    if inline:
        table_source_report[table_name] = "yaml_inline"
        return inline
    # 2) CSV file path
    csv_path = TABLE_FILE_PATHS.get(table_name)
    if csv_path:
        mapping = try_load_csv_to_mapping(csv_path)
        if mapping:
            table_source_report[table_name] = f"csv:{csv_path}"
            return mapping
        else:
            table_source_report[table_name] = f"csv_not_readable:{csv_path}"
            return None
    # 3) not found
    table_source_report[table_name] = "not_provided"
    return None

def get_table_or_heuristic(table_name, key, heuristic_fn):
    """
    Generic accessor:
      - If inline table value present for key -> return it
      - Else if CSV mapping has nearest/exact key -> return it
      - Else return heuristic_fn(key) if USE_HEUR True
    Also logs which source used.
    """
    tbl = load_table_from_config(table_name)
    # exact match in inline or csv mapping
    if tbl:
        # if key is numeric, try numeric lookup, else string based
        if isinstance(key, (int,float)) and key in tbl:
            source = table_source_report.get(table_name, "csv_inline")
            logger.debug(f"Table {table_name}: exact match for {key} from {source}")
            return tbl[key], source
        # if mapping keys numeric and no exact match, choose nearest numeric
        try:
            numeric_keys = [k for k in tbl.keys() if isinstance(k, (int,float))]
            if numeric_keys and isinstance(key, (int,float)):
                nearest = min(numeric_keys, key=lambda k: abs(k - key))
                source = table_source_report.get(table_name, "csv_inline")
                logger.debug(f"Table {table_name}: nearest match {nearest} for {key} from {source}")
                return tbl[nearest], source
        except Exception:
            pass
    # fallback to heuristic
    if USE_HEUR:
        val = heuristic_fn(key)
        table_source_report[table_name] = table_source_report.get(table_name, "heuristic")
        logger.debug(f"Table {table_name}: using heuristic for key={key}, val={val}")
        return val, "heuristic"
    else:
        logger.warning(f"Table {table_name} missing and heuristics disabled.")
        return None, "missing"

# -----------------------------
# Heuristics (same as earlier, but now used via get_table_or_heuristic)
# -----------------------------
def lewis_Y_heuristic(z):
    z = max(6, int(round(z)))
    if z <= 10:
        base = 0.40 - 0.01*(z - 6)
    elif z <= 20:
        base = 0.36 - 0.005*(z - 10)
    elif z <= 40:
        base = 0.31 - 0.003*(z - 20)
    else:
        base = 0.25 - 0.001*(z - 40)
    return round(max(DEFAULTS.get("lewis_Y_default_min", 0.06), base), 5)

def Kb_heuristic(face_width_mm, module_mm):
    ratio = max(1.0, face_width_mm / max(0.1, module_mm))
    if ratio < 8:
        return 1.25
    elif ratio < 12:
        return 1.12
    elif ratio < 20:
        return 1.03
    else:
        return 1.0

def Kv_heuristic(speed_rpm):
    # read breakpoints from defaults if present
    bps = DEFAULTS.get("dynamic_default_breakpoints", [])
    if not bps:
        # fallback to simple
        if speed_rpm < 200: return 1.0
        if speed_rpm < 1000: return 1.1
        if speed_rpm < 2000: return 1.2
        return 1.35
    # bps is list of dicts {"speed_rpm": x, "Kv": y}
    kv = None
    for bp in sorted(bps, key=lambda d: d["speed_rpm"]):
        if speed_rpm <= bp["speed_rpm"]:
            kv = bp["Kv"]
            break
    if kv is None:
        kv = bps[-1]["Kv"]
    return kv

def Yj_heuristic(z, helix_angle_deg=0.0):
    base = 0.9 - 0.002*(z/5.0)
    if helix_angle_deg:
        base *= (1.0 + 0.005 * math.sin(math.radians(helix_angle_deg)))
    return round(max(0.5, base), 4)

def zE_heuristic(hv=200):
    return round(DEFAULTS.get("zE_default", 1900.0) * (1.0 + 0.001*(hv - 200)), 2)

def zN_heuristic(cycles):
    if cycles < 1e5: return 1.0
    if cycles < 1e6: return 0.95
    if cycles < 1e7: return 0.88
    return 0.8

# -----------------------------
# Wrapper helpers used in generator
# -----------------------------
def get_lewis_Y(z, pressure_angle_deg=20, gear_type='spur'):
    val, src = get_table_or_heuristic("Lewis_Y", z, lewis_Y_heuristic)
    # small helical adjustment if needed
    if gear_type == 'helical':
        val = val * 1.05
    return val, src

def get_Kb(face_width_mm, module_mm):
    # Kb table rarely directly tabulated; use heuristic or CSV if provided
    # We'll attempt to find "Ks" or "Kb" in config; fallback to heuristic
    val, src = get_table_or_heuristic("Ks", face_width_mm / max(0.1, module_mm), lambda k: Kb_heuristic(face_width_mm, module_mm))
    return val, src

def get_Kv(speed_rpm):
    val, src = get_table_or_heuristic("Kv", speed_rpm, Kv_heuristic)
    return val, src

def get_Yj(z, helix_angle_deg):
    val, src = get_table_or_heuristic("Geometry_J", z, lambda k: Yj_heuristic(z, helix_angle_deg))
    return val, src

def get_zE(hv):
    val, src = get_table_or_heuristic("Ze", hv, zE_heuristic)
    return val, src

def get_zN(cycles):
    val, src = get_table_or_heuristic("Zn", cycles, zN_heuristic)
    return val, src

# -----------------------------
# Generator (adapted from earlier code)
# -----------------------------
def pitch_diameter(m_mm, z): return m_mm * z
def center_distance_from_d(d1_mm, d2_mm): return 0.5 * (d1_mm + d2_mm)
def circular_pitch_mm(m_mm): return math.pi * m_mm

def recommend_face_width(module_mm, power_kw=None, torque_Nm=None, dp_mm=None):
    base = module_mm * 10.0
    if torque_Nm and dp_mm:
        q = torque_Nm / max(1.0, dp_mm)
        factor = clamp(1 + (q / 500.0), 0.8, 2.5)
        return round(base * factor, 3)
    return round(base, 3)

def recommend_lubrication(torque_Nm, speed_rpm, temperature_c=40):
    if torque_Nm < 50 and speed_rpm > 1000: return "ISO VG 150"
    if torque_Nm < 200: return "ISO VG 220"
    if torque_Nm < 1000: return "ISO VG 320"
    return "ISO VG 460 (heavy)"

def helical_z_eq(z, helix_angle_deg):
    psi = math.radians(helix_angle_deg)
    c = math.cos(psi)
    if c <= 0.1: return z
    return max(8, int(round(z / (c**3))))

def contact_stress(Ft_N, b_mm, d_mm, zE):
    if b_mm <= 0 or d_mm <= 0 or Ft_N <= 0: return None
    val = zE * math.sqrt(Ft_N / (b_mm * d_mm))
    return round(val, 3)

def estimated_life_hours(safety_factor_b, speed_rpm, duty_cycle=0.5):
    if safety_factor_b is None: safety_factor_b = 1.0
    life = 10000 * safety_factor_b / (1 + speed_rpm/1000.0) * duty_cycle
    return int(max(10, life))

# Material DB (same as before)
MATERIAL_DB = [
    {"name":"AISI 4140 Alloy Steel (QT)", "bending_allowable_MPa":180, "contact_allowable_MPa":950, "typical_HV":250},
    {"name":"42CrMo4 (QT)", "bending_allowable_MPa":200, "contact_allowable_MPa":1000, "typical_HV":260},
    {"name":"C45/1045 (through-hardened)", "bending_allowable_MPa":150, "contact_allowable_MPa":900, "typical_HV":200},
    {"name":"Case carburized (20MnCr5)", "bending_allowable_MPa":260, "contact_allowable_MPa":1300, "typical_HV":550},
    {"name":"18CrNiMo7 (through-hardened)", "bending_allowable_MPa":220, "contact_allowable_MPa":1100, "typical_HV":300}
]

def pick_material(est_b, est_c):
    for m in MATERIAL_DB:
        if m['bending_allowable_MPa'] >= (est_b * 1.2 if est_b else 0) and m['contact_allowable_MPa'] >= (est_c * 1.2 if est_c else 0):
            return m
    return MATERIAL_DB[-1]


def contact_ratio_estimate(m_mm, z1, z2, pressure_angle_deg=20, helix_angle_deg=0):
    # simple fallback heuristic
    base = 1.0 + 0.02 * (min(z1, z2) / 10.0)
    helical_boost = 1.0 + 0.02 * math.tan(math.radians(helix_angle_deg))
    return round(base * helical_boost, 3)
def overlap_ratio(m_mm, z1, z2, face_width_mm, pressure_angle_deg=20, helix_angle_deg=0):
    base = 1.0 + (face_width_mm / (10.0 * m_mm)) * 0.05
    heli = 1.0 + 0.02 * math.tan(math.radians(helix_angle_deg))
    teeth_factor = min(z1, z2) / 40.0
    return round(base * heli * (1 + teeth_factor), 3)
# Generator single example
def generate_single_example():
    gear_type = random.choice(["spur", "helical", "bevel", "worm"])
    module_mm = float(random.choice([0.5,0.8,1.0,1.25,1.5,2.0,2.5,3.0,4.0,5.0]))
    if gear_type == "worm":
        z_pinion = random.randint(1,4)
        z_gear = random.randint(30, 300)
    elif gear_type == "bevel":
        z_pinion = random.randint(12, 36)
        z_gear = random.randint(12, 60)
    else:
        z_pinion = random.randint(12, 40)
        ratio = random.choice([1,2,3,4,5])
        z_gear = max(z_pinion, z_pinion * ratio)
    helix_angle_deg = float(random.choice([0,8,15,20,25])) if gear_type == "helical" else 0.0
    pressure_angle_deg = random.choice([14.5, 20, 25])
    face_width_mm = recommend_face_width(module_mm, torque_Nm=None, dp_mm=None)
    face_width_mm = round(face_width_mm * random.uniform(0.8, 1.4), 3)
    addendum_mm = round(module_mm, 4)
    dedendum_mm = round(module_mm * 1.25, 4)
    dp_pinion = pitch_diameter(module_mm, z_pinion)
    dp_gear = pitch_diameter(module_mm, z_gear)
    center_distance_mm = round(center_distance_from_d(dp_pinion, dp_gear), 4)
    torque_Nm = round(random.uniform(2.0, 8000.0), 4)
    speed_rpm = int(random.uniform(5, 4000))
    power_kw = round((torque_Nm * speed_rpm * 2 * math.pi / 60.0) / 1000.0, 4)
    Ft_N = round((2.0 * torque_Nm * 1000.0) / dp_pinion if dp_pinion>0 else 0.0, 4)
    hv_guess = random.choice([200, 250, 300, 350, 550])
    # get Lewis Y (table -> csv -> heuristic)
    if gear_type == "helical":
        z_eq = helical_z_eq(z_pinion, helix_angle_deg)
        Y_val, Y_src = get_lewis_Y(z_eq, pressure_angle_deg, 'helical')
    else:
        Y_val, Y_src = get_lewis_Y(z_pinion, pressure_angle_deg, 'spur')
    Kb_val, Kb_src = get_Kb(face_width_mm, module_mm)
    Yj_val, Yj_src = get_Yj(z_pinion, helix_angle_deg)
    Kv_val, Kv_src = get_Kv(speed_rpm)
    sigma_b = None
    if Ft_N and face_width_mm and module_mm and Y_val:
        sigma_basic = Ft_N / (face_width_mm * module_mm * Y_val)
        sigma_b = round(sigma_basic * Kb_val * Kv_val / Yj_val, 3)
    zE_val, zE_src = get_zE(hv_guess)
    sigma_c = contact_stress(Ft_N, face_width_mm, dp_pinion if dp_pinion>0 else dp_gear, zE_val)
    cycles_est = speed_rpm * 60 * 10000
    zN_val, zN_src = get_zN(cycles_est)
    material = pick_material(sigma_b if sigma_b else 0.0, sigma_c if sigma_c else 0.0)
    sf_b = round(material['bending_allowable_MPa'] / sigma_b, 3) if sigma_b and sigma_b>0 else None
    sf_c = round(material['contact_allowable_MPa'] / sigma_c, 3) if sigma_c and sigma_c>0 else None
    # contact ratio & overlap using earlier heuristics
    contact_ratio = contact_ratio_estimate(module_mm, z_pinion, z_gear, pressure_angle_deg=pressure_angle_deg, helix_angle_deg=helix_angle_deg)
    overlap = overlap_ratio(module_mm, z_pinion, z_gear, face_width_mm, pressure_angle_deg, helix_angle_deg)
    lubrication = recommend_lubrication(torque_Nm, speed_rpm)
    heat_treatment = "case-carburized" if material['typical_HV'] > 500 else "through-hardened" if material['typical_HV'] >= 250 else "annealed/normalized"
    life_hours = estimated_life_hours(sf_b if sf_b else 1.0, speed_rpm)
    standard_modules = [0.5,0.8,1.0,1.25,1.5,2.0,2.5,3.0,4.0,5.0]
    module_candidates = [m for m in standard_modules if 0.8*module_mm <= m <= 1.2*module_mm]
    if not module_candidates:
        module_candidates = [module_mm]
    example = {
        "gear_design": {
            # "source_pdf": str(PDF_PATH),
            "gear_type": gear_type,
            "module_mm": module_mm,
            "number_of_teeth_pinion": int(z_pinion),
            "number_of_teeth_gear": int(z_gear),
            "pitch_diameter_pinion_mm": round(dp_pinion, 4),
            "pitch_diameter_gear_mm": round(dp_gear, 4),
            "face_width_mm": round(face_width_mm, 4),
            "addendum_mm": addendum_mm,
            "dedendum_mm": dedendum_mm,
            "pressure_angle_deg": pressure_angle_deg,
            "center_distance_mm": center_distance_mm,
            "input_torque_Nm": torque_Nm,
            "input_speed_rpm": speed_rpm,
            "input_power_kW": power_kw,
            "helix_angle_deg": helix_angle_deg if helix_angle_deg else None,
            "recommended_face_width_mm": recommend_face_width(module_mm, torque_Nm, torque_Nm, dp_pinion),
            "module_candidates": module_candidates,
            "lubrication_recommendation": lubrication
        },
        "calculated_values": {
            "tangential_force_N": Ft_N,
            "used_lewis_Y": Y_val,
            "used_lewis_Y_source": Y_src,
            "used_Kb": Kb_val,
            "used_Kb_source": Kb_src,
            "used_Yj": Yj_val,
            "used_Yj_source": Yj_src,
            "used_Kv": Kv_val,
            "used_Kv_source": Kv_src,
            "used_zE": zE_val,
            "used_zE_source": zE_src,
            "used_zN": zN_val,
            "used_zN_source": zN_src,
            "estimated_bending_stress_MPa": sigma_b,
            "estimated_contact_stress_MPa": sigma_c,
            "contact_ratio": contact_ratio,
            "overlap_ratio": overlap,
            "recommended_material": material['name'],
            "allowable_bending_stress_MPa": material['bending_allowable_MPa'],
            "allowable_contact_stress_MPa": material['contact_allowable_MPa'],
            "estimated_bending_safety_factor": sf_b,
            "estimated_contact_safety_factor": sf_c,
            "estimated_life_hours": life_hours,
            "recommended_heat_treatment": heat_treatment,
            "estimated_surface_hardness_HV": hv_guess
        }
    }
    return example

# -----------------------------
# Dataset generator & save
# -----------------------------
def generate_dataset(n=NUM_EXAMPLES, out_dir=OUT_DIR):
    examples = []
    for i in range(n):
        ex = generate_single_example()
        examples.append(ex)
        if (i+1) % 100 == 0:
            logger.info(f"Generated {i+1}/{n}")
    jpath = out_dir / "synthetic_gear_design_dataset.json"
    with open(jpath, "w") as f:
        json.dump(examples, f, indent=2)
    jlpath = out_dir / "synthetic_gear_design_dataset.jsonl"
    with open(jlpath, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    rows = []
    for e in examples:
        row = {}
        for k, v in e['gear_design'].items():
            row[f"gd_{k}"] = v
        for k, v in e['calculated_values'].items():
            row[f"calc_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    cpath = out_dir / "synthetic_gear_design_dataset.csv"
    df.to_csv(cpath, index=False)
    # Save config report/log
    report_path = out_dir / "config_table_source_report.json"
    with open(report_path, "w") as rp:
        json.dump(table_source_report, rp, indent=2)
    # README
    readme_path = out_dir / "README_dataset_notes.txt"
    with open(readme_path, "w") as f:
        f.write("Synthetic Gear Dataset - Detailed Notes\n")
        f.write("======================================\n\n")
        f.write(f"Config used: {CONFIG_PATH}\n\n")
        f.write("This dataset was generated using a blend of heuristics and approximations.\n")
        f.write("Table source report (saved in config_table_source_report.json) indicates which tables\n")
        f.write("were loaded from YAML, CSV, or heuristics.\n\n")
        f.write("Key assumptions and formulas used:\n")
        f.write("- Ft (N) = 2*T(Nm)*1000 / d(mm)\n")
        f.write("- Bending stress (approx): sigma_b = Ft/(b * m * Y) * Kb * Kv / Yj\n")
        f.write("- Contact stress (heuristic): sigma_c = zE * sqrt( Ft/(b * d) )\n")
        f.write("- Lewis Y: from YAML/CSV or heuristic\n")
        f.write("- Kb: from YAML/CSV or heuristic\n")
        f.write("- Kv: from YAML/CSV or heuristic\n")
        f.write("- zE, zN: from YAML/CSV or heuristic\n")
        f.write("- Helical equivalent teeth: z_eq = z / cos^3(psi)\n\n")
        f.write("Caveats:\n- These outputs are for ML dataset generation and prototyping. Replace the heuristics with exact\n  AGMA/ISO tables and run FEA + standards checks for real-world designs.\n")
    logger.info("Saved dataset JSON/JSONL/CSV and README to %s", out_dir)
    logger.info("Table source report saved to %s", report_path)
    return {"json": str(jpath), "jsonl": str(jlpath), "csv": str(cpath), "readme": str(readme_path), "report": str(report_path)}

# -----------------------------
# Run as script
# -----------------------------
if __name__ == "__main__":
    logger.info("Generating synthetic gear dataset with config-driven tables...")
    out_files = generate_dataset(NUM_EXAMPLES)
    logger.info("Done. Files: %s", out_files)
