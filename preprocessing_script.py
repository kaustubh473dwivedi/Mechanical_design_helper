#!/usr/bin/env python3
#Kaustubh Dwivedi
#this file is for gears
"""
create_gear_manifest.py

Generates a LLaMA/Alpaca-style training manifest (instruction/output JSONL)
from an existing synthetic gear dataset JSON.

Outputs:
  ./output/gear_training_manifest.jsonl      (full manifest)
  ./output/manifest_sample.jsonl            (first 50 entries for quick check)

Provenance: uses /mnt/data/Roadmap_Gear Design.pdf (kept in prompts/metadata)

Usage:
  python create_gear_manifest.py
"""
import json
import random
from pathlib import Path
from typing import Dict, Any, List

random.seed(12345)

# ---------- Config ----------
INPUT_CANDIDATES = [
    
Path("./output_final_gears/synthetic_gear_design_dataset.json")

]
OUT_DIR = Path("./trainable_input_gears")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "gear_training_manifest.jsonl"
SAMPLE_PATH = OUT_DIR / "manifest_sample.jsonl"

# Provenance file (the PDF you uploaded) — included in some prompts
PROVENANCE_PDF_PATH = "/mnt/data/Roadmap_Gear Design.pdf"

# Practical requirement patterns (curated). Each pattern describes which fields to include.
# Keys prefixed with "gd:" come from gear_design; "cv:" from calculated_values.
PATTERNS = [
    ("torque_speed_type", ["gd:gear_type", "gd:input_torque_Nm", "gd:input_speed_rpm"]),
    ("torque_contact_req", ["gd:input_torque_Nm", "cv:estimated_contact_stress_MPa"]),
    ("contact_and_bending_limits", ["cv:estimated_contact_stress_MPa", "cv:estimated_bending_stress_MPa"]),
    ("target_safety_factor", ["cv:estimated_bending_safety_factor", "cv:estimated_contact_safety_factor"]),
    ("module_and_ratio", ["gd:module_mm", "gd:number_of_teeth_pinion", "gd:number_of_teeth_gear"]),
    ("center_distance_and_torque", ["gd:center_distance_mm", "gd:input_torque_Nm"]),
    ("life_and_contact", ["cv:estimated_life_hours", "cv:estimated_contact_safety_factor"]),
    ("material_pref_and_torque", ["cv:recommended_material", "gd:input_torque_Nm"]),
    ("geometry_only", ["gd:center_distance_mm", "gd:module_mm", "gd:face_width_mm"]),
    ("module_and_pressure_angle", ["gd:module_mm", "gd:pressure_angle_deg"]),
    ("facewidth_and_lubrication", ["gd:face_width_mm", "gd:lubrication_recommendation"]),
    ("ratio_and_speed", ["gd:number_of_teeth_pinion", "gd:number_of_teeth_gear", "gd:input_speed_rpm"])
]

# How many QA variants per example (random between __)
MIN_PAIRS = 3
MAX_PAIRS = 8

# Utility - turn a key spec into value from sample (if present)
def get_field(sample: Dict[str, Any], key_spec: str):
    """
    key_spec format: 'gd:field' or 'cv:field'
    Returns (present:Boolean, field_name, value)
    """
    part, field = key_spec.split(":", 1)
    block = "gear_design" if part == "gd" else "calculated_values"
    if block in sample and field in sample[block]:
        return True, f"{part}:{field}", sample[block][field]
    return False, f"{part}:{field}", None

# Natural-language composition helpers
def fmt_val(v):
    if v is None:
        return "unspecified"
    if isinstance(v, float):
        # short float display
        return f"{v:.4g}"
    return str(v)

def build_requirement_sentence(sample: Dict[str, Any], chosen_keys: List[str]) -> str:
    # build a natural, professional instruction using the chosen keys and their values
    parts = []
    for ks in chosen_keys:
        ok, kname, val = get_field(sample, ks)
        if not ok:
            continue
        # humanize
        if ks.startswith("gd:"):
            label = kname.split(":",1)[1].replace("_", " ")
            parts.append(f"{label} = {fmt_val(val)}")
        else:
            label = kname.split(":",1)[1].replace("_", " ")
            # for calculated values, phrase as "target" or "required"
            parts.append(f"{label} <= {fmt_val(val)}")
    if not parts:
        return "Design a gear meeting typical industrial requirements."
    # join elegantly
    if len(parts) == 1:
        body = parts[0]
    elif len(parts) == 2:
        body = " and ".join(parts)
    else:
        body = ", ".join(parts[:-1]) + ", and " + parts[-1]
    # Full instruction
    return f"Design a gear that satisfies the following requirements: {body}. Provide the complete gear parameters (module, teeth counts, diameters, face width, addendum/dedendum, pressure angle, center distance, lubrication) as a JSON gear_design object."

def preserve_given_gd_fields(requirement_keys: List[str], sample_gd: Dict[str,Any]) -> Dict[str,Any]:
    """
    If the requirement contained any gear_design keys, those should appear exactly as given.
    We'll produce full gear_design by starting from sample_gd (original) — since instructions are generated from that sample,
    this ensures given fields remain unchanged.
    """
    # in our generation we will simply output sample_gd (complete) — because instruction values are drawn from it.
    # this function is a placeholder for more advanced merging if needed.
    return sample_gd.copy()

def create_pairs_for_sample(sample: Dict[str,Any], examples_per_sample: int) -> List[Dict[str,str]]:
    pairs = []
    # ensure we only use patterns that have at least one present field in sample
    valid_patterns = []
    for name, keys in PATTERNS:
        has_any = any(get_field(sample, k)[0] for k in keys)
        if has_any:
            valid_patterns.append((name, keys))
    if not valid_patterns:
        # fallback: simple torque+speed if present otherwise full geometry
        if get_field(sample, "gd:input_torque_Nm")[0] and get_field(sample, "gd:input_speed_rpm")[0]:
            valid_patterns.append(("torque_speed_type", ["gd:input_torque_Nm","gd:input_speed_rpm"]))
        else:
            valid_patterns.append(("geometry_only", ["gd:module_mm","gd:face_width_mm","gd:center_distance_mm"]))
    # sample patterns randomly (with replacement allowed) up to examples_per_sample
    for _ in range(examples_per_sample):
        pattern_name, keys = random.choice(valid_patterns)
        instruction = build_requirement_sentence(sample, keys)
        # output must be gear_design only (JSON string). Use the sample's gear_design as the authoritative result.
        gear_design = preserve_given_gd_fields(keys, sample.get("gear_design", {}))
        output_json_string = json.dumps({"gear_design": gear_design}, separators=(',', ':'))
        pairs.append({"instruction": instruction, "output": output_json_string})
    return pairs

def find_input_file() -> Path:
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"No input synthetic dataset found. Tried: {INPUT_CANDIDATES}")

def main():
    input_path = find_input_file()
    print("Using input dataset:", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    manifest_lines = []
    for i, sample in enumerate(samples):
        # decide how many QA pairs for this sample: practical set between MIN_PAIRS..MAX_PAIRS
        n = random.randint(MIN_PAIRS, MAX_PAIRS)
        pairs = create_pairs_for_sample(sample, n)
        manifest_lines.extend(pairs)
    # save manifest as LLaMA/Alpaca style (instruction/output)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as out:
        for item in manifest_lines:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    # write a small sample file
    with open(SAMPLE_PATH, "w", encoding="utf-8") as s:
        for item in manifest_lines[:50]:
            s.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote manifest ({len(manifest_lines)} entries) to {MANIFEST_PATH}")
    print(f"Wrote sample (first 50) to {SAMPLE_PATH}")
    # provenance note
    prov_path = OUT_DIR / "manifest_provenance.txt"
    prov_path.write_text(f"Input dataset: {input_path}\nSource PDF: {PROVENANCE_PDF_PATH}\nPatterns used: {len(PATTERNS)}\n")
    print("Provenance written to:", prov_path)

if __name__ == "__main__":
    main()

