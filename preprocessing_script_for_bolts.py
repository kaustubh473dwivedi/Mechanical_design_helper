#!/usr/bin/env python3
#kaustubh Dwivedi
"""
create_nuts_and_bolts_manifest.py

Generates an Alpaca/LLaMA-style JSONL manifest from an existing synthetic nuts & bolts dataset.

Input: a JSON file containing a list of bolt examples like the sample you provided:
[
  {
    "bolt_spec": { "size":"M20", "grade":"12.9", "pitch_mm":2.5, "stress_area_mm2":245.0 },
    "calculated_values": { ... }
  }, ...
]

Output (per line):
{"instruction": "...", "output": "{\"bolt_spec\":{...}}"}  # output is JSON string of bolt_spec only

Usage:
  - Place your bolt JSON at one of the INPUT_CANDIDATES or edit INPUT_CANDIDATES to point to your file.
  - Run: python create_nuts_and_bolts_manifest.py
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List

random.seed(12345)

# --- Edit this to point to your bolt dataset if needed ---
INPUT_CANDIDATES = [
    Path("./output_nuts_and_bolts/nuts_and_bolts_dataset_final.json")
    
]

OUT_DIR = Path("./trainable_input_nuts_and_bolts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "nuts_and_bolts_manifest.jsonl"
SAMPLE_PATH = OUT_DIR / "nuts_and_bolts_manifest_sample.jsonl"

# Provenance (keeps same PDF path you uploaded previously)
PROVENANCE_PDF_PATH = "/mnt/data/Roadmap_Gear Design.pdf"

# Practical requirement patterns (curated). Keys: 'bs:' bolt_spec, 'cv:' calculated_values
PATTERNS = [
    ("size_grade", ["bs:size", "bs:grade"]),
    ("size_and_preload", ["bs:size", "cv:recommended_preload_N"]),
    ("torque_requirement", ["cv:estimated_tightening_torque_Nm"]),
    ("shear_requirement", ["cv:shear_capacity_N"]),
    ("tensile_capacity", ["cv:tensile_strength_MPa", "bs:stress_area_mm2"]),
    ("thread_and_engagement", ["bs:pitch_mm", "cv:thread_engagement_min_diameters"]),
    ("full_spec", ["bs:size", "bs:grade", "bs:pitch_mm", "bs:stress_area_mm2"]),
]

MIN_PAIRS = 3
MAX_PAIRS = 6

# Helper to access fields
def get_field(sample: Dict[str, Any], key_spec: str):
    part, field = key_spec.split(":", 1)
    block = "bolt_spec" if part == "bs" else "calculated_values"
    if block in sample and field in sample[block]:
        return True, f"{part}:{field}", sample[block][field]
    return False, f"{part}:{field}", None

def fmt_val(v):
    if v is None:
        return "unspecified"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)

def build_instruction(sample: Dict[str, Any], chosen_keys: List[str]) -> str:
    parts = []
    for ks in chosen_keys:
        ok, kname, val = get_field(sample, ks)
        if not ok:
            continue
        label = kname.split(":",1)[1].replace("_", " ")
        # make wording natural
        if ks.startswith("bs:"):
            parts.append(f"{label} = {fmt_val(val)}")
        else:
            # calculated values -> desired limits or provide capacity
            if "torque" in label or "preload" in label or "shear" in label:
                parts.append(f"{label} >= {fmt_val(val)}")
            else:
                parts.append(f"{label} = {fmt_val(val)}")
    if not parts:
        return "Select a suitable bolt for general-purpose mechanical joining under moderate load."
    if len(parts) == 1:
        body = parts[0]
    elif len(parts) == 2:
        body = " and ".join(parts)
    else:
        body = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return f"Select/design a bolt that meets: {body}. Return only the bolt_spec as JSON."

def preserve_bolt_spec(sample_bolt_spec: Dict[str,Any]) -> Dict[str,Any]:
    # In our manifest the authoritative bolt_spec is the example's bolt_spec
    return sample_bolt_spec.copy()

def create_pairs_for_sample(sample: Dict[str,Any], pairs_count: int) -> List[Dict[str,str]]:
    pairs = []
    valid_patterns = []
    for name, keys in PATTERNS:
        if any(get_field(sample, k)[0] for k in keys):
            valid_patterns.append((name, keys))
    if not valid_patterns:
        # fallback to basic size+grade if available
        if get_field(sample, "bs:size")[0] or get_field(sample, "bs:grade")[0]:
            valid_patterns.append(("basic", ["bs:size","bs:grade"]))
        else:
            valid_patterns.append(("generic", ["bs:size"]))
    for _ in range(pairs_count):
        pattern_name, keys = random.choice(valid_patterns)
        instruction = build_instruction(sample, keys)
        bolt_spec = preserve_bolt_spec(sample.get("bolt_spec", {}))
        output_json_string = json.dumps({"bolt_spec": bolt_spec}, separators=(',', ':'))
        pairs.append({"instruction": instruction, "output": output_json_string})
    return pairs

def find_input_file() -> Path:
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"No input bolt dataset found. Tried: {INPUT_CANDIDATES}")

def main():
    input_path = find_input_file()
    print("Using input dataset:", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    manifest_lines = []
    for sample in samples:
        n = random.randint(MIN_PAIRS, MAX_PAIRS)
        pairs = create_pairs_for_sample(sample, n)
        manifest_lines.extend(pairs)
    # write manifest
    with open(MANIFEST_PATH, "w", encoding="utf-8") as out:
        for item in manifest_lines:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    # sample
    with open(SAMPLE_PATH, "w", encoding="utf-8") as s:
        for item in manifest_lines[:50]:
            s.write(json.dumps(item, ensure_ascii=False) + "\n")
    # provenance
    prov = OUT_DIR / "manifest_provenance.txt"
    prov.write_text(f"Input dataset: {input_path}\nSource PDF (if any): {PROVENANCE_PDF_PATH}\nPatterns used: {len(PATTERNS)}\n")
    print(f"Wrote manifest ({len(manifest_lines)} entries) to {MANIFEST_PATH}")
    print(f"Wrote sample (first 50) to {SAMPLE_PATH}")
    print("Provenance written to:", prov)

if __name__ == "__main__":
    main()
