

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types


# ============================================================
# Paths (relative to the script location in the Sitting folder)
# ============================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
VIDEOS_DIR = SCRIPT_DIR / "videos"
RESULTS_DIR = SCRIPT_DIR / "Gemini_export"
GROUND_TRUTH_FILE = SCRIPT_DIR / "SIT_export.csv"
CODING_MANUAL_FILE = SCRIPT_DIR / "sitting_manual.pdf"

# ============================================================
# Model configuration
# ============================================================

MODEL_NAME = "gemini-3.1-pro-preview"
PAUSE_BETWEEN_CALLS = 5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (5, 15, 45)
POSTURE_CODES = ["0", "1", "2", "3", "F", "N"]

# ============================================================
# System instruction
# ============================================================

SYSTEM_INSTRUCTION = (
    "You are analyzing a one-minute video of an infant during a free play session on "
    "the floor. A coding manual for infant sitting postures has been provided alongside "
    "this video. Use the definitions, illustrations, and decision rules in that manual "
    "as your primary reference for all coding decisions.\n\n"
    "The coding scheme has the following posture codes:\n"
    "- Code 0 (Supported sitting): the infant is held upright or propped by an EXTERNAL "
    "agent, such as a caregiver's hands or body, a seat or device, or furniture. The infant "
    "is not supporting their own trunk.\n"
    "- Code 1 (Floor-based sitting): the infant is seated on the floor without upright "
    "trunk control (slumped or collapsed at the trunk, leaning heavily without using arms "
    "for support). Do NOT use Code 1 for lying down.\n"
    "- Code 2 (Tripod sitting): the infant maintains partial trunk control by SELF-supporting "
    "with one or both of their own hands or arms on the floor. Distinguish from Code 0: "
    "support comes from the infant's own arms, not from an external agent.\n"
    "- Code 3 (Independent sitting): the infant sits upright with self-controlled trunk and "
    "no external support; both hands are free of the floor.\n"
    "- Code F (Falling/transition): the infant is actively falling or moving between postures.\n"
    "- Code N (Not observable): the posture cannot be coded (e.g., infant out of frame, "
    "obscured, or in a non-sitting posture such as lying or being carried).\n\n"
    "Base every observation strictly on what is visible in the video. Do not infer diagnoses "
    "or conditions. When a behavior is partially obscured or ambiguous, prefer Code N over "
    "guessing. Refer to the coding manual for boundary cases between adjacent codes."
)

# ============================================================
# Task prompts and schemas
# ============================================================

TASKS = {
    "T1_posture_identification": {
        "prompt": (
            "Using the coding manual provided, identify which sitting postures the infant "
            "displays in the video. Return all distinct codes observed, the single code that "
            "best characterizes the infant's most frequent posture, and a 1-2 sentence "
            "description of the sitting behavior."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "postures_observed": {
                    "type": "array",
                    "items": {"type": "string", "enum": POSTURE_CODES},
                },
                "primary_posture": {"type": "string", "enum": POSTURE_CODES},
                "brief_description": {"type": "string"},
            },
            "required": ["postures_observed", "primary_posture", "brief_description"],
        },
    },

    "T2_temporal_estimation": {
        "prompt": (
            "Estimate the proportion of the one-minute video the infant spends in each "
            "sitting-posture code from the coding manual. Each percentage must be an integer "
            "in [0, 100], and the six values must sum to exactly 100."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "code_0_supported_pct": {"type": "integer", "minimum": 0, "maximum": 100},
                "code_1_floor_based_pct": {"type": "integer", "minimum": 0, "maximum": 100},
                "code_2_tripod_pct": {"type": "integer", "minimum": 0, "maximum": 100},
                "code_3_independent_pct": {"type": "integer", "minimum": 0, "maximum": 100},
                "code_F_transition_pct": {"type": "integer", "minimum": 0, "maximum": 100},
                "code_N_not_observable_pct": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": [
                "code_0_supported_pct", "code_1_floor_based_pct", "code_2_tripod_pct",
                "code_3_independent_pct", "code_F_transition_pct", "code_N_not_observable_pct",
            ],
        },
    },

    "T3_temporal_localization": {
        "prompt": (
            "Identify the infant's posture at the start of the video, every transition "
            "between sitting-posture codes (as defined in the coding manual), and the posture "
            "at the end of the video. Report each transition with its timestamp in seconds "
            "from the start of the video (0-60). If no transitions occur, return an empty "
            "transitions list."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "starting_posture": {"type": "string", "enum": POSTURE_CODES},
                "transitions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp_seconds": {"type": "number", "minimum": 0, "maximum": 60},
                            "from_code": {"type": "string", "enum": POSTURE_CODES},
                            "to_code": {"type": "string", "enum": POSTURE_CODES},
                        },
                        "required": ["timestamp_seconds", "from_code", "to_code"],
                    },
                },
                "ending_posture": {"type": "string", "enum": POSTURE_CODES},
                "total_transitions": {"type": "integer", "minimum": 0},
            },
            "required": [
                "starting_posture", "transitions", "ending_posture", "total_transitions",
            ],
        },
    },

    "T4_sitter_categorization": {
        "prompt": (
            "Based on the coding manual and the sitting behavior observed, classify the "
            "infant into one developmental category:\n"
            "- 'Independent sitter': predominantly sits upright (Code 3) without external "
            "support or self-propping; sitting is stable and sustained.\n"
            "- 'Emergent sitter': mixes supported and unsupported sitting; some Code 3 is "
            "present but not dominant; frequently uses arms (Code 2) or alternates with "
            "supported postures.\n"
            "- 'Non-sitter': predominantly requires external support (Code 0) or shows "
            "floor-based postures (Code 1); independent sitting is absent or very brief.\n"
            "Provide a confidence rating and a 2-3 sentence rationale grounded in the "
            "observed behavior."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "enum": ["Independent sitter", "Emergent sitter", "Non-sitter"],
                },
                "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
                "reasoning": {"type": "string"},
            },
            "required": ["classification", "confidence", "reasoning"],
        },
    },

    "T5_age_estimation": {
        "prompt": (
            "Estimate the infant's age in months based ONLY on the sitting behavior "
            "observed and the developmental progression described in the coding manual. "
            "Constrain your estimate to the typical sitting-development window of 3 to 18 "
            "months, inclusive. Provide a point estimate and a plausible lower-upper range "
            "(both within 3-18). Consider trunk-control quality and stability, presence or "
            "absence of independent sitting, smoothness of postural transitions, and other "
            "visible motor cues. Justify with 2-3 sentences referencing the specific cues "
            "you used."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "age_estimate_months": {"type": "number", "minimum": 3, "maximum": 18},
                "lower_bound_months": {"type": "number", "minimum": 3, "maximum": 18},
                "upper_bound_months": {"type": "number", "minimum": 3, "maximum": 18},
                "reasoning": {"type": "string"},
            },
            "required": [
                "age_estimate_months", "lower_bound_months", "upper_bound_months", "reasoning",
            ],
        },
    },

    "T7_developmental_justification": {
        "prompt": (
            "In one paragraph (4-6 sentences), describe what the infant's sitting behavior "
            "suggests about current motor development. Use the coding manual as your reference "
            "for developmental progression. Address: (a) the level of trunk control and "
            "postural stability demonstrated; (b) where this places the infant along the "
            "typical progression from fully supported to independent sitting; (c) any notable "
            "strengths or emerging skills visible in the video. Base observations only on what "
            "is visible. Do not speculate about diagnoses or conditions."
        ),
        "schema": None,
    },

    "T8_caregiving_recommendations": {
        "prompt": (
            "In one paragraph (4-6 sentences), recommend caregiving strategies and "
            "environmental supports that would benefit this infant's sitting development. "
            "Address: (a) appropriate physical support or positioning during floor play; "
            "(b) activities or play interactions that could encourage the next stage of "
            "sitting development; (c) safety considerations relevant to the infant's "
            "current sitting ability. Recommendations should be practical for a home "
            "setting and grounded in what is visible in the video."
        ),
        "schema": None,
    },
}

# ============================================================
# Hallucination probes (Task 6)
# ============================================================

PROBE_TEMPLATES = {
    "T6_probe_crawling": (
        "Did the infant crawl or move on hands and knees at any point during this video?"
    ),
    "T6_probe_standing": (
        "Did the infant pull to stand or bear weight on their feet at any point during "
        "this video?"
    ),
    "T6_probe_arm_support": (
        "Did the infant use one or both hands on the floor for support while sitting at "
        "any point during this video?"
    ),
    "T6_probe_caregiver_support": (
        "Did a caregiver physically support or hold the infant in a sitting position at "
        "any point during this video?"
    ),
    "T6_probe_falling": (
        "Did the infant fall over or lose balance while sitting at any point during this "
        "video?"
    ),
    "T6_probe_toy_interaction": (
        "Did the infant reach for, grasp, or manipulate a toy or object while sitting "
        "during this video?"
    ),
}

PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["YES", "NO"]},
        "explanation": {"type": "string"},
    },
    "required": ["answer", "explanation"],
}

PROBE_INSTRUCTION_SUFFIX = (
    " Answer strictly YES or NO based only on what is visible. Provide a one-sentence "
    "explanation grounded in observed behavior."
)

# ============================================================
# Parse video filenames
# ============================================================

def parse_video_filename(filename):
    """
    Parse 'LOG - 003_SIT_06M_V1.mp4' into components.
    Returns dict with subject_id, age_months, visit, and the video_id string.
    """
    stem = Path(filename).stem  # 'LOG - 003_SIT_06M_V1'
    match = re.match(
        r"LOG\s*-\s*(\d+)_SIT_(\d+)M_V(\d+)",
        stem,
        re.IGNORECASE,
    )
    if not match:
        print(f"  WARNING: Could not parse filename '{filename}'. Using stem as video_id.")
        return {
            "video_id": stem,
            "subject_id": None,
            "age_months": None,
            "visit": None,
        }
    return {
        "video_id": stem,
        "subject_id": match.group(1),       # '003'
        "age_months": int(match.group(2)),   # 6
        "visit": int(match.group(3)),        # 1
    }

# ============================================================
# Load and process ground truth
# ============================================================

def load_ground_truth(csv_path):
    """
    Load SIT_export.csv (tab-separated) and organize by subject + visit.

    Expected columns: subj, visit, onset, offset, posture
    The visit column may contain extra info (e.g., "1, ,ED" where 1 is the
    visit number and ED is the coder). We extract just the visit number.

    Returns a dict keyed by (subject_id, visit) with:
        - bouts: list of {onset_ms, offset_ms, posture}
        - postures_present: set of posture codes present
        - duration_proportions: dict of code -> proportion
        - sitter_category: derived classification
        - transitions: list of {timestamp_seconds, from_code, to_code}
    """
    ground_truth = {}

    with open(csv_path, "r") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            row = dict(zip(header, parts))
            subj = row["subj"].strip().zfill(3)  # Ensure 3-digit padding

            # Parse visit number from potentially messy column (e.g., "1, ,ED")
            visit_raw = row["visit"].strip()
            visit_match = re.match(r"(\d+)", visit_raw)
            visit = int(visit_match.group(1)) if visit_match else 1

            key = (subj, visit)

            if key not in ground_truth:
                ground_truth[key] = {"bouts": []}

            ground_truth[key]["bouts"].append({
                "onset_ms": int(row["onset"]),
                "offset_ms": int(row["offset"]),
                "posture": row["posture"],
            })

    # Derive metrics for each subject-visit
    for key, data in ground_truth.items():
        bouts = data["bouts"]
        bouts.sort(key=lambda b: b["onset_ms"])

        # Postures present
        data["postures_present"] = set(b["posture"] for b in bouts)

        # Duration proportions
        total_ms = sum(b["offset_ms"] - b["onset_ms"] + 1 for b in bouts)
        duration_by_code = {}
        for b in bouts:
            dur = b["offset_ms"] - b["onset_ms"] + 1
            duration_by_code[b["posture"]] = duration_by_code.get(b["posture"], 0) + dur

        data["duration_proportions"] = {}
        for code in ["0", "1", "2", "3", "F", "N"]:
            data["duration_proportions"][code] = round(
                duration_by_code.get(code, 0) / total_ms * 100
            ) if total_ms > 0 else 0

        # Sitter classification (using proportion-of-codable-time rule)
        # Codable time excludes F and N
        codable_ms = sum(
            b["offset_ms"] - b["onset_ms"] + 1
            for b in bouts if b["posture"] not in ("F", "N")
        )
        if codable_ms > 0:
            code3_ms = duration_by_code.get("3", 0)
            code2_ms = duration_by_code.get("2", 0)
            code0_ms = duration_by_code.get("0", 0)
            code1_ms = duration_by_code.get("1", 0)

            pct_3 = code3_ms / codable_ms
            pct_23 = (code2_ms + code3_ms) / codable_ms
            pct_01 = (code0_ms + code1_ms) / codable_ms

            if pct_3 >= 0.50:
                data["sitter_category"] = "Independent sitter"
            elif pct_23 >= 0.50:
                data["sitter_category"] = "Emergent sitter"
            else:
                data["sitter_category"] = "Non-sitter"
        else:
            data["sitter_category"] = "Non-sitter"

        # Transitions (for temporal localization ground truth)
        data["transitions"] = []
        for i in range(1, len(bouts)):
            if bouts[i]["posture"] != bouts[i-1]["posture"]:
                data["transitions"].append({
                    "timestamp_seconds": round(bouts[i]["onset_ms"] / 1000, 1),
                    "from_code": bouts[i-1]["posture"],
                    "to_code": bouts[i]["posture"],
                })

        data["starting_posture"] = bouts[0]["posture"] if bouts else None
        data["ending_posture"] = bouts[-1]["posture"] if bouts else None

    return ground_truth


def determine_probes(ground_truth_entry):
    """
    Determine which hallucination probes are valid (behavior absent) and which
    are present-behavior probes (behavior confirmed present) for a given video.
    Returns (absent_probes, present_probes).
    """
    postures = ground_truth_entry["postures_present"]

    # Map probe names to the posture codes that indicate the behavior IS present
    probe_to_present_codes = {
        "T6_probe_arm_support": {"2"},        # Tripod = arm support present
        "T6_probe_caregiver_support": {"0"},   # Supported = caregiver support present
        "T6_probe_falling": {"F"},             # Fall code present
    }

    # These probes test behaviors NOT in the posture coding scheme
    # (crawling, standing, toy interaction). We assume these are absent
    # unless you add additional columns to the CSV later.
    always_absent_probes = [
        "T6_probe_crawling",
        "T6_probe_standing",
        "T6_probe_toy_interaction",
    ]

    absent_probes = list(always_absent_probes)
    present_probes = []

    for probe_name, present_codes in probe_to_present_codes.items():
        if postures & present_codes:
            # Behavior IS present in this video
            present_probes.append(probe_name)
        else:
            # Behavior is absent, use as hallucination probe
            absent_probes.append(probe_name)

    return absent_probes, present_probes

# ============================================================
# Upload coding manual
# ============================================================

_manual_cache = {}

def get_or_upload_manual(client):
    """Upload the coding manual once and cache the file handle."""
    if "file" in _manual_cache:
        try:
            f = client.files.get(name=_manual_cache["file"].name)
            if f.state.name == "ACTIVE":
                return f
        except Exception:
            pass

    if not CODING_MANUAL_FILE.exists():
        print(f"  WARNING: Coding manual not found at {CODING_MANUAL_FILE}")
        print("  Proceeding without coding manual.")
        return None

    print(f"  Uploading coding manual: {CODING_MANUAL_FILE}")
    f = client.files.upload(file=str(CODING_MANUAL_FILE))
    while f.state.name == "PROCESSING":
        time.sleep(3)
        f = client.files.get(name=f.name)
    if f.state.name == "FAILED":
        print("  WARNING: Coding manual upload failed. Proceeding without it.")
        return None
    print(f"  Coding manual ready.")
    _manual_cache["file"] = f
    return f


# ============================================================
# Video upload with caching
# ============================================================

def video_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache(cache_path):
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return {}


def save_cache(cache_path, cache):
    cache_path.write_text(json.dumps(cache, indent=2))


def get_or_upload_video(client, video_path, cache_path):
    vhash = video_sha256(video_path)
    cache = load_cache(cache_path)
    cached = cache.get(vhash)

    if cached:
        try:
            f = client.files.get(name=cached["name"])
            if f.state.name == "ACTIVE":
                print(f"  Reusing cached upload: {f.name}")
                return f, vhash
        except Exception:
            pass

    print(f"  Uploading video: {video_path}")
    f = client.files.upload(file=str(video_path))
    while f.state.name == "PROCESSING":
        print("  Waiting for video processing...")
        time.sleep(5)
        f = client.files.get(name=f.name)
    if f.state.name == "FAILED":
        raise RuntimeError(f"Video processing failed: {f.name}")
    print(f"  Video ready.")

    cache[vhash] = {"name": f.name, "uri": f.uri, "mime_type": f.mime_type}
    save_cache(cache_path, cache)
    return f, vhash


# ============================================================
# Prompt execution
# ============================================================

def build_config(schema, max_output_tokens=2048):
    kwargs = dict(
        temperature=0.2,
        max_output_tokens=max_output_tokens,
        system_instruction=SYSTEM_INSTRUCTION,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    )
    if schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = schema
    return types.GenerateContentConfig(**kwargs)


def extract_response(response):
    finish_reason = "unknown"
    text = None
    if response.candidates:
        cand = response.candidates[0]
        if cand.finish_reason is not None:
            finish_reason = str(cand.finish_reason)
        if cand.content and cand.content.parts:
            collected = [p.text for p in cand.content.parts if getattr(p, "text", None)]
            if collected:
                text = "\n".join(collected)
    return text, finish_reason


def run_prompt(client, video_file, manual_file, task_name, prompt_text, schema,
               expected=None, category=None):
    """Send a prompt with video + coding manual to Gemini."""
    print(f"  Running task: {task_name}")
    config = build_config(schema)

    # Build content parts: coding manual (if available) + video + prompt
    parts = []
    if manual_file:
        parts.append(types.Part.from_uri(
            file_uri=manual_file.uri,
            mime_type=manual_file.mime_type,
        ))
    parts.append(types.Part.from_uri(
        file_uri=video_file.uri,
        mime_type=video_file.mime_type,
    ))
    parts.append(types.Part.from_text(text=prompt_text))

    contents = [types.Content(role="user", parts=parts)]

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config,
            )
            text, finish_reason = extract_response(response)
            parsed = None
            parse_error = None
            if schema is not None and text:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as e:
                    parse_error = str(e)

            result = {
                "task": task_name,
                "category": category,
                "expected": expected,
                "response_text": text,
                "parsed": parsed,
                "parse_error": parse_error,
                "finish_reason": finish_reason,
                "attempt": attempt + 1,
                "timestamp": datetime.now().isoformat(),
            }
            print(f"  Task complete: {task_name}")
            return result
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_SECONDS[attempt]
                print(f"  ERROR on {task_name} (attempt {attempt + 1}): {e}; "
                      f"retrying in {wait}s.")
                time.sleep(wait)
            else:
                print(f"  ERROR on {task_name} (final): {e}")

    return {
        "task": task_name,
        "category": category,
        "expected": expected,
        "response_text": None,
        "parsed": None,
        "parse_error": None,
        "finish_reason": "error",
        "error": str(last_err),
        "attempt": MAX_RETRIES,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# Process a single video
# ============================================================

def process_video(client, video_path, manual_file, ground_truth, task_filter=None):
    """Run all tasks for one video. Returns the results dict."""
    info = parse_video_filename(video_path.name)
    video_id = info["video_id"]
    subj_id = info["subject_id"]
    age_months = info["age_months"]
    visit = info["visit"]

    print(f"\n{'='*60}")
    print(f"Processing: {video_id}")
    print(f"  Subject: {subj_id}, Age: {age_months}M, Visit: V{visit}")
    print(f"{'='*60}")

    # Upload video
    cache_path = RESULTS_DIR / ".upload_cache.json"
    video_file, video_hash = get_or_upload_video(client, video_path, cache_path)

    # Look up ground truth
    gt_key = (subj_id, visit) if subj_id and visit else None
    gt_entry = ground_truth.get(gt_key) if gt_key else None

    if gt_entry:
        print(f"  Ground truth found: {len(gt_entry['bouts'])} bouts, "
              f"category = {gt_entry['sitter_category']}")
    else:
        print(f"  WARNING: No ground truth found for subject {subj_id}, visit {visit}")

    # Build task plan
    task_plan = []

    # Core tasks (T1-T5, T7, T8)
    for name, spec in TASKS.items():
        if task_filter and name not in task_filter:
            continue
        task_plan.append((name, spec["prompt"], spec["schema"], None, None))

    # Hallucination probes (T6) - auto-determined from ground truth
    if gt_entry and (task_filter is None or "T6" in task_filter):
        absent_probes, present_probes = determine_probes(gt_entry)
        for probe_name in absent_probes:
            prompt = PROBE_TEMPLATES[probe_name] + PROBE_INSTRUCTION_SUFFIX
            task_plan.append((probe_name, prompt, PROBE_SCHEMA, "NO", "absent_probe"))
        for probe_name in present_probes:
            prompt = PROBE_TEMPLATES[probe_name] + PROBE_INSTRUCTION_SUFFIX
            task_plan.append((probe_name, prompt, PROBE_SCHEMA, "YES", "present_probe"))
    elif task_filter is None or "T6" in task_filter:
        print("  Skipping hallucination probes (no ground truth for probe assignment)")

    # Prepare results structure
    all_results = {
        "video_id": video_id,
        "video_path": str(video_path),
        "video_sha256": video_hash,
        "subject_id": subj_id,
        "age_months": age_months,
        "visit": visit,
        "model": MODEL_NAME,
        "coding_manual_included": manual_file is not None,
        "generation_config": {
            "temperature": 0.2,
            "max_output_tokens": 2048,
            "media_resolution": "MEDIA_RESOLUTION_HIGH",
        },
        "ground_truth": {
            "sitter_category": gt_entry["sitter_category"] if gt_entry else None,
            "duration_proportions": gt_entry["duration_proportions"] if gt_entry else None,
            "postures_present": list(gt_entry["postures_present"]) if gt_entry else None,
            "transitions": gt_entry["transitions"] if gt_entry else None,
            "starting_posture": gt_entry["starting_posture"] if gt_entry else None,
            "ending_posture": gt_entry["ending_posture"] if gt_entry else None,
            "age_months": age_months,
        },
        "run_timestamp": datetime.now().isoformat(),
        "tasks": {},
    }

    output_file = RESULTS_DIR / f"{video_id}_results.json"

    # Run tasks
    for i, (name, prompt, schema, expected, category) in enumerate(task_plan):
        result = run_prompt(
            client, video_file, manual_file, name, prompt, schema,
            expected=expected, category=category,
        )
        all_results["tasks"][name] = result

        # Incremental save
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        if i < len(task_plan) - 1:
            time.sleep(PAUSE_BETWEEN_CALLS)

    print(f"  Results saved to: {output_file}")
    return all_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Gemini sitting evaluation on LogOn study videos."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video", type=str,
        help="Single video filename (e.g., 'LOG - 003_SIT_06M_V1.mp4')"
    )
    group.add_argument(
        "--all", action="store_true",
        help="Process all .mp4 videos in the videos/ folder"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Specific tasks to run (e.g., T1_posture_identification T4_sitter_categorization). "
             "Use 'T6' to include all hallucination probes. Default: all tasks."
    )
    parser.add_argument(
        "--api_key", default=None,
        help="Gemini API key. If omitted, reads GEMINI_API_KEY environment variable."
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip videos that already have a results file in the Gemini_export/ folder."
    )
    args = parser.parse_args()

    # Setup
    # NOTE: Replace this key with your own or set GEMINI_API_KEY environment variable.
    # Regenerate this key if it has been shared or exposed.
    DEFAULT_API_KEY = "AIzaSyC7z7-twZJIYvgthG9PBmlVaUGkqfknMnk"
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or DEFAULT_API_KEY
    if not api_key:
        raise ValueError(
            "No API key. Set GEMINI_API_KEY environment variable or pass --api_key."
        )

    client = genai.Client(api_key=api_key)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    if GROUND_TRUTH_FILE.exists():
        print(f"Loading ground truth from: {GROUND_TRUTH_FILE}")
        ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
        print(f"  Found data for {len(ground_truth)} subject-visit combinations.")
    else:
        print(f"WARNING: Ground truth file not found at {GROUND_TRUTH_FILE}")
        ground_truth = {}

    # Upload coding manual once
    manual_file = get_or_upload_manual(client)

    # Determine task filter
    task_filter = None
    if args.tasks:
        task_filter = set(args.tasks)

    # Determine which videos to process
    if args.all:
        video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
        if not video_files:
            print(f"No .mp4 files found in {VIDEOS_DIR}")
            return
        print(f"\nFound {len(video_files)} videos to process.")
    else:
        video_path = VIDEOS_DIR / args.video
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        video_files = [video_path]

    # Process videos
    summary = {"processed": 0, "skipped": 0, "errors": 0}

    for video_path in video_files:
        video_id = video_path.stem
        result_file = RESULTS_DIR / f"{video_id}_results.json"

        if args.skip_existing and result_file.exists():
            print(f"\nSkipping (results exist): {video_id}")
            summary["skipped"] += 1
            continue

        try:
            process_video(client, video_path, manual_file, ground_truth, task_filter)
            summary["processed"] += 1
        except Exception as e:
            print(f"\nERROR processing {video_id}: {e}")
            summary["errors"] += 1

    # Aggregate all individual JSONs into a single combined CSV
    aggregate_results(RESULTS_DIR)

    # Print summary
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"  Processed: {summary['processed']}")
    print(f"  Skipped:   {summary['skipped']}")
    print(f"  Errors:    {summary['errors']}")
    print(f"  Results in: {RESULTS_DIR}")
    print(f"{'='*60}")


# ============================================================
# Aggregation: combine all per-video JSONs into one CSV
# ============================================================

def aggregate_results(results_dir):
    """
    Read all *_results.json files in results_dir and produce a single
    sitting_results_combined.csv with one row per video per task.

    Columns:
        video_id, subject_id, age_months, visit, model, sitter_category_gt,
        task, category, expected, finish_reason, attempt,
        # Task-specific parsed fields (flattened from JSON):
        postures_observed, primary_posture, brief_description,
        code_0_supported_pct, code_1_floor_based_pct, code_2_tripod_pct,
        code_3_independent_pct, code_F_transition_pct, code_N_not_observable_pct,
        starting_posture, transitions_json, ending_posture, total_transitions,
        classification, confidence, reasoning,
        age_estimate_months, lower_bound_months, upper_bound_months,
        answer, explanation,
        response_text,
        # Ground truth fields for easy comparison:
        gt_sitter_category, gt_duration_0, gt_duration_1, gt_duration_2,
        gt_duration_3, gt_duration_F, gt_duration_N,
        gt_postures_present, gt_starting_posture, gt_ending_posture,
        gt_transitions_json, gt_age_months
    """
    import csv as csv_module

    json_files = sorted(results_dir.glob("*_results.json"))
    if not json_files:
        print("  No result files found to aggregate.")
        return

    print(f"\n  Aggregating {len(json_files)} result files...")

    # Define all possible columns in a stable order
    meta_cols = [
        "video_id", "subject_id", "age_months", "visit", "model",
    ]
    task_meta_cols = [
        "task", "category", "expected", "finish_reason", "attempt",
    ]
    # All possible parsed fields across all task schemas
    parsed_cols = [
        "postures_observed", "primary_posture", "brief_description",
        "code_0_supported_pct", "code_1_floor_based_pct", "code_2_tripod_pct",
        "code_3_independent_pct", "code_F_transition_pct", "code_N_not_observable_pct",
        "starting_posture", "transitions_json", "ending_posture", "total_transitions",
        "classification", "confidence", "reasoning",
        "age_estimate_months", "lower_bound_months", "upper_bound_months",
        "answer", "explanation",
        "response_text",
    ]
    gt_cols = [
        "gt_sitter_category",
        "gt_duration_0", "gt_duration_1", "gt_duration_2",
        "gt_duration_3", "gt_duration_F", "gt_duration_N",
        "gt_postures_present", "gt_starting_posture", "gt_ending_posture",
        "gt_transitions_json", "gt_age_months",
    ]
    all_cols = meta_cols + task_meta_cols + parsed_cols + gt_cols

    rows = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARNING: Could not read {jf.name}: {e}")
            continue

        gt = data.get("ground_truth", {}) or {}

        for task_name, task_data in data.get("tasks", {}).items():
            row = {}

            # Meta
            row["video_id"] = data.get("video_id", "")
            row["subject_id"] = data.get("subject_id", "")
            row["age_months"] = data.get("age_months", "")
            row["visit"] = data.get("visit", "")
            row["model"] = data.get("model", "")

            # Task meta
            row["task"] = task_data.get("task", task_name)
            row["category"] = task_data.get("category", "")
            row["expected"] = task_data.get("expected", "")
            row["finish_reason"] = task_data.get("finish_reason", "")
            row["attempt"] = task_data.get("attempt", "")

            # Parsed fields
            parsed = task_data.get("parsed") or {}
            for col in parsed_cols:
                if col == "response_text":
                    row[col] = task_data.get("response_text", "")
                elif col == "transitions_json":
                    # Serialize the transitions list as JSON string for CSV
                    val = parsed.get("transitions", "")
                    row[col] = json.dumps(val) if val else ""
                elif col == "postures_observed":
                    val = parsed.get("postures_observed", "")
                    row[col] = ",".join(val) if isinstance(val, list) else str(val) if val else ""
                else:
                    row[col] = parsed.get(col, "")

            # Ground truth
            row["gt_sitter_category"] = gt.get("sitter_category", "")
            dur_props = gt.get("duration_proportions", {}) or {}
            row["gt_duration_0"] = dur_props.get("0", "")
            row["gt_duration_1"] = dur_props.get("1", "")
            row["gt_duration_2"] = dur_props.get("2", "")
            row["gt_duration_3"] = dur_props.get("3", "")
            row["gt_duration_F"] = dur_props.get("F", "")
            row["gt_duration_N"] = dur_props.get("N", "")
            gt_postures = gt.get("postures_present", "")
            row["gt_postures_present"] = ",".join(sorted(gt_postures)) if isinstance(gt_postures, list) else ""
            row["gt_starting_posture"] = gt.get("starting_posture", "")
            row["gt_ending_posture"] = gt.get("ending_posture", "")
            gt_trans = gt.get("transitions", "")
            row["gt_transitions_json"] = json.dumps(gt_trans) if gt_trans else ""
            row["gt_age_months"] = gt.get("age_months", "")

            # Ensure all values are strings for CSV
            row = {k: "" if v is None else v for k, v in row.items()}
            rows.append(row)

    # Write combined CSV
    output_csv = results_dir / "sitting_results_combined.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Combined CSV saved: {output_csv}")
    print(f"  {len(rows)} rows across {len(json_files)} videos")


if __name__ == "__main__":
    main()
