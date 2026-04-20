"""
Sitting Evaluation: Scoring and Analysis
==========================================
Reads all *_results.json files from Gemini_export/ and computes
evaluation metrics for each task tier.

Usage:
    cd ~/Desktop/Sitting
    python3 sitting_analysis.py

Output (all saved to Gemini_export/):
    - sitting_analysis_report.txt    Summary report with all metrics
    - scoring_details.csv            Per-video per-task scoring breakdown
    - confusion_matrix_T1.csv        Posture identification confusion matrix
    - confusion_matrix_T4.csv        Sitter categorization confusion matrix

Requirements:
    pip3 install scikit-learn numpy
    (If not installed, the script will still run but skip kappa/F1 metrics)
"""

import json
import csv
import os
import math
from pathlib import Path
from collections import Counter, defaultdict

# Try importing sklearn; gracefully degrade if not available
try:
    from sklearn.metrics import (
        cohen_kappa_score, confusion_matrix, classification_report,
        f1_score, accuracy_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not installed. Install with: pip3 install scikit-learn")
    print("  Some metrics (kappa, F1) will be skipped.\n")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not installed. Install with: pip3 install numpy")
    print("  Some metrics (correlation, RMSE) will be skipped.\n")

# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "Gemini_export"
REPORT_FILE = RESULTS_DIR / "sitting_analysis_report.txt"
SCORING_CSV = RESULTS_DIR / "scoring_details.csv"

# ============================================================
# Load all results
# ============================================================

def load_all_results(results_dir):
    """Load all *_results.json files and return as a list of dicts."""
    results = []
    for jf in sorted(results_dir.glob("*_results.json")):
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARNING: Could not read {jf.name}: {e}")
    return results


# ============================================================
# Helper functions
# ============================================================

def safe_pearson(x, y):
    """Compute Pearson correlation without numpy if needed."""
    if HAS_NUMPY:
        if len(x) < 3:
            return float("nan")
        r = np.corrcoef(x, y)
        return r[0, 1]
    else:
        n = len(x)
        if n < 3:
            return float("nan")
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
        if den_x == 0 or den_y == 0:
            return float("nan")
        return num / (den_x * den_y)


def mean(values):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def rmse(values):
    if not values:
        return float("nan")
    return math.sqrt(sum(v ** 2 for v in values) / len(values))


def median(values):
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


# ============================================================
# T1: Posture Identification
# ============================================================

def score_t1(results):
    """Score posture identification: compare primary posture and posture sets."""
    scores = []
    gt_primaries = []
    pred_primaries = []

    for r in results:
        gt = r.get("ground_truth", {})
        t1 = r.get("tasks", {}).get("T1_posture_identification", {})
        parsed = t1.get("parsed")
        if not parsed or not gt.get("postures_present"):
            continue

        video_id = r["video_id"]
        gt_postures = set(gt["postures_present"])

        # Determine ground truth primary posture (highest duration)
        dur = gt.get("duration_proportions", {})
        gt_primary = max(dur, key=lambda k: dur[k]) if dur else None

        pred_postures = set(parsed.get("postures_observed", []))
        pred_primary = parsed.get("primary_posture")

        # Primary posture match
        primary_match = 1 if gt_primary == pred_primary else 0

        # Set overlap metrics
        intersection = gt_postures & pred_postures
        union = gt_postures | pred_postures
        jaccard = len(intersection) / len(union) if union else 1.0

        # False positives and false negatives in posture set
        false_pos = pred_postures - gt_postures
        false_neg = gt_postures - pred_postures

        scores.append({
            "video_id": video_id,
            "task": "T1",
            "gt_primary": gt_primary,
            "pred_primary": pred_primary,
            "primary_match": primary_match,
            "gt_postures": ",".join(sorted(gt_postures)),
            "pred_postures": ",".join(sorted(pred_postures)),
            "jaccard": round(jaccard, 3),
            "false_positives": ",".join(sorted(false_pos)) if false_pos else "",
            "false_negatives": ",".join(sorted(false_neg)) if false_neg else "",
        })

        if gt_primary and pred_primary:
            gt_primaries.append(gt_primary)
            pred_primaries.append(pred_primary)

    return scores, gt_primaries, pred_primaries


# ============================================================
# T2: Temporal Estimation
# ============================================================

def score_t2(results):
    """Score temporal estimation: compare duration proportions."""
    scores = []
    code_map = {
        "0": "code_0_supported_pct",
        "1": "code_1_floor_based_pct",
        "2": "code_2_tripod_pct",
        "3": "code_3_independent_pct",
        "F": "code_F_transition_pct",
        "N": "code_N_not_observable_pct",
    }

    all_gt = []
    all_pred = []

    for r in results:
        gt = r.get("ground_truth", {})
        t2 = r.get("tasks", {}).get("T2_temporal_estimation", {})
        parsed = t2.get("parsed")
        gt_dur = gt.get("duration_proportions", {})
        if not parsed or not gt_dur:
            continue

        video_id = r["video_id"]
        errors = {}
        abs_errors = []

        for code, field in code_map.items():
            gt_val = gt_dur.get(code, 0)
            pred_val = parsed.get(field, 0)
            error = pred_val - gt_val
            errors[f"error_{code}"] = error
            abs_errors.append(abs(error))
            all_gt.append(gt_val)
            all_pred.append(pred_val)

        mae = mean(abs_errors)
        row = {
            "video_id": video_id,
            "task": "T2",
            "mae": round(mae, 2),
        }
        row.update({k: round(v, 2) for k, v in errors.items()})
        scores.append(row)

    return scores, all_gt, all_pred


# ============================================================
# T3: Temporal Localization
# ============================================================

def score_t3(results):
    """Score temporal localization: compare transitions."""
    scores = []

    for r in results:
        gt = r.get("ground_truth", {})
        t3 = r.get("tasks", {}).get("T3_temporal_localization", {})
        parsed = t3.get("parsed")
        if not parsed:
            continue

        video_id = r["video_id"]
        gt_transitions = gt.get("transitions", [])
        pred_transitions = parsed.get("transitions", [])
        gt_start = gt.get("starting_posture")
        pred_start = parsed.get("starting_posture")
        gt_end = gt.get("ending_posture")
        pred_end = parsed.get("ending_posture")

        gt_n = len(gt_transitions)
        pred_n = len(pred_transitions)

        # Starting and ending posture accuracy
        start_match = 1 if gt_start == pred_start else 0
        end_match = 1 if gt_end == pred_end else 0

        # Transition count error
        count_error = pred_n - gt_n

        # Transition matching with tolerance windows
        matched_2s = 0
        matched_5s = 0
        direction_correct = 0

        for gt_t in gt_transitions:
            gt_ts = gt_t["timestamp_seconds"]
            gt_from = gt_t["from_code"]
            gt_to = gt_t["to_code"]

            best_match = None
            best_dist = float("inf")
            for pred_t in pred_transitions:
                dist = abs(pred_t["timestamp_seconds"] - gt_ts)
                if dist < best_dist:
                    best_dist = dist
                    best_match = pred_t

            if best_match and best_dist <= 2:
                matched_2s += 1
            if best_match and best_dist <= 5:
                matched_5s += 1
                if best_match["from_code"] == gt_from and best_match["to_code"] == gt_to:
                    direction_correct += 1

        # Precision and recall for transition detection (within 5s)
        precision_5s = matched_5s / pred_n if pred_n > 0 else (1.0 if gt_n == 0 else 0.0)
        recall_5s = matched_5s / gt_n if gt_n > 0 else (1.0 if pred_n == 0 else 0.0)

        scores.append({
            "video_id": video_id,
            "task": "T3",
            "gt_transitions": gt_n,
            "pred_transitions": pred_n,
            "count_error": count_error,
            "start_match": start_match,
            "end_match": end_match,
            "matched_within_2s": matched_2s,
            "matched_within_5s": matched_5s,
            "precision_5s": round(precision_5s, 3),
            "recall_5s": round(recall_5s, 3),
            "direction_correct": direction_correct,
        })

    return scores


# ============================================================
# T4: Sitter Categorization
# ============================================================

def score_t4(results):
    """Score sitter categorization: classification metrics."""
    scores = []
    gt_labels = []
    pred_labels = []

    for r in results:
        gt = r.get("ground_truth", {})
        t4 = r.get("tasks", {}).get("T4_sitter_categorization", {})
        parsed = t4.get("parsed")
        if not parsed or not gt.get("sitter_category"):
            continue

        video_id = r["video_id"]
        gt_cat = gt["sitter_category"]
        pred_cat = parsed.get("classification")
        confidence = parsed.get("confidence")
        match = 1 if gt_cat == pred_cat else 0

        scores.append({
            "video_id": video_id,
            "task": "T4",
            "gt_category": gt_cat,
            "pred_category": pred_cat,
            "confidence": confidence,
            "match": match,
        })

        gt_labels.append(gt_cat)
        pred_labels.append(pred_cat)

    return scores, gt_labels, pred_labels


# ============================================================
# T5: Age Estimation
# ============================================================

def score_t5(results):
    """Score age estimation: compare estimated vs actual age."""
    scores = []
    gt_ages = []
    pred_ages = []

    for r in results:
        gt = r.get("ground_truth", {})
        t5 = r.get("tasks", {}).get("T5_age_estimation", {})
        parsed = t5.get("parsed")
        actual_age = r.get("age_months") or gt.get("age_months")
        if not parsed or actual_age is None:
            continue

        video_id = r["video_id"]
        est_age = parsed.get("age_estimate_months")
        lower = parsed.get("lower_bound_months")
        upper = parsed.get("upper_bound_months")

        if est_age is None:
            continue

        error = est_age - actual_age
        abs_error = abs(error)
        within_1m = 1 if abs_error <= 1 else 0
        within_2m = 1 if abs_error <= 2 else 0
        range_captures = 1 if (lower is not None and upper is not None
                               and lower <= actual_age <= upper) else 0

        scores.append({
            "video_id": video_id,
            "task": "T5",
            "actual_age": actual_age,
            "estimated_age": est_age,
            "lower_bound": lower,
            "upper_bound": upper,
            "error": round(error, 1),
            "abs_error": round(abs_error, 1),
            "within_1_month": within_1m,
            "within_2_months": within_2m,
            "range_captures_actual": range_captures,
        })

        gt_ages.append(actual_age)
        pred_ages.append(est_age)

    return scores, gt_ages, pred_ages


# ============================================================
# T6: Hallucination Probes
# ============================================================

def score_t6(results):
    """Score hallucination probes: false positive and hit rates."""
    scores = []
    absent_results = []  # (expected NO, did model say NO?)
    present_results = []  # (expected YES, did model say YES?)
    probe_breakdown = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0})

    for r in results:
        for task_name, task_data in r.get("tasks", {}).items():
            if not task_name.startswith("T6_probe_"):
                continue

            parsed = task_data.get("parsed")
            expected = task_data.get("expected")
            category = task_data.get("category")
            if not parsed or expected is None:
                continue

            answer = parsed.get("answer")
            correct = 1 if answer == expected else 0

            scores.append({
                "video_id": r["video_id"],
                "task": task_name,
                "category": category,
                "expected": expected,
                "answer": answer,
                "correct": correct,
                "explanation": parsed.get("explanation", ""),
            })

            probe_breakdown[task_name]["total"] += 1
            if correct:
                probe_breakdown[task_name]["correct"] += 1
            else:
                probe_breakdown[task_name]["incorrect"] += 1

            if category == "absent_probe":
                absent_results.append(correct)
            elif category == "present_probe":
                present_results.append(correct)

    return scores, absent_results, present_results, dict(probe_breakdown)


# ============================================================
# T7 & T8: Collect for rubric scoring
# ============================================================

def collect_t7_t8(results):
    """Collect developmental justification and caregiving responses for manual rubric scoring."""
    t7_responses = []
    t8_responses = []

    for r in results:
        video_id = r["video_id"]
        gt = r.get("ground_truth", {})

        t7 = r.get("tasks", {}).get("T7_developmental_justification", {})
        if t7.get("response_text"):
            t7_responses.append({
                "video_id": video_id,
                "sitter_category": gt.get("sitter_category", ""),
                "age_months": r.get("age_months", ""),
                "response": t7["response_text"],
            })

        t8 = r.get("tasks", {}).get("T8_caregiving_recommendations", {})
        if t8.get("response_text"):
            t8_responses.append({
                "video_id": video_id,
                "sitter_category": gt.get("sitter_category", ""),
                "age_months": r.get("age_months", ""),
                "response": t8["response_text"],
            })

    return t7_responses, t8_responses


# ============================================================
# Performance by sitter category
# ============================================================

def performance_by_category(results):
    """Break down key metrics by sitter category."""
    by_cat = defaultdict(lambda: {
        "t1_primary_matches": [],
        "t2_maes": [],
        "t4_matches": [],
        "t5_abs_errors": [],
        "t6_absent_correct": [],
    })

    for r in results:
        gt = r.get("ground_truth", {})
        cat = gt.get("sitter_category")
        if not cat:
            continue

        # T1
        t1 = r.get("tasks", {}).get("T1_posture_identification", {})
        if t1.get("parsed"):
            dur = gt.get("duration_proportions", {})
            gt_primary = max(dur, key=lambda k: dur[k]) if dur else None
            pred_primary = t1["parsed"].get("primary_posture")
            if gt_primary and pred_primary:
                by_cat[cat]["t1_primary_matches"].append(1 if gt_primary == pred_primary else 0)

        # T2
        t2 = r.get("tasks", {}).get("T2_temporal_estimation", {})
        if t2.get("parsed"):
            code_map = {
                "0": "code_0_supported_pct", "1": "code_1_floor_based_pct",
                "2": "code_2_tripod_pct", "3": "code_3_independent_pct",
                "F": "code_F_transition_pct", "N": "code_N_not_observable_pct",
            }
            gt_dur = gt.get("duration_proportions", {})
            abs_errors = []
            for code, field in code_map.items():
                abs_errors.append(abs(t2["parsed"].get(field, 0) - gt_dur.get(code, 0)))
            by_cat[cat]["t2_maes"].append(mean(abs_errors))

        # T4
        t4 = r.get("tasks", {}).get("T4_sitter_categorization", {})
        if t4.get("parsed"):
            by_cat[cat]["t4_matches"].append(
                1 if t4["parsed"].get("classification") == cat else 0
            )

        # T5
        t5 = r.get("tasks", {}).get("T5_age_estimation", {})
        actual_age = r.get("age_months") or gt.get("age_months")
        if t5.get("parsed") and actual_age is not None:
            est = t5["parsed"].get("age_estimate_months")
            if est is not None:
                by_cat[cat]["t5_abs_errors"].append(abs(est - actual_age))

        # T6 absent probes
        for task_name, task_data in r.get("tasks", {}).items():
            if task_name.startswith("T6_probe_") and task_data.get("category") == "absent_probe":
                parsed = task_data.get("parsed")
                if parsed:
                    by_cat[cat]["t6_absent_correct"].append(
                        1 if parsed.get("answer") == "NO" else 0
                    )

    return dict(by_cat)


# ============================================================
# Generate report
# ============================================================

def generate_report(results):
    """Generate the full analysis report."""
    lines = []
    scoring_rows = []

    def add(text=""):
        lines.append(text)

    n_videos = len(results)
    add("=" * 70)
    add("GEMINI SITTING EVALUATION: ANALYSIS REPORT")
    add("=" * 70)
    add(f"Total videos analyzed: {n_videos}")
    models = set(r.get("model", "unknown") for r in results)
    add(f"Model(s): {', '.join(models)}")
    add("")

    # ----------------------------------------------------------
    # TIER 1: PERCEPTUAL CODING
    # ----------------------------------------------------------
    add("-" * 70)
    add("TIER 1: PERCEPTUAL CODING")
    add("-" * 70)

    # T1: Posture Identification
    add("\n--- T1: Posture Identification ---")
    t1_scores, gt_pri, pred_pri = score_t1(results)
    scoring_rows.extend(t1_scores)

    if t1_scores:
        primary_acc = mean([s["primary_match"] for s in t1_scores])
        avg_jaccard = mean([s["jaccard"] for s in t1_scores])
        add(f"  N = {len(t1_scores)}")
        add(f"  Primary posture accuracy: {primary_acc:.1%}")
        add(f"  Mean Jaccard similarity (posture sets): {avg_jaccard:.3f}")

        if HAS_SKLEARN and gt_pri and pred_pri:
            all_labels = sorted(set(gt_pri + pred_pri))
            kappa = cohen_kappa_score(gt_pri, pred_pri)
            add(f"  Cohen's kappa (primary posture): {kappa:.3f}")
            add(f"\n  Classification report (primary posture):")
            report = classification_report(gt_pri, pred_pri, labels=all_labels, zero_division=0)
            for line in report.split("\n"):
                add(f"    {line}")

            cm = confusion_matrix(gt_pri, pred_pri, labels=all_labels)
            add(f"\n  Confusion matrix (rows=GT, cols=Pred):")
            add(f"    Labels: {all_labels}")
            for i, row in enumerate(cm):
                add(f"    {all_labels[i]}: {list(row)}")

            # Save confusion matrix CSV
            cm_file = RESULTS_DIR / "confusion_matrix_T1.csv"
            with open(cm_file, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([""] + all_labels)
                for i, row in enumerate(cm):
                    w.writerow([all_labels[i]] + list(row))

    # T2: Temporal Estimation
    add("\n--- T2: Temporal Estimation ---")
    t2_scores, t2_gt, t2_pred = score_t2(results)
    scoring_rows.extend(t2_scores)

    if t2_scores:
        overall_mae = mean([s["mae"] for s in t2_scores])
        overall_rmse = rmse([s["mae"] for s in t2_scores])
        add(f"  N = {len(t2_scores)}")
        add(f"  Mean absolute error (avg across codes): {overall_mae:.2f} percentage points")
        add(f"  RMSE of per-video MAE: {overall_rmse:.2f}")

        if t2_gt and t2_pred:
            r_val = safe_pearson(t2_gt, t2_pred)
            add(f"  Pearson r (all code estimates vs GT): {r_val:.3f}")

        # Per-code error breakdown
        code_errors = defaultdict(list)
        for s in t2_scores:
            for key, val in s.items():
                if key.startswith("error_"):
                    code_errors[key].append(val)
        add(f"\n  Per-code mean error (positive = overestimate):")
        for code in ["error_0", "error_1", "error_2", "error_3", "error_F", "error_N"]:
            if code in code_errors:
                vals = code_errors[code]
                add(f"    {code}: mean = {mean(vals):+.1f}, "
                    f"MAE = {mean([abs(v) for v in vals]):.1f}")

    # T3: Temporal Localization
    add("\n--- T3: Temporal Localization ---")
    t3_scores = score_t3(results)
    scoring_rows.extend(t3_scores)

    if t3_scores:
        start_acc = mean([s["start_match"] for s in t3_scores])
        end_acc = mean([s["end_match"] for s in t3_scores])
        avg_count_error = mean([s["count_error"] for s in t3_scores])
        avg_abs_count_error = mean([abs(s["count_error"]) for s in t3_scores])

        has_gt_transitions = [s for s in t3_scores if s["gt_transitions"] > 0]
        has_no_gt = [s for s in t3_scores if s["gt_transitions"] == 0]

        add(f"  N = {len(t3_scores)}")
        add(f"  Starting posture accuracy: {start_acc:.1%}")
        add(f"  Ending posture accuracy: {end_acc:.1%}")
        add(f"  Mean transition count error: {avg_count_error:+.1f} "
            f"(positive = Gemini reports more)")
        add(f"  Mean absolute count error: {avg_abs_count_error:.1f}")

        if has_gt_transitions:
            avg_p = mean([s["precision_5s"] for s in has_gt_transitions])
            avg_r = mean([s["recall_5s"] for s in has_gt_transitions])
            add(f"\n  Videos WITH ground truth transitions (n={len(has_gt_transitions)}):")
            add(f"    Mean precision (5s window): {avg_p:.3f}")
            add(f"    Mean recall (5s window): {avg_r:.3f}")

        if has_no_gt:
            false_alarm_rate = mean([1 if s["pred_transitions"] > 0 else 0 for s in has_no_gt])
            avg_false = mean([s["pred_transitions"] for s in has_no_gt])
            add(f"\n  Videos WITHOUT ground truth transitions (n={len(has_no_gt)}):")
            add(f"    False alarm rate: {false_alarm_rate:.1%}")
            add(f"    Mean false transitions reported: {avg_false:.1f}")

    # ----------------------------------------------------------
    # TIER 2: INTEGRATIVE ASSESSMENT
    # ----------------------------------------------------------
    add("\n" + "-" * 70)
    add("TIER 2: INTEGRATIVE ASSESSMENT")
    add("-" * 70)

    # T4: Sitter Categorization
    add("\n--- T4: Sitter Categorization ---")
    t4_scores, t4_gt, t4_pred = score_t4(results)
    scoring_rows.extend(t4_scores)

    if t4_scores:
        acc = mean([s["match"] for s in t4_scores])
        add(f"  N = {len(t4_scores)}")
        add(f"  Overall accuracy: {acc:.1%}")

        # Confidence calibration
        for conf in ["High", "Medium", "Low"]:
            conf_scores = [s for s in t4_scores if s["confidence"] == conf]
            if conf_scores:
                conf_acc = mean([s["match"] for s in conf_scores])
                add(f"  Accuracy when confidence = {conf}: {conf_acc:.1%} "
                    f"(n={len(conf_scores)})")

        if HAS_SKLEARN and t4_gt and t4_pred:
            all_labels = ["Non-sitter", "Emergent sitter", "Independent sitter"]
            present_labels = sorted(set(t4_gt + t4_pred))
            kappa = cohen_kappa_score(t4_gt, t4_pred)
            add(f"  Cohen's kappa: {kappa:.3f}")
            add(f"\n  Classification report:")
            report = classification_report(t4_gt, t4_pred, labels=all_labels, zero_division=0)
            for line in report.split("\n"):
                add(f"    {line}")

            cm = confusion_matrix(t4_gt, t4_pred, labels=all_labels)
            add(f"\n  Confusion matrix (rows=GT, cols=Pred):")
            add(f"    Labels: {all_labels}")
            for i, row in enumerate(cm):
                add(f"    {all_labels[i]}: {list(row)}")

            cm_file = RESULTS_DIR / "confusion_matrix_T4.csv"
            with open(cm_file, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([""] + all_labels)
                for i, row in enumerate(cm):
                    w.writerow([all_labels[i]] + list(row))

    # T5: Age Estimation
    add("\n--- T5: Age Estimation ---")
    t5_scores, t5_gt, t5_pred = score_t5(results)
    scoring_rows.extend(t5_scores)

    if t5_scores:
        errors = [s["error"] for s in t5_scores]
        abs_errors = [s["abs_error"] for s in t5_scores]
        within_1 = mean([s["within_1_month"] for s in t5_scores])
        within_2 = mean([s["within_2_months"] for s in t5_scores])
        range_cap = mean([s["range_captures_actual"] for s in t5_scores])

        add(f"  N = {len(t5_scores)}")
        add(f"  Mean error: {mean(errors):+.1f} months (positive = overestimate)")
        add(f"  Mean absolute error: {mean(abs_errors):.1f} months")
        add(f"  Median absolute error: {median(abs_errors):.1f} months")
        add(f"  RMSE: {rmse(errors):.1f} months")
        add(f"  Within 1 month: {within_1:.1%}")
        add(f"  Within 2 months: {within_2:.1%}")
        add(f"  Confidence range captures actual age: {range_cap:.1%}")

        if t5_gt and t5_pred:
            r_val = safe_pearson(t5_gt, t5_pred)
            add(f"  Pearson r (estimated vs actual): {r_val:.3f}")

    # T6: Hallucination Probes
    add("\n--- T6: Hallucination Probes ---")
    t6_scores, absent_res, present_res, probe_breakdown = score_t6(results)
    scoring_rows.extend(t6_scores)

    if t6_scores:
        add(f"  Total probes administered: {len(t6_scores)}")

        if absent_res:
            absent_acc = mean(absent_res)
            hallucination_rate = 1 - absent_acc
            add(f"\n  Absent-behavior probes (expected NO):")
            add(f"    N = {len(absent_res)}")
            add(f"    Correct rejection rate: {absent_acc:.1%}")
            add(f"    Hallucination rate (false positives): {hallucination_rate:.1%}")

        if present_res:
            present_acc = mean(present_res)
            miss_rate = 1 - present_acc
            add(f"\n  Present-behavior probes (expected YES):")
            add(f"    N = {len(present_res)}")
            add(f"    Hit rate (sensitivity): {present_acc:.1%}")
            add(f"    Miss rate (false negatives): {miss_rate:.1%}")

        add(f"\n  Per-probe breakdown:")
        for probe_name in sorted(probe_breakdown.keys()):
            pb = probe_breakdown[probe_name]
            acc = pb["correct"] / pb["total"] if pb["total"] > 0 else 0
            add(f"    {probe_name}: {pb['correct']}/{pb['total']} correct ({acc:.1%})")

    # ----------------------------------------------------------
    # TIER 3: DEVELOPMENTAL REASONING
    # ----------------------------------------------------------
    add("\n" + "-" * 70)
    add("TIER 3: DEVELOPMENTAL REASONING")
    add("-" * 70)

    t7_responses, t8_responses = collect_t7_t8(results)
    add(f"\n--- T7: Developmental Justification ---")
    add(f"  Responses collected: {len(t7_responses)}")
    add(f"  These require manual rubric scoring (1-5 scale).")
    add(f"  Responses exported to: T7_rubric_scoring.csv")

    add(f"\n--- T8: Caregiving Recommendations ---")
    add(f"  Responses collected: {len(t8_responses)}")
    add(f"  These require manual rubric scoring (1-5 scale).")
    add(f"  Responses exported to: T8_rubric_scoring.csv")

    # Export T7 and T8 for rubric scoring
    for filename, responses in [("T7_rubric_scoring.csv", t7_responses),
                                 ("T8_rubric_scoring.csv", t8_responses)]:
        filepath = RESULTS_DIR / filename
        if responses:
            with open(filepath, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "video_id", "sitter_category", "age_months",
                    "response", "rater1_score", "rater2_score"
                ])
                w.writeheader()
                for resp in responses:
                    resp["rater1_score"] = ""
                    resp["rater2_score"] = ""
                    w.writerow(resp)

    # ----------------------------------------------------------
    # PERFORMANCE BY SITTER CATEGORY
    # ----------------------------------------------------------
    add("\n" + "-" * 70)
    add("PERFORMANCE BY SITTER CATEGORY")
    add("-" * 70)

    by_cat = performance_by_category(results)
    for cat in ["Non-sitter", "Emergent sitter", "Independent sitter"]:
        data = by_cat.get(cat)
        if not data:
            continue
        add(f"\n  {cat}:")
        if data["t1_primary_matches"]:
            add(f"    T1 primary posture accuracy: "
                f"{mean(data['t1_primary_matches']):.1%} "
                f"(n={len(data['t1_primary_matches'])})")
        if data["t2_maes"]:
            add(f"    T2 mean absolute error: "
                f"{mean(data['t2_maes']):.1f}% "
                f"(n={len(data['t2_maes'])})")
        if data["t4_matches"]:
            add(f"    T4 categorization accuracy: "
                f"{mean(data['t4_matches']):.1%} "
                f"(n={len(data['t4_matches'])})")
        if data["t5_abs_errors"]:
            add(f"    T5 age MAE: "
                f"{mean(data['t5_abs_errors']):.1f} months "
                f"(n={len(data['t5_abs_errors'])})")
        if data["t6_absent_correct"]:
            add(f"    T6 hallucination rate: "
                f"{1 - mean(data['t6_absent_correct']):.1%} "
                f"(n={len(data['t6_absent_correct'])})")

    # ----------------------------------------------------------
    # TASK DIFFICULTY GRADIENT
    # ----------------------------------------------------------
    add("\n" + "-" * 70)
    add("TASK DIFFICULTY GRADIENT")
    add("-" * 70)
    add("\n  Tier 1 (Perceptual Coding):")
    if t1_scores:
        add(f"    T1 primary posture accuracy: {mean([s['primary_match'] for s in t1_scores]):.1%}")
    if t2_scores:
        add(f"    T2 temporal estimation MAE: {mean([s['mae'] for s in t2_scores]):.1f}%")
    if t3_scores:
        add(f"    T3 starting posture accuracy: {mean([s['start_match'] for s in t3_scores]):.1%}")

    add("\n  Tier 2 (Integrative Assessment):")
    if t4_scores:
        add(f"    T4 categorization accuracy: {mean([s['match'] for s in t4_scores]):.1%}")
    if t5_scores:
        add(f"    T5 age estimation MAE: {mean([s['abs_error'] for s in t5_scores]):.1f} months")
    if absent_res:
        add(f"    T6 hallucination rate: {1 - mean(absent_res):.1%}")

    add("\n  Tier 3 (Developmental Reasoning):")
    add(f"    T7 and T8 require manual rubric scoring")

    add("\n" + "=" * 70)
    add("END OF REPORT")
    add("=" * 70)

    return "\n".join(lines), scoring_rows


# ============================================================
# Main
# ============================================================

def main():
    print("Loading results...")
    results = load_all_results(RESULTS_DIR)
    print(f"  Loaded {len(results)} result files.\n")

    if not results:
        print("No results found. Run sitting_eval.py first.")
        return

    print("Generating analysis report...")
    report_text, scoring_rows = generate_report(results)

    # Save report
    with open(REPORT_FILE, "w") as f:
        f.write(report_text)
    print(f"  Report saved to: {REPORT_FILE}")

    # Save scoring details CSV
    if scoring_rows:
        all_keys = set()
        for row in scoring_rows:
            all_keys.update(row.keys())
        fieldnames = ["video_id", "task"] + sorted(all_keys - {"video_id", "task"})

        with open(SCORING_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(scoring_rows)
        print(f"  Scoring details saved to: {SCORING_CSV}")

    # Print report to console
    print("\n" + report_text)


if __name__ == "__main__":
    main()
