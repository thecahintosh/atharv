# metrics.py
# Clean table-based metrics for OCR, Parser, Matcher, and RAG

import re
import sys
import time
import json
import difflib
from pathlib     import Path
from fuzzywuzzy  import fuzz
from dataclasses import dataclass
from typing      import List, Optional
from dotenv      import load_dotenv

load_dotenv()

# ── Path setup ───────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ── Imports ──────────────────────────────────────────────────────────
print("\n" + "═" * 72)
print("  🔌  IMPORTING PROJECT MODULES")
print("═" * 72)

try:
    from parser     import extract_medicines
    print("  ✅ parser       → extract_medicines()")
except ImportError as e:
    print(f"  ❌ parser       FAILED: {e}")
    extract_medicines = None

try:
    from matcher    import verify_medicine
    print("  ✅ matcher      → verify_medicine()")
except ImportError as e:
    print(f"  ❌ matcher      FAILED: {e}")
    verify_medicine = None

try:
    from rag_engine import explain_prescription
    print("  ✅ rag_engine   → explain_prescription()")
except ImportError as e:
    print(f"  ❌ rag_engine   FAILED: {e}")
    explain_prescription = None


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def divider(char="─", width=72):
    print(char * width)


def header(title):
    print()
    divider("═")
    print(f"  {title}")
    divider("═")


def character_error_rate(hypothesis, reference):
    if not reference:
        return 1.0
    m = difflib.SequenceMatcher(None, hypothesis, reference)
    return round(1.0 - m.ratio(), 4)


def word_error_rate(hypothesis, reference):
    ref = reference.upper().split()
    hyp = hypothesis.upper().split()
    if not ref:
        return 1.0
    r, c = len(ref) + 1, len(hyp) + 1
    d = [[0] * c for _ in range(r)]
    for i in range(r): d[i][0] = i
    for j in range(c): d[0][j] = j
    for i in range(1, r):
        for j in range(1, c):
            cost    = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )
    return round(d[len(ref)][len(hyp)] / len(ref), 4)


def readability(text):
    sentences = [
        s.strip() for s in re.split(r'[.!?]+', text) if s.strip()
    ]
    if not sentences:
        return 0.0
    return round(
        sum(len(s.split()) for s in sentences) / len(sentences), 1
    )


def has_keywords(response, keywords):
    low = response.lower()
    return all(k.lower() in low for k in keywords)


def no_hallucination(response):
    flags = [
        "i think", "i believe", "i'm not sure", "i am not sure",
        "probably", "i cannot verify", "i don't know", "i do not know"
    ]
    low = response.lower()
    return not any(f in low for f in flags)


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — OCR
# ══════════════════════════════════════════════════════════════════════

OCR_TESTS = [
    {
        "test"    : "Clean line",
        "ocr"     : "1) CALCIN K 27 0-1-0 After Food Daily 4 Weeks",
        "expected": "1) CALCIN K 27 0-1-0 After Food Daily 4 Weeks",
    },
    {
        "test"    : "O → 0 in dosage",
        "ocr"     : "2) COLAHYAL CAPSULE 1-O-1 After Food Daily",
        "expected": "2) COLAHYAL CAPSULE 1-0-1 After Food Daily",
    },
    {
        "test"    : "Missing hyphens",
        "ocr"     : "3) MVL Q10 0 1 0 After Food Daily 4 Weeks",
        "expected": "3) MVL Q10 0-1-0 After Food Daily 4 Weeks",
    },
    {
        "test"    : "Garbled name",
        "ocr"     : "4) CALC1DOL D3 0-1-0 Aftor Food Weekty",
        "expected": "4) CALCIDOL D3 0-1-0 After Food Weekly",
    },
    {
        "test"    : "Noise symbols",
        "ocr"     : "5) ##COLAHYAL## CAPSULE 1-0-1 After Food",
        "expected": "5) COLAHYAL CAPSULE 1-0-1 After Food",
    },
]


def run_ocr_metrics(threshold_cer=0.15):
    header("SECTION 1 — OCR METRICS")

    col = [22, 7, 7, 6, 6]
    print(
        f"  {'Test':<{col[0]}} {'CER':>{col[1]}} "
        f"{'WER':>{col[2]}} {'Sim':>{col[3]}} {'Pass':>{col[4]}}"
    )
    divider()

    rows   = []
    passed = 0

    for t in OCR_TESTS:
        cer    = character_error_rate(t["ocr"], t["expected"])
        wer    = word_error_rate(t["ocr"], t["expected"])
        sim    = fuzz.ratio(t["ocr"].upper(), t["expected"].upper())
        ok     = cer <= threshold_cer
        passed += int(ok)
        mark   = "✅" if ok else "❌"

        rows.append({
            "Test": t["test"], "CER": cer,
            "WER": wer, "Sim": sim, "Pass": mark
        })

        print(
            f"  {t['test']:<{col[0]}} {cer:>{col[1]}.3f} "
            f"{wer:>{col[2]}.3f} {sim:>{col[3]}}% {mark:>{col[4]}}"
        )

    divider()
    avg_cer = round(sum(r["CER"] for r in rows) / len(rows), 3)
    avg_wer = round(sum(r["WER"] for r in rows) / len(rows), 3)
    avg_sim = round(sum(r["Sim"] for r in rows) / len(rows), 1)
    print(
        f"  {'AVERAGE':<{col[0]}} {avg_cer:>{col[1]}.3f} "
        f"{avg_wer:>{col[2]}.3f} {avg_sim:>{col[3]}}%"
    )
    print(
        f"\n  Passed : {passed}/{len(OCR_TESTS)}"
        f"   Threshold : CER ≤ {threshold_cer}"
    )

    return rows, passed, len(OCR_TESTS)


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — PARSER
# ══════════════════════════════════════════════════════════════════════

PARSER_TESTS = [
    {
        "test"  : "Standard format",
        "input" : """
1) CALCIN K 27 0-1-0 After Food - Daily - 4 Weeks
2) COLAHYAL CAPSULE 1-0-1 After Food - Daily - 4 Weeks
3) MVL Q10 0-1-0 After Food - Daily - 4 Weeks
""",
        "expected": [
            {"name": "CALCIN K 27",      "dosage": "0-1-0",
             "timing": "After Food - Daily - 4 Weeks"},
            {"name": "COLAHYAL CAPSULE", "dosage": "1-0-1",
             "timing": "After Food - Daily - 4 Weeks"},
            {"name": "MVL Q10",          "dosage": "0-1-0",
             "timing": "After Food - Daily - 4 Weeks"},
        ],
    },
    {
        "test"  : "Dot numbered list",
        "input" : """
1. CALCIDOL D3 0-1-0 After Food - Weekly - 4 Weeks
2. MVL Q10 0-1-0 After Food - Daily - 4 Weeks
""",
        "expected": [
            {"name": "CALCIDOL D3", "dosage": "0-1-0",
             "timing": "After Food - Weekly - 4 Weeks"},
            {"name": "MVL Q10",     "dosage": "0-1-0",
             "timing": "After Food - Daily - 4 Weeks"},
        ],
    },
    {
        "test"  : "Brace numbered list",
        "input" : """
1} CALCIN K 27 0-1-0 After Food - Daily - 4 Weeks
2} MVL Q10 0-1-0 After Food - Daily - 4 Weeks
""",
        "expected": [
            {"name": "CALCIN K 27", "dosage": "0-1-0",
             "timing": "After Food - Daily - 4 Weeks"},
            {"name": "MVL Q10",     "dosage": "0-1-0",
             "timing": "After Food - Daily - 4 Weeks"},
        ],
    },
    {
        "test"  : "With notes line",
        "input" : """
1) CALCIDOL D3 0-1-0 After Food - Weekly - 4 Weeks
Notes: Vitamin D3 60000 IU
2) MVL Q10 0-1-0 After Food - Daily - 4 Weeks
""",
        "expected": [
            {"name": "CALCIDOL D3", "dosage": "0-1-0",
             "timing": "After Food - Weekly - 4 Weeks"},
            {"name": "MVL Q10",     "dosage": "0-1-0",
             "timing": "After Food - Daily - 4 Weeks"},
        ],
    },
    {
        "test"  : "Mixed clean OCR",
        "input" : """
1) COLAHYAL CAPSULE 1-0-1 After Food - Daily - 4 Weeks
2) CALCIN K 27 0-1-0 After Food - Daily - 4 Weeks
3) CALCIDOL D3 0-1-0 After Food - Weekly - 4 Weeks
""",
        "expected": [
            {"name": "COLAHYAL CAPSULE", "dosage": "1-0-1",
             "timing": "After Food - Daily - 4 Weeks"},
            {"name": "CALCIN K 27",      "dosage": "0-1-0",
             "timing": "After Food - Daily - 4 Weeks"},
            {"name": "CALCIDOL D3",      "dosage": "0-1-0",
             "timing": "After Food - Weekly - 4 Weeks"},
        ],
    },
]


def score_parser_test(got_meds, exp_meds):
    """Score one parser test. Returns (name_rate, dosage_rate, timing_rate)."""
    n = len(exp_meds)
    if n == 0:
        return 0.0, 0.0, 0.0

    name_scores    = []
    dosage_correct = []
    timing_scores  = []

    for exp in exp_meds:
        best_score = 0
        best_got   = None

        for got in got_meds:
            s = fuzz.token_set_ratio(
                exp["name"].upper(), got["name"].upper()
            )
            if s > best_score:
                best_score = s
                best_got   = got

        name_scores.append(best_score / 100)

        if best_got:
            dosage_correct.append(
                1.0 if best_got["dosage"] == exp["dosage"] else 0.0
            )
            timing_scores.append(
                fuzz.partial_ratio(
                    exp["timing"].upper(),
                    best_got["timing"].upper()
                ) / 100
            )
        else:
            dosage_correct.append(0.0)
            timing_scores.append(0.0)

    return (
        sum(name_scores)    / n,
        sum(dosage_correct) / n,
        sum(timing_scores)  / n,
    )


def run_parser_metrics(threshold=0.80):
    header("SECTION 2 — PARSER METRICS")

    if extract_medicines is None:
        print("  ⏭️  Skipped — extract_medicines not imported")
        return [], 0, 0

    col = [22, 8, 8, 8, 9, 6]
    print(
        f"  {'Test':<{col[0]}} {'Names':>{col[1]}} {'Dosage':>{col[2]}} "
        f"{'Timing':>{col[3]}} {'Overall':>{col[4]}} {'Pass':>{col[5]}}"
    )
    divider()

    rows   = []
    passed = 0

    for t in PARSER_TESTS:
        got_meds   = extract_medicines(t["input"])
        nr, dr, tr = score_parser_test(got_meds, t["expected"])
        overall    = (nr + dr + tr) / 3
        ok         = overall >= threshold
        passed    += int(ok)
        mark       = "✅" if ok else "❌"

        rows.append({
            "Test": t["test"], "Names": nr, "Dosage": dr,
            "Timing": tr, "Overall": overall, "Pass": mark
        })

        print(
            f"  {t['test']:<{col[0]}} {nr:>{col[1]}.0%} "
            f"{dr:>{col[2]}.0%} {tr:>{col[3]}.0%} "
            f"{overall:>{col[4]}.0%} {mark:>{col[5]}}"
        )

    divider()
    avg_n = sum(r["Names"]   for r in rows) / len(rows)
    avg_d = sum(r["Dosage"]  for r in rows) / len(rows)
    avg_t = sum(r["Timing"]  for r in rows) / len(rows)
    avg_o = sum(r["Overall"] for r in rows) / len(rows)
    print(
        f"  {'AVERAGE':<{col[0]}} {avg_n:>{col[1]}.0%} "
        f"{avg_d:>{col[2]}.0%} {avg_t:>{col[3]}.0%} "
        f"{avg_o:>{col[4]}.0%}"
    )
    print(
        f"\n  Passed : {passed}/{len(PARSER_TESTS)}"
        f"   Threshold : overall ≥ {threshold:.0%}"
    )

    return rows, passed, len(PARSER_TESTS)


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — MATCHER
# ══════════════════════════════════════════════════════════════════════

PRESCRIPTION = [
    {"name": "CRWITH BK FIBER CAST APPLIED LT SIDE",
     "dosage": "1-0-0", "timing": "4 Weeks", "notes": []},
    {"name": "CALCIN K 27",
     "dosage": "0-1-0", "timing": "After Food - Daily - 4 Weeks",
     "notes": []},
    {"name": "CALCIDOL D3",
     "dosage": "0-1-0", "timing": "After Food - Weekly - 4 Weeks",
     "notes": []},
    {"name": "COLAHYAL CAPSULE",
     "dosage": "1-0-1", "timing": "After Food - Daily - 4 Weeks",
     "notes": []},
    {"name": "MVL Q10",
     "dosage": "0-1-0", "timing": "After Food - Daily - 4 Weeks",
     "notes": []},
]

MATCHER_TESTS = [
    # (label, scanned, expect_match, expect_medicine)
    ("Perfect match",      "COLAHYAL",          True,  "COLAHYAL CAPSULE"),
    ("Capsule suffix",     "COLAHYAL CAPSULE",  True,  "COLAHYAL CAPSULE"),
    ("Truncated OCR",      "COLAHYAL CAPSU",    True,  "COLAHYAL CAPSULE"),
    ("Lowercase input",    "colahyal capsule",  True,  "COLAHYAL CAPSULE"),
    ("Noise chars",        "##COLAHYAL##",      True,  "COLAHYAL CAPSULE"),
    ("Space in number",    "CALCIN K 27",       True,  "CALCIN K 27"),
    ("No space in number", "CALCINK27",         True,  "CALCIN K 27"),
    ("Partial name",       "MVL",               True,  "MVL Q10"),
    ("Similar wrong name", "CALCIDAY D3",       True,  "CALCIDOL D3"),
    ("Wrong medicine",     "PARACETAMOL",       False, None),
    ("Random noise",       "XJ92 KKP",          False, None),
    ("Empty string",       "",                  False, None),
]


def run_matcher_metrics(threshold=75):
    header("SECTION 3 — MATCHER METRICS")

    if verify_medicine is None:
        print("  ⏭️  Skipped — verify_medicine not imported")
        return [], 0, 0

    col = [22, 16, 16, 7, 6]
    print(
        f"  {'Test':<{col[0]}} {'Expected':<{col[1]}} "
        f"{'Got':<{col[2]}} {'Score':>{col[3]}} {'Pass':>{col[4]}}"
    )
    divider()

    rows   = []
    passed = 0

    for label, scanned, exp_match, exp_med in MATCHER_TESTS:
        matched, score = verify_medicine(
            scanned, PRESCRIPTION, threshold, silent=True
        )
        got_match = matched is not None
        got_med   = matched["name"] if matched else None

        correct    = (got_match == exp_match)
        if correct and exp_med:
            correct = (got_med == exp_med)

        passed += int(correct)
        mark    = "✅" if correct else "❌"

        exp_str = (exp_med[:14] if exp_med else "—")
        got_str = (got_med[:14] if got_med else "—")

        rows.append({
            "Test"     : label,
            "Expected" : exp_str,
            "Got"      : got_str,
            "Score"    : score,
            "Pass"     : mark,
            "exp_match": exp_match,
            "got_match": got_match
        })

        print(
            f"  {label:<{col[0]}} {exp_str:<{col[1]}} "
            f"{got_str:<{col[2]}} {score:>{col[3]}} {mark:>{col[4]}}"
        )

    divider()

    tp = sum(1 for r in rows if     r["exp_match"] and     r["got_match"])
    fp = sum(1 for r in rows if not r["exp_match"] and     r["got_match"])
    fn = sum(1 for r in rows if     r["exp_match"] and not r["got_match"])
    tn = sum(1 for r in rows if not r["exp_match"] and not r["got_match"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    accuracy  = (tp + tn) / len(rows) if rows else 0.0

    print(f"\n  Passed    : {passed}/{len(MATCHER_TESTS)}")
    print(f"  Precision : {precision:.1%}")
    print(f"  Recall    : {recall:.1%}")
    print(f"  F1 Score  : {f1:.1%}")
    print(f"  Accuracy  : {accuracy:.1%}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return rows, passed, len(MATCHER_TESTS)


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — RAG
# ══════════════════════════════════════════════════════════════════════

RAG_MEDICINES = [
    {
        "name"   : "CALCIDOL D3",
        "dosage" : "0-1-0",
        "timing" : "After Food - Weekly - 4 Weeks",
        "notes"  : ["Composition: Vitamin D3 60000 IU"]
    },
    {
        "name"   : "COLAHYAL CAPSULE",
        "dosage" : "1-0-1",
        "timing" : "After Food - Daily - 4 Weeks",
        "notes"  : []
    },
    {
        "name"   : "CALCIN K 27",
        "dosage" : "0-1-0",
        "timing" : "After Food - Daily - 4 Weeks",
        "notes"  : []
    },
]

RAG_TESTS = [
    {
        "test"    : "General explanation",
        "question": None,
        "keywords": ["food", "week", "take"],
    },
    {
        "test"    : "Missed dose",
        "question": "I forgot my afternoon medicine. What should I do?",
        "keywords": ["forget", "next", "doctor"],
    },
    {
        "test"    : "Food interaction",
        "question": "Should I take these with food?",
        "keywords": ["food", "after", "take"],
    },
    {
        "test"    : "Side effects",
        "question": "What side effects should I watch for?",
        "keywords": ["side", "effect", "doctor"],
    },
    {
        "test"    : "Duration",
        "question": "How long do I need to take these?",
        "keywords": ["week", "doctor"],
    },
]


def run_rag_metrics(threshold_read=20.0):
    header("SECTION 4 — RAG METRICS")

    if explain_prescription is None:
        print("  ⏭️  Skipped — explain_prescription not imported")
        return [], 0, 0

    col = [22, 6, 8, 7, 8, 6]
    print(
        f"  {'Test':<{col[0]}} {'Info':>{col[1]}} {'Halluc':>{col[2]}} "
        f"{'Read':>{col[3]}} {'Time(s)':>{col[4]}} {'Pass':>{col[5]}}"
    )
    divider()

    rows   = []
    passed = 0

    for t in RAG_TESTS:
        start    = time.time()
        response = explain_prescription(
            RAG_MEDICINES,
            user_question=t["question"],
            silent=True
        )
        elapsed  = round(time.time() - start, 2)

        info    = has_keywords(response, t["keywords"])
        halluc  = no_hallucination(response)
        read    = readability(response)
        ok      = info and halluc and read <= threshold_read
        passed += int(ok)
        mark    = "✅" if ok else "❌"

        info_m   = "✅" if info   else "❌"
        halluc_m = "✅" if halluc else "⚠️"

        rows.append({
            "Test"  : t["test"],
            "Info"  : info,
            "Halluc": halluc,
            "Read"  : read,
            "Time"  : elapsed,
            "Pass"  : mark
        })

        print(
            f"  {t['test']:<{col[0]}} {info_m:>{col[1]}} "
            f"{halluc_m:>{col[2]}} {read:>{col[3]}.1f} "
            f"{elapsed:>{col[4]}.1f} {mark:>{col[5]}}"
        )

        # Show debug info for any failed test
        if not ok:
            print()
            if not info:
                missing = [
                    k for k in t["keywords"]
                    if k.lower() not in response.lower()
                ]
                print(f"  ⚠️  Missing keywords : {missing}")
            if read > threshold_read:
                print(
                    f"  ⚠️  Readability {read} "
                    f"> threshold {threshold_read}"
                )
            snippet = response[:150].replace('\n', ' ')
            print(f"  ⚠️  Preview : \"{snippet}\"")
            print()

    divider()
    avg_read = round(sum(r["Read"] for r in rows) / len(rows), 1)
    avg_time = round(sum(r["Time"] for r in rows) / len(rows), 1)
    print(
        f"  {'AVERAGE':<{col[0]}} {'':>{col[1]}} {'':>{col[2]}} "
        f"{avg_read:>{col[3]}.1f} {avg_time:>{col[4]}.1f}"
    )
    print(
        f"\n  Passed : {passed}/{len(RAG_TESTS)}"
        f"   Threshold : readability ≤ {threshold_read} words/sentence"
    )

    return rows, passed, len(RAG_TESTS)


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════

def print_summary(results):
    header("OVERALL SYSTEM HEALTH")

    col = [12, 10, 8, 22]
    print(
        f"  {'Component':<{col[0]}} {'Passed':>{col[1]}} "
        f"{'Rate':>{col[2]}} {'Bar':<{col[3]}}"
    )
    divider()

    for name, passed, total in results:
        if total == 0:
            print(f"  {name:<{col[0]}} {'not run':>{col[1]+col[2]+2}}")
            continue
        pct    = passed / total
        filled = int(pct * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        mark   = "✅" if pct == 1.0 else ("⚠️ " if pct >= 0.75 else "❌")
        print(
            f"  {name:<{col[0]}} {passed:>{col[1]-4}}/{total:<4} "
            f"{pct:>{col[2]}.0%} {bar} {mark}"
        )

    divider()


def save_report(results_map, path="metrics_report.json"):
    with open(path, "w") as f:
        json.dump(results_map, f, indent=2)
    print(f"\n  📄 Report saved → {path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print()
    divider("═")
    print("  🏥  MEDICINE VERIFICATION SYSTEM — METRICS")
    divider("═")

    # 1. OCR
    _, ocr_p, ocr_t = run_ocr_metrics(threshold_cer=0.15)

    # 2. Parser
    _, par_p, par_t = run_parser_metrics(threshold=0.80)

    # 3. Matcher
    _, mat_p, mat_t = run_matcher_metrics(threshold=75)

    # 4. RAG
    run_rag = input(
        "\n  Run RAG evaluation? (requires internet)  [y/n]: "
    ).strip().lower()

    rag_p = rag_t = 0
    if run_rag == "y":
        _, rag_p, rag_t = run_rag_metrics(threshold_read=20.0)
    else:
        print("  ⏭️  RAG skipped.")

    # Summary
    print_summary([
        ("OCR",     ocr_p, ocr_t),
        ("Parser",  par_p, par_t),
        ("Matcher", mat_p, mat_t),
        ("RAG",     rag_p, rag_t),
    ])

    save_report({
        "ocr"    : {"passed": ocr_p, "total": ocr_t},
        "parser" : {"passed": par_p, "total": par_t},
        "matcher": {"passed": mat_p, "total": mat_t},
        "rag"    : {"passed": rag_p, "total": rag_t},
    })