# test_pipeline.py
# Structured testing of the full pipeline components

import re
from fuzzywuzzy import fuzz

# ── Copy normalize and verify_medicine from pipeline.py ──

def normalize(text):
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r'[^\w\s-]', '', text)
    fillers = ['CAPSULE', 'CAPSULES', 'TABLET', 'TABLETS',
               'SYRUP', 'INJECTION', 'MG', 'ML', 'IU']
    for word in fillers:
        text = re.sub(rf'\b{word}\b', '', text)
    return ' '.join(text.split()).strip()


def match_score(name_a, name_b):
    a = normalize(name_a)
    b = normalize(name_b)
    return max(
        fuzz.ratio(a, b),
        fuzz.partial_ratio(a, b),
        fuzz.token_sort_ratio(a, b),
        fuzz.token_set_ratio(a, b)
    )


# ── Test Data ─────────────────────────────────────────────

# Simulated prescription from our real image
PRESCRIPTION = [
    {'name': 'CRWITH BK FIBER CAST APPLIED LT SIDE',
     'dosage': '1-0-0', 'timing': '4 Weeks'},
    {'name': 'CALCIN K 27',
     'dosage': '0-1-0', 'timing': 'After Food - Daily - 4 Weeks'},
    {'name': 'CALCIDOL D3',
     'dosage': '0-1-0', 'timing': 'After Food - Weekly - 4 Weeks'},
    {'name': 'COLAHYAL CAPSULE',
     'dosage': '1-0-1', 'timing': 'After Food - Daily - 4 Weeks'},
    {'name': 'MVL Q10',
     'dosage': '0-1-0', 'timing': 'After Food - Daily - 4 Weeks'},
]

# ── Test Suite ────────────────────────────────────────────

TESTS = [
    # (test_name, packaging_ocr_output, expected_match, expected_medicine)
    ("Perfect match",           "COLAHYAL",          True,  "COLAHYAL CAPSULE"),
    ("With capsule suffix",     "COLAHYAL CAPSULE",  True,  "COLAHYAL CAPSULE"),
    ("Truncated by OCR",        "COLAHYAL CAPSU",    True,  "COLAHYAL CAPSULE"),
    ("Lowercase OCR output",    "colahyal capsule",  True,  "COLAHYAL CAPSULE"),
    ("Extra noise chars",       "##COLAHYAL##",      True,  "COLAHYAL CAPSULE"),
    ("Space in number",         "CALCIN K 27",       True,  "CALCIN K 27"),
    ("No space in number",      "CALCINK27",         True,  "CALCIN K 27"),
    ("Wrong medicine",          "PARACETAMOL",       False, None),
    ("Random noise",            "XJ92 KKP",          False, None),
    ("Empty string",            "",                  False, None),
    ("Partial name only",       "MVL",               True,  "MVL Q10"),
    ("Similar but wrong",       "CALCIDAY D3",       True,  "CALCIDOL D3"),
]


def run_tests():
    print("\n" + "=" * 70)
    print("🧪 PIPELINE TEST SUITE")
    print("=" * 70)
    print(f"{'#':<3} {'Test Name':<25} {'Scanned':<20} "
          f"{'Expected':<8} {'Got':<8} {'Score':<6} {'Pass?'}")
    print("-" * 70)

    passed = 0
    failed = 0
    failures = []

    for i, (test_name, scanned, expect_match, expect_med) in enumerate(TESTS, 1):

        # Find best match
        best_match = None
        best_score = 0

        for med in PRESCRIPTION:
            score = match_score(scanned, med['name'])
            if score > best_score:
                best_score = score
                best_match = med

        threshold = 75
        got_match = best_score >= threshold
        got_med   = best_match['name'] if got_match else None

        # Did we get the right answer?
        correct = (got_match == expect_match)
        if correct and expect_med:
            correct = (got_med == expect_med)

        status = "✅" if correct else "❌"
        if correct:
            passed += 1
        else:
            failed += 1
            failures.append({
                'test'     : test_name,
                'scanned'  : scanned,
                'expected' : expect_med,
                'got'      : got_med,
                'score'    : best_score
            })

        expect_str = expect_med[:10] if expect_med else "None"
        got_str    = got_med[:10]    if got_med    else "None"

        print(f"{i:<3} {test_name:<25} {scanned:<20} "
              f"{expect_str:<12} {got_str:<12} {best_score:<6} {status}")

    print("-" * 70)
    print(f"\n📊 RESULTS: {passed}/{len(TESTS)} passed  "
          f"| {failed} failed\n")

    if failures:
        print("❌ FAILED TESTS — details:")
        print("-" * 40)
        for f in failures:
            print(f"  Test    : {f['test']}")
            print(f"  Scanned : '{f['scanned']}'")
            print(f"  Expected: {f['expected']}")
            print(f"  Got     : {f['got']}")
            print(f"  Score   : {f['score']}/100")
            print()

    return passed, failed


if __name__ == "__main__":
    run_tests()