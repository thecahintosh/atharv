# matcher.py
# Goal: Match extracted packaging name against prescription medicine list

from fuzzywuzzy import fuzz
import re


def normalize(text):
    """
    Normalize text for comparison:
    - Uppercase
    - Remove extra spaces
    - Remove punctuation
    - Remove common filler words
    """
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
    """
    Calculate fuzzy match scores between two medicine names.
    Returns a dict with individual scores and the best score.
    """
    a = normalize(name_a)
    b = normalize(name_b)

    if not a or not b:
        return {
            'ratio'      : 0,
            'partial'    : 0,
            'token_sort' : 0,
            'token_set'  : 0,
            'best'       : 0
        }

    ratio      = fuzz.ratio(a, b)
    partial    = fuzz.partial_ratio(a, b)
    token_sort = fuzz.token_sort_ratio(a, b)
    token_set  = fuzz.token_set_ratio(a, b)
    best       = max(ratio, partial, token_sort, token_set)

    return {
        'ratio'      : ratio,
        'partial'    : partial,
        'token_sort' : token_sort,
        'token_set'  : token_set,
        'best'       : best
    }


def find_best_match(packaging_name, prescription_medicines,
                    threshold=75, silent=False):
    """
    Find the best matching medicine in prescription list.

    Args:
        packaging_name        : name scanned from packaging
        prescription_medicines: list of medicine dicts
        threshold             : minimum score to count as match
        silent                : suppress print output

    Returns:
        (matched_medicine_dict, best_score)
        or (None, best_score) if below threshold
    """
    if not packaging_name or not packaging_name.strip():
        return None, 0

    if not prescription_medicines:
        return None, 0

    if not silent:
        print(f"\n🔍 Matching packaging name: '{packaging_name}'")
        print(f"   Against {len(prescription_medicines)} prescription medicines")
        print(f"   Threshold: {threshold}/100")
        print("-" * 50)

    best_match = None
    best_score = 0

    for med in prescription_medicines:
        scores = match_score(packaging_name, med['name'])
        score  = scores['best']

        if not silent:
            print(f"  vs '{med['name']}'")
            print(
                f"     ratio={scores['ratio']}  "
                f"partial={scores['partial']}  "
                f"token_sort={scores['token_sort']}  "
                f"token_set={scores['token_set']}"
            )
            print(f"     → Best: {score}")

        if score > best_score:
            best_score = score
            best_match = med

    if not silent:
        print("-" * 50)
        if best_score >= threshold:
            print(
                f"✅ MATCH FOUND: '{best_match['name']}' "
                f"(score: {best_score})"
            )
        else:
            print(
                f"❌ NO MATCH — best score was {best_score} "
                f"(below threshold {threshold})"
            )

    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


def get_confidence_level(score):
    """Translate numeric score into human readable confidence label."""
    if score >= 95:
        return "🟢 VERY HIGH"
    elif score >= 85:
        return "🟡 HIGH"
    elif score >= 75:
        return "🟠 MODERATE"
    else:
        return "🔴 LOW"


def verify_medicine(packaging_name, prescription_medicines,
                    threshold=75, silent=False):
    """
    Full verification pipeline.

    Args:
        packaging_name        : name scanned from medicine packaging
        prescription_medicines: list of medicine dicts from prescription
        threshold             : minimum score to consider a match
        silent                : if True suppress all print output

    Returns:
        (matched_medicine_dict, score)
        or (None, score) if no match found
    """
    if not silent:
        print("\n" + "=" * 60)
        print("💊 MEDICINE VERIFICATION")
        print("=" * 60)
        print(f"📦 Scanned from packaging : {packaging_name}")

    if not packaging_name or not packaging_name.strip():
        if not silent:
            print("❌ Empty input — cannot match")
            print("=" * 60)
        return None, 0

    matched_med, score = find_best_match(
        packaging_name, prescription_medicines, threshold, silent=silent
    )

    if not silent:
        print("\n📋 VERIFICATION RESULT:")
        print("-" * 60)

        if matched_med:
            confidence = get_confidence_level(score)
            print(f"  ✅ STATUS     : MATCH FOUND")
            print(f"  💊 Medicine   : {matched_med['name']}")
            print(f"  💉 Dosage     : {matched_med['dosage']}")
            print(f"  🕐 Timing     : {matched_med['timing']}")
            print(f"  📊 Score      : {score}/100")
            print(f"  📈 Confidence : {confidence}")
            if matched_med.get('notes'):
                for note in matched_med['notes']:
                    print(f"  📝 Note       : {note}")
            if score < 85:
                print(
                    f"\n  ⚠️  Moderate confidence — "
                    f"please double check manually"
                )
        else:
            print(f"  ❌ STATUS  : NO MATCH IN PRESCRIPTION")
            print(f"     '{packaging_name}' was NOT prescribed")
            print(f"     Best score: {score}/100")

        print("=" * 60)

    return matched_med, score


# ──────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":

    prescription = [
        {
            'name'   : 'COLAHYAL CAPSULE',
            'dosage' : '1-0-1',
            'timing' : 'After Food - Daily - 4 Weeks',
            'notes'  : []
        },
        {
            'name'   : 'CALCIN K 27',
            'dosage' : '0-1-0',
            'timing' : 'After Food - Daily - 4 Weeks',
            'notes'  : []
        },
        {
            'name'   : 'MVL Q10',
            'dosage' : '0-1-0',
            'timing' : 'After Food - Daily - 4 Weeks',
            'notes'  : []
        },
    ]

    print("\n--- Test 1: Verbose mode ---")
    verify_medicine("COLAHYAL", prescription)

    print("\n--- Test 2: Silent mode ---")
    matched, score = verify_medicine(
        "COLAHYAL", prescription, silent=True
    )
    print(f"matched : {matched['name'] if matched else None}")
    print(f"score   : {score}")

    print("\n--- Test 3: Empty string ---")
    matched, score = verify_medicine("", prescription, silent=True)
    print(f"matched : {matched}")
    print(f"score   : {score}")

    print("\n--- Test 4: Wrong medicine ---")
    matched, score = verify_medicine("PARACETAMOL", prescription)