# parse_prescription.py
# Goal: Take raw OCR text and extract structured medicine data

import re


def clean_text(raw_text):
    """Basic cleanup of common OCR errors."""

    # FIX 1: Replace letter O with 0 in dosage-like patterns
    def fix_dosage_o(text):
        return re.sub(
            r'\b([O0])\s*-\s*([O0-9])\s*-\s*([O0-9])\b',
            lambda m: f"{m.group(1).replace('O','0')}-"
                      f"{m.group(2).replace('O','0')}-"
                      f"{m.group(3).replace('O','0')}",
            text
        )

    raw_text = fix_dosage_o(raw_text)

    replacements = {
        'Daity'    : 'Daily',
        'Datty'    : 'Daily',
        'FOLLO\\Y' : 'FOLLOW',
        'Tatar'    : 'After',
        'sutete'   : 'sulfate',
        # FIX 3: Remove stray quotes/symbols
        '"'        : '',
        "'"        : '',
        '='        : '',
    }

    cleaned = raw_text
    for wrong, correct in replacements.items():
        cleaned = cleaned.replace(wrong, correct)

    return cleaned


def extract_medicines(raw_text):
    """
    Parse raw OCR text into a structured list of medicines.

    Expected OCR pattern:
    1) MEDICINE NAME   1-0-0   After Food - Daily - 4 Weeks
    2) MEDICINE NAME   0-1-0   After Food - Weekly - 4 Weeks
    """

    cleaned = clean_text(raw_text)
    medicines = []

    lines = cleaned.split('\n')
    current_medicine = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Pattern: line starts with a number like "1)" or "1." or "5}"
        medicine_start = re.match(r'^[\d]+[\)\.\}]\s*(.+)', line)

        if medicine_start:
            # Save previous medicine if exists
            if current_medicine:
                medicines.append(current_medicine)

            content = medicine_start.group(1).strip()

            # Extract dosage pattern: digits-digits-digits (e.g. 1-0-0)
            dosage_match = re.search(r'(\d+\s*-\s*\d+\s*-\s*\d+)', content)

            if dosage_match:
                dosage_raw = dosage_match.group(1)

                # Normalize dosage: remove spaces around dashes
                dosage = re.sub(r'\s*-\s*', '-', dosage_raw)

                # Medicine name = everything BEFORE the dosage
                dosage_start = content.index(dosage_raw)
                name = content[:dosage_start].strip()

                # Timing = everything AFTER the dosage
                timing = content[dosage_start + len(dosage_raw):].strip()
                timing = timing.lstrip('-').strip()

            else:
                # FIX 2: Try splitting by 2+ spaces (table columns)
                parts = re.split(r'\s{2,}', content)
                if len(parts) >= 2:
                    name   = parts[0].strip()
                    timing = ' '.join(parts[1:]).strip()
                else:
                    name   = content
                    timing = "Not found"
                dosage = "Not found"

            current_medicine = {
                'name'   : name,
                'dosage' : dosage,
                'timing' : timing,
                'notes'  : []
            }

        # Pattern: composition or notes line
        elif current_medicine and re.match(
                r'^(Notes?|Composition|Compos)', line, re.IGNORECASE):
            current_medicine['notes'].append(line.strip())

    # Don't forget the last medicine
    if current_medicine:
        medicines.append(current_medicine)

    return medicines


def display_medicines(medicines):
    """Pretty print the extracted medicine list."""
    print("\n💊 EXTRACTED MEDICINES:")
    print("=" * 60)

    if not medicines:
        print("❌ No medicines found. Check OCR output quality.")
        return

    for i, med in enumerate(medicines, 1):
        print(f"\n[{i}] Medicine : {med['name']}")
        print(f"    Dosage  : {med['dosage']}")
        print(f"    Timing  : {med['timing']}")
        if med['notes']:
            for note in med['notes']:
                print(f"    Note    : {note}")

    print("\n" + "=" * 60)
    print(f"✅ Total medicines found: {len(medicines)}")


# ── Test with your actual OCR output ──────────────────────
if __name__ == "__main__":

    raw_ocr_text = """
Medicine Dosage Timing - Freq. - Duration
1) CRWITH BK FIBER CAST APPLIED LT SIDE 1- 0-0 - 4 Weeks
Notas : FOLLO\Y PLASTER INSTRUCTION
2) CALCIN K 27 0- 1-0 After Food - Daity - 4 Weeks
3) CALCIDOL D3 = Tatar Foods Weekly 4 Weeks
Compasson Vitamin D3 60000 IU
4) "COLAHYAL CAPSULE 1- 0-1 After Food - Daity - 4 Weeks
Compos von _ Chonironn sutete 200 MG + Colagen peptide 300 RAG + Sodium hyaluronate 40 MG
5} MVL Q10 O- 1-0 After Food - Datty - 4 Weeks
$$ ODE Day 4 Weeks
    """

    medicines = extract_medicines(raw_ocr_text)
    display_medicines(medicines)