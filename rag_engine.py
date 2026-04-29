# rag_engine.py
# RAG engine — Groq API version
# Retrieves medicine context and generates plain English explanation

import json
import os
from pathlib    import Path
from groq       import Groq
from fuzzywuzzy import fuzz
from dotenv     import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

GROQ_API_KEY        = os.environ.get("GROQ_API_KEY")
GROQ_MODEL          = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
KNOWLEDGE_BASE_PATH = "knowledge_base/medicines.json"


# ──────────────────────────────────────────────
# KNOWLEDGE BASE
# ──────────────────────────────────────────────

def load_knowledge_base(silent=False):
    """Load local medicine knowledge base from JSON."""
    kb_path = Path(KNOWLEDGE_BASE_PATH)
    if not kb_path.exists():
        if not silent:
            print("⚠️  Knowledge base not found — using general knowledge only")
        return {}
    with open(kb_path, 'r') as f:
        data = json.load(f)
    if not silent:
        print(f"✅ Knowledge base loaded — {len(data)} medicines")
    return data


def retrieve_medicine_info(medicine_name, knowledge_base):
    """
    Fuzzy match medicine name against knowledge base keys.
    Returns best matching entry or None.
    """
    if not knowledge_base:
        return None, None, 0

    best_key   = None
    best_score = 0

    for key in knowledge_base:
        score = max(
            fuzz.ratio(medicine_name.upper(), key.upper()),
            fuzz.partial_ratio(medicine_name.upper(), key.upper()),
            fuzz.token_set_ratio(medicine_name.upper(), key.upper())
        )
        if score > best_score:
            best_score = score
            best_key   = key

    if best_score >= 70 and best_key:
        return knowledge_base[best_key], best_key, best_score

    return None, None, 0


def decode_dosage(dosage_str):
    """
    Convert 1-0-1 format into human readable string.
    1-0-1 → Morning and Night
    0-1-0 → Afternoon only
    """
    if dosage_str == "Not found":
        return "as directed by your doctor"

    parts = dosage_str.split('-')
    if len(parts) != 3:
        return dosage_str

    labels = ['Morning', 'Afternoon', 'Night']
    times  = []

    for i, part in enumerate(parts):
        if part.strip() == '1':
            times.append(labels[i])

    if not times:
        return "not needed today"

    return ' and '.join(times)


def build_context(medicines, knowledge_base):
    """
    Build retrieval context block for the LLM.
    Combines prescription data with knowledge base info.
    """
    context_blocks = []

    for med in medicines:
        block = []

        readable_dose = decode_dosage(med['dosage'])
        block.append(f"Medicine      : {med['name']}")
        block.append(f"When to take  : {readable_dose}")
        block.append(f"Timing        : {med['timing']}")

        if med.get('notes'):
            block.append(f"Doctor notes  : {'; '.join(med['notes'])}")

        kb_info, matched_key, score = retrieve_medicine_info(
            med['name'], knowledge_base
        )

        if kb_info:
            block.append(
                f"Generic name  : {kb_info.get('generic_name', 'N/A')}"
            )
            block.append(
                f"Drug class    : {kb_info.get('drug_class', 'N/A')}"
            )
            block.append(
                f"Used for      : {kb_info.get('used_for', 'N/A')}"
            )
            block.append(
                f"Side effects  : {kb_info.get('common_side_effects', 'N/A')}"
            )
            block.append(
                f"With food     : {kb_info.get('food_interaction', 'N/A')}"
            )
            block.append(
                f"Special notes : {kb_info.get('special_notes', 'N/A')}"
            )
        else:
            block.append(
                "(No additional information found in knowledge base)"
            )

        context_blocks.append('\n'.join(block))

    return '\n\n---\n\n'.join(context_blocks)


# ──────────────────────────────────────────────
# PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a kind, patient, and helpful medical assistant.
Your job is to explain doctor prescriptions to elderly patients
who find medical terms confusing and difficult to understand.

Your strict rules:
- Use very simple plain English only
- Write SHORT sentences — maximum 10 words per sentence
- Never use medical jargon without immediately explaining it
- Be warm, calm, and reassuring in your tone
- Only use information given to you in the context
- Never invent drug names, doses, side effects, or interactions
- Structure your response so each medicine has its own clear section
- Always end with a reminder to follow the doctor instructions exactly
- If the patient asks a question answer that specific question first"""


def build_user_message(medicines, knowledge_base, user_question=None):
    """Build the full user message with retrieved context."""
    context = build_context(medicines, knowledge_base)

    if user_question:
        message = f"""Here is the patient's prescription information:

{context}

The elderly patient is asking this question:
"{user_question}"

Please answer their specific question simply and clearly first.
Then briefly remind them of the key points about their medicines.
Use only the information from the prescription context above."""

    else:
        message = f"""Here is the patient's prescription information:

{context}

Please explain this full prescription to the patient in simple words.

For each medicine please cover:
1. The medicine name in simple terms
2. Exactly when to take it — morning, afternoon, or night
3. Whether to take it with food
4. What it is for in one simple sentence
5. How long to take it
6. Any important warnings or special notes

Keep the tone warm and reassuring.
The patient is elderly and may be worried.
Help them feel calm and confident."""

    return message


# ──────────────────────────────────────────────
# GROQ API CALL
# ──────────────────────────────────────────────

def call_groq(system_prompt, user_message):
    """Call Groq API. Returns generated text string."""
    if not GROQ_API_KEY:
        return (
            "⚠️ No API key found.\n"
            "Please check your .env file has GROQ_API_KEY set correctly."
        )

    try:
        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ],
            temperature = 0.3,
            max_tokens  = 800,
            top_p       = 0.9,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        error_msg = str(e)

        if "invalid_api_key" in error_msg.lower():
            return (
                "⚠️ Invalid API key.\n"
                "Please check your GROQ_API_KEY in the .env file."
            )
        elif "rate_limit" in error_msg.lower():
            return (
                "⚠️ Rate limit reached.\n"
                "Please wait a minute and try again."
            )
        elif "model_decommissioned" in error_msg.lower():
            return (
                "⚠️ Model no longer available.\n"
                "Please update GROQ_MODEL in your .env file.\n"
                "Current working model: llama-3.3-70b-versatile"
            )
        elif "connection" in error_msg.lower():
            return (
                "⚠️ No internet connection.\n"
                "Please check your connection and try again."
            )
        else:
            return f"⚠️ Unexpected error: {error_msg}"


# ──────────────────────────────────────────────
# MAIN RAG FUNCTION
# ──────────────────────────────────────────────

def explain_prescription(medicines, user_question=None, silent=False):
    """
    Full RAG pipeline:
    1. Load knowledge base
    2. Build context from prescription + knowledge base
    3. Build prompt
    4. Call Groq
    5. Return plain English explanation

    Args:
        medicines     : list of medicine dicts from parser
        user_question : optional specific question from patient
        silent        : if True suppress all print output (used by metrics)
    """
    knowledge_base = load_knowledge_base(silent=silent)
    user_message   = build_user_message(
        medicines, knowledge_base, user_question
    )
    explanation    = call_groq(SYSTEM_PROMPT, user_message)
    return explanation


# ──────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":

    test_medicines = [
        {
            'name'   : 'CALCIDOL D3',
            'dosage' : '0-1-0',
            'timing' : 'After Food - Weekly - 4 Weeks',
            'notes'  : ['Composition: Vitamin D3 60000 IU']
        },
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

    print("\n🧪 TEST 1 — General explanation")
    print("=" * 60)
    result = explain_prescription(test_medicines)
    print(result)

    print("\n\n🧪 TEST 2 — Specific question")
    print("=" * 60)
    result = explain_prescription(
        test_medicines,
        user_question="Can I take all my medicines together at the same time?"
    )
    print(result)

    print("\n\n🧪 TEST 3 — Missed dose")
    print("=" * 60)
    result = explain_prescription(
        test_medicines,
        user_question="I forgot to take my afternoon medicine. What should I do?"
    )
    print(result)