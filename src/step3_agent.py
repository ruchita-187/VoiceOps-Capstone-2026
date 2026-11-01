
import os
import sys
import re
from pathlib import Path
from datetime import date, timedelta

from dotenv import load_dotenv
from google import genai
from google.genai import types

# --------------------------------------------------
# PROJECT PATHS AND IMPORTS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import retrieve
from tools.tools import lookup_order, create_ticket

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=120000)
)

MODEL_NAME = "gemini-3.5-flash-lite"
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DATE = date(2026, 10, 20)

# --------------------------------------------------
# CONVERSATION STATE
# --------------------------------------------------

conversation_history = []
verified_order = None
pending_order_id = None
pending_issue = None
pending_intent = None

awaiting_order_id = False
awaiting_verification = False

verification_attempts = 0
MAX_VERIFICATION_ATTEMPTS = 2

created_handover_keys = set()


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are the ZentroMart customer-support assistant.

Answer using the supplied company knowledge-base documents
and verified order information.

Rules:
1. Do not invent policies, refund amounts, delivery dates or exceptions.
2. Current policies take precedence over superseded policies.
3. Never disclose specific order details before verification.
4. Never reveal a full phone number, email or street address.
5. Never claim an action succeeded unless the relevant tool confirms it.
6. Do not promise compensation or exceptions not allowed by policy.
7. Be polite, concise and conversational.
8. General policy questions do not require order verification.
9. Do not invent information if the documents do not contain the answer.
"""


# --------------------------------------------------
# GENERAL HELPERS
# --------------------------------------------------

def clean_digits(value):
    return "".join(
        char for char in str(value)
        if "0" <= char <= "9"
    )


def normalise_text(value):
    return str(value or "").strip().lower()


def extract_spoken_digit_sequence(value):
    text = normalise_text(value)

    digit_words = {
        "zero": "0", "oh": "0", "one": "1",
        "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9"
    }

    tokens = re.findall(r"[a-z]+|\d+", text)
    sequences = []
    current_sequence = []

    for token in tokens:
        if token in digit_words:
            current_sequence.append(digit_words[token])
        elif token.isdigit():
            current_sequence.extend(list(token))
        else:
            if current_sequence:
                sequences.append("".join(current_sequence))
                current_sequence = []

    if current_sequence:
        sequences.append("".join(current_sequence))

    for sequence in sequences:
        if len(sequence) == 8:
            return sequence

    return ""


def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_date(value):
    if not value:
        return None

    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


# --------------------------------------------------
# HOLIDAY CALENDAR
# --------------------------------------------------

def load_holidays():
    import pandas as pd

    holiday_file = DATA_DIR / "holidays.csv"
    holidays_df = pd.read_csv(holiday_file, dtype=str)

    return {
        safe_date(value)
        for value in holidays_df["date"].dropna()
        if safe_date(value)
    }


HOLIDAYS = load_holidays()


def business_days_after(start_date, end_date):
    if not start_date or not end_date or end_date <= start_date:
        return 0

    count = 0
    current = start_date + timedelta(days=1)

    while current <= end_date:
        if current.weekday() != 6 and current not in HOLIDAYS:
            count += 1
        current += timedelta(days=1)

    return count


# --------------------------------------------------
# RETRIEVAL AND GEMINI
# --------------------------------------------------

def get_retrieved_context(question):
    try:
        results = retrieve(question, top_k=5)
    except TypeError:
        results = retrieve(question)

    context_parts = []
    sources = []

    for item in results or []:
        if isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("content")
                or item.get("chunk")
                or ""
            )
            source = (
                item.get("source")
                or item.get("file")
                or item.get("document")
                or item.get("source_file")
                or "Knowledge base"
            )
            section = item.get("section_id") or item.get("section")
            if section:
                source = f"{source} ({section})"

        elif isinstance(item, (tuple, list)):
            text = str(item[0]) if item else ""
            source = str(item[1]) if len(item) > 1 else "Knowledge base"

        else:
            text = str(item)
            source = "Knowledge base"

        if text:
            context_parts.append(text)
            sources.append(source)

    return "\n\n".join(context_parts), sources


def get_ai_response(user_message):
    context, sources = get_retrieved_context(user_message)

    enriched_message = f"""
CUSTOMER MESSAGE:
{user_message}

RELEVANT COMPANY DOCUMENTS:
{context if context else "No relevant document sections were retrieved."}

VERIFIED ORDER INFORMATION:
{verified_order if verified_order else "No verified order information is available."}

Answer using the company documents. Do not invent policies or details.
"""

    conversation_history.append({
        "role": "user",
        "parts": [{"text": enriched_message}]
    })

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=conversation_history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2
        )
    )

    answer = response.text or "I'm sorry, I couldn't generate a response."

    conversation_history.append({
        "role": "model",
        "parts": [{"text": answer}]
    })

    if sources:
        print("\nSources:")
        for source in dict.fromkeys(sources):
            print(f"- {source}")

    return answer


# --------------------------------------------------
# INTENT DETECTION
# --------------------------------------------------

def detect_intent(message):
    text = normalise_text(message)

    if any(word in text for word in ["refund", "money back", "payment back"]):
        return "refund"

    if any(word in text for word in [
        "damage", "damaged", "broken", "defective",
        "wrong item", "incorrect item", "faulty"
    ]):
        return "damaged_wrong_item"

    if any(word in text for word in [
        "delivery", "delivered", "shipment", "shipping",
        "where is my order", "track"
    ]):
        return "order_status"

    if any(word in text for word in ["cancel", "cancellation"]):
        return "cancellation"

    if any(word in text for word in ["return", "exchange"]):
        return "return"

    return "general_support"


# --------------------------------------------------
# GENERAL POLICY VS PERSONAL ORDER DETECTION
# --------------------------------------------------

def is_general_policy_question(message):
    """
    General questions about policy, eligibility or timelines
    must not trigger personal order verification.
    """
    text = normalise_text(message)

    policy_phrases = [
        "return policy",
        "refund policy",
        "exchange policy",
        "return period",
        "return window",
        "refund timeline",
        "refund process",
        "how many days",
        "within how many days",
        "how long do i have",
        "am i eligible",
        "is it returnable",
        "can i return a",
        "can customers return",
        "what is the policy",
        "what are the rules",
        "policy for",
        "days to return",
        "days for return"
    ]

    return any(phrase in text for phrase in policy_phrases)


def is_personal_order_request(message):
    """
    Require verification for requests concerning a customer's
    own order, rather than every mention of a product or policy.
    """
    text = normalise_text(message)

    personal_phrases = [
        "my order",
        "my delivery",
        "my shipment",
        "my refund",
        "my payment",
        "my return",
        "my cancellation",
        "my parcel",
        "my package",
        "where is my",
        "track my",
        "check my",
        "status of my",
        "update on my",
        "i ordered",
        "i purchased",
        "i bought",
        "i received",
        "i have not received",
        "i haven't received",
        "i did not receive",
        "i didn't receive",
        "i want to cancel my",
        "cancel my order",
        "return my order",
        "return the item i",
        "refund for my"
    ]

    return any(phrase in text for phrase in personal_phrases)


def is_order_related(message):
    """
    Only explicit personal order requests require verification.
    General policy questions are handled by RAG without an order ID.
    """
    if is_general_policy_question(message):
        return False

    return is_personal_order_request(message)


def reset_pending_order_flow():
    global awaiting_order_id
    global awaiting_verification
    global pending_order_id
    global pending_issue
    global pending_intent
    global verification_attempts

    awaiting_order_id = False
    awaiting_verification = False
    pending_order_id = None
    pending_issue = None
    pending_intent = None
    verification_attempts = 0


# --------------------------------------------------
# IMMEDIATE ESCALATION DETECTION
# --------------------------------------------------

def detect_immediate_escalation(message):
    text = normalise_text(message)

    human_phrases = [
        "human agent", "talk to a human", "speak to a human",
        "talk to a person", "speak to a person",
        "human representative", "real person",
        "customer care executive", "supervisor",
        "human support", "connect me to an agent"
    ]

    if any(phrase in text for phrase in human_phrases):
        return "customer_requested_human"

    unauthorized_phrases = [
        "i did not place", "i didn't place", "not my order",
        "unauthorised", "unauthorized", "without my permission",
        "i did not make", "i didn't make",
        "someone used my account", "someone accessed my account",
        "fraudulent transaction", "unknown transaction",
        "payment i did not make"
    ]

    if any(phrase in text for phrase in unauthorized_phrases):
        return "unauthorized_activity"

    legal_phrases = [
        "consumer court", "police complaint",
        "file a police report", "legal action",
        "take legal action", "sue you", "lawyer", "legal notice"
    ]

    if any(phrase in text for phrase in legal_phrases):
        return "legal_action"

    safety_phrases = [
        "swelling battery", "swollen battery",
        "battery is swollen", "battery is swelling",
        "battery swelling", "battery has swollen",
        "battery started swelling", "battery swelling up",
        "battery is bulging", "bulging battery",
        "burning smell", "smell of burning",
        "smells like burning", "smells like it is burning",
        "smells like it's burning", "burning odour",
        "burning odor", "smoke from", "smoke coming from",
        "sparks", "electric shock", "caught fire", "on fire",
        "product injured me", "injury from product",
        "overheating dangerously"
    ]

    if any(phrase in text for phrase in safety_phrases):
        return "product_safety_risk"

    battery_swelling = (
        "battery" in text
        and any(word in text for word in [
            "swelling", "swollen", "bulging", "bulged"
        ])
    )

    burning_risk = (
        "burning" in text
        and any(word in text for word in [
            "smell", "smells", "smoke", "odor", "odour"
        ])
    )

    fire_or_electrical_risk = any(
        phrase in text for phrase in [
            "electric shock", "sparking", "sparks",
            "caught fire", "on fire", "flames coming",
            "product is burning"
        ]
    )

    if battery_swelling or burning_risk or fire_or_electrical_risk:
        return "product_safety_risk"

    return None


def detect_damage_issue(message):
    text = normalise_text(message)

    return any(word in text for word in [
        "damaged", "damage", "broken", "defective",
        "faulty", "wrong item", "incorrect item"
    ])


# --------------------------------------------------
# REFUND ESCALATION
# --------------------------------------------------

def get_refund_max_business_days(method):
    method = normalise_text(method)

    timelines = {
        "upi": 3,
        "card": 7,
        "netbanking": 5,
        "wallet": 0,
        "cod_wallet": 0,
        "cod_bank": 7
    }

    return timelines.get(method)


def find_overdue_refund(order):
    refund_types = [
        (
            "cancellation",
            "cancellation_refund_initiated_on",
            "cancellation_refund_method",
            "cancellation_rrn",
            None
        ),
        (
            "RTO",
            "rto_refund_initiated_on",
            "rto_refund_method",
            "rto_rrn",
            None
        ),
        (
            "return",
            "return_refund_initiated_on",
            "return_refund_method",
            "return_rrn",
            "return_refund_completed_on"
        )
    ]

    for (
        refund_name,
        initiated_field,
        method_field,
        rrn_field,
        completed_field
    ) in refund_types:

        initiated_on = safe_date(order.get(initiated_field))
        method = order.get(method_field)

        if not initiated_on or not method:
            continue

        if completed_field and order.get(completed_field):
            continue

        if not order.get(rrn_field):
            continue

        max_days = get_refund_max_business_days(method)
        if max_days is None:
            continue

        elapsed = business_days_after(initiated_on, REFERENCE_DATE)

        if elapsed > max_days + 3:
            return {
                "refund_type": refund_name,
                "reason": (
                    f"{refund_name} refund is more than 3 business days "
                    "past its maximum policy timeline."
                )
            }

    return None


# --------------------------------------------------
# HIGH-VALUE DAMAGED ITEM CHECK
# --------------------------------------------------

def get_damaged_item_value(order):
    damage_sku = order.get("damage_sku")
    if not damage_sku:
        return None

    for item in order.get("items", []):
        if str(item.get("sku")) == str(damage_sku):
            unit_price = safe_int(item.get("unit_price_inr"))
            quantity = safe_int(item.get("qty"), 1)
            return unit_price * quantity

    return None


def check_high_value_damage(order, customer_message):
    damage_issue = order.get("damage_issue")

    if not damage_issue and not detect_damage_issue(customer_message):
        return False

    item_value = get_damaged_item_value(order)
    return item_value is not None and item_value > 10000


# --------------------------------------------------
# TICKET CREATION
# --------------------------------------------------

def create_handover_ticket(reason, customer_message, intent=None):
    order_id = None
    if verified_order:
        order_id = str(verified_order.get("order_id") or "") or None

    detected_intent = intent or detect_intent(customer_message)

    handover_key = (
        str(reason or ""),
        str(order_id or ""),
        normalise_text(customer_message)
    )

    if handover_key in created_handover_keys:
        return "A handover ticket has already been created for this request."

    checks = []

    if verified_order:
        checks.append("Customer identity verification succeeded.")
        checks.append(
            f"Verified order status: "
            f"{verified_order.get('status', 'not available')}."
        )

        contacts = safe_int(verified_order.get("contacts_last_7_days"))
        checks.append(f"Earlier contacts in the last 7 days: {contacts}.")

        damage_value = get_damaged_item_value(verified_order)
        if damage_value is not None:
            checks.append(f"Verified affected item value: ₹{damage_value}.")

        overdue = find_overdue_refund(verified_order)
        if overdue:
            checks.append(overdue["reason"])
    else:
        checks.append("No verified order details were accessed or disclosed.")

    summary = (
        f"Issue: {customer_message}. "
        f"Detected intent: {detected_intent}. "
        f"Order ID: {order_id or 'Not provided'}. "
        f"Customer information: {customer_message}. "
        f"Checks already made: {' '.join(checks)} "
        f"Escalation reason: {reason}."
    )

    result = create_ticket(
        summary=summary,
        order_id=order_id,
        reason=reason,
        intent=detected_intent
    )

    if result.get("created") is True:
        created_handover_keys.add(handover_key)
        return (
            "I have created a handover ticket for you. "
            f"Your ticket ID is {result.get('ticket_id')}. "
            "A human colleague will follow up with you. "
            "I cannot promise a specific outcome or timeline."
        )

    return (
        "I could not create the handover ticket successfully. "
        "I won't claim that the handover is complete. "
        "Please try again or contact customer support."
    )


# --------------------------------------------------
# ORDER VERIFICATION
# --------------------------------------------------

def start_order_verification(message):
    global awaiting_order_id
    global pending_issue
    global pending_intent
    global verification_attempts

    pending_issue = message
    pending_intent = detect_intent(message)
    awaiting_order_id = True
    verification_attempts = 0

    return (
        "I can help you with that. "
        "Please share your 8-digit order ID first."
    )


def ask_for_verification(order_id):
    global awaiting_verification
    global awaiting_order_id
    global pending_order_id

    pending_order_id = order_id
    result = lookup_order(order_id)

    if result.get("verification_required"):
        awaiting_order_id = False
        awaiting_verification = True
        return (
            "Thank you. To verify your order, please share either "
            "the last 4 digits of your registered phone number "
            "or the 6-digit delivery pincode."
        )

    return (
        "I couldn't start order verification. "
        "Please check the order ID and try again."
    )


def verify_customer(verification_value):
    global verified_order
    global awaiting_verification
    global verification_attempts

    verification_attempts += 1

    result = lookup_order(
        pending_order_id,
        verification_value=verification_value
    )

    if result.get("found") is True and result.get("verified") is True:
        verified_order = result
        awaiting_verification = False
        return True, None

    if verification_attempts < MAX_VERIFICATION_ATTEMPTS:
        return (
            False,
            "That verification value did not match. "
            "You have one more attempt. Please enter the registered "
            "phone's last 4 digits or the delivery pincode."
        )

    awaiting_verification = False
    verified_order = None

    return (
        False,
        "I couldn't verify the order after two attempts, so I cannot "
        "share its details. I can still help with general policies. "
        "You can also contact support using your registered phone number."
    )


# --------------------------------------------------
# ORDER ESCALATION CHECKS
# --------------------------------------------------

def check_verified_order_escalations(customer_message):
    if not verified_order:
        return None

    earlier_contacts = safe_int(
        verified_order.get("contacts_last_7_days")
    )

    if earlier_contacts >= 2:
        return create_handover_ticket(
            reason="repeat_contact_same_issue",
            customer_message=customer_message,
            intent=detect_intent(customer_message)
        )

    if check_high_value_damage(verified_order, customer_message):
        return create_handover_ticket(
            reason="high_value_damaged_defective_or_wrong_item",
            customer_message=customer_message,
            intent="damaged_wrong_item"
        )

    overdue_refund = find_overdue_refund(verified_order)
    if overdue_refund:
        return create_handover_ticket(
            reason="refund_overdue_beyond_escalation_limit",
            customer_message=customer_message,
            intent="refund"
        )

    return None


# --------------------------------------------------
# SAFE ORDER DETAILS
# --------------------------------------------------

def get_safe_order_details():
    if not verified_order:
        return None

    allowed_fields = [
        "order_id", "status", "payment_method", "delivery_type",
        "promised_date", "shipped_on", "delivered_on", "total_inr",
        "timeline", "return_status", "return_requested_on",
        "return_pickup_on", "return_refund_initiated_on",
        "return_refund_completed_on", "return_refund_amount_inr",
        "return_rrn", "cancellation_refund_initiated_on",
        "cancellation_refund_amount_inr", "cancellation_rrn",
        "rto_refund_initiated_on", "rto_refund_amount_inr", "rto_rrn"
    ]

    safe_details = {
        key: verified_order.get(key)
        for key in allowed_fields
        if verified_order.get(key) is not None
    }

    safe_details["items"] = [
        {
            "sku": item.get("sku"),
            "title": item.get("title"),
            "qty": item.get("qty"),
            "unit_price_inr": item.get("unit_price_inr")
        }
        for item in verified_order.get("items", [])
    ]

    return safe_details


# --------------------------------------------------
# MAIN MESSAGE PROCESSING
# --------------------------------------------------

def process_message(user_message):
    global awaiting_order_id
    global awaiting_verification
    global pending_issue
    global pending_intent
    global verified_order
    global pending_order_id
    global verification_attempts

    message = str(user_message or "").strip()
    if not message:
        return "Please enter a message."

    # Immediate escalations always take priority.
    immediate_reason = detect_immediate_escalation(message)
    if immediate_reason:
        return create_handover_ticket(
            reason=immediate_reason,
            customer_message=message,
            intent=detect_intent(message)
        )

    # A general policy question can interrupt an unfinished
    # order-verification flow. It does not need an order ID.
    if is_general_policy_question(message) and not is_personal_order_request(message):
        if awaiting_order_id or awaiting_verification:
            reset_pending_order_flow()

        return get_ai_response(message)

    # Accept numeric or spoken-digit order IDs while awaiting the ID.
    if awaiting_order_id:
        order_id = clean_digits(message)

        if len(order_id) != 8:
            order_id = extract_spoken_digit_sequence(message)

        if len(order_id) == 8:
            return ask_for_verification(order_id)

        return (
            "I couldn't identify an 8-digit order ID. "
            "Please say or enter all 8 digits, one at a time."
        )

    # Verification value is accepted only after an order ID was supplied.
    if awaiting_verification:
        proof = clean_digits(message)

        if len(proof) in (4, 6):
            success, error_message = verify_customer(proof)
            if not success:
                return error_message

            escalation_response = check_verified_order_escalations(
                pending_issue or "Customer requested order information."
            )
            if escalation_response:
                return escalation_response

            safe_details = get_safe_order_details()
            question = pending_issue or "Please help me with my order."
            pending_issue = None

            return get_ai_response(
                f"{question}\n\nVerified order details:\n{safe_details}"
            )

        return (
            "Please enter either exactly 4 digits for your registered "
            "phone suffix or exactly 6 digits for your delivery pincode."
        )

    # Only explicit personal order requests trigger verification.
    if is_order_related(message):
        if verified_order:
            escalation_response = check_verified_order_escalations(message)
            if escalation_response:
                return escalation_response

            safe_details = get_safe_order_details()
            return get_ai_response(
                f"{message}\n\nVerified order details:\n{safe_details}"
            )

        return start_order_verification(message)

    # Damage complaints are order-specific and need verification.
    if detect_damage_issue(message):
        return start_order_verification(message)

    # All other questions go to knowledge-base retrieval.
    return get_ai_response(message)


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():
    print("=" * 60)
    print("ZENTROMART CUSTOMER SUPPORT ASSISTANT")
    print("Step 3: Multi-turn conversation and order tools")
    print("=" * 60)

    print("\nYou can ask about policies, orders, returns and refunds.")
    print("Type 'exit' to stop.\n")

    while True:
        try:
            user_message = input("You: ").strip()

            if user_message.lower() in ["exit", "quit", "bye"]:
                print(
                    "\nAssistant: Thank you for contacting "
                    "ZentroMart. Goodbye!"
                )
                break

            response = process_message(user_message)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\nAssistant: Goodbye!")
            break

        except Exception as error:
            print(
                "\nAssistant: Something went wrong while processing "
                "your request. Please try again."
            )
            print(f"\nTechnical error: {error}\n")


if __name__ == "__main__":
    main()
