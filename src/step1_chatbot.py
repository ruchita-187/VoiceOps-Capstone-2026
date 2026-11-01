
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from retrieve import retrieve

# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

# Connect to Gemini
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=30000,
        retry_options=types.HttpRetryOptions(attempts=1)
    )
)

SYSTEM_PROMPT = """
You are ZentroMart's AI Customer Support Assistant.

Your responsibilities:
1. Help customers with questions about orders, deliveries,
   returns, refunds and cancellations.
2. Be polite, professional and helpful.
3. Keep answers clear and simple.
4. Ask one clear question when information is missing.
5. Never invent policies, dates, order details or amounts.
6. Never claim to have checked an order or created a ticket
   unless the relevant tool has actually been used.
7. Answer policy questions using only the supplied
   knowledge-base context.
8. If the context does not contain the answer, clearly say
   that you cannot confirm the information from the available
   company documents.
9. Follow the current company policy when documents conflict.
   Do not treat superseded policies as current policy.

At the beginning, introduce yourself as ZentroMart's
AI Customer Support Assistant.
"""

print("=" * 55)
print("       ZENTROMART AI CUSTOMER SUPPORT")
print("=" * 55)

print("\nHello! I am ZentroMart's AI Customer Support Assistant.")
print("How can I help you today?")
print("\nType 'exit' to close the chatbot.\n")

while True:

    user_message = input("Customer: ").strip()

    if user_message.lower() == "exit":
        print("\nThank you for contacting ZentroMart. Goodbye!")
        break

    if not user_message:
        print("Please enter a question.")
        continue

    try:
        print("\nSearching company documents...", flush=True)

        start_time = time.time()

        # Retrieve relevant knowledge-base sections
        retrieved_docs = retrieve(user_message, top_k=5)

        context = ""

        for doc in retrieved_docs:
            context += (
                f"\nSource: {doc['source_file']}\n"
                f"Section: {doc['section_id']}\n"
                f"Content:\n{doc['text']}\n"
                + "-" * 50 + "\n"
            )

        if not context:
            context = "No relevant company documents were found."

        # Send question and retrieved context to Gemini
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=f"""
Company knowledge-base context:
{context}

Customer question:
{user_message}

Answer the customer using the knowledge-base context
and the system instructions.
""",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                )
            )
        )

        end_time = time.time()

        print("\nZentroMart AI:", response.text)

        print("\nRetrieved sources:")
        for doc in retrieved_docs:
            print(f"- {doc['source_file']} ({doc['section_id']})")

        print(
            f"\nResponse time: {end_time - start_time:.2f} seconds"
        )
        print()

    except Exception as e:
        print("\nError:", str(e))
        print("Please try again.\n")
