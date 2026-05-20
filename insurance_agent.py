"""
insurance_agent.py — Agent 3: The Insurance Clerk
==================================================
Answers patient queries about insurance coverage, accepted plans,
rates, and policies using RAG over a MongoDB Atlas vector store.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pymongo import MongoClient


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are the Insurance Information Clerk at Harram Hospital, Mianwali, Pakistan. "
    "You answer patient questions about insurance coverage, accepted plans, and billing policies "
    "using ONLY the provided context documents.\n\n"

    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "WHAT YOU CAN DO — nothing else, ever:\n"
    "  1. Answer questions about which insurance plans are accepted\n"
    "  2. Explain coverage details, cashless treatment eligibility\n"
    "  3. Explain billing policies and admission fees\n"
    "  4. Tell patients what documents to bring for insurance claims\n"
    "  5. Clarify what is and isn't covered under specific plans\n\n"

    "WHAT YOU CANNOT DO — never suggest, offer, or imply these:\n"
    "  ✗ Contact any insurance company or manager on the patient's behalf\n"
    "  ✗ Contact the front desk, any doctor, or any hospital staff\n"
    "  ✗ File, process, or submit any insurance claim\n"
    "  ✗ Access the patient's policy, records, or personal data\n"
    "  ✗ Approve or reject any claim or coverage\n"
    "  ✗ Send emails, messages, or make calls of any kind\n"
    "  ✗ Book appointments — direct to the booking section\n"
    "  ✗ Do ANYTHING outside of providing insurance information\n\n"

    "If the patient asks for something outside this list, say:\n"
    "  'I can only provide insurance information. For [their request], please visit\n"
    "   the hospital billing desk or contact your insurance provider directly.'\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    "ACCEPTED PLANS (cashless treatment only):\n"
    "  - EFU: Corporate Healthcare, Mukammal Sehat, Rahbar Health Cover.\n"
    "  - Jubilee: Family Health, Lifestyle Care, Personal Health.\n"
    "  All other providers (Adamjee, State Life, etc.) are OUT-OF-NETWORK "
    "and require 100% upfront payment.\n\n"

    "ADMISSION FILE FEE: Rs. 500 applies to ALL patients and is NOT covered by any insurance.\n\n"

    "STRICT RULES:\n"
    "1. ONLY use information from the context documents provided below.\n"
    "2. NEVER ask for policy numbers, CNICs, or any private patient data.\n"
    "3. If a detail is not in the documents, say exactly:\n"
    "   'I cannot find that exact information in our current guidelines. "
    "Please contact our billing desk for clarification.'\n"
    "4. Never make up coverage details, amounts, or plan names.\n"
    "5. Be professional, warm, and concise. Address the patient by name when natural.\n"
    "6. If the patient mentions an emergency or distress, respond with empathy first.\n\n"

    "Context from hospital documents:\n{context}"
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


class InsuranceAgent:
    def __init__(self, groq_api_key: str, mongo_uri: str, patient_name: str):
        self.patient_name  = patient_name
        self._groq_api_key = groq_api_key
        self._mongo_uri    = mongo_uri
        self.history: list = []

        # Lazy-loaded on first respond() call
        self._retriever = None
        self._llm       = None

    def _ensure_ready(self):
        if self._retriever is not None:
            return

        print("⏳ Loading insurance knowledge base (first use only)...")

        mongo_client = MongoClient(self._mongo_uri)
        collection   = mongo_client["patient_db"]["insurance_knowledge_base"]

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index"
        )
        self._retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        self._llm = ChatGroq(
            groq_api_key=self._groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=512
        )
        print("✅ Insurance knowledge base ready.")

    def _trimmed_history(self, max_turns: int = 10) -> list:
        return self.history[-(max_turns * 2):]

    def respond(self, user_message: str) -> str:
        self._ensure_ready()

        try:
            docs = self._retriever.invoke(user_message)
            context_text = "\n\n".join(doc.page_content for doc in docs)
        except Exception:
            context_text = ""

        formatted = _PROMPT.format_messages(
            context=context_text,
            chat_history=self._trimmed_history(),
            input=user_message
        )

        try:
            response = self._llm.invoke(formatted)
            reply    = response.content.strip()
        except Exception as e:
            reply = (
                f"I'm sorry, {self.patient_name}, I'm having trouble accessing "
                f"the insurance information right now. Please try again in a moment."
            )

        self.history.append(HumanMessage(content=user_message))
        self.history.append(AIMessage(content=reply))

        return reply

    def reset(self):
        self.history = []
