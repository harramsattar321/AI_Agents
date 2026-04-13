"""
insurance_agent.py — Agent 3: The Insurance Clerk
==================================================
Answers patient queries about insurance coverage, accepted plans,
rates, and policies using RAG over a MongoDB Atlas vector store.

Pattern mirrors BookingAgent:
  - __init__(groq_api_key, mongo_uri, patient_name)
  - respond(user_message) → reply_text
  - reset()
  - Sliding window history (last 10 turns)
  - Never terminal — always returns to CHAT via router timeout logic
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
    "You are the official Virtual Assistant for Virtual Hospital, located in Mianwali, Pakistan. "
    "Your job is to answer patient queries about insurance policies, accepted plans, rates, "
    "and coverage using ONLY the provided context.\n\n"

    "CRITICAL RULES:\n"
    "1. IDENTITY: You represent Virtual Hospital — NOT any insurance company.\n"
    "2. ACCEPTED PLANS (cashless treatment only):\n"
    "   - EFU: Corporate Healthcare, Mukammal Sehat, Rahbar Health Cover.\n"
    "   - Jubilee: Family Health, Lifestyle Care, Personal Health.\n"
    "   All other providers (Adamjee, State Life, etc.) are OUT-OF-NETWORK "
    "   and require 100% upfront payment.\n"
    "3. ADMISSION FILE FEE: Rs. 500 applies to ALL patients and is NOT covered by any insurance.\n"
    "4. NO PERSONAL DATA: Never ask for policy numbers, CNICs, or private IDs.\n"
    "5. EMPATHY: If the patient mentions an emergency or distress, respond with empathy first.\n"
    "6. STRICT CONTEXT: If a detail isn't in the documents, say: "
    "   'I cannot find that exact information in our current guidelines. "
    "   Please contact our billing desk for clarification.'\n"
    "7. TONE: Professional, warm, concise. Address the patient by name when natural.\n\n"
    "Context from hospital documents:\n{context}"
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


class InsuranceAgent:
    def __init__(self, groq_api_key: str, mongo_uri: str, patient_name: str):
        """
        groq_api_key : Groq API key (can share with chat agent or be separate)
        mongo_uri    : MongoDB Atlas URI — must have the insurance_knowledge_base collection
        patient_name : Carried over from the router / login

        NOTE: Heavy objects (HuggingFace embeddings, vector store, LLM) are
        lazy-loaded on the FIRST respond() call — __init__ returns instantly
        so the program starts without any delay.
        """
        self.patient_name  = patient_name
        self._groq_api_key = groq_api_key
        self._mongo_uri    = mongo_uri
        self.history: list = []

        # Lazy-loaded — None until first respond() call
        self._retriever = None
        self._llm       = None

    # ── Lazy initialiser — called once, on first real query ──────────────────
    def _ensure_ready(self):
        """Build the retriever and LLM the first time they are needed."""
        if self._retriever is not None:
            return   # already initialised

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

    # ── Sliding-window history (last 10 turns = 20 messages) ─────────────────
    def _trimmed_history(self, max_turns: int = 10) -> list:
        return self.history[-(max_turns * 2):]

    # ── Core turn method ──────────────────────────────────────────────────────
    def respond(self, user_message: str) -> str:
        """
        Process one user turn.
        Returns reply string. Never terminal — router decides when to leave.
        """
        self._ensure_ready()   # no-op after first call; loads model on first use

        # Retrieve relevant document chunks
        try:
            docs = self._retriever.invoke(user_message)
            context_text = "\n\n".join(doc.page_content for doc in docs)
        except Exception:
            context_text = ""   # degrade gracefully if vector store is unreachable

        # Build the prompt with trimmed history
        formatted = _PROMPT.format_messages(
            context=context_text,
            chat_history=self._trimmed_history(),
            input=user_message
        )

        try:
            response = self._llm.invoke(formatted)
            reply = response.content.strip()
        except Exception as e:
            reply = (
                f"I'm sorry, {self.patient_name}, I'm having trouble accessing "
                f"the insurance information right now. Please try again in a moment. ({e})"
            )

        # Append to sliding-window history
        self.history.append(HumanMessage(content=user_message))
        self.history.append(AIMessage(content=reply))

        return reply

    def reset(self):
        """Clear history between sessions."""
        self.history = []