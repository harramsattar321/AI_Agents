"""
report_analyzer.py — Medical Report Analyzer
=============================================
Handles all logic for the /analyze-report route:
  1. File validation   (type, size)
  2. Text extraction   (PDF via pdfplumber)
  3. LLM summarization (Groq API)

Used by app.py — no Flask imports here, just pure logic.
"""

import io
import json
import requests
import pdfplumber


class ReportAnalyzer:

    ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
    MAX_FILE_BYTES     = 10 * 1024 * 1024          # 10 MB
    GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL         = "llama-3.3-70b-versatile"

    SYSTEM_PROMPT = """You are a helpful medical report assistant for Harram Hospital's virtual patient support system.

When given a medical report, you must respond in this EXACT JSON format (no extra text, no markdown):
{
  "report_type": "Type of report e.g. Blood Test, Urine Analysis, Radiology, Discharge Summary, Other",
  "summary": "2-3 sentence plain English explanation of what the report shows overall",
  "abnormal_values": [
    {"name": "parameter name", "value": "patient value", "normal_range": "normal range", "status": "HIGH or LOW"}
  ],
  "key_observations": ["observation 1", "observation 2"],
  "advice": "One sentence on what the patient should do next e.g. consult a doctor, repeat the test, etc.",
  "disclaimer": "This summary is for informational purposes only and is not a substitute for professional medical advice. Please consult your doctor."
}

If there are no abnormal values, return an empty array for abnormal_values.
Always be clear, simple, and avoid medical jargon."""

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key

    # ── Public entry point ────────────────────────────────────────────────────
    def analyze(self, file) -> tuple[bool, int, dict | str]:
        """
        Main method called by app.py.
        Returns: (success, http_status_code, result_or_error_message)
        """

        # Step 1 — Validate file
        error = self._validate(file)
        if error:
            return False, 400, error

        file_bytes = file.read()

        if len(file_bytes) > self.MAX_FILE_BYTES:
            return False, 400, "File too large. Maximum size is 10 MB."

        # Step 2 — Extract text
        ext = file.filename.rsplit(".", 1)[1].lower()

        if ext != "pdf":
            return False, 422, (
                "Image upload is not yet supported. "
                "Please upload a text-based PDF for best results."
            )

        report_text = self._extract_text(file_bytes)
        if not report_text:
            return False, 422, (
                "Could not extract text from this PDF. "
                "It may be a scanned image. "
                "Please upload a JPG or PNG of the report instead."
            )

        # Step 3 — Summarize with Groq
        try:
            result = self._summarize(report_text)
            return True, 200, result

        except requests.exceptions.Timeout:
            return False, 504, "Analysis timed out. Please try again."
        except requests.exceptions.HTTPError as e:
            print(f"❌ Groq API error: {e}")
            return False, 502, "AI service error. Please try again later."
        except Exception as e:
            print(f"❌ Summarization error: {e}")
            return False, 500, "Failed to analyze the report. Please try again."

    # ── Private helpers ───────────────────────────────────────────────────────
    def _validate(self, file) -> str | None:
        """Returns an error string if invalid, None if OK."""
        if not file or file.filename == "":
            return "No file uploaded. Please attach a PDF or image."
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in self.ALLOWED_EXTENSIONS:
            return "Unsupported file type. Please upload a PDF, PNG, or JPG."
        return None

    def _extract_text(self, file_bytes: bytes) -> str:
        """Extract all text from a PDF using pdfplumber."""
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts).strip()

    def _summarize(self, report_text: str) -> dict:
        """Send report text to Groq and return structured summary dict."""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": self.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": f"Please analyze this medical report:\n\n{report_text}"},
            ],
            "temperature": 0.2,
            "max_tokens":  1024,
        }

        response = requests.post(self.GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        raw = response.json()["choices"][0]["message"]["content"].strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback if model didn't return clean JSON
            return {
                "report_type":      "Unknown",
                "summary":          raw,
                "abnormal_values":  [],
                "key_observations": [],
                "advice":           "Please consult your doctor for a detailed explanation.",
                "disclaimer":       "This summary is for informational purposes only and is not a substitute for professional medical advice.",
            }