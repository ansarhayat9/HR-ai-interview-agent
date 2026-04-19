import os
import json
import requests
import numpy as np
import streamlit as st
import faiss
import PyPDF2
import io
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBED_DIM = 384

# ============================================================
# EMBEDDING MODEL  (locally cached — never reloaded)
# ============================================================
@st.cache_resource(show_spinner="Loading local embedding model…")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# IN-MEMORY VECTOR STORE
# ============================================================
class DocumentStore:
    """Session-scoped FAISS store for JD + CV chunks."""

    def __init__(self, model: SentenceTransformer):
        self._model = model
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        self.chunks: list[dict] = []   # [{text, source}]

    def add_chunk(self, text: str, source: str):
        emb = self._model.encode(text, convert_to_numpy=True).astype("float32")
        self.index.add(np.array([emb]))
        self.chunks.append({"text": text, "source": source})

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        if not self.chunks:
            return []
        emb = self._model.encode(query, convert_to_numpy=True).astype("float32")
        k = min(k, len(self.chunks))
        _, I = self.index.search(np.array([emb]), k)
        return [self.chunks[i] for i in I[0] if 0 <= i < len(self.chunks)]

# ============================================================
# SESSION STATE  INITIALISER
# ============================================================
def init_state():
    """Initialise all session-state keys if they don't exist."""
    defaults = {
        "interview_state": "SETUP",   # SETUP | INTERVIEW | EVALUATION
        "questions": [],
        "q_index": 0,
        "chat_history": [],
        "evaluation_report": "",
        "raw_jd": "",
        "raw_cv": "",
        "candidate_name": "",
        "doc_store": None,            # Created after model loads
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_state(model: SentenceTransformer):
    """Full reset — call when starting a brand-new interview."""
    st.session_state.interview_state = "SETUP"
    st.session_state.questions = []
    st.session_state.q_index = 0
    st.session_state.chat_history = []
    st.session_state.evaluation_report = ""
    st.session_state.raw_jd = ""
    st.session_state.raw_cv = ""
    st.session_state.candidate_name = ""
    st.session_state.doc_store = DocumentStore(model)

# ============================================================
# DOCUMENT HELPERS
# ============================================================
def extract_text(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF or TXT file."""
    raw = uploaded_file.read()
    if uploaded_file.name.lower().endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            ).strip()
            if not text:
                return ""
            return text
        except Exception as e:
            st.error(f"PDF parse error ({uploaded_file.name}): {e}")
            return ""
    else:
        try:
            return raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            return raw.decode("latin-1").strip()

def chunk_text(text: str, chunk_size: int = 250) -> list[str]:
    """Sliding-window word chunker with 50-word overlap."""
    words = text.split()
    step = chunk_size - 50          # 50-word overlap
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, max(len(words), 1), step)
        if words[i : i + chunk_size]
    ]

# ============================================================
# GROQ API WRAPPER
# ============================================================
def call_groq(
    system_prompt: str,
    user_prompt: str,
    require_json: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.4,
) -> str | None:
    if not GROQ_API_KEY:
        st.error("⚠️  GROQ_API_KEY is missing. Add it to your .env file.")
        return None

    # For JSON mode we reinforce the instruction INSIDE the system message
    if require_json:
        system_prompt = system_prompt.rstrip() + (
            "\n\nIMPORTANT: Respond ONLY with a single valid JSON object. "
            "Do not include any explanation, markdown fences, or extra text."
        )

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if require_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.HTTPError:
        st.error(f"Groq API error {r.status_code}: {r.text[:300]}")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None

# ============================================================
# AGENT LOGIC
# ============================================================
def generate_questions(jd: str, cv: str) -> list[str]:
    """Ask Groq to return exactly 3 role-specific interview questions as JSON."""
    sys_p = """\
You are an expert technical recruiter and hiring manager.

Given a Job Description and a Candidate's CV, generate exactly 3 targeted interview questions.

RULES — follow each one strictly:
1. Questions must test competency for the Job Description requirements, NOT just CV content.
2. If the CV has ZERO overlap with the JD field (e.g. Civil Engineer applying for an AI role):
   - Q1 MUST challenge the discrepancy directly.
   - The other 2 questions should probe any transferable technical skills and motivation.
3. If the CV is partially relevant, probe the GAPS in required skills.
4. Each question must be specific, not generic (no "Tell me about yourself").

Return JSON: {"questions": ["<Q1>", "<Q2>", "<Q3>"]}"""

    user_p = f"## Job Description\n{jd}\n\n## Candidate CV\n{cv}"
    resp = call_groq(sys_p, user_p, require_json=True, max_tokens=512)
    if not resp:
        return []
    try:
        data = json.loads(resp)
        qs = data.get("questions", [])
        return [str(q) for q in qs if q][:3]
    except json.JSONDecodeError:
        st.error("Could not parse the questions from the AI response. Please try again.")
        return []

def evaluate_interview(jd: str, cv: str, history: list[dict], name: str) -> str:
    """Generate a detailed evaluation report from the full context."""
    transcript = "\n\n".join(
        f"{'🤖 INTERVIEWER' if m['role'] == 'assistant' else '👤 CANDIDATE'}: {m['content']}"
        for m in history
    )

    sys_p = """\
You are a highly experienced, critical technical recruiter writing an official interview evaluation.

GRADING CONTRACT — you MUST follow these rules:
- Score 1–3/10: Wrong answers, evasive responses, irrelevant background.
- Score 4–6/10: Partial knowledge; candidate understands basics but lacks depth.
- Score 7–9/10: Correct, detailed, thoughtful answers.
- Score 10/10: Only for exceptional, expert-level responses.
- Penalise "I don't know" or empty responses heavily.
- If the candidate's CV is entirely unrelated to the JD, the max possible fit score is 4/10 no matter how well they answer.

Format your report using Markdown:
## Candidate Evaluation Report
### Score: X / 10
### Strengths
### Gaps & Areas for Improvement
### Interview Performance Summary
### Final Recommendation
**Proceed to next round** OR **Do Not Proceed** — with a one-sentence justification."""

    user_p = (
        f"**Candidate Name:** {name or 'Anonymous'}\n\n"
        f"## Job Description\n{jd}\n\n"
        f"## Candidate CV\n{cv}\n\n"
        f"## Interview Transcript\n{transcript}\n\n"
        "Write the evaluation report now."
    )
    return call_groq(sys_p, user_p, require_json=False, max_tokens=1200, temperature=0.3) or "❌ Failed to generate report."

# ============================================================
# PAGE CONFIG  &  PREMIUM CSS
# ============================================================
st.set_page_config(
    page_title="AI Pre-Screening Interviewer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@500;700;800&display=swap');

/* ── Global Styles ── */
html, body { 
    background: #020617;
}

/* Target specific text elements for font to avoid breaking Streamlit internal icons */
h1, h2, h3, p, li, span, label, div, small, .stMarkdown { 
    font-family: 'Inter', sans-serif; 
    color: #f1f5f9 !important; 
}

.stApp { 
    background: radial-gradient(circle at top right, #0f172a, #020617);
    background-attachment: fixed;
}

/* Make sure all paragraph and list text is bright silver/white */
p, li, span, label, .stMarkdown { 
    color: #cbd5e1 !important; 
    line-height: 1.6;
}

/* ── Sidebar Redesign ── */
[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, 0.8) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(34, 211, 238, 0.15);
}
[data-testid="stSidebarContent"] { padding: 2rem 1rem; }
[data-testid="stSidebar"] h2 { 
    color: #22d3ee !important; 
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.02em;
}

/* ── Headings ── */
h1, h2, h3 { 
    font-family: 'Outfit', sans-serif; 
    color: #f8fafc !important; 
    letter-spacing: -0.01em;
}
h1 { 
    font-size: 2.8rem !important; 
    font-weight: 800 !important; 
    background: linear-gradient(90deg, #22d3ee, #06b6d4); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
}

/* ── Cyber Cards ── */
.ag-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(34, 211, 238, 0.2);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5), inset 0 0 20px rgba(34,211,238,0.02);
    transition: transform 0.3s ease, border-color 0.3s ease;
}
.ag-card:hover {
    border-color: rgba(34, 211, 238, 0.4);
}

/* ── Chat Interface ── */
.chat-ai {
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.08), rgba(6, 182, 212, 0.03));
    border: 1px solid rgba(34, 211, 238, 0.3);
    border-left: 4px solid #22d3ee;
    border-radius: 4px 16px 16px 16px;
    padding: 1.2rem;
    margin: 1rem 0;
    color: #f1f5f9;
}
.chat-user {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(5, 150, 105, 0.03));
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-right: 4px solid #10b981;
    border-radius: 16px 4px 16px 16px;
    padding: 1.2rem;
    margin: 1rem 0 1rem 3rem;
    color: #ecfdf5;
}
.bubble-label { 
    font-size: 0.7rem; 
    font-weight: 800; 
    letter-spacing: 0.1em; 
    text-transform: uppercase; 
    color: #94a3b8;
    margin-bottom: 0.5rem; 
}

/* ── Progress & Badges ── */
.progress-wrap { 
    background: rgba(255,255,255,0.05); 
    border-radius: 999px; 
    height: 6px; 
    margin: 1rem 0; 
    overflow: hidden;
}
.progress-fill { 
    height: 100%; 
    background: linear-gradient(90deg, #22d3ee, #06b6d4); 
    box-shadow: 0 0 15px rgba(34,211,238,0.5);
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); 
}

.stat-chip {
    background: rgba(34, 211, 238, 0.1);
    border: 1px solid rgba(34, 211, 238, 0.2);
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    display: inline-block;
    font-size: 0.8rem;
    font-weight: 600;
    color: #67e8f9;
    margin: 0.3rem 0.3rem 0 0;
}

/* ── Form Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: rgba(2, 6, 23, 0.5) !important;
    border: 1px solid rgba(34, 211, 238, 0.2) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
}

/* File Uploader Correction */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.5) !important;
    border-radius: 14px;
    padding: 10px;
}
[data-testid="stFileUploader"] section {
    background-color: rgba(2, 6, 23, 0.5) !important;
    border: 1px dashed rgba(34, 211, 238, 0.4) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] p, [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small {
    color: #f8fafc !important;
}
[data-testid="stFileUploadDropzone"] button {
    background-color: #0891b2 !important;
    color: white !important;
    border-radius: 8px !important;
}
.stTextInput input:focus {
    border-color: #22d3ee !important;
    box-shadow: 0 0 10px rgba(34,211,238,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0891b2, #06b6d4) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(6,182,212,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(6,182,212,0.5) !important;
    opacity: 0.95 !important;
}

/* ── Hide Defaults ── */
#MainMenu, footer { visibility: hidden; }
header { 
    background-color: rgba(0,0,0,0) !important; 
    border: none !important;
}
header * { color: #22d3ee !important; } /* Sidebar toggle button color */

/* ── Chat Input Redesign ── */
.stChatInputContainer {
    background-color: transparent !important;
}
[data-testid="stChatInput"] {
    background-color: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid rgba(34, 211, 238, 0.4) !important;
    border-radius: 16px !important;
    padding: 8px !important;
    backdrop-filter: blur(15px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}
[data-testid="stChatInput"] div {
    background-color: transparent !important;
}
[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
}
[data-testid="stChatInput"] button {
    color: #22d3ee !important;
    background-color: rgba(34, 211, 238, 0.1) !important;
    border-radius: 10px !important;
}
/* Ensure the placeholder is also readable */
[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255, 255, 255, 0.4) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(2,6,23,0.5); }
::-webkit-scrollbar-thumb { background: rgba(34,211,238,0.2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(34,211,238,0.4); }

/* ── Setup Steps ── */
.step-item { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1.2rem; }
.step-num { 
    background: linear-gradient(135deg, #0891b2, #06b6d4); 
    color: white; 
    border-radius: 50%; 
    width: 32px; height: 32px; 
    display: flex; align-items: center; justify-content: center; 
    font-weight: 800; font-size: 0.9rem; 
    flex-shrink: 0; 
    box-shadow: 0 0 15px rgba(34,211,238,0.3);
}
.step-txt { color: #cbd5e1; font-size: 1rem; line-height: 1.6; }

/* ── Alerts Override ── */
.stAlert {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(34, 211, 238, 0.2) !important;
    border-radius: 14px !important;
    color: #f1f5f9 !important;
}
.stAlert p { color: #f1f5f9 !important; }

.chat-scroll { max-height: 65vh; overflow-y: auto; padding-right: 10px; }
.report-wrap h2, .report-wrap h3 { color: #22d3ee !important; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL  →  INIT STATE
# ============================================================
emb_model = load_embedding_model()
init_state()
if st.session_state.doc_store is None:
    st.session_state.doc_store = DocumentStore(emb_model)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎯 Interview Setup")
    st.caption("Upload documents and launch the AI interviewer.")
    st.markdown("---")

    candidate_name = st.text_input(
        "Candidate Name (optional)",
        placeholder="e.g. John Smith",
        key="candidate_name_input",
    )

    jd_file = st.file_uploader("📄 Job Description (PDF / TXT)", type=["pdf", "txt"])
    cv_file = st.file_uploader("📑 Candidate CV (PDF / TXT)", type=["pdf", "txt"])

    ready = bool(jd_file and cv_file and GROQ_API_KEY)

    if st.button("🚀  Launch Interview Agent", type="primary", use_container_width=True, disabled=not ready):
        with st.spinner("Reading & indexing documents…"):
            jd_text = extract_text(jd_file)
            cv_text = extract_text(cv_file)

        if not jd_text or not cv_text:
            st.error("Could not extract text from one or both files. Please check the files and try again.")
        else:
            # Full state reset so we never bleed data from a prior session
            reset_state(emb_model)
            st.session_state.raw_jd = jd_text
            st.session_state.raw_cv = cv_text
            st.session_state.candidate_name = candidate_name.strip()

            with st.spinner("Indexing into FAISS vector store…"):
                for chunk in chunk_text(jd_text):
                    st.session_state.doc_store.add_chunk(chunk, "Job Description")
                for chunk in chunk_text(cv_text):
                    st.session_state.doc_store.add_chunk(chunk, "Candidate CV")

            with st.spinner("Agent generating targeted questions…"):
                qs = generate_questions(jd_text, cv_text)

            if qs:
                st.session_state.questions = qs
                st.session_state.interview_state = "INTERVIEW"
                st.session_state.chat_history = [{"role": "assistant", "content": qs[0]}]
                st.rerun()
            else:
                st.error("Failed to generate questions. Re-check your API key or try again.")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in .env")

    st.markdown("---")

    state = st.session_state.interview_state
    if state == "INTERVIEW":
        qi     = st.session_state.q_index
        total  = len(st.session_state.questions)
        pct    = int((qi / total) * 100)
        st.markdown(f"**Progress — Question {qi + 1} of {total}**")
        st.markdown(
            f'<div class="progress-wrap"><div class="progress-fill" style="width:{pct}%"></div></div>',
            unsafe_allow_html=True,
        )
    elif state == "EVALUATION":
        st.success("✅ Interview complete")
        if st.button("🔄  New Interview", use_container_width=True):
            reset_state(emb_model)
            st.rerun()

# ============================================================
# MAIN PANEL
# ============================================================
st.markdown(
    "<h1>AI Pre-Screening Interviewer</h1>"
    "<p style='color:#94a3b8; font-family:\"Inter\"; font-weight:500; font-size:1rem; margin-top:-0.5rem; margin-bottom:2rem;'>"
    "⚡ Powered by Groq · LLaMA 3.1 · FAISS · SentenceTransformers</p>",
    unsafe_allow_html=True,
)

# ── SETUP PAGE ──────────────────────────────────────────────
if st.session_state.interview_state == "SETUP":
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="ag-card">', unsafe_allow_html=True)
        st.markdown("### How it works")
        steps = [
            ("Upload", "Drag in the Job Description & Candidate CV — PDF or plain text."),
            ("Index",  "Documents are chunked and embedded locally in a FAISS vector store."),
            ("Ask",    "The AI generates 3 role-specific questions and conducts the interview."),
            ("Evaluate","An evaluator agent scores the interview against the actual JD requirements."),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(
                f'<div class="step-item">'
                f'  <div class="step-num">{i}</div>'
                f'  <div class="step-txt"><strong style="color:#22d3ee">{title}</strong> — {desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="ag-card">', unsafe_allow_html=True)
        st.markdown("### Tech Stack")
        chips = ["🐍 Python", "🧠 FAISS (local)", "🔗 Sentence-Transformers", "⚡ Groq API", "🦙 LLaMA 3.1 8B", "📊 Streamlit"]
        for chip in chips:
            st.markdown(f'<span class="stat-chip">{chip}</span>', unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("👈  Upload a Job Description and CV in the sidebar to begin.")
        st.markdown('</div>', unsafe_allow_html=True)

# ── INTERVIEW PAGE ───────────────────────────────────────────
elif st.session_state.interview_state == "INTERVIEW":
    qi    = st.session_state.q_index
    total = len(st.session_state.questions)

    # Progress header
    pct = int((qi / total) * 100)
    st.markdown(
        f"<p style='color:#94a3b8;font-size:0.88rem;margin-bottom:0.25rem;'>"
        f"Question <strong style='color:#22d3ee'>{qi + 1}</strong> of {total}</p>"
        f'<div class="progress-wrap"><div class="progress-fill" style="width:{pct}%"></div></div>',
        unsafe_allow_html=True,
    )

    # Chat history
    st.markdown('<div class="ag-card chat-scroll">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            st.markdown(
                f'<div class="chat-ai"><div class="bubble-label">🤖 AI Interviewer</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-user"><div class="bubble-label">👤 Candidate</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # Input
    answer = st.chat_input("Type your answer and press Enter…")
    if answer and answer.strip():
        st.session_state.chat_history.append({"role": "user", "content": answer.strip()})
        next_qi = qi + 1
        st.session_state.q_index = next_qi

        if next_qi < total:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": st.session_state.questions[next_qi]}
            )
            st.rerun()
        else:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "Thank you! The interview is now complete. Generating your evaluation report…"}
            )
            st.session_state.interview_state = "EVALUATION"
            st.rerun()

# ── EVALUATION PAGE ──────────────────────────────────────────
elif st.session_state.interview_state == "EVALUATION":
    # Generate report exactly once
    if not st.session_state.evaluation_report:
        with st.spinner("Evaluating interview transcript against JD requirements…"):
            st.session_state.evaluation_report = evaluate_interview(
                jd=st.session_state.raw_jd,
                cv=st.session_state.raw_cv,
                history=st.session_state.chat_history,
                name=st.session_state.candidate_name,
            )

    name_label = st.session_state.candidate_name or "Candidate"
    st.markdown(f"### 📋 Evaluation for **{name_label}**")
    st.caption(f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}")

    # Show full transcript in expander  
    with st.expander("📝 Full Interview Transcript"):
        for msg in st.session_state.chat_history:
            label = "🤖 AI Interviewer" if msg["role"] == "assistant" else "👤 Candidate"
            st.markdown(f"**{label}:** {msg['content']}")
            st.markdown("---")

    # Report card
    st.markdown('<div class="ag-card report-wrap">', unsafe_allow_html=True)
    st.markdown(st.session_state.evaluation_report)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄  Start New Interview", type="primary"):
        reset_state(emb_model)
        st.rerun()