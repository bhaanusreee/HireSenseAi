"""
HireSense AI — Simple Resume Analyzer (Streamlit)
Beginner-friendly: upload PDF, extract text, predict role, skills, ATS-style score,
plus a tiny CSV-backed “recruiter log” and Admin Dashboard.
"""

import hashlib
import html
import io
import re
import time
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Paths & constants (easy to tweak)
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "dataset.csv"
# All candidate uploads are appended here (simple “database” for learning projects).
SUBMISSIONS_CSV = Path(__file__).parent / "candidates_log.csv"

# Column order for the recruiter CSV (keep stable so pandas read/write stays predictable).
CANDIDATE_CSV_COLUMNS = [
    "id",
    "timestamp",
    "candidate_name",
    "candidate_email",
    "resume_filename",
    "predicted_role",
    "ats_score",
    "extracted_skills",
    "top_probability",
]

# ---------------------------------------------------------------------------
# Recruiter login (demo only — real apps should use secrets + HTTPS + real auth)
# ---------------------------------------------------------------------------
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Practical email shape check (good enough for a beginner demo; not perfect RFC compliance).
EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)

# Common skills we look for in resume text (simple substring match, case-insensitive)
SKILL_KEYWORDS = [
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "SQL",
    "React",
    "Node",
    "AWS",
    "Docker",
    "Kubernetes",
    "Machine Learning",
    "TensorFlow",
    "pandas",
    "scikit-learn",
    "Excel",
    "Tableau",
    "Figma",
    "Agile",
    "Scrum",
    "Jira",
    "Linux",
    "Git",
    "REST",
    "API",
    "Snowflake",
    "ETL",
    "Leadership",
    "Communication",
]

# Words that suggest good resume structure (used for a simple ATS-style score)
STRUCTURE_KEYWORDS = [
    "experience",
    "education",
    "skills",
    "summary",
    "objective",
    "project",
    "certification",
    "work",
    "employment",
]

# Short, beginner-friendly blurbs for each label the model can output (edit freely).
JOB_ROLE_DESCRIPTIONS = {
    "Software Engineer": (
        "Designs, builds, and tests software—often using languages, frameworks, and APIs. "
        "Typical focus areas include web, mobile, backend services, and shipping reliable features."
    ),
    "Data Scientist": (
        "Uses data, statistics, and machine learning to find patterns, build models, and support decisions. "
        "Often works with Python/SQL, experiments, metrics, and storytelling from data."
    ),
    "Product Manager": (
        "Connects user problems, business goals, and engineering work. "
        "Owns prioritization, roadmaps, requirements, and measuring outcomes after launch."
    ),
    "UX Designer": (
        "Improves how a product feels and flows through research, wireframes, and prototypes. "
        "Collaborates with users and engineers to make experiences clear and accessible."
    ),
    "Marketing Specialist": (
        "Helps grow awareness and demand through campaigns, content, channels, and analytics. "
        "Often tracks performance and iterates messaging based on results."
    ),
    "Financial Analyst": (
        "Builds forecasts, budgets, and reports to explain financial performance. "
        "Works with spreadsheets, accounting data, and variance analysis for planning."
    ),
    "Customer Support": (
        "Helps customers solve issues via tickets, chat, or phone. "
        "Uses CRM tools, clear communication, and follow-up to improve satisfaction."
    ),
    "HR Specialist": (
        "Supports hiring, onboarding, policies, and employee programs. "
        "Coordinates interviews, HR systems, and day-to-day people operations."
    ),
    "DevOps Engineer": (
        "Improves how software is deployed and kept healthy in production. "
        "Focuses on automation, monitoring, infrastructure, and safe release practices."
    ),
    "QA Engineer": (
        "Protects quality through test planning, manual checks, and automation. "
        "Finds bugs early, tracks issues, and helps teams ship more reliable software."
    ),
    "Security Analyst": (
        "Helps protect systems and data through risk review, monitoring, and incident response. "
        "Works with policies, tooling, and investigations to reduce security threats."
    ),
    "Content Writer": (
        "Creates clear written content for websites, blogs, emails, or marketing. "
        "Often edits for tone, SEO, and consistency with brand guidelines."
    ),
    "Sales Representative": (
        "Drives revenue by finding prospects, running demos, and closing deals. "
        "Uses CRM pipelines, follow-ups, and negotiation to hit sales goals."
    ),
}

DEFAULT_ROLE_DESCRIPTION = (
    "This label came from the model, but there is no custom description yet. "
    "Add it to JOB_ROLE_DESCRIPTIONS in app.py."
)


# ---------------------------------------------------------------------------
# Theme tokens (CSS variables) — dark + light “SaaS” skins
# ---------------------------------------------------------------------------
THEME_VARS = {
    "dark": """
        --hs-bg-a: #030712;
        --hs-bg-b: #0b1020;
        --hs-bg-c: #120a22;
        --hs-text: #e8eaf0;
        --hs-muted: #a5b4fc;
        --hs-card: rgba(15,23,42,0.55);
        --hs-card-border: rgba(129,140,248,0.45);
        --hs-shadow: 0 18px 60px rgba(0,0,0,0.55);
        --hs-accent: #6366f1;
        --hs-accent2: #22d3ee;
        --hs-glow1: rgba(99,102,241,0.55);
        --hs-glow2: rgba(34,211,238,0.35);
        --hs-input-bg: rgba(15,23,42,0.65);
        --hs-sidebar: linear-gradient(185deg, #0a0f1f 0%, #0f0a18 100%);
        --hs-hero-tint: linear-gradient(135deg, rgba(99,102,241,0.14), rgba(15,23,42,0.65));
    """,
    "light": """
        --hs-bg-a: #ffffff;
        --hs-bg-b: #f8fafc;
        --hs-bg-c: #f1f5f9;
        --hs-text: #0f172a;
        --hs-muted: #475569;
        --hs-card: #ffffff;
        --hs-card-border: #e2e8f0;
        --hs-shadow: 0 4px 24px rgba(15, 23, 42, 0.08);
        --hs-accent: #4f46e5;
        --hs-accent2: #2563eb;
        --hs-glow1: rgba(79,70,229,0.2);
        --hs-glow2: rgba(37,99,235,0.15);
        --hs-input-bg: #ffffff;
        --hs-sidebar: #fafbfc;
        --hs-hero-tint: linear-gradient(135deg, #f5f3ff 0%, #eff6ff 50%, #f0f9ff 100%);
    """,
}


# ---------------------------------------------------------------------------
# CSV “backend” — one file, pandas only (beginner friendly)
# ---------------------------------------------------------------------------
def ensure_candidates_csv() -> None:
    """Create an empty CSV with headers if it does not exist yet."""
    if not SUBMISSIONS_CSV.exists():
        pd.DataFrame(columns=CANDIDATE_CSV_COLUMNS).to_csv(SUBMISSIONS_CSV, index=False)


def load_candidates() -> pd.DataFrame:
    """Read all saved candidates into a DataFrame (always returns the expected columns)."""
    ensure_candidates_csv()
    df = pd.read_csv(SUBMISSIONS_CSV)
    # If someone edits the CSV by hand, this keeps columns aligned for the UI.
    for col in CANDIDATE_CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[CANDIDATE_CSV_COLUMNS]


def normalize_email(email: str) -> str:
    """Normalize email for comparisons + storage (lowercase, trimmed)."""
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    """Return True if the email looks structurally valid (simple regex check)."""
    e = email.strip()
    if not e:
        return False
    return bool(EMAIL_PATTERN.match(e))


def submission_fingerprint(
    email_key: str,
    name_clean: str,
    file_bytes: bytes,
    predicted_role: str,
    ats_score: int,
    skills: List[str],
) -> str:
    """
    Build a small fingerprint for “did we already save this exact analysis?”.

    Streamlit reruns the script a lot; this prevents rewriting the CSV on every rerun.
    """
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    skills_part = "|".join(skills)
    payload = (
        f"{email_key}|{name_clean}|{file_hash}|{predicted_role}|{ats_score}|{skills_part}".encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def upsert_candidate_record(
    uploaded_file,
    candidate_name: str,
    candidate_email: str,
    predicted_role: str,
    ats_score: int,
    skills: List[str],
    top_probability: float,
) -> bool:
    """
    Insert OR update a candidate row keyed by email (one record per email).

    Returns True if the CSV was written, False if nothing changed (rerun dedupe).
    """
    email_key = normalize_email(candidate_email)
    name_clean = candidate_name.strip()
    file_bytes = uploaded_file.getvalue()

    fp = submission_fingerprint(
        email_key=email_key,
        name_clean=name_clean,
        file_bytes=file_bytes,
        predicted_role=str(predicted_role),
        ats_score=int(ats_score),
        skills=skills,
    )
    if st.session_state.get("last_persisted_digest") == fp:
        return False

    df = load_candidates()
    mask = df["candidate_email"].astype(str).str.strip().str.lower() == email_key

    # Reuse the same row id when updating so links feel stable in demos.
    row_id = str(uuid.uuid4())
    if mask.any():
        row_id = str(df.loc[df.index[mask][0], "id"])

    # Remove any old rows for this email (handles legacy duplicates safely).
    df = df[~mask]

    new_row = {
        "id": row_id,
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "candidate_name": name_clean,
        "candidate_email": email_key,
        "resume_filename": uploaded_file.name,
        "predicted_role": str(predicted_role),
        "ats_score": int(ats_score),
        "extracted_skills": "|".join(skills),
        "top_probability": round(float(top_probability), 4),
    }

    df = pd.concat([df, pd.DataFrame([new_row], columns=CANDIDATE_CSV_COLUMNS)], ignore_index=True)
    df[CANDIDATE_CSV_COLUMNS].to_csv(SUBMISSIONS_CSV, index=False)

    st.session_state.last_persisted_digest = fp
    st.toast("Candidate profile saved (CSV).", icon="📋")
    return True


def delete_candidate(candidate_id: str) -> None:
    """Remove a candidate row by id and rewrite the CSV."""
    df = load_candidates()
    df = df[df["id"].astype(str) != str(candidate_id)]
    df.to_csv(SUBMISSIONS_CSV, index=False)


def search_candidates(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Very simple search: any field contains the query (case-insensitive)."""
    if df.empty or not query.strip():
        return df
    q = query.strip().lower()
    mask = pd.Series(False, index=df.index)
    text_cols = [
        "candidate_name",
        "candidate_email",
        "resume_filename",
        "predicted_role",
        "extracted_skills",
    ]
    for col in text_cols:
        mask = mask | df[col].astype(str).str.lower().str.contains(q, na=False)
    return df[mask]


def extract_text_from_pdf(uploaded_file) -> str:
    """Read uploaded PDF bytes and return plain text from all pages."""
    # BytesIO keeps PyPDF2 happy regardless of file pointer position on reruns
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def normalize_text(text: str) -> str:
    """Lowercase and collapse extra spaces — helps matching and ML."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_skills_in_resume(text: str) -> List[str]:
    """Return skills from SKILL_KEYWORDS that appear in the resume (simple search)."""
    lower = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        if skill.lower() in lower:
            found.append(skill)
    # Remove duplicates while keeping order
    seen = set()
    unique = []
    for s in found:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def compute_ats_score(text: str, skills_found: List[str]) -> Tuple[int, dict]:
    """
    Very simple 'ATS-style' score (0–100) for learning/demo — not a real ATS.
    Combines length, structure keywords, and matched skills.
    """
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)

    # Length: ideal band roughly 150–600 words gets full points
    if word_count < 50:
        length_score = 10
    elif word_count < 150:
        length_score = 15 + int((word_count - 50) / 10)
    elif word_count <= 600:
        length_score = 30
    else:
        length_score = 25

    lower = text.lower()
    structure_hits = sum(1 for kw in STRUCTURE_KEYWORDS if kw in lower)
    structure_score = min(structure_hits * 8, 40)

    skill_score = min(len(skills_found) * 4, 30)

    total = length_score + structure_score + skill_score
    total = max(0, min(100, total))

    breakdown = {
        "Length & detail": length_score,
        "Section keywords": structure_score,
        "Skills matched": skill_score,
    }
    return total, breakdown


@st.cache_resource
def load_model_pipeline():
    """
    Load CSV, build TF-IDF + Logistic Regression pipeline, fit once per session.
    Cached so Streamlit reruns do not retrain every interaction.
    """
    df = pd.read_csv(DATA_PATH)
    # Simple column check for beginners
    if "resume_text" not in df.columns or "job_role" not in df.columns:
        raise ValueError("dataset.csv must have columns: resume_text, job_role")

    X = df["resume_text"].astype(str)
    y = df["job_role"].astype(str)

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=300,
                    stop_words="english",
                    ngram_range=(1, 2),
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, C=1.0),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def predict_role(pipeline: Pipeline, resume_text: str) -> Tuple[str, List[Tuple[str, float]]]:
    """Predict single label; return role name and top (label, probability) pairs."""
    probs = pipeline.predict_proba([resume_text])[0]
    classes = pipeline.classes_
    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    top_role = ranked[0][0]
    top_five = [(str(lbl), float(p)) for lbl, p in ranked[:5]]
    return top_role, top_five


def top_resume_phrases_for_class(
    pipeline: Pipeline, resume_text: str, predicted_role: str, top_n: int = 8
) -> List[str]:
    """
    For the predicted class, list TF-IDF terms in *this* resume that pushed the score up most.

    Beginner intuition: TF-IDF highlights important words in your text; logistic regression learns
    weights per role. Multiplying (word strength in your resume) * (weight for this role) gives a
    simple local explanation—larger values matter more for the prediction.
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    classes = np.asarray(clf.classes_)
    pred = str(predicted_role)

    # Find the row in coef_ that matches the predicted label
    match_idx = np.flatnonzero(classes.astype(str) == pred)
    if match_idx.size == 0:
        return []

    class_row = int(match_idx[0])
    X = tfidf.transform([resume_text])
    # Dense vector is small here because max_features is capped (good enough for learning projects)
    xi = np.asarray(X.toarray()).ravel()
    coef = np.asarray(clf.coef_[class_row]).ravel()
    contributions = xi * coef

    feature_names = tfidf.get_feature_names_out()
    order = np.argsort(contributions)[::-1]

    phrases: List[str] = []
    for idx in order:
        if contributions[idx] <= 0:
            break
        phrases.append(str(feature_names[idx]))
        if len(phrases) >= top_n:
            break
    return phrases


def floating_particles_html() -> str:
    """Lightweight floating particles layer (CSS-only, pointer-events none)."""
    parts = []
    for i in range(24):
        left = (i * 4.1) % 96
        delay = round((i % 10) * 0.37, 2)
        parts.append(
            f'<i style="left:{left}%; animation-delay:{delay}s; width:{4 + (i % 3)}px; height:{4 + (i % 3)}px;"></i>'
        )
    return '<div class="hs-particles" aria-hidden="true">' + "".join(parts) + "</div>"


def apply_theme_css(theme: str) -> None:
    """Load fonts + CSS variables for dark/light themes (smooth transitions on .stApp)."""
    theme_key = theme if theme in THEME_VARS else "dark"
    tokens = THEME_VARS[theme_key]

    # Light mode needs extra Streamlit widget overrides (defaults assume dark text on dark chrome).
    light_widget_css = ""
    if theme_key == "light":
        light_widget_css = """
        /* ----- Light theme: readable sidebar ----- */
        [data-testid="stSidebar"] {
          color: #0f172a !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {
          color: #0f172a !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] .stCaption {
          color: #475569 !important;
        }

        /* ----- Widget labels (main + sidebar) ----- */
        label[data-testid="stWidgetLabel"] p,
        label[data-testid="stWidgetLabel"] span,
        .stWidget label {
          color: #0f172a !important;
          opacity: 1 !important;
        }

        /* ----- Text inputs & text areas ----- */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
          background-color: #ffffff !important;
          color: #0f172a !important;
          -webkit-text-fill-color: #0f172a !important;
          caret-color: #0f172a !important;
          border: 1px solid #94a3b8 !important;
        }
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {
          color: #64748b !important;
          opacity: 1 !important;
        }

        /* ----- File uploader: readable hint + visible Browse button -----
           Never use "section * { color: dark }" — it paints the Browse label dark on a dark pill. */
        [data-testid="stFileUploader"] section {
          background: #ffffff !important;
          border: 1px dashed #64748b !important;
          color: #334155 !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] [data-testid="stCaptionContainer"] p {
          color: #475569 !important;
        }
        [data-testid="stFileUploader"] a {
          color: #1d4ed8 !important;
        }
        /* Browse files control (BaseWeb) — high contrast label */
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] [data-baseweb="button"] {
          background: linear-gradient(120deg, #4338ca, #6366f1) !important;
          color: #ffffff !important;
          -webkit-text-fill-color: #ffffff !important;
          border: none !important;
        }
        [data-testid="stFileUploader"] button:hover,
        [data-testid="stFileUploader"] [data-baseweb="button"]:hover {
          filter: brightness(1.06);
        }
        [data-testid="stFileUploader"] button *,
        [data-testid="stFileUploader"] [data-baseweb="button"] * {
          color: #ffffff !important;
          -webkit-text-fill-color: #ffffff !important;
        }
        [data-testid="stFileUploader"] [role="button"] {
          background: linear-gradient(120deg, #4338ca, #6366f1) !important;
          color: #ffffff !important;
          -webkit-text-fill-color: #ffffff !important;
        }
        [data-testid="stFileUploader"] [role="button"] * {
          color: #ffffff !important;
          -webkit-text-fill-color: #ffffff !important;
        }
        [data-testid="stFileUploader"] input[type="file"]::file-selector-button {
          background: linear-gradient(120deg, #4338ca, #6366f1) !important;
          color: #ffffff !important;
          border: none !important;
          border-radius: 10px !important;
          padding: 0.45rem 1rem !important;
          font-weight: 600 !important;
          cursor: pointer !important;
        }

        /* ----- Alerts: keep text readable on Streamlit default backgrounds ----- */
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] span {
          color: #0f172a !important;
        }

        /* ----- Headings in main area ----- */
        .main h1, .main h2, .main h3, .main h4, .main h5 {
          color: #0f172a !important;
        }
        .main .stCaption, .main [data-testid="stCaptionContainer"] p {
          color: #475569 !important;
        }

        /* ----- Skill pills: dark text on light ----- */
        .skill-pill {
          color: #0f172a !important;
          background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(14,165,233,0.1)) !important;
          box-shadow: 0 1px 3px rgba(15,23,42,0.12) !important;
        }

        /* ----- Progress bar label readability ----- */
        [data-testid="stProgress"] label {
          color: #334155 !important;
        }

        /* ----- Radio labels in sidebar ----- */
        [data-testid="stSidebar"] [data-baseweb="radio"] label,
        [data-testid="stSidebar"] [data-baseweb="radio"] div {
          color: #0f172a !important;
        }

        /* Softer floating particles in light mode */
        .hs-particles i {
          opacity: 0.1 !important;
        }

        /* ----- Fallback: text-like inputs anywhere in main ----- */
        .stApp input[type="text"],
        .stApp input[type="email"],
        .stApp input[type="password"],
        .stApp textarea {
          background-color: #ffffff !important;
          color: #0f172a !important;
          -webkit-text-fill-color: #0f172a !important;
        }
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        :root {{
            {tokens}
        }}
        html, body, [class*="css"] {{
            font-family: "Plus Jakarta Sans", system-ui, sans-serif;
        }}
        .stApp {{
            background: linear-gradient(145deg, var(--hs-bg-a), var(--hs-bg-b), var(--hs-bg-c));
            color: var(--hs-text);
            transition: background 0.45s ease, color 0.35s ease;
        }}
        .block-container {{
            padding-top: 1.15rem;
            max-width: 1180px;
        }}
        [data-testid="stSidebar"] {{
            background: var(--hs-sidebar);
            border-right: 1px solid var(--hs-card-border);
        }}
        .hs-particles {{
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
        }}
        .hs-particles i {{
            position: absolute;
            bottom: -14px;
            border-radius: 999px;
            background: radial-gradient(circle, var(--hs-glow1), transparent 70%);
            opacity: 0.55;
            animation: hs-rise 14s linear infinite;
        }}
        @keyframes hs-rise {{
            0% {{ transform: translateY(0) scale(1); opacity: 0; }}
            12% {{ opacity: 0.75; }}
            100% {{ transform: translateY(-115vh) scale(0.35); opacity: 0; }}
        }}
        .hs-glass {{
            position: relative;
            z-index: 1;
            background: var(--hs-card);
            border: 1px solid var(--hs-card-border);
            border-radius: 18px;
            box-shadow: var(--hs-shadow);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }}
        .hs-glass-pad {{ padding: 1.3rem 1.45rem; margin-bottom: 1rem; }}
        .hs-glass h3, div.hs-glass.hs-glass-pad h3 {{
            margin: 0 0 0.55rem 0;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            background: linear-gradient(90deg, var(--hs-accent), var(--hs-accent2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .hs-hero-title {{
            font-size: clamp(2.2rem, 4.2vw, 3.75rem);
            font-weight: 800;
            margin: 0;
            line-height: 1.05;
            background: linear-gradient(115deg, #f8fafc, var(--hs-accent2), var(--hs-accent), #e9d5ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: hs-glow-pulse 3.6s ease-in-out infinite;
        }}
        @keyframes hs-glow-pulse {{
            0%, 100% {{ filter: drop-shadow(0 0 10px var(--hs-glow1)); }}
            50% {{ filter: drop-shadow(0 0 24px var(--hs-glow2)); }}
        }}
        .hs-typing {{
            display: inline-block;
            overflow: hidden;
            white-space: nowrap;
            max-width: 0;
            animation: hs-typing 2.8s steps(40, end) 0.4s forwards, hs-blink 0.8s step-end 3;
            border-right: 2px solid var(--hs-accent2);
        }}
        @keyframes hs-typing {{
            to {{ max-width: 42ch; }}
        }}
        @keyframes hs-blink {{
            50% {{ border-color: transparent; }}
        }}
        .stButton > button {{
            border-radius: 12px !important;
            border: none !important;
            font-weight: 600 !important;
            transition: transform 0.18s ease, box-shadow 0.2s ease, filter 0.18s ease;
            background: linear-gradient(120deg, var(--hs-accent), #7c3aed) !important;
            color: #fff !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 12px 32px rgba(99,102,241,0.38);
            filter: brightness(1.05);
        }}
        [data-testid="stFileUploader"] section {{
            border-radius: 14px !important;
            border: 1px dashed var(--hs-card-border) !important;
            background: var(--hs-input-bg) !important;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--hs-accent) !important;
        }}
        .skill-pill {{
            display: inline-block;
            padding: 0.2rem 0.65rem;
            margin: 0.25rem 0.3rem 0 0;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            border: 1px solid var(--hs-card-border);
            background: linear-gradient(135deg, rgba(99,102,241,0.28), rgba(34,211,238,0.12));
            box-shadow: 0 0 14px rgba(99,102,241,0.35);
        }}
        div.admin-kpi {{
            background: var(--hs-card);
            border: 1px solid var(--hs-card-border);
            border-radius: 16px;
            padding: 0.95rem 1.05rem;
            margin-bottom: 0.65rem;
            box-shadow: var(--hs-shadow);
            backdrop-filter: blur(12px);
        }}
        div.admin-kpi .kpi-label {{
            color: var(--hs-muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        div.admin-kpi .kpi-value {{
            font-size: 1.32rem;
            font-weight: 800;
            color: var(--hs-text);
        }}
        .hs-login-card {{
            max-width: 420px;
            margin: 0 auto;
            padding: 2rem 1.75rem 1.75rem 1.75rem;
            border-radius: 22px;
            border: 1px solid var(--hs-card-border);
            background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(99,102,241,0.08));
            box-shadow: var(--hs-shadow);
            backdrop-filter: blur(22px);
            -webkit-backdrop-filter: blur(22px);
        }}

        /* ----- Reference UI: top bar + hero + feature grid ----- */
        .hs-main-topbar {{
            display: flex;
            align-items: center;
            justify-content: flex-end;
            margin-bottom: 1rem;
            position: relative;
            z-index: 2;
        }}
        .hs-pill-workspace {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.45rem 0.95rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--hs-text);
            background: var(--hs-card);
            border: 1px solid var(--hs-card-border);
            box-shadow: var(--hs-shadow);
        }}
        .hs-hero-banner {{
            position: relative;
            z-index: 2;
            border-radius: 20px;
            overflow: hidden;
            margin-bottom: 1.35rem;
            background: var(--hs-hero-tint, linear-gradient(135deg, rgba(99,102,241,0.08), rgba(14,165,233,0.06)));
            border: 1px solid var(--hs-card-border);
            box-shadow: var(--hs-shadow);
        }}
        .hs-hero-banner-inner {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 1.25rem;
            padding: 1.5rem 1.65rem;
        }}
        .hs-hero-text {{
            flex: 1 1 280px;
            max-width: 38rem;
        }}
        .hs-hero-kicker {{
            margin: 0 0 0.35rem;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--hs-muted);
        }}
        .hs-hero-h1 {{
            margin: 0;
            font-size: clamp(1.45rem, 2.8vw, 2.05rem);
            font-weight: 800;
            color: var(--hs-text);
            line-height: 1.2;
        }}
        .hs-hero-sub {{
            margin: 0.65rem 0 0;
            color: var(--hs-muted);
            line-height: 1.55;
            font-size: 0.95rem;
        }}
        .hs-hero-art {{
            flex: 0 0 auto;
            width: 140px;
            height: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.5);
            border: 1px dashed var(--hs-card-border);
        }}
        .hs-section-intake {{
            margin: 1.5rem 0 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--hs-text);
        }}
        .hs-why-section {{
            margin-top: 2rem;
            position: relative;
            z-index: 2;
        }}
        .hs-why-title {{
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--hs-text);
            margin: 0 0 1rem;
        }}
        .hs-feature-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
        }}
        .hs-feature-card {{
            background: var(--hs-card);
            border: 1px solid var(--hs-card-border);
            border-radius: 16px;
            padding: 1.1rem 1.15rem;
            box-shadow: var(--hs-shadow);
        }}
        .hs-feature-icon {{
            width: 42px;
            height: 42px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.35rem;
            margin-bottom: 0.65rem;
        }}
        .hs-feature-card h4 {{
            margin: 0 0 0.35rem;
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--hs-text);
        }}
        .hs-feature-card p {{
            margin: 0;
            font-size: 0.82rem;
            color: var(--hs-muted);
            line-height: 1.5;
        }}
        .hs-sidebar-brand {{
            padding: 0.25rem 0 1rem;
            border-bottom: 1px solid var(--hs-card-border);
            margin-bottom: 1rem;
        }}
        .hs-sidebar-brand-row {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .hs-sidebar-logo {{
            font-size: 1.5rem;
            line-height: 1;
        }}
        .hs-sidebar-title {{
            font-size: 1.15rem;
            font-weight: 800;
            color: var(--hs-text);
            margin: 0;
            letter-spacing: -0.02em;
        }}
        .hs-sidebar-tagline {{
            margin: 0.2rem 0 0;
            font-size: 0.78rem;
            color: var(--hs-muted);
            font-weight: 500;
        }}
        .hs-sidebar-promo {{
            margin-top: 1.25rem;
            padding: 1rem;
            border-radius: 14px;
            background: linear-gradient(145deg, rgba(99,102,241,0.1), rgba(14,165,233,0.08));
            border: 1px solid var(--hs-card-border);
        }}
        .hs-sidebar-promo-emoji {{
            font-size: 1.75rem;
            margin-bottom: 0.35rem;
        }}
        .hs-sidebar-promo p {{
            margin: 0;
            font-size: 0.78rem;
            line-height: 1.45;
            color: var(--hs-muted);
        }}
        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
            background: var(--hs-card) !important;
            color: var(--hs-text) !important;
            border: 1px solid var(--hs-card-border) !important;
            box-shadow: none !important;
        }}
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
            border-color: var(--hs-accent) !important;
            background: rgba(99,102,241,0.06) !important;
        }}
        {light_widget_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body_html: str) -> None:
    """Glassmorphism-style content card."""
    safe_title = html.escape(title)
    st.markdown(
        f'<div class="hs-glass hs-glass-pad"><h3>{safe_title}</h3>{body_html}</div>',
        unsafe_allow_html=True,
    )


def skill_pills_html(skills: List[str]) -> str:
    """Render matched skills as glowing pills."""
    if not skills:
        return "<p style='color:var(--hs-muted); margin:0;'>No keyword matches from the built-in list.</p>"
    return "".join(f'<span class="skill-pill">{html.escape(s)}</span>' for s in skills)


def render_main_topbar() -> None:
    """Right-aligned pill matching reference header."""
    st.markdown(
        '<div class="hs-main-topbar">'
        '<span class="hs-pill-workspace">✨ AI screening workspace</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_analyzer_hero() -> None:
    """Two-column hero: headline + decorative art (reference layout)."""
    st.markdown(
        """
        <div class="hs-hero-banner">
          <div class="hs-hero-banner-inner">
            <div class="hs-hero-text">
              <p class="hs-hero-kicker">Screening workspace</p>
              <h1 class="hs-hero-h1">Calibrate your story before recruiters do.</h1>
              <p class="hs-hero-sub">
                One PDF • structured signals • instant feedback loop.
                Your profile syncs to the recruiter log (CSV) when analysis succeeds.
              </p>
            </div>
            <div class="hs-hero-art" aria-hidden="true" title="Resume + scan">
              <div style="text-align:center;line-height:1.1;">
                <span style="font-size:2.4rem;">📄</span><br/>
                <span style="font-size:1.6rem;">🔍</span>
                <span style="font-size:1rem;vertical-align:super;">✨</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_resume_intake_heading() -> None:
    """Section title with icon (matches reference)."""
    st.markdown(
        '<div class="hs-section-intake"><span>📄</span> Resume intake</div>',
        unsafe_allow_html=True,
    )


def render_why_hiresense_grid() -> None:
    """Four-column feature strip from reference design."""
    st.markdown(
        """
        <div class="hs-why-section">
          <h2 class="hs-why-title">Why HireSense AI?</h2>
          <div class="hs-feature-row">
            <div class="hs-feature-card">
              <div class="hs-feature-icon" style="background:linear-gradient(135deg,#ede9fe,#ddd6fe);">🧠</div>
              <h4>AI insights</h4>
              <p>Advanced AI analyzes your resume and extracts key skills and signals.</p>
            </div>
            <div class="hs-feature-card">
              <div class="hs-feature-icon" style="background:linear-gradient(135deg,#dcfce7,#bbf7d0);">✅</div>
              <h4>ATS score</h4>
              <p>Get accurate ATS-style score feedback based on simple, explainable heuristics.</p>
            </div>
            <div class="hs-feature-card">
              <div class="hs-feature-icon" style="background:linear-gradient(135deg,#ffedd5,#fed7aa);">📊</div>
              <h4>Role prediction</h4>
              <p>AI predicts the most suitable job roles for your profile from a demo dataset.</p>
            </div>
            <div class="hs-feature-card">
              <div class="hs-feature-icon" style="background:linear-gradient(135deg,#dbeafe,#bfdbfe);">⚡</div>
              <h4>Instant feedback</h4>
              <p>Receive actionable feedback to improve your resume in seconds.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Logo + tagline in sidebar (reference)."""
    st.markdown(
        """
        <div class="hs-sidebar-brand">
          <div class="hs-sidebar-brand-row">
            <span class="hs-sidebar-logo">✨</span>
            <div>
              <p class="hs-sidebar-title">HireSense AI</p>
              <p class="hs-sidebar-tagline">AI powered screening</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_promo() -> None:
    """Bottom promo card in sidebar."""
    st.markdown(
        """
        <div class="hs-sidebar-promo">
          <div class="hs-sidebar-promo-emoji">🤖</div>
          <p><strong style="color:var(--hs-text);">Hire smarter. Faster.</strong><br/>
          AI screening that helps recruiters find the best candidates effortlessly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Create friendly session defaults (Streamlit reruns need stable keys)."""
    defaults = {
        "admin_logged_in": False,
        "admin_area": "dashboard",
        "last_persisted_digest": None,
        # First launch: premium landing + timed entry.
        "has_seen_landing": False,
        # "dark" | "light" — persisted for the browser session only.
        "ui_theme": "dark",
        # guest navigation: home | upload | results | about | admin_login
        "nav_page": "home",
        # Last successful candidate analysis (for the Results screen).
        "last_analysis": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def recruiter_logout() -> None:
    """End the recruiter session and return the app to candidate-only mode."""
    st.session_state.admin_logged_in = False
    st.session_state.admin_area = "dashboard"
    st.session_state.last_persisted_digest = None
    st.session_state.nav_page = "home"
    for k in ("recruiter_nav_radio", "recruiter_username", "recruiter_password"):
        if k in st.session_state:
            del st.session_state[k]


def recruiter_credentials_ok(username: str, password: str) -> bool:
    """Very small demo auth check (not secure for real production)."""
    return username.strip() == ADMIN_USERNAME and password == ADMIN_PASSWORD


def inject_landing_splash_css(theme: str) -> None:
    """Minimal premium splash — light lavender / dark navy variants (HTML + CSS only)."""
    is_dark = theme == "dark"
    app_bg = (
        "background: radial-gradient(120% 80% at 50% 100%, rgba(124,58,237,0.12) 0%, transparent 50%), "
        "linear-gradient(180deg, #0b0f1a 0%, #070b14 100%) !important;"
        if is_dark
        else "background: linear-gradient(165deg, #ffffff 0%, #faf8ff 45%, #f3f0ff 100%) !important;"
    )
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@500;600;700&family=Sora:wght@500;600;700&display=swap');

        .stApp {{
            {app_bg}
        }}
        [data-testid="stSidebar"] {{ display: none !important; }}
        [data-testid="collapsedControl"] {{ display: none !important; }}
        .block-container {{
            padding-top: 0 !important;
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}

        /* ----- Shell ----- */
        .hs-splash-v2 {{
            min-height: 92vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: "Poppins", "Inter", system-ui, sans-serif;
            position: relative;
            overflow: hidden;
        }}
        .hs-splash-v2 * {{
            box-sizing: border-box;
        }}

        /* Soft ambient glow (both themes) */
        .hs-splash-v2 .hs-splash-ambient {{
            position: absolute;
            inset: 0;
            pointer-events: none;
        }}
        .hs-splash-v2[data-theme="light"] .hs-splash-ambient {{
            background: radial-gradient(ellipse 70% 55% at 50% 0%, rgba(99,102,241,0.08) 0%, transparent 60%),
                        radial-gradient(ellipse 60% 50% at 80% 100%, rgba(59,130,246,0.06) 0%, transparent 50%);
        }}
        .hs-splash-v2[data-theme="dark"] .hs-splash-ambient {{
            background: radial-gradient(ellipse 80% 60% at 50% 100%, rgba(124,58,237,0.18) 0%, transparent 55%),
                        radial-gradient(ellipse 50% 40% at 20% 20%, rgba(59,130,246,0.08) 0%, transparent 50%);
        }}

        /* Centered panel — card feel, soft shadow */
        .hs-splash-v2 .hs-splash-panel {{
            position: relative;
            z-index: 2;
            text-align: center;
            max-width: 420px;
            width: 100%;
            padding: 2.5rem 2rem 2.25rem;
            border-radius: 24px;
            animation: hs-splash-fadein 0.9s ease-out forwards;
            opacity: 0;
        }}
        .hs-splash-v2[data-theme="light"] .hs-splash-panel {{
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(226, 232, 240, 0.9);
            box-shadow: 0 4px 32px rgba(15, 23, 42, 0.06), 0 1px 0 rgba(255,255,255,0.8) inset;
            backdrop-filter: blur(12px);
        }}
        .hs-splash-v2[data-theme="dark"] .hs-splash-panel {{
            background: rgba(15, 23, 42, 0.45);
            border: 1px solid rgba(148, 163, 184, 0.12);
            box-shadow: 0 8px 40px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(99, 102, 241, 0.08) inset;
            backdrop-filter: blur(16px);
        }}

        @keyframes hs-splash-fadein {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* ----- HS logo (gradient + sparkle + float) ----- */
        .hs-splash-v2 .hs-splash-logo-wrap {{
            display: flex;
            justify-content: center;
            margin-bottom: 1.35rem;
            animation: hs-splash-float 3.2s ease-in-out infinite;
        }}
        @keyframes hs-splash-float {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-6px); }}
        }}

        .hs-splash-v2 .hs-splash-hs {{
            position: relative;
            display: inline-flex;
            align-items: flex-end;
            font-family: "Sora", "Inter", sans-serif;
            font-weight: 700;
            font-size: clamp(2.5rem, 8vw, 3.25rem);
            line-height: 1;
            letter-spacing: -0.06em;
        }}
        .hs-splash-v2 .hs-splash-hs .hs-h {{
            background: linear-gradient(145deg, #6366f1 0%, #4f46e5 50%, #4338ca 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .hs-splash-v2 .hs-splash-hs .hs-s {{
            margin-left: 2px;
            background: linear-gradient(145deg, #8b5cf6 0%, #6366f1 45%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .hs-splash-v2 .hs-splash-hs .hs-spark {{
            position: absolute;
            top: -2px;
            right: -14px;
            font-size: 0.65rem;
            color: #a855f7;
            filter: drop-shadow(0 0 6px rgba(168, 85, 247, 0.55));
            line-height: 1;
        }}

        /* ----- Title & subtitle (staggered fade) ----- */
        .hs-splash-v2 .hs-splash-title {{
            font-family: "Sora", "Inter", sans-serif;
            font-weight: 700;
            font-size: clamp(1.55rem, 4.5vw, 2rem);
            letter-spacing: -0.03em;
            margin: 0;
            line-height: 1.2;
            animation: hs-splash-fadein 0.85s ease-out 0.15s forwards;
            opacity: 0;
        }}
        .hs-splash-v2[data-theme="light"] .hs-splash-title .t1 {{ color: #0f172a; }}
        .hs-splash-v2[data-theme="dark"] .hs-splash-title .t1 {{ color: #f1f5f9; }}
        .hs-splash-v2 .hs-splash-title .t2 {{
            background: linear-gradient(90deg, #4f46e5, #7c3aed, #2563eb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .hs-splash-v2 .hs-splash-sub {{
            margin: 0.85rem 0 0;
            font-size: 0.98rem;
            font-weight: 500;
            letter-spacing: 0.01em;
            animation: hs-splash-fadein 0.85s ease-out 0.28s forwards;
            opacity: 0;
        }}
        .hs-splash-v2[data-theme="light"] .hs-splash-sub {{ color: #64748b; }}
        .hs-splash-v2[data-theme="dark"] .hs-splash-sub {{ color: #94a3b8; }}

        /* ----- Loading row ----- */
        .hs-splash-v2 .hs-splash-load {{
            margin-top: 2.25rem;
            animation: hs-splash-fadein 0.85s ease-out 0.4s forwards;
            opacity: 0;
        }}
        .hs-splash-v2 .hs-splash-load-text {{
            margin: 0 0 0.85rem;
            font-size: 0.82rem;
            font-weight: 500;
        }}
        .hs-splash-v2[data-theme="light"] .hs-splash-load-text {{ color: #64748b; }}
        .hs-splash-v2[data-theme="dark"] .hs-splash-load-text {{ color: #94a3b8; }}

        .hs-splash-v2 .hs-splash-dots {{
            display: flex;
            justify-content: center;
            gap: 10px;
        }}
        .hs-splash-v2 .hs-splash-dots span {{
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            animation: hs-splash-dotglow 1.35s ease-in-out infinite;
        }}
        .hs-splash-v2 .hs-splash-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
        .hs-splash-v2 .hs-splash-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
        @keyframes hs-splash-dotglow {{
            0%, 100% {{ transform: scale(0.85); opacity: 0.45; box-shadow: 0 0 0 0 rgba(99,102,241,0); }}
            50% {{ transform: scale(1.1); opacity: 1; box-shadow: 0 0 14px 2px rgba(99,102,241,0.45); }}
        }}

        /* Minimal progress line — smooth handoff cue */
        .hs-splash-v2 .hs-splash-progress {{
            margin: 1.75rem auto 0;
            max-width: 200px;
            height: 3px;
            border-radius: 999px;
            overflow: hidden;
            opacity: 0.85;
        }}
        .hs-splash-v2[data-theme="light"] .hs-splash-progress {{
            background: rgba(226, 232, 240, 0.9);
        }}
        .hs-splash-v2[data-theme="dark"] .hs-splash-progress {{
            background: rgba(51, 65, 85, 0.6);
        }}
        .hs-splash-v2 .hs-splash-progress-bar {{
            height: 100%;
            width: 0%;
            border-radius: 999px;
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #3b82f6);
            animation: hs-splash-prog 2.85s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }}
        @keyframes hs-splash-prog {{
            to {{ width: 100%; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_landing_entry() -> None:
    """Elegant opening splash — works in light & dark; continues to Home after load."""
    theme = st.session_state.get("ui_theme", "dark")
    inject_landing_splash_css(theme)

    st.markdown(
        f"""
        <div class="hs-splash-v2" data-theme="{html.escape(theme)}">
          <div class="hs-splash-ambient" aria-hidden="true"></div>
          <div class="hs-splash-panel">
            <div class="hs-splash-logo-wrap">
              <span class="hs-splash-hs" aria-label="HireSense">
                <span class="hs-h">H</span><span class="hs-s">S</span>
                <span class="hs-spark">✦</span>
              </span>
            </div>
            <h1 class="hs-splash-title">
              <span class="t1">HireSense </span><span class="t2">AI</span>
            </h1>
            <p class="hs-splash-sub">Smarter Hiring Starts Here.</p>
            <div class="hs-splash-load">
              <p class="hs-splash-load-text">Initializing AI Screening Engine…</p>
              <div class="hs-splash-dots" aria-hidden="true"><span></span><span></span><span></span></div>
            </div>
            <div class="hs-splash-progress" aria-hidden="true">
              <div class="hs-splash-progress-bar"></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    time.sleep(2.9)
    st.session_state.has_seen_landing = True
    st.session_state.nav_page = "home"
    st.rerun()


def render_home() -> None:
    """Home hub — matches reference: top bar + hero + quick actions."""
    st.markdown(floating_particles_html(), unsafe_allow_html=True)
    render_main_topbar()
    st.markdown(
        """
        <div class="hs-hero-banner">
          <div class="hs-hero-banner-inner">
            <div class="hs-hero-text">
              <p class="hs-hero-kicker">Welcome back</p>
              <h1 class="hs-hero-h1">Hire faster with calmer, clearer signals.</h1>
              <p class="hs-hero-sub">
                Structured resume screening, ATS-style scoring, and a lightweight CSV log —
                ideal for demos, classrooms, and rapid hiring prototypes.
              </p>
            </div>
            <div class="hs-hero-art" aria-hidden="true">✨</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⬆️ Upload resume", use_container_width=True, key="home_upload"):
            st.session_state.nav_page = "upload"
            st.rerun()
    with c2:
        if st.button("📊 View analysis", use_container_width=True, key="home_results"):
            st.session_state.nav_page = "results"
            st.rerun()
    with c3:
        if st.button("ℹ️ About", use_container_width=True, key="home_about"):
            st.session_state.nav_page = "about"
            st.rerun()
    render_why_hiresense_grid()


def render_results() -> None:
    """Read-only recap of the last successful screening in this session."""
    render_main_topbar()
    st.subheader("Analysis results")
    data = st.session_state.get("last_analysis")
    if not data:
        st.info("Run an upload from **Upload resume** — your latest scorecards will appear here.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("ATS-style score", f"{data['ats_total']}/100")
    with c2:
        st.metric("Predicted role", data["top_role"])
    with c3:
        st.metric("Model confidence", f"{data['top_prob'] * 100:.1f}%")

    st.markdown("#### Matched skills")
    st.markdown(skill_pills_html(data.get("skills", [])), unsafe_allow_html=True)

    st.markdown("#### Role snapshot")
    st.write(data.get("role_description", ""))

    with st.expander("Why this prediction?", expanded=False):
        st.write(data.get("why_text", ""))


def render_about() -> None:
    """Lightweight product / stack description."""
    render_main_topbar()
    st.subheader("About HireSense AI")
    st.markdown(
        """
This is a **learning-friendly** resume lab: TF-IDF + logistic regression on a tiny CSV,
ATS-style heuristics, skill keyword matching, and a CSV-backed recruiter log.

**Not** a certified ATS or legal hiring tool — use it to explore UX + ML wiring in Streamlit.
        """
    )


def render_recruiter_login_main() -> None:
    """Glassmorphism admin login (demo credentials in source — replace for production)."""
    st.markdown(floating_particles_html(), unsafe_allow_html=True)
    render_main_topbar()
    _, mid, _ = st.columns([1, 1.15, 1])
    with mid:
        st.markdown(
            """
            <div class="hs-login-card">
              <p style="margin:0; font-size:0.75rem; letter-spacing:0.16em; text-transform:uppercase; color:var(--hs-muted);">
                Secure access
              </p>
              <h2 style="margin:0.35rem 0 0.25rem; font-size:1.55rem; font-weight:800; color:var(--hs-text);">
                Recruiter console
              </h2>
              <p style="margin:0 0 1.1rem; color:var(--hs-muted); font-size:0.92rem;">
                Authenticate to open the hiring command center. Sessions stay in-browser only.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        username = st.text_input("Username", key="recruiter_username", placeholder="admin")
        password = st.text_input("Password", type="password", key="recruiter_password")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔐 Sign in", type="primary", use_container_width=True):
                if recruiter_credentials_ok(username, password):
                    st.session_state.admin_logged_in = True
                    st.session_state.admin_area = "dashboard"
                    st.session_state.nav_page = "home"
                    st.toast("Access granted — loading dashboard.", icon="✅")
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
        with b2:
            if st.button("← Back", use_container_width=True):
                st.session_state.nav_page = "home"
                st.rerun()


def admin_kpi_tile(label: str, value: str) -> None:
    """Small HTML tile for the admin dashboard header row."""
    st.markdown(
        f'<div class="admin-kpi"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def style_plotly(fig: go.Figure, theme: str, height: int = 320) -> go.Figure:
    """Match charts to the active UI theme."""
    if theme == "light":
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.55)",
            font=dict(color="#0f172a"),
            height=height,
            margin=dict(l=10, r=10, t=40, b=10),
        )
    else:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb"),
            height=height,
            margin=dict(l=10, r=10, t=40, b=10),
        )
    return fig


def page_admin() -> None:
    """Admin dashboard: KPIs, charts, searchable table, delete, CSV export."""
    render_main_topbar()
    st.markdown(
        "<h1 style='margin-bottom:0.15rem;color:var(--hs-text);'>Recruitment Command Center</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:var(--hs-muted); font-size:1.02rem;'>"
        "Monitor uploads, ATS-style scores, and predicted roles — powered by a simple CSV log.</p>",
        unsafe_allow_html=True,
    )

    df = load_candidates()

    # ----- KPI row (SaaS-style summary) -----
    total_profiles = int(len(df))
    avg_ats = float(df["ats_score"].astype(float).mean()) if total_profiles else 0.0
    top_role_name = "—"
    if total_profiles:
        top_role_name = str(df["predicted_role"].astype(str).value_counts().idxmax())

    k1, k2, k3 = st.columns(3, gap="small")
    with k1:
        admin_kpi_tile("Total uploads", str(total_profiles))
    with k2:
        admin_kpi_tile("Average ATS-style score", f"{avg_ats:.1f}" if total_profiles else "—")
    with k3:
        admin_kpi_tile("Top predicted role", top_role_name if len(top_role_name) < 36 else top_role_name[:33] + "…")

    st.caption(
        "Each email maps to one CSV row (updates overwrite). "
        f"Unique emails tracked: **{int(df['candidate_email'].astype(str).str.lower().str.strip().nunique()) if total_profiles else 0}**."
    )

    st.markdown("---")

    # ----- Recent uploads (quick glance) -----
    st.subheader("Recent uploads")
    if total_profiles:
        recent = df.sort_values("timestamp", ascending=False).head(12)[
            ["timestamp", "candidate_name", "candidate_email", "resume_filename", "predicted_role", "ats_score"]
        ]
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("No uploads yet — candidates appear here after a successful screening.")

    st.markdown("---")

    # ----- Charts -----
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Top predicted roles")
        if total_profiles:
            counts = df["predicted_role"].astype(str).value_counts().head(10)
            fig_roles = go.Figure(
                data=[
                    go.Bar(
                        x=counts.values,
                        y=counts.index.astype(str),
                        orientation="h",
                        marker=dict(
                            color=counts.values,
                            colorscale=[[0, "#1e3a5f"], [1, "#7c3aed"]],
                        ),
                    )
                ]
            )
            fig_roles.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Count")
            st.plotly_chart(style_plotly(fig_roles, st.session_state.ui_theme), use_container_width=True)
        else:
            st.info("Upload resumes on the analyzer page to see role trends here.")

    with c2:
        st.subheader("ATS-style score distribution")
        if total_profiles:
            fig_hist = go.Figure(
                data=[
                    go.Histogram(
                        x=df["ats_score"].astype(float),
                        nbinsx=min(20, max(5, total_profiles)),
                        marker=dict(color="#6366f1", line=dict(color="#312e81", width=1)),
                    )
                ]
            )
            fig_hist.update_layout(xaxis_title="ATS-style score", yaxis_title="Candidates")
            st.plotly_chart(style_plotly(fig_hist, st.session_state.ui_theme), use_container_width=True)
        else:
            st.info("No scores yet — your histogram will appear after the first save.")

    st.markdown("---")

    # ----- Search + table -----
    st.subheader("Candidate registry")
    q = st.text_input("Search (name, email, filename, role, skills)", placeholder="Type to filter…")
    filtered = search_candidates(df, q)

    st.caption(f"Showing **{len(filtered)}** / **{len(df)}** rows")

    display_cols = [
        "timestamp",
        "candidate_name",
        "candidate_email",
        "resume_filename",
        "predicted_role",
        "ats_score",
        "top_probability",
        "extracted_skills",
        "id",
    ]
    show_df = filtered[display_cols].copy()
    # Make skills easier to read in the table (pipes → commas for display only).
    show_df["extracted_skills"] = show_df["extracted_skills"].astype(str).str.replace("|", ", ", regex=False)

    st.dataframe(
        show_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Saved at"),
            "candidate_name": st.column_config.TextColumn("Candidate"),
            "candidate_email": st.column_config.TextColumn("Email"),
            "resume_filename": st.column_config.TextColumn("Resume file"),
            "predicted_role": st.column_config.TextColumn("Predicted role"),
            "ats_score": st.column_config.NumberColumn("ATS-style"),
            "top_probability": st.column_config.NumberColumn("Top prob.", format="%.3f"),
            "extracted_skills": st.column_config.TextColumn("Skills", width="large"),
            "id": st.column_config.TextColumn("Row ID", width="medium"),
        },
    )

    st.markdown("---")
    st.subheader("Row actions")

    act1, act2 = st.columns((1.1, 1), gap="large")
    with act1:
        st.markdown("**Delete an entry** (rewrites the CSV)")
        if filtered.empty:
            st.caption("Nothing to delete yet.")
            chosen_id = None
        else:
            options = filtered["id"].astype(str).tolist()

            def _fmt(rid: str) -> str:
                hit = filtered.loc[filtered["id"].astype(str) == rid].iloc[0]
                return f"{hit['candidate_name']} — {hit['predicted_role']} — {hit['resume_filename']}"

            chosen_id = st.selectbox("Pick a row", options=options, format_func=_fmt)

        delete_clicked = st.button(
            "Delete selected entry",
            type="primary",
            disabled=filtered.empty or (chosen_id is None),
        )
        if delete_clicked and chosen_id:
            delete_candidate(str(chosen_id))
            # Allow the same analysis fingerprint to be saved again after a manual delete.
            st.session_state.last_persisted_digest = None
            st.success("Deleted. Refreshing…")
            st.rerun()

    with act2:
        st.markdown("**Export**")
        export_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download candidates_log.csv",
            data=export_bytes,
            file_name="candidates_log_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Tip: the live file on disk is `candidates_log.csv` next to `app.py`.")


def page_analyzer() -> None:
    """Candidate workspace — layout aligned to premium SaaS reference (hero, intake, feature grid)."""
    st.markdown(floating_particles_html(), unsafe_allow_html=True)
    render_main_topbar()
    render_analyzer_hero()
    render_resume_intake_heading()

    cname, cemail = st.columns(2, gap="medium")
    with cname:
        candidate_name = st.text_input("👤 Full name *", placeholder="e.g., Jordan Lee")
    with cemail:
        candidate_email = st.text_input("✉️ Work email *", placeholder="name@company.com")

    uploaded = st.file_uploader("📤 Drop your resume (PDF, single file)", type=["pdf"])
    st.caption("Limit 200MB per file · PDF")

    name_ok = bool(candidate_name.strip())
    email_nonempty = bool(candidate_email.strip())
    email_ok = is_valid_email(candidate_email) if email_nonempty else False
    ready_for_analysis = name_ok and email_ok

    intake_hints: List[str] = []
    if not name_ok:
        intake_hints.append("Enter your **full name** to unlock analysis.")
    if not email_nonempty:
        intake_hints.append("Enter your **work email**.")
    elif not email_ok:
        intake_hints.append("Fix your email format (example: `name@company.com`).")

    if intake_hints:
        st.info("\n\n".join(intake_hints))

    if uploaded is None:
        st.info("👆 Upload **one** PDF after your details are complete.")
        render_why_hiresense_grid()
        return

    if not ready_for_analysis:
        st.error("Analysis stays locked until both **name** and a **valid email** are valid.")
        render_why_hiresense_grid()
        return

    try:
        raw_text = extract_text_from_pdf(uploaded)
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        return

    if not raw_text:
        st.warning("No text found in this PDF. Try a text-based PDF (not only scanned images).")
        return

    st.toast("Resume ingested — models are running.", icon="🤖")

    col_left, col_right = st.columns((1.1, 1), gap="large")

    with col_left:
        card("Extracted text (preview)", "")
        preview = raw_text[:2500] + ("…" if len(raw_text) > 2500 else "")
        st.text_area("preview", preview, height=220, label_visibility="collapsed")

    try:
        pipeline = load_model_pipeline()
    except Exception as e:
        st.error(f"Model load error: {e}")
        return

    top_role, top_ranked = predict_role(pipeline, raw_text)
    skills = find_skills_in_resume(raw_text)
    ats_total, ats_breakdown = compute_ats_score(raw_text, skills)
    top_prob = float(top_ranked[0][1])
    role_str = str(top_role)

    written = upsert_candidate_record(
        uploaded_file=uploaded,
        candidate_name=candidate_name,
        candidate_email=candidate_email,
        predicted_role=role_str,
        ats_score=int(ats_total),
        skills=skills,
        top_probability=top_prob,
    )
    if written:
        st.balloons()

    st.progress(min(max(ats_total / 100.0, 0.0), 1.0))
    st.caption(f"ATS-style signal strength — **{ats_total}/100**")

    with col_right:
        safe_role = html.escape(role_str)
        card(
            "Predicted role",
            f"<p style='margin:0;font-size:1.35rem;font-weight:800;color:var(--hs-text)'>{safe_role}</p>",
        )
        st.caption("Demo classifier trained on `dataset.csv`.")

        m1, m2, m3 = st.columns(3)
        m1.metric("ATS-style score", f"{ats_total}/100")
        m2.metric("Skills matched", len(skills))
        m3.metric("Words", len(re.findall(r"\b\w+\b", raw_text)))

    st.markdown("---")
    st.subheader("Signal cards")
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        card(
            "ATS-style score",
            f"<p style='margin:0;font-size:2.1rem;font-weight:800;color:var(--hs-accent);'>{ats_total}</p>"
            f"<p style='margin:0.25rem 0 0;color:var(--hs-muted);'>out of 100 heuristic</p>",
        )
    with g2:
        card(
            "Predicted role (ML)",
            f"<p style='margin:0;font-size:1.35rem;font-weight:800;color:var(--hs-text);'>{safe_role}</p>",
        )

    st.markdown("### Matched skills")
    st.markdown(skill_pills_html(skills), unsafe_allow_html=True)

    st.markdown("### Resume tips")
    st.markdown(
        """
<div class="hs-glass hs-glass-pad">
  <ul style="margin:0; padding-left:1.1rem; color:var(--hs-muted); line-height:1.55;">
    <li>Mirror the language of the target role without stuffing keywords.</li>
    <li>Quantify outcomes (%, $, time saved) wherever you can.</li>
    <li>Use crisp section headers: Experience, Projects, Education, Skills.</li>
    <li>Export PDFs with embedded text — scans can break parsers.</li>
  </ul>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Role profile & why this prediction")

    st.markdown(f"**What “{html.escape(role_str)}” means**")
    st.write(JOB_ROLE_DESCRIPTIONS.get(role_str, DEFAULT_ROLE_DESCRIPTION))

    runner = top_ranked[1] if len(top_ranked) > 1 else None
    key_phrases = top_resume_phrases_for_class(pipeline, raw_text, role_str)

    why_lines = [
        f"It gave **{html.escape(role_str)}** the **highest probability ({top_prob * 100:.1f}%)** among the "
        "labels it was trained on in `dataset.csv` (demo only).",
    ]
    if runner is not None:
        why_lines.append(
            f"Runner-up: **{runner[0]}** at **{runner[1] * 100:.1f}%** — more training data would smooth this curve."
        )
    why_lines.append(
        "TF-IDF highlights important phrases; logistic regression scores each possible role. "
        "Below are the phrases that most increased the predicted label in *your* resume."
    )
    why_text = "\n\n".join(why_lines)

    st.markdown("**Why the model predicted this role**")
    st.markdown(why_text)
    if key_phrases:
        safe_list = ", ".join(f"“{html.escape(p)}”" for p in key_phrases)
        st.markdown(f"**Top phrases from your resume:** {safe_list}")
    else:
        st.info(
            "No standout phrases were found by this simple explanation (common with very short text or niche wording)."
        )

    st.session_state.last_analysis = {
        "candidate_name": candidate_name.strip(),
        "candidate_email": candidate_email.strip(),
        "filename": uploaded.name,
        "top_role": role_str,
        "top_prob": top_prob,
        "ats_total": int(ats_total),
        "skills": skills,
        "role_description": JOB_ROLE_DESCRIPTIONS.get(role_str, DEFAULT_ROLE_DESCRIPTION),
        "why_text": why_text,
    }

    st.markdown("<br>", unsafe_allow_html=True)

    th = st.session_state.ui_theme
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("Top role probabilities")
        labels = [x[0] for x in top_ranked]
        values = [x[1] for x in top_ranked]
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker=dict(
                        color=values,
                        colorscale=[[0, "#1e3a5f"], [1, "#7c3aed"]],
                    ),
                )
            ]
        )
        fig_bar.update_layout(xaxis_title="Probability", yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_plotly(fig_bar, th), use_container_width=True)

    with c2:
        st.subheader("ATS-style score breakdown")
        names = list(ats_breakdown.keys())
        vals = list(ats_breakdown.values())
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=names,
                    values=vals,
                    hole=0.55,
                    marker=dict(colors=["#3b82f6", "#8b5cf6", "#22d3ee"]),
                    textinfo="label+percent",
                )
            ]
        )
        fig_pie.update_layout(showlegend=True)
        st.plotly_chart(style_plotly(fig_pie, th), use_container_width=True)

    render_why_hiresense_grid()
    st.caption("Run locally: `streamlit run app.py`")


def main() -> None:
    st.set_page_config(
        page_title="HireSense AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    apply_theme_css(st.session_state.ui_theme)

    # Splash screen once per browser session.
    if not st.session_state.has_seen_landing:
        render_landing_entry()
        return

    # ----- Sidebar navigation -----
    with st.sidebar:
        render_sidebar_brand()

        st.markdown('<p style="margin:0.5rem 0 0.35rem;font-size:0.72rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--hs-muted);">Theme</p>', unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            kwargs = {"use_container_width": True}
            if st.session_state.ui_theme == "dark":
                kwargs["type"] = "primary"
            if st.button("🌙 Dark", **kwargs):
                st.session_state.ui_theme = "dark"
                st.rerun()
        with t2:
            kwargs2 = {"use_container_width": True}
            if st.session_state.ui_theme == "light":
                kwargs2["type"] = "primary"
            if st.button("☀️ Light", **kwargs2):
                st.session_state.ui_theme = "light"
                st.rerun()

        st.markdown("---")

        if st.session_state.admin_logged_in:
            st.success("Recruiter session active")
            nav_choice = st.radio(
                "Workspace",
                ["Command center", "Candidate lab"],
                index=0 if st.session_state.admin_area == "dashboard" else 1,
                key="recruiter_nav_radio",
            )
            st.session_state.admin_area = "dashboard" if nav_choice == "Command center" else "analyzer"

            if st.button("🚪 Logout", use_container_width=True):
                recruiter_logout()
                st.rerun()
        else:
            nav = st.session_state.nav_page
            st.markdown('<p style="margin:0 0 0.35rem;font-size:0.72rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--hs-muted);">Navigate</p>', unsafe_allow_html=True)
            if st.button("🏠 Home", use_container_width=True, type="primary" if nav == "home" else "secondary"):
                st.session_state.nav_page = "home"
                st.rerun()
            if st.button("📤 Upload resume", use_container_width=True, type="primary" if nav == "upload" else "secondary"):
                st.session_state.nav_page = "upload"
                st.rerun()
            if st.button("📊 Analysis results", use_container_width=True, type="primary" if nav == "results" else "secondary"):
                st.session_state.nav_page = "results"
                st.rerun()
            if st.button("🔐 Admin login", use_container_width=True, type="primary" if nav == "admin_login" else "secondary"):
                st.session_state.nav_page = "admin_login"
                st.rerun()
            if st.button("ℹ️ About project", use_container_width=True, type="primary" if nav == "about" else "secondary"):
                st.session_state.nav_page = "about"
                st.rerun()

            render_sidebar_promo()

        st.markdown("---")
        st.caption("Streamlit · scikit-learn · Plotly · PyPDF2 · pandas")

    # ----- Main routing -----
    if st.session_state.admin_logged_in and st.session_state.admin_area == "dashboard":
        page_admin()
        return

    if st.session_state.admin_logged_in and st.session_state.admin_area == "analyzer":
        page_analyzer()
        return

    nav = st.session_state.nav_page
    if nav == "admin_login":
        render_recruiter_login_main()
        return
    if nav == "home":
        render_home()
        return
    if nav == "upload":
        page_analyzer()
        return
    if nav == "results":
        render_results()
        return
    if nav == "about":
        render_about()
        return

    # Safe fallback
    render_home()


if __name__ == "__main__":
    main()
