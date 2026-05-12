# HireSense AI

**HireSense AI** is a beginner-friendly, **Streamlit** demo of an AI-assisted resume screening flow: upload a PDF, get ATS-style scoring, skill highlights, a **demo job-role prediction**, and optional **recruiter analytics** — all with a **modern SaaS-style UI** (splash screen, light/dark themes, glass-style panels).

> **Disclaimer:** This is a **learning / portfolio project**, not a certified ATS or legal hiring tool. Model outputs are trained on a **small sample dataset** and should not be used for real hiring decisions.

---

## What it does

### For candidates

- **Splash screen** — Minimal branded intro (HS logo, tagline, loading state), then the main app.
- **Resume intake** — Full name + work email (validated); **one PDF** per analysis.
- **PDF text extraction** via **PyPDF2** (text-based PDFs work best; scanned-only PDFs may be empty).
- **ATS-style score** — Simple heuristic (length, section-like keywords, matched skills); **not** a real applicant tracking system score.
- **Skill tags** — Highlights common keywords found in the resume from a built-in list.
- **Role prediction** — **TF-IDF + Logistic Regression** trained on `dataset.csv`; includes short **role description** and a plain-language **“why this prediction”** explanation.
- **Charts** — Plotly visuals for top role probabilities and score breakdown.
- **Persistence** — Successful runs **upsert** one row per email into **`candidates_log.csv`** (same email updates the existing row).

### For recruiters (demo admin)

- **Admin login** in the sidebar (credentials are **hard-coded for demos** — change them in production).
- **Dashboard** — Totals, average ATS-style score, top role, recent uploads, searchable table, delete row, **download CSV** export.
- **Session-only “security”** — Hides the dashboard behind login; **not** production-grade auth.

---

## Tech stack

| Layer | Choice |
|--------|--------|
| UI | **Streamlit**, custom **HTML/CSS** (themes, splash, cards) |
| ML | **scikit-learn** — `TfidfVectorizer` + `LogisticRegression` |
| Data | **pandas** — `dataset.csv` (training snippets), `candidates_log.csv` (submissions) |
| PDF | **PyPDF2** |
| Charts | **Plotly** |

**Python 3.10+** recommended (works with current Streamlit / sklearn releases).

---

## Quick start

### 1. Clone and enter the project

```bash
git clone https://github.com/bhaanusreee/HireSenseAi.git
cd HireSenseAi
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually **http://localhost:8501**).

---

## Project layout

```
hireai/
├── app.py              # Full app: UI, ML pipeline, CSV helpers, admin flow
├── requirements.txt    # Python dependencies
├── dataset.csv         # Tiny labeled snippets → trains the role classifier
├── candidates_log.csv  # Created/updated: one row per candidate email (after analysis)
├── README.md
└── .gitignore
```

- **`dataset.csv`** — Columns: `resume_text`, `job_role`. Expand this to improve predictions (more rows, consistent labels).
- **`candidates_log.csv`** — Optional to commit; contains PII from your tests. Add to `.gitignore` if you prefer it local-only.

---

## Using the app

1. **First launch** — Splash screen, then **Home** (or go to **Upload resume** from the sidebar).
2. **Theme** — Toggle **Dark** / **Light** in the sidebar (session preference).
3. **Upload flow** — Enter **name** + **valid email**, upload a **PDF**, review scores, role, skills, and tips.
4. **Analysis results** — Sidebar link shows the **last successful** run in this browser session.
5. **Admin** — **Admin login** → dashboard. **Change default credentials** in `app.py` (`ADMIN_USERNAME` / `ADMIN_PASSWORD`) before any real deployment.

---

## How the “AI” part works (short)

1. Resume text is vectorized with **TF-IDF** (word and short phrases, English stop words removed, capped features).
2. A **logistic regression** model predicts **job_role** from `dataset.csv`.
3. **ATS-style score** and **skills** use separate, simple rules — they are **not** produced by the classifier.
4. **Explanations** for the predicted role use a lightweight linear-model cue (important terms in *your* resume for that label).

---

## Customization ideas

- Add rows to **`dataset.csv`** and redeploy / restart the app (model is refit per session via Streamlit cache).
- Adjust **`SKILL_KEYWORDS`** and **`JOB_ROLE_DESCRIPTIONS`** in `app.py`.
- Replace demo auth with **Streamlit secrets**, OAuth, or a real backend before production.

---

## Troubleshooting

| Issue | What to try |
|--------|-------------|
| Empty PDF text | Use a PDF with selectable text, not a flat scan. |
| `streamlit` not found | Use `python -m streamlit run app.py`. |
| Port in use | `streamlit run app.py --server.port 8502` |
| Push / Git errors | Ensure `git add` and `git commit` succeed **before** `git push`. |

---

## License

Use and modify freely for **education and demos**. If you ship a product, replace demo credentials, add proper security, and comply with hiring and privacy laws in your jurisdiction.

---

## Acknowledgements

Built as a **minimal, readable** example of wiring **Streamlit + classical ML + CSV storage** for resume screening UX experiments.
