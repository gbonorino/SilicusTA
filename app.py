import streamlit as st

import streamlit as st

st.set_page_config(page_title="Silicus TA 2.0", page_icon="🎓")

st.title("Silicus TA 2.0  🎓🤖")
st.markdown(
"""
Welcome to **Silicus TA 2.0**, your course‑aware teaching assistant.

* **Chat page** – ask anything about the uploaded lecture slides.  
* **Admin page** – professors can upload PDF decks, rebuild embeddings, and
  push updates to GitHub with a single click.

### What Silicus *can* do
* Answer conceptual questions tied to specific lecture pages.
* Remember the last few turns of conversation for follow‑up queries.
* Cite exactly which PDF page(s) it used.

### What Silicus *cannot* do (yet)
* Reason about material **outside** the uploaded PDFs.  
* Handle very large (> 8 K‑token) single pages without truncating.  
* Guarantee 100 % accuracy—always verify critical answers.

### Student tips
1. Be specific: “Explain slide 12’s likelihood analogy” beats “Explain MLE.”
2. Use follow‑ups instead of repeating the full question.
3. Open the citation expander to read the source excerpt.

**How it works**  
1. 📄 We pre‑OCR all lecture PDFs and store page‑level embeddings.  
2. 🔍 When you ask a question, we pull the 10 most relevant pages.  
3. 🤖 Mistral’s chat model answers using those excerpts (RAG).

*No personal data is stored.* You can inspect the source on GitHub and fork it for your own courses!
---

Head to **➡️ Chat** in the sidebar to start, or **➡️ Admin** if you’re an
instructor.
""")

st.page_link("pages/1_Silicus_TA.py", label="👉 Go to the Chat page")
