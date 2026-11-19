"""
app.py — Step 9 of the pipeline: a simple web app for the assistant.

Type a question about the Kenya Data Protection Act, 2019 and the app shows:
  - Claude's answer with section citations (only if an API key is set up)
  - the sections of the Act it found, so you can check the answer yourself

Without an API key the app still works: it shows the matching sections only.

Run it from the project root with:
    streamlit run app.py
It opens in your browser at http://localhost:8501
"""

import os
import sys

import streamlit as st

# Our pipeline code lives in src/, and those files import each other like
# "from config import ...". Adding src/ to Python's search path lets the
# same imports work when we start from here.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import anthropic  # noqa: E402  (must come after the sys.path line above)

from answer import answer  # noqa: E402  (this also loads .env)
from retrieve import retrieve  # noqa: E402

# Is an API key available? If not, the app runs in "search only" mode.
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Kenya DPA Assistant", page_icon="⚖️")
st.title("⚖️ Kenya Data Protection Act Assistant")
st.caption("Ask a question about the Data Protection Act, 2019. "
           "Answers come only from the text of the Act and cite their sections.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Sections to search", min_value=3, max_value=8, value=5,
                      help="How many chunks of the Act to find and give to Claude.")
    if HAS_KEY:
        st.success("Claude answers: on")
    else:
        st.warning("Claude answers: off (no API key found). "
                   "Showing matching sections only. "
                   "Add ANTHROPIC_API_KEY to a .env file to turn answers on.")
    st.markdown("---")
    st.caption("This tool gives information about the Act, not legal advice.")

# A form groups the text box and button, so the search only runs when
# the button is pressed (or Enter), not on every key press.
with st.form("question_form"):
    question = st.text_input("Your question",
                             placeholder="e.g. Can I ask a company to delete my data?")
    submitted = st.form_submit_button("Ask")


# ---------------------------------------------------------------------------
# Show the results
# ---------------------------------------------------------------------------

def show_sources(chunks, cited=()):
    """Show each retrieved chunk in a box you can open to read the full text."""
    st.subheader("Sources found in the Act")
    for c in chunks:
        star = " ⭐ cited" if c["section_number"] in cited else ""
        label = (f"Section {c['section_number']}: {c['section_title']}"
                 f"  (match {c['score']:.2f}){star}")
        with st.expander(label):
            if c["part"]:
                st.caption(c["part"])
            # The first line of each chunk is its label, which we already
            # show in the box title, so we skip it here.
            st.write(c["text"].split("\n", 1)[-1])


if submitted and question.strip():
    if HAS_KEY:
        with st.spinner("Searching the Act and asking Claude..."):
            try:
                result = answer(question, top_k=top_k)
            except anthropic.APIError as e:
                # e.g. wrong key or no credit left: say so, but still show sources.
                st.error(f"Couldn't get an answer from Claude: {e}")
                result = None

        if result:
            st.subheader("Answer")
            if result["said_idk"]:
                st.info(result["answer"])
            else:
                st.markdown(result["answer"])
                if result["cited_sections"]:
                    st.caption("Cited sections: " + ", ".join(result["cited_sections"]))
            if result["bad_citations"]:
                st.warning("The answer cites sections that were not in the search "
                           f"results: {', '.join(result['bad_citations'])}. "
                           "Check them carefully.")
            show_sources(result["retrieved"], cited=result["cited_sections"])
        else:
            show_sources(retrieve(question, top_k=top_k))
    else:
        with st.spinner("Searching the Act..."):
            chunks = retrieve(question, top_k=top_k)
        st.info("Answers are off (no API key), so here are the parts of the Act "
                "that best match your question.")
        show_sources(chunks)
elif submitted:
    st.warning("Please type a question first.")
