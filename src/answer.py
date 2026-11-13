"""
answer.py — Step 7 of the pipeline: answer a question with Claude.

How it works:
  1. retrieve() finds the chunks of the Act that best match the question
  2. We put those chunks and the question into a prompt
  3. Claude answers using ONLY those chunks, cites the sections it used,
     and says "I don't know" if the chunks don't contain the answer
  4. We return the answer plus the list of sections it cited

Why does Claude decide "I don't know", and not the search scores?
On Day 2 we saw that off-topic questions and real questions can get similar
scores, so a score cut-off can't tell them apart. Claude can read the chunks
and judge whether they actually answer the question.

Run it from the project root:
    python src/answer.py                          # type questions one by one
    python src/answer.py "Can I delete my data?"  # ask a single question
    python src/answer.py --dry-run "..."          # show the prompt, don't call Claude
"""

import re
import sys

import anthropic
from dotenv import load_dotenv

from retrieve import retrieve

# Read ANTHROPIC_API_KEY from the .env file into the environment.
# The anthropic library then picks it up automatically, so the key
# never appears in our code.
load_dotenv()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

MODEL = "claude-opus-5-5"

# How many chunks to give Claude. More chunks = more context to answer from.
# (Retrieval is still MEASURED on the top 3 in evaluate.py.)
TOP_K = 5

# The exact words Claude must start with when it can't answer.
# evaluate.py and app.py check for this phrase, so keep it in one place.
IDK_PHRASE = "I don't know"

# The "system prompt" holds the rules Claude must follow for every question.
SYSTEM_PROMPT = f"""You answer questions about the Kenya Data Protection Act, 2019.

You are given excerpts from the Act inside <excerpt> tags, each labelled with its section number and title. Follow these rules:

1. Answer ONLY from the excerpts. Do not use outside knowledge about the Act or any other law, even if you know it.
2. After every statement, cite the section it came from in brackets, like (Section 26) or (Section 2). Only cite sections that appear in the excerpts.
3. If the excerpts do not contain the answer, reply with exactly: "{IDK_PHRASE} — the provided sections of the Data Protection Act don't cover this." Do not guess.
4. If the excerpts answer only part of the question, answer that part with citations and say clearly which part is not covered.
5. Keep answers short and plain: a few sentences or a short list. Quote the Act's exact words when precision matters (for example amounts, time limits, or definitions).
6. You give information about the Act, not legal advice."""


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def build_prompt(question, chunks):
    """Put the retrieved chunks and the question into one message for Claude."""
    excerpts = []
    for c in chunks:
        # Wrapping each chunk in tags with its section number makes it easy
        # for Claude to see where one excerpt ends and which section it is.
        excerpts.append(
            f'<excerpt section="{c["section_number"]}" title="{c["section_title"]}">\n'
            f'{c["text"]}\n'
            f"</excerpt>"
        )
    return "\n\n".join(excerpts) + f"\n\nQuestion: {question}"


def find_cited_sections(answer_text):
    """
    Pull the section numbers out of citations like "(Section 26)" or
    "(Sections 26 and 40)". Returns a sorted list without duplicates.
    """
    cited = set()
    # Find every "Section ..." / "Sections ..." mention, then take the numbers in it.
    for match in re.finditer(r"Sections?\s+([\d\w,\s]+?)(?:\)|$|\.)", answer_text):
        cited.update(re.findall(r"\b\d+[A-Z]?\b", match.group(1)))
    return sorted(cited, key=lambda s: int(re.match(r"\d+", s).group()))


def answer(question, top_k=TOP_K):
    """
    Answer a question about the Act.

    Returns a dictionary:
      answer           - Claude's answer text
      cited_sections   - section numbers Claude cited, e.g. ["26", "40"]
      said_idk         - True if Claude said "I don't know"
      retrieved        - the chunks we gave Claude (for checking and the app)
      bad_citations    - cited sections that were NOT in the chunks we gave
                         (should always be empty; if not, Claude went off-script)
    """
    chunks = retrieve(question, top_k=top_k)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        # "effort" controls how hard Claude thinks before answering.
        # "medium" is plenty for reading a few excerpts and answering.
        output_config={"effort": "medium"},
        # Safety net: if this model ever declines a request, the API retries
        # it on a suitable fallback model automatically.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        answer_text = "Sorry, this question couldn't be answered."
    else:
        # The response can contain several "blocks" (e.g. thinking + text).
        # We only want the text the user should see.
        answer_text = "".join(b.text for b in response.content if b.type == "text").strip()

    cited = find_cited_sections(answer_text)
    retrieved_sections = {c["section_number"] for c in chunks}

    return {
        "answer": answer_text,
        "cited_sections": cited,
        "said_idk": answer_text.startswith(IDK_PHRASE),
        "retrieved": chunks,
        "bad_citations": [s for s in cited if s not in retrieved_sections],
    }


def print_answer(question, result):
    """Print an answer in a readable way."""
    print(f"\nQ: {question}")
    print(f"\nA: {result['answer']}")
    print(f"\n   Cited sections:     {result['cited_sections'] or '-'}")
    print(f"   Retrieved sections: {[c['section_number'] for c in result['retrieved']]}")
    if result["bad_citations"]:
        print(f"   ⚠ Cited sections that were not retrieved: {result['bad_citations']}")


def main():
    args = sys.argv[1:]

    # --dry-run: show exactly what would be sent to Claude, without calling it.
    # Handy for checking the prompt, and works without an API key.
    if args and args[0] == "--dry-run":
        question = " ".join(args[1:]) or "Can I ask a company to delete my data?"
        print("=== SYSTEM PROMPT ===\n" + SYSTEM_PROMPT)
        print("\n=== USER MESSAGE ===\n" + build_prompt(question, retrieve(question, TOP_K)))
        return

    if args:
        question = " ".join(args)
        print_answer(question, answer(question))
        return

    print("Ask a question about the Kenya Data Protection Act "
          "(press Enter on an empty line to quit).")
    while True:
        question = input("\n> ").strip()
        if not question:
            break
        print_answer(question, answer(question))


if __name__ == "__main__":
    main()
