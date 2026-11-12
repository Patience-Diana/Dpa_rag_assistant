"""
ingest.py — Step 2 of the pipeline: turn the PDF into clean text.

What it does:
  1. Opens the Data Protection Act PDF from data/raw/
  2. Reads the text of each page (skipping the cover and table of contents)
  3. Removes clutter: the header line on every page, page numbers,
     and special "joined letter" characters (like "ﬁ")
  4. Saves the clean text to data/processed/dpa_clean.txt

Run it from the project root with:
    python src/ingest.py
"""

import re
from pathlib import Path

from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Where the input PDF is, and where to save the output text.
PDF_PATH = Path("data/raw/data_protection_act_2019.pdf")
OUTPUT_PATH = Path("data/processed/dpa_clean.txt")

# Pages 1-5 are the cover, the license notice and the table of contents.
# The Act itself starts on page 6. (Python counts from 0, so page 6 = index 5.)
FIRST_ACT_PAGE_INDEX = 5

# The header line printed at the top of every page of the Act.
PAGE_HEADER = "Data Protection Act (Cap. 411C)"

# Some PDFs store common letter pairs as ONE special character (a "ligature").
# We swap them back to normal letters, so a search for "specific" also
# finds "speciﬁc".
LIGATURES = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def read_pdf_pages(pdf_path):
    """Return a list with the raw text of every page in the PDF."""
    reader = PdfReader(pdf_path)
    return [page.extract_text() for page in reader.pages]


def clean_page(page_text):
    """Clean the text of ONE page and return it."""

    # 1. Replace the special joined letters with normal letters.
    for special, normal in LIGATURES.items():
        page_text = page_text.replace(special, normal)

    clean_lines = []
    for line in page_text.split("\n"):
        line = line.strip()  # remove spaces at the start and end of the line

        # 2. Skip the header line that repeats on every page.
        if line.startswith(PAGE_HEADER):
            continue

        # 3. Skip lines that are ONLY a number. These are page numbers.
        #    (Real section headings look like "29. Duty to notify",
        #     with a dot and a title, so they are not affected.)
        if re.fullmatch(r"\d+", line):
            continue

        clean_lines.append(line)

    return "\n".join(clean_lines)


def clean_text(pages):
    """Clean every page of the Act and join them into one text."""
    act_pages = pages[FIRST_ACT_PAGE_INDEX:]  # skip cover + table of contents
    cleaned = [clean_page(p) for p in act_pages]
    text = "\n".join(cleaned)

    # Squash runs of 3+ line breaks down to a single blank line,
    # and runs of spaces down to one space.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def main():
    print(f"Reading {PDF_PATH} ...")
    pages = read_pdf_pages(PDF_PATH)
    print(f"  Found {len(pages)} pages (using pages {FIRST_ACT_PAGE_INDEX + 1}-{len(pages)}).")

    text = clean_text(pages)

    # Make sure the output folder exists, then save the text.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(text, encoding="utf-8")

    word_count = len(text.split())
    print(f"Saved clean text to {OUTPUT_PATH} ({word_count:,} words).")
    print("\n--- First 600 characters ---")
    print(text[:600])


# This line means: only run main() when the file is run directly
# (not when another file imports a function from it).
if __name__ == "__main__":
    main()
