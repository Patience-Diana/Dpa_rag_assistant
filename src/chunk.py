"""
chunk.py — Step 3 of the pipeline: split the clean text into labelled chunks.

Why chunks? Later we'll search for the pieces of the Act that best match a
question. Small pieces ("chunks") make that search more precise, and they
fit easily into the prompt we send to Claude.

What it does:
  1. Reads data/processed/dpa_clean.txt (made by ingest.py)
  2. Splits it into sections, using headings like "29. Duty to notify"
  3. Cuts long sections into overlapping chunks of about CHUNK_SIZE words
  4. Labels every chunk with its section number, section title and Part
  5. Saves all chunks to data/processed/chunks.json and prints a few samples

Run it from the project root with:
    python src/chunk.py
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

INPUT_PATH = Path("data/processed/dpa_clean.txt")
OUTPUT_PATH = Path("data/processed/chunks.json")

# How many words go in each chunk, and how many words each chunk shares
# with the one before it. The overlap makes sure a sentence that falls on a
# chunk boundary still appears whole in at least one chunk.
# (On Day 4 we'll try a different CHUNK_SIZE and compare the results.)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# --- Patterns ("regular expressions") for the headings we look for ---
# A section heading is a whole line like "29. Duty to notify".
#   ^ and $       = start and end of a line
#   (\d+[A-Z]?)   = the section number, e.g. "29" (or "29A" if ever added)
#   \.\s+         = a dot followed by a space
#   (.+)          = the title (the rest of the line)
SECTION_HEADING = re.compile(r"^(\d+[A-Z]?)\.\s+(.+)$")

# A Schedule heading, e.g. "FIRST SCHEDULE [s. 15]".
SCHEDULE_HEADING = re.compile(r"^([A-Z]+ SCHEDULE)\b.*$")

# A Part heading, e.g. "Part IV – PRINCIPLES AND OBLIGATIONS OF ...".
PART_HEADING = re.compile(r"^Part [IVXL]+ – .+$")


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def split_into_sections(text):
    """
    Walk through the text line by line and group the lines into sections.

    Returns a list of dictionaries, one per section, like:
      {"section_number": "29", "section_title": "Duty to notify",
       "part": "Part IV – ...", "text": "A data controller ..."}
    """
    sections = []
    current = None       # the section we are currently filling
    current_part = ""    # the most recent Part heading we have seen

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # A Part heading: remember it, but don't add it to any section's text.
        if PART_HEADING.match(line):
            current_part = line
            continue

        section_match = SECTION_HEADING.match(line)
        schedule_match = SCHEDULE_HEADING.match(line)

        if section_match or schedule_match:
            # A new section starts here, so save the previous one first.
            if current:
                sections.append(current)

            if section_match:
                number = section_match.group(1)
                title = section_match.group(2)
                part = current_part
            else:
                # e.g. "FIRST SCHEDULE" -> number "First Schedule".
                # Its title is on the next line, so we fill it in below.
                number = schedule_match.group(1).title()
                title = ""
                part = ""  # Schedules sit outside the numbered Parts

            current = {
                "section_number": number,
                "section_title": title,
                "part": part,
                "lines": [],
            }
        elif current is None:
            # Text before section 1 (the Act's title and long title).
            # We skip it: it only repeats the purpose stated in section 3.
            continue
        elif current["section_title"] == "":
            # The first line after a Schedule heading is its title,
            # e.g. "OATH OF OFFICE".
            current["section_title"] = line.title()
        else:
            current["lines"].append(line)

    if current:
        sections.append(current)  # don't forget the very last section

    # Join each section's lines into one block of text.
    for s in sections:
        s["text"] = " ".join(s.pop("lines"))
    return sections


def split_into_chunks(words, size, overlap):
    """
    Cut a list of words into overlapping windows.

    Example with size=5, overlap=2:
      words 0-4, then words 3-7, then words 6-10, ...
    Each new chunk starts (size - overlap) words after the previous one.
    """
    if len(words) <= size:
        return [words]  # short section: keep it as a single chunk

    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(words[start:start + size])
        if start + size >= len(words):
            break  # this chunk already reached the end
    return chunks


def make_chunks(sections, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Turn every section into one or more labelled chunks."""
    all_chunks = []
    for s in sections:
        words = s["text"].split()
        pieces = split_into_chunks(words, size, overlap)

        for i, piece in enumerate(pieces, start=1):
            # Put the section label at the top of the chunk's text, so the
            # search step and Claude both "see" which section it came from.
            label = f"Section {s['section_number']}: {s['section_title']}"
            if s["section_number"].endswith("Schedule"):
                label = f"{s['section_number']}: {s['section_title']}"

            all_chunks.append({
                "chunk_id": f"s{s['section_number']}-c{i}".replace(" ", "_"),
                "section_number": s["section_number"],
                "section_title": s["section_title"],
                "part": s["part"],
                "chunk_index": i,          # 1st, 2nd, ... chunk of this section
                "chunk_count": len(pieces),  # how many chunks this section has
                "text": label + "\n" + " ".join(piece),
            })
    return all_chunks


def main():
    text = INPUT_PATH.read_text(encoding="utf-8")

    sections = split_into_sections(text)
    chunks = make_chunks(sections)

    OUTPUT_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    # --- Print a short report so we can check the result by eye ---
    sizes = [len(c["text"].split()) for c in chunks]
    print(f"Found {len(sections)} sections -> {len(chunks)} chunks "
          f"(chunk size {CHUNK_SIZE} words, overlap {CHUNK_OVERLAP}).")
    print(f"Chunk length in words: smallest {min(sizes)}, "
          f"largest {max(sizes)}, average {sum(sizes) // len(sizes)}.")
    print(f"Saved to {OUTPUT_PATH}\n")

    # Show three sample chunks: first, one from the middle, and the last.
    for c in [chunks[0], chunks[len(chunks) // 2], chunks[-1]]:
        print(f"--- {c['chunk_id']} | {c['part'] or '(no Part)'} "
              f"| chunk {c['chunk_index']} of {c['chunk_count']} ---")
        print(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))
        print()


if __name__ == "__main__":
    main()
