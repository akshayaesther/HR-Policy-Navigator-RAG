"""Text cleaning and normalization utilities for document ingestion.

Handles de-hyphenation, header/footer/page-number removal,
whitespace normalization, and text extraction quality scoring.
"""

import re
from typing import List, Set


# Common legitimate hyphenated compound prefixes/words that should NOT be merged
LEGITIMATE_COMPOUNDS = {
    "full-time",
    "part-time",
    "clean-desk",
    "self-service",
    "self-certified",
    "one-on-one",
    "one-time",
    "pre-approval",
    "pre-planned",
    "cross-document",
    "health-care",
    "work-life",
    "check-ins",
    "non-essential",
    "non-profit",
    "non-domestic",
    "in-office",
    "twenty-four",
    "forty-five",
    "sixty-five",
    "long-term",
    "short-term",
    "out-of-pocket",
    "out-of-state",
    "all-in",
    "post-departure",
    "semi-monthly",
    "co-pay",
    "open-toed",
    "vapor-free",
}

# Regex to detect standard page number patterns
PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*\[?\s*Page\s*\d+\s*\]?\s*$", re.IGNORECASE),
]

# Static recurring enterprise headers/footers
STATIC_RECURRING_PHRASES = [
    "Enterprise HR Global Policy Management System",
    "CONFIDENTIAL & PROPRIETARY",
    "Subject to corporate revision and jurisdictional regulations",
    "OFFICIAL CORPORATE HR MEMORANDUM",
    "CONFIDENTIAL & ARCHIVED",
]


def fix_hyphenated_line_breaks(text: str) -> str:
    """Rejoin words split across line breaks by hyphens and repair artificial syllable breaks.

    Example:
        'employ-\\n  ment' -> 'employment'
        'or-gani-\\n  zation' -> 'organization'
    """
    if not text:
        return ""

    # Strip soft hyphens
    text = text.replace("\xad", "")

    # 1. Standard newline hyphenation: word- \n word
    def _unbreak_newline_hyphen(match: re.Match) -> str:
        part1, part2 = match.group(1), match.group(2)
        candidate = f"{part1}{part2}".lower()
        hyphenated = f"{part1}-{part2}".lower()
        if hyphenated in LEGITIMATE_COMPOUNDS:
            return f"{part1}-{part2}"
        return f"{part1}{part2}"

    text = re.sub(
        r"(\b[A-Za-z]+)-\s*\n\s*([A-Za-z]+\b)",
        _unbreak_newline_hyphen,
        text,
    )

    # 2. Fix artificial inline syllable hyphens (e.g. em-ployees, de-part-ments, reg-ulations)
    def _unbreak_inline_hyphen(match: re.Match) -> str:
        word = match.group(0)
        # Never merge uppercase codes, acronyms, or tokens containing numbers
        if word.isupper() or any(c.isdigit() for c in word):
            return word
        lower_word = word.lower()
        if lower_word in LEGITIMATE_COMPOUNDS:
            return word
        parts = word.split("-")
        # If parts represent broken lowercase syllables (e.g. hand-book, em-ploy-ees)
        if len(parts) >= 2 and all(len(p) >= 2 for p in parts):
            return "".join(parts)
        return word

    text = re.sub(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b", _unbreak_inline_hyphen, text)

    return text


def normalize_whitespace(text: str) -> str:
    """Normalize tabs, non-breaking spaces, consecutive spaces, and excessive linebreaks."""
    if not text:
        return ""

    # Normalize CRLF and non-breaking spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\u200b", "")

    # Strip trailing whitespace on each line and collapse tabs/spaces
    lines: List[str] = []
    for line in text.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)

    text = "\n".join(lines)

    # Collapse 3 or more consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_page_number_line(line: str) -> bool:
    """Return True if line matches common page number patterns."""
    line_clean = line.strip()
    return any(p.match(line_clean) for p in PAGE_NUMBER_PATTERNS)


def detect_and_remove_headers_footers(
    pages_text: List[str],
    repetition_threshold: float = 0.5,
) -> List[str]:
    """Detect and strip running headers, footers, and page numbers across pages of a document."""
    if not pages_text:
        return []

    num_pages = len(pages_text)
    if num_pages == 1:
        # Single page: only strip explicit page numbers and static phrases
        cleaned_single: List[str] = []
        for line in pages_text[0].split("\n"):
            line_str = line.strip()
            if is_page_number_line(line_str) or any(phrase in line_str for phrase in STATIC_RECURRING_PHRASES):
                continue
            cleaned_single.append(line)
        return ["\n".join(cleaned_single)]

    # Collect candidate header (top 3) and footer (bottom 3) lines per page
    candidate_lines: List[str] = []
    page_split_lines: List[List[str]] = []

    for page_text in pages_text:
        lines = [l.strip() for l in page_text.split("\n") if l.strip()]
        page_split_lines.append(lines)
        if not lines:
            continue
        top_candidates = lines[:3]
        bottom_candidates = lines[-3:] if len(lines) >= 3 else lines
        candidate_lines.extend(top_candidates + bottom_candidates)

    # Count occurrences across pages
    line_page_counts: dict[str, int] = {}
    for line in set(candidate_lines):
        occurrences = 0
        for lines in page_split_lines:
            if line in lines:
                occurrences += 1
        line_page_counts[line] = occurrences

    # Lines repeating across >= repetition_threshold * num_pages (min 2) are marked
    min_repeats = max(2, int(repetition_threshold * num_pages))
    repeated_lines_to_remove: Set[str] = {
        line for line, count in line_page_counts.items() if count >= min_repeats
    }

    # Clean each page
    cleaned_pages: List[str] = []
    for page_text in pages_text:
        lines = page_text.split("\n")
        retained_lines: List[str] = []
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                retained_lines.append("")
                continue

            # Check if page number line
            if is_page_number_line(line_clean):
                continue

            # Check if static recurring phrase
            if any(phrase in line_clean for phrase in STATIC_RECURRING_PHRASES):
                continue

            # Check if repeated line across threshold
            if line_clean in repeated_lines_to_remove:
                continue

            retained_lines.append(line)

        cleaned_page = "\n".join(retained_lines)
        cleaned_pages.append(normalize_whitespace(cleaned_page))

    return cleaned_pages


def compute_quality_score(text: str, extraction_method: str = "text") -> float:
    """Compute text extraction quality score between 0.0 and 1.0.

    Evaluates text density, printable character ratio, word validity,
    and extraction pipeline reliability.
    """
    if not text or not text.strip():
        return 0.0

    clean = text.strip()
    total_chars = len(clean)

    # Printable character ratio
    printable_chars = sum(1 for c in clean if c.isprintable() and not c.isspace())
    if total_chars == 0 or printable_chars == 0:
        return 0.0

    printable_ratio = printable_chars / total_chars

    # Alphanumeric ratio
    alnum_chars = sum(1 for c in clean if c.isalnum())
    alnum_ratio = alnum_chars / total_chars

    # Average word length sanity (realistic English words avg 4-8 chars)
    words = [w for w in clean.split() if w.isalnum()]
    word_count = len(words)
    if word_count == 0:
        return 0.1

    avg_word_len = sum(len(w) for w in words) / word_count
    word_len_penalty = 1.0
    if avg_word_len < 2.5 or avg_word_len > 15.0:
        word_len_penalty = 0.7

    # Base score
    score = (0.5 * printable_ratio) + (0.3 * min(1.0, alnum_ratio * 1.4)) + (0.2 * word_len_penalty)

    # Length consideration: very short pages (< 40 chars) receive a small penalty
    if total_chars < 50:
        score *= 0.8

    # OCR method adjustment (OCR text may have minor artifacts)
    if extraction_method == "ocr":
        score = min(0.95, score * 0.98)

    return round(max(0.0, min(1.0, score)), 2)
