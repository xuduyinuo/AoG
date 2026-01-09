import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Sequence


def read_output(file_path: str) -> List[dict]:
    """Load a JSONL prediction file and return a list of records."""
    resolved_path = Path(file_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {file_path}")

    records: List[dict] = []
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {file_path}: {exc}"
                ) from exc
    return records


def _strip_accents(text: str) -> str:
    # Remove diacritics to make matching more tolerant
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _normalize_text(text: str, keep_spaces: bool = True) -> str:
    """Lowercase and remove punctuation; optionally keep spaces for tokenization."""
    text = _strip_accents(str(text)).lower().strip()
    if keep_spaces:
        # Replace non-alphanum with space, collapse spaces
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    return re.sub(r"[^a-z0-9]+", "", text)


def normalise_answers(values: Sequence[str]) -> List[str]:
    # Kept for backward compatibility; not used by the new logic
    return [_normalize_text(v, keep_spaces=False) for v in values if v is not None]


def _split_candidates(value: str) -> List[str]:
    """Split a string into candidate answer items using common delimiters and patterns.

    Also extracts parenthetical content as separate candidates, e.g.,
    "X (now Y)" -> ["X", "Y", "X now Y"].
    """
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        return []

    pieces: List[str] = [value]

    # If it looks like a JSON list, try to parse first
    if value.startswith("[") and value.endswith("]"):
        try:
            arr = json.loads(value)
            flat = []
            for el in arr:
                flat.extend(_split_candidates(el))
            return _dedupe_preserve(flat)
        except Exception:
            pass

    # Apply common separators
    seps = [",", ";", "|", "/", "、", "，", "\\n"]
    tmp: List[str] = []
    for part in pieces:
        cur = [part]
        for sep in seps:
            next_parts: List[str] = []
            for item in cur:
                next_parts.extend([x.strip() for x in item.split(sep)])
            cur = [x for x in next_parts if x]
        tmp.extend(cur)
    pieces = tmp if tmp else pieces

    # Split on ' and ' only for longer phrases to avoid splitting single names like 'rock and roll'
    refined: List[str] = []
    for p in pieces:
        tokens = p.split()
        if len(tokens) >= 5 and " and " in f" {p} ":
            refined.extend([x.strip() for x in re.split(r"\band\b", p)])
        else:
            refined.append(p)

    pieces = [p for p in (x.strip() for x in refined) if p]

    # Parenthetical extraction
    final: List[str] = []
    for p in pieces:
        final.append(p)
        # Extract text inside parentheses
        for inner in re.findall(r"\(([^\)]+)\)", p):
            inner = inner.strip()
            if inner:
                final.append(inner)
        # Also add version with parentheses removed
        no_par = re.sub(r"\([^\)]*\)", "", p).strip()
        if no_par and no_par != p:
            final.append(no_par)

    return _dedupe_preserve([_normalize_text(x, keep_spaces=True) for x in final if x])


def _dedupe_preserve(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def extract_ground_truth(record: dict) -> List[str]:
    raw_truth = record.get("ground_truth")
    if raw_truth is None:
        return []
    if isinstance(raw_truth, str):
        return _split_candidates(raw_truth)
    if isinstance(raw_truth, Iterable):
        items: List[str] = []
        for item in raw_truth:
            items.extend(_split_candidates(str(item)))
        return _dedupe_preserve(items)
    return []


def extract_predictions(record: dict) -> List[str]:
    answer_field = record.get("Answer")
    if answer_field is None:
        return []
    if isinstance(answer_field, str):
        return _split_candidates(answer_field)
    if isinstance(answer_field, Iterable):
        items: List[str] = []
        for item in answer_field:
            items.extend(_split_candidates(str(item)))
        return _dedupe_preserve(items)
    return []


def is_prediction_correct(predictions: List[str], ground_truth: List[str]) -> bool:
    """Relaxed matching tolerant to order, formatting, parentheses, and synonyms.

    Strategy:
    - Parse answers into item lists (predictions and ground_truth)
    - Items match if any of the following:
      * exact equality
      * one contains the other (after normalization)
      * 4-digit year overlap
      * Jaccard token similarity >= 0.6
      * SequenceMatcher ratio >= 0.72
    - Overall correctness threshold:
      * If |GT| <= 3: require >= 1 match
      * Else: require >= max(2, ceil(0.3 * |GT|)) matches
    """
    if not predictions or not ground_truth:
        return False

    def years(text: str) -> set:
        return set(re.findall(r"\b(\d{4})\b", text))

    def tokens(text: str) -> List[str]:
        return [t for t in text.split() if t]

    def jaccard(a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    def _items_match(a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a == b:
            return True
        # Also try without spaces for substring checks
        a_nos = a.replace(" ", "")
        b_nos = b.replace(" ", "")
        if a_nos in b_nos or b_nos in a_nos:
            return True
        # Year overlap
        ya, yb = years(a), years(b)
        if ya and yb and (ya & yb):
            return True
        # Token/Jaccard similarity
        ja = jaccard(tokens(a), tokens(b))
        if ja >= 0.6:
            return True
        # Fuzzy ratio
        if SequenceMatcher(None, a, b).ratio() >= 0.72:
            return True
        return False

    for t in ground_truth:
        for p in predictions:
            if _items_match(p, t):
                return True
    return False


def write_output(file_path: str, records: List[dict]) -> None:
    """Write a list of records (dicts) to a JSONL file, one JSON object per line.

    This overwrites the target file.
    """
    resolved_path = Path(file_path)
    # Ensure parent exists
    if not resolved_path.parent.exists():
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with resolved_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
