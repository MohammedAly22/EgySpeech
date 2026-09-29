"""Transcript utilities: script detection, tags, normalization for comparison, sanity checks."""

import re
import unicodedata
from dataclasses import dataclass

ARABIC_LETTER = re.compile(r"[ء-يٱ-ۓۺ-ۿ]")
LATIN_LETTER = re.compile(r"[A-Za-z]")
DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
TAG = re.compile(r"\[\s*(/?)\s*([A-Za-z_ ]+?)\s*\]")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def arabic_ratio(text: str) -> float:
    ar = len(ARABIC_LETTER.findall(text))
    la = len(LATIN_LETTER.findall(text))
    return ar / (ar + la) if ar + la else 0.0


def latin_words(text: str) -> list[str]:
    return LATIN_WORD.findall(strip_tags(text))


def is_code_switched(text: str) -> bool:
    return bool(latin_words(text)) and arabic_ratio(text) > 0


@dataclass
class TagSet:
    events: list[str]
    spans: list[str]
    styles: list[str]

    @property
    def all(self) -> set[str]:
        return set(self.events) | set(self.spans) | set(self.styles)

    def prompt_list(self) -> str:
        items = [f"[{t}]" for t in self.styles]
        items += [f"[{t}]" for t in self.events]
        items += [f"[{t}] ... [/{t}]" for t in self.spans]
        return ", ".join(items)


def _norm_tag(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


def clean_tags(text: str, tags: TagSet) -> tuple[str, list[str]]:
    """Keep only allowed tags (canonical spelling), drop unknown ones, fix unbalanced spans.

    Returns (text_with_tags, list of tag names in order).
    """
    allowed = tags.all
    spans = set(tags.spans)
    open_spans: list[str] = []
    found: list[str] = []
    out: list[str] = []
    pos = 0
    for m in TAG.finditer(text):
        out.append(text[pos : m.start()])
        pos = m.end()
        closing, name = m.group(1) == "/", _norm_tag(m.group(2))
        if name not in allowed:
            continue
        if closing:
            if name in open_spans:
                open_spans.remove(name)
                out.append(f"[/{name}]")
            continue
        if name in spans:
            open_spans.append(name)
        found.append(name)
        out.append(f"[{name}]")
    out.append(text[pos:])
    result = "".join(out)
    for name in reversed(open_spans):  # close spans the model left open
        result += f" [/{name}]"
    return re.sub(r"\s+", " ", result).strip(), found


def strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", TAG.sub(" ", text)).strip()


def clean_llm_output(text: str) -> str:
    """Remove wrappers a chat model sometimes adds around the transcript."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    text = re.sub(r"^(transcript|transcription|النص|التفريغ)\s*[:：]\s*", "", text, flags=re.I).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'«»“”":
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text)


def normalize_for_compare(text: str) -> str:
    """Orthography-insensitive form used to compare two transcripts (CER)."""
    text = unicodedata.normalize("NFKC", strip_tags(text)).lower()
    text = DIACRITICS.sub("", text)
    text = re.sub("[أإآٱ]", "ا", text)
    text = text.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cer(ref: str, hyp: str) -> float:
    """Character error rate of hyp against ref (Levenshtein / len(ref))."""
    a, b = ref, hyp
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


def has_repetition_loop(text: str, max_repeat: int = 6) -> bool:
    """True for the typical ASR/LLM hallucination: one word or short phrase looping."""
    words = strip_tags(text).split()
    for n in (1, 2, 3):
        run = 1
        for i in range(n, len(words) - n + 1, n):
            if words[i : i + n] == words[i - n : i]:
                run += 1
                if run >= max_repeat:
                    return True
            else:
                run = 1
    return False


def letters_count(text: str) -> int:
    t = strip_tags(text)
    return len(ARABIC_LETTER.findall(t)) + len(LATIN_LETTER.findall(t))
