"""
claim_extractor.py
==================
Extracts atomic, verifiable factual claims from any input paragraph.
Two strategies are provided and can be combined:

  1. HeuristicExtractor  — fast, rule-based, no API calls needed.
                           Good for filtering and sentence-level splitting.

  2. LLMExtractor        — uses an LLM (local or API) to decompose
                           complex sentences into clean atomic claims.
                           More accurate but slower.

  3. ClaimExtractor      — combines both: heuristics filter first,
                           then LLM refines the remaining sentences.

Usage
-----
    from claim_extractor import ClaimExtractor

    extractor = ClaimExtractor(use_llm=True)  # set False for heuristics only
    claims = extractor.extract("Einstein was born in Germany in 1879. He loved physics.")
    # → ["Einstein was born in Germany.", "Einstein was born in 1879.", "Einstein loved physics."]
"""

import re
import json
import spacy
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# Shared utility
# ─────────────────────────────────────────────────────────────────

def load_spacy():
    """Load spaCy model. Falls back to blank English if model missing."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("[claim_extractor] spaCy model not found.")
        print("  Run: python -m spacy download en_core_web_sm")
        print("  Falling back to rule-based sentence splitter.\n")
        return None


# ─────────────────────────────────────────────────────────────────
# Strategy 1 — Heuristic extractor
# ─────────────────────────────────────────────────────────────────

# Patterns that strongly suggest a sentence is NOT a factual claim.
# These are things we want to DISCARD before passing to DeBERTa.
NON_CLAIM_PATTERNS = [
    # Questions
    r"^\s*(?:who|what|when|where|why|how|is|are|was|were|do|does|did|can|could|will|would|should)\b.+\?",
    # Exclamations with no factual content
    r"^\s*(?:wow|oh|hey|hmm|well|look|note|notice)\b",
    # Pure opinions / subjectivity markers
    r"\b(?:i think|i believe|i feel|in my opinion|i guess|i reckon|i suppose)\b",
    # Recommendations / imperatives
    r"^\s*(?:please|kindly|make sure|ensure|try|consider|remember|note that)\b",
    # Filler / transition phrases
    r"^\s*(?:however|moreover|furthermore|additionally|therefore|thus|hence|in conclusion|in summary|as a result|for example|for instance|such as|namely)\s*[,.]?\s*$",
    # Purely temporal/location anchors with no predicate
    r"^\s*(?:in \d{4}|back in|last year|this year|recently|soon|later|earlier|afterwards)\s*[,.]?\s*$",
    # URLs / citations
    r"https?://",
    r"\[\d+\]|\(\d{4}\)",
]

NON_CLAIM_REGEX = [re.compile(p, re.IGNORECASE) for p in NON_CLAIM_PATTERNS]


class HeuristicExtractor:
    """
    Fast claim extraction using sentence splitting + rule-based filters.

    Steps:
      1. Split paragraph into sentences (spaCy or regex fallback)
      2. Discard sentences that match known non-claim patterns
      3. Apply minimum length filter
      4. Strip leading/trailing noise tokens
    """

    MIN_WORDS  = 5    # sentences shorter than this are likely fragments
    MIN_CHARS  = 20   # characters minimum

    def __init__(self):
        self.nlp = load_spacy()

    # ── Public API ───────────────────────────────────────────────

    def extract(self, text: str) -> list[str]:
        """Return a list of candidate claim strings from `text`."""
        sentences = self._split_sentences(text)
        claims    = []

        for sent in sentences:
            sent = self._clean(sent)
            if not sent:
                continue
            if len(sent) < self.MIN_CHARS:
                continue
            if len(sent.split()) < self.MIN_WORDS:
                continue
            if self._is_non_claim(sent):
                continue
            claims.append(sent)

        return claims

    # ── Internals ────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using spaCy or regex fallback."""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        # Fallback: split on . ! ? followed by space + capital
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [p.strip() for p in parts if p.strip()]

    def _clean(self, sent: str) -> str:
        """Normalise whitespace, strip markdown, remove leading bullet chars."""
        sent = re.sub(r'\s+', ' ', sent).strip()
        sent = re.sub(r'^[-*•·▪–—]+\s*', '', sent)         # bullet points
        sent = re.sub(r'\*\*(.+?)\*\*', r'\1', sent)        # **bold**
        sent = re.sub(r'`(.+?)`', r'\1', sent)               # `code`
        sent = re.sub(r'^\d+[.)]\s*', '', sent)              # "1. " or "1) "
        return sent.strip()

    def _is_non_claim(self, sent: str) -> bool:
        """Return True if the sentence matches any non-claim pattern."""
        for pattern in NON_CLAIM_REGEX:
            if pattern.search(sent):
                return True
        return False


# ─────────────────────────────────────────────────────────────────
# Strategy 2 — LLM-based extractor
# ─────────────────────────────────────────────────────────────────

LLM_SYSTEM_PROMPT = """You are a claim extraction assistant. Your job is to break a given text into a list of atomic, self-contained, verifiable factual claims.

Rules:
- Each claim must express ONE fact only (atomic).
- Each claim must be a complete sentence that could stand alone.
- Remove all opinions, questions, rhetorical statements, and filler phrases.
- Resolve pronouns: replace "he/she/it/they" with the actual entity name where possible.
- If a sentence contains multiple facts joined by "and", split them into separate claims.
- Preserve exact numbers, dates, and proper nouns.
- Do NOT infer or add information that is not explicitly stated.
- Do NOT include meta-commentary like "The text states that..."
- Return ONLY a valid JSON array of strings. No preamble, no explanation, no markdown.

Example input:
  "Einstein, who was born in Germany in 1879, developed the theory of relativity and later moved to the United States."

Example output:
  ["Einstein was born in Germany.", "Einstein was born in 1879.", "Einstein developed the theory of relativity.", "Einstein moved to the United States."]
"""


class LLMExtractor:
    """
    LLM-based claim extraction using any OpenAI-compatible API.

    Supports:
      - Local models via Ollama (http://localhost:11434)
      - Anthropic Claude via the claude-compatible wrapper
      - OpenAI / OpenAI-compatible APIs

    Parameters
    ----------
    provider : str
        'ollama'    — local Ollama server (recommended for offline use)
        'openai'    — OpenAI API (needs OPENAI_API_KEY env var)
        'anthropic' — Anthropic API (needs ANTHROPIC_API_KEY env var)

    model : str
        Model name. Defaults:
          ollama    → 'llama3'
          openai    → 'gpt-4o-mini'
          anthropic → 'claude-haiku-4-5-20251001'

    fallback : HeuristicExtractor or None
        If the LLM call fails or returns unparseable JSON, fall back
        to this extractor. Strongly recommended.
    """

    DEFAULT_MODELS = {
        "ollama":    "llama3",
        "openai":    "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
    }

    def __init__(
        self,
        provider:  str = "ollama",
        model:     Optional[str] = None,
        api_key:   Optional[str] = None,
        base_url:  Optional[str] = None,
        fallback:  Optional[HeuristicExtractor] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.provider    = provider
        self.model       = model or self.DEFAULT_MODELS.get(provider, "llama3")
        self.api_key     = api_key
        self.base_url    = base_url
        self.fallback    = fallback or HeuristicExtractor()
        self.max_tokens  = max_tokens
        self.temperature = temperature

        self._client = self._build_client()

    # ── Public API ───────────────────────────────────────────────

    def extract(self, text: str) -> list[str]:
        """Send text to LLM and parse returned claim list."""
        try:
            raw = self._call_llm(text)
            claims = self._parse_json(raw)
            return [c.strip() for c in claims if isinstance(c, str) and c.strip()]
        except Exception as e:
            print(f"[LLMExtractor] Error: {e}. Falling back to heuristics.")
            return self.fallback.extract(text)

    # ── Client builders ──────────────────────────────────────────

    def _build_client(self):
        if self.provider == "anthropic":
            try:
                import anthropic
                key = self.api_key or __import__('os').environ.get("ANTHROPIC_API_KEY")
                return anthropic.Anthropic(api_key=key)
            except ImportError:
                raise ImportError("pip install anthropic")

        elif self.provider in ("openai", "ollama"):
            try:
                import openai
                url = self.base_url or (
                    "http://localhost:11434/v1" if self.provider == "ollama" else None
                )
                key = self.api_key or (
                    "ollama" if self.provider == "ollama"
                    else __import__('os').environ.get("OPENAI_API_KEY")
                )
                return openai.OpenAI(api_key=key, base_url=url)
            except ImportError:
                raise ImportError("pip install openai")

        raise ValueError(f"Unknown provider: {self.provider}")

    # ── LLM call ────────────────────────────────────────────────

    def _call_llm(self, text: str) -> str:
        user_message = f"Extract all atomic factual claims from the following text:\n\n{text}"

        if self.provider == "anthropic":
            import anthropic
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=LLM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text

        else:  # openai / ollama
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )
            return response.choices[0].message.content

    # ── JSON parsing ────────────────────────────────────────────

    def _parse_json(self, raw: str) -> list[str]:
        """Extract the JSON array from LLM output even if wrapped in prose."""
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON array inside the response
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try stripping markdown code fences
        cleaned = re.sub(r'```(?:json)?', '', raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Last resort: treat each non-empty line as a claim
        print("[LLMExtractor] Could not parse JSON, using line-split fallback.")
        lines = [
            l.strip().strip('"').strip("'").strip('-').strip()
            for l in raw.split('\n')
            if l.strip() and not l.strip().startswith('[') and not l.strip().startswith(']')
        ]
        return [l for l in lines if len(l) > 10]


# ─────────────────────────────────────────────────────────────────
# Strategy 3 — Combined extractor (recommended)
# ─────────────────────────────────────────────────────────────────

class ClaimExtractor:
    """
    Production-ready claim extractor.

    Pipeline:
      1. Heuristic pre-filter  — removes obvious non-claims quickly
      2. LLM decomposition     — splits compound sentences into atomics
                                 (only when use_llm=True)
      3. Deduplication         — removes near-duplicate claims

    Parameters
    ----------
    use_llm : bool
        If True, use LLM for atomic decomposition (recommended).
        If False, use heuristics only (faster, less accurate).

    llm_provider : str
        'ollama' | 'openai' | 'anthropic'

    llm_model : str or None
        Specific model name, or None for the provider default.

    api_key : str or None
        API key for cloud providers. Falls back to env vars.

    dedup_threshold : float
        Jaccard similarity above which two claims are considered
        duplicates. Default 0.85.
    """

    def __init__(
        self,
        use_llm:          bool = True,
        llm_provider:     str  = "ollama",
        llm_model:        Optional[str] = None,
        api_key:          Optional[str] = None,
        dedup_threshold:  float = 0.85,
    ):
        self.dedup_threshold = dedup_threshold
        self.heuristic = HeuristicExtractor()

        if use_llm:
            self.llm = LLMExtractor(
                provider=llm_provider,
                model=llm_model,
                api_key=api_key,
                fallback=self.heuristic,
            )
        else:
            self.llm = None

    # ── Public API ───────────────────────────────────────────────

    def extract(self, text: str, verbose: bool = False) -> list[str]:
        """
        Extract clean, atomic, factual claims from `text`.

        Parameters
        ----------
        text : str
            Any paragraph or multi-sentence input.
        verbose : bool
            Print intermediate steps for debugging.

        Returns
        -------
        list[str]
            Deduplicated list of atomic factual claims.
        """
        text = text.strip()
        if not text:
            return []

        # Step 1: heuristic pre-filter
        heuristic_candidates = self.heuristic.extract(text)
        if verbose:
            print(f"[Step 1 - Heuristic] {len(heuristic_candidates)} candidates:")
            for c in heuristic_candidates:
                print(f"  • {c}")

        if not heuristic_candidates:
            return []

        # Step 2: LLM decomposition
        if self.llm:
            # Re-join the filtered sentences and send to LLM
            # (this way the LLM only processes what passed the filter)
            filtered_text = " ".join(heuristic_candidates)
            claims = self.llm.extract(filtered_text)
            if verbose:
                print(f"\n[Step 2 - LLM] {len(claims)} atomic claims:")
                for c in claims:
                    print(f"  • {c}")
        else:
            claims = heuristic_candidates

        # Step 3: deduplication
        unique_claims = self._deduplicate(claims)
        if verbose:
            print(f"\n[Step 3 - Dedup] {len(unique_claims)} unique claims")

        return unique_claims

    def extract_batch(self, texts: list[str], verbose: bool = False) -> list[list[str]]:
        """Extract claims from a list of texts. Returns a list of lists."""
        return [self.extract(t, verbose=verbose) for t in texts]

    # ── Deduplication ────────────────────────────────────────────

    def _jaccard(self, a: str, b: str) -> float:
        """Jaccard similarity between two strings (word-level)."""
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _deduplicate(self, claims: list[str]) -> list[str]:
        """Remove claims that are near-duplicates of an already-kept claim."""
        unique = []
        for candidate in claims:
            is_dup = any(
                self._jaccard(candidate, kept) >= self.dedup_threshold
                for kept in unique
            )
            if not is_dup:
                unique.append(candidate)
        return unique


# ─────────────────────────────────────────────────────────────────
# Quick demo — run this file directly to test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_paragraphs = [
        # Paragraph 1: mixed facts and opinions
        """
        Albert Einstein, who was born in Ulm, Germany in 1879, developed the special
        theory of relativity in 1905. I think he was one of the greatest scientists
        ever. He also won the Nobel Prize in Physics in 1921, not for relativity but
        for his discovery of the law of the photoelectric effect. He later moved to
        the United States and worked at Princeton. What do you think about his legacy?
        """,

        # Paragraph 2: compound sentences and pronouns
        """
        The Eiffel Tower was built by Gustave Eiffel and completed in 1889. It stands
        330 metres tall and was originally meant to be a temporary structure. Paris
        has about 2 million residents, and it is the capital of France.
        """,

        # Paragraph 3: hallucinated-style text (for testing the pipeline)
        """
        Python was invented by Guido van Rossum and first released in 1991. The language
        was named after the BBC comedy series Monty Python. Python is currently the most
        popular programming language according to the TIOBE index. It was released in
        2022 and has over 500 million users worldwide.
        """,
    ]

    print("=" * 60)
    print("CLAIM EXTRACTION DEMO (heuristics only, use_llm=False)")
    print("=" * 60)

    extractor = ClaimExtractor(use_llm=False)

    for i, para in enumerate(test_paragraphs, 1):
        print(f"\n--- Paragraph {i} ---")
        print("INPUT:", para.strip()[:120], "...")
        print()
        claims = extractor.extract(para, verbose=False)
        print(f"EXTRACTED {len(claims)} CLAIMS:")
        for j, claim in enumerate(claims, 1):
            print(f"  {j}. {claim}")

    print("\n" + "=" * 60)
    print("To use LLM-based extraction (better results):")
    print("  extractor = ClaimExtractor(use_llm=True, llm_provider='ollama')")
    print("  claims = extractor.extract(your_text, verbose=True)")
    print("=" * 60)
