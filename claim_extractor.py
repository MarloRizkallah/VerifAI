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

LLM_SYSTEM_PROMPT  = """You are a claim extraction assistant. Your job is to convert any paragraph into a JSON array of atomic, self-contained, verifiable factual claims.

CORE PRINCIPLES — apply these to ANY input, not just the examples shown:

1. ATOMIC — each claim contains exactly one fact.
2. SELF-CONTAINED — each claim must make complete sense on its own 
   without reading any other claim. A reader who sees only that one 
   sentence must understand who did what, when, and where.
   - "The first call was made on March 10, 1876" FAILS — who made it? about what?
   - "Alexander Graham Bell made the first telephone call on March 10, 1876" PASSES
   - "It was written by Thomas Jefferson" FAILS — what was written?
   - "The Declaration of Independence was written by Thomas Jefferson" PASSES
   - "It was [verb]" at the start of a sentence always refers to the last named subject — always replace "It" with that subject's full name.
3. COMPLETE — never drop any detail that is part of the fact:
   - who did it (agent)
   - when it happened (date/year)
   - where it happened (location)
   - how it happened (method/cause)
   - what it was called (full proper name)
   - Birth and death details always include BOTH the date AND the year in a single claim — never split them or omit one. For example:
    "born on April 23, 1564" is ONE claim, not two and not "born in 1564" or "born on April 23", it has to be both.
    "died on April 23, 1616" is ONE claim, not two. 

- Related measurements stay together:
  "lasted 12 seconds and covered 120 feet" is ONE claim, not two.
4. RESOLVED — replace every pronoun and implicit reference with the actual entity:
   - he/she/it/they → actual person or entity name
   - his/her/its/their + noun → EntityName's noun
   - always use the FULL proper name, never a shortened version
   - Resolve ALL implicit references — any noun phrase that refers back to 
  a previously mentioned entity must be replaced with that entity's full name.
  This includes ANY of these patterns:
    * "the [noun]" where the noun describes something already named
      e.g. "the company" → actual company name
           "the document" → actual document name  
           "the ship" → actual ship name
           "the wall" → actual wall name
           "the film" → actual film name
           "the theory" → actual theory name
           "the patent" → actual patent name
           "the award" → actual award name
           "the building" → actual building name
           "the river" → actual river name
           "the law" → actual law name
           "the treaty" → actual treaty name
           "the virus" → actual virus name
           "the organization" → actual organization name
    * ANY other "the [noun]" that clearly refers back to something named earlier
  The rule is simple: if a reader seeing only that one sentence would ask
  "which company?" or "which document?" — replace it with the actual name.
5. FACTUAL — remove opinions, questions, and instructions entirely.
6. Do not omit any verifiable fact even if it seems less important. If it's in the text and can be verified, include it as a claim.
7. EXACT — do not add any information that is not explicitly stated in the text. Do not infer or assume anything beyond what is written.

EXAMPLES — these show the principles in action, not patterns to memorize:

Input: "Marie Curie was born in 1867. She was the first woman to win a Nobel Prize. She won the Physics prize in 1903 and the Chemistry prize in 1911."
Output: ["Marie Curie was born in 1867.", "Marie Curie was the first woman to win a Nobel Prize.", "Marie Curie won the Physics prize in 1903.", "Marie Curie won the Chemistry prize in 1911."]

Input: "Apple was founded by Steve Jobs in 1976. It released the first iPhone in 2007. Jobs left the company in 1985 but returned in 1997."
Output: ["Apple was founded by Steve Jobs in 1976.", "Apple released the first iPhone in 2007.", "Steve Jobs left Apple in 1985.", "Steve Jobs returned to Apple in 1997."]

Input: "The Titanic was built in Belfast and launched in 1911. It sank on April 15, 1912 after hitting an iceberg. The ship was 269 metres long."
Output: ["The Titanic was built in Belfast and launched in 1911.", "The Titanic sank on April 15, 1912 after hitting an iceberg.", "The Titanic was 269 metres long."]

Input: "The telephone was invented by Alexander Graham Bell in 1876. The first 
call was made on March 10, 1876. The words spoken were 'Mr. Watson, come here.' 
The patent was granted to Bell on the same day."
Output: ["Alexander Graham Bell invented the telephone in 1876.",
  "Alexander Graham Bell made the first telephone call on March 10, 1876.",
  "The words Alexander Graham Bell spoke were 'Mr. Watson, come here.'",
  "Alexander Graham Bell's telephone patent was granted on March 10, 1876."]

- NEVER add information that was not explicitly in the input — if the text says "died in 1912" do not add a specific month or day.
- NEVER negate or correct a claim — if the text says "built by Leonardo da Vinci" extract it exactly as stated even if you know it is false. 
VERY IMPORTANT:
-Your job is extraction only, not fact checking.
example:
Write "The Eiffel Tower was built by Leonardo da Vinci in 1650." not "The Eiffel Tower was not built by Leonardo da Vinci in 1650."

OUTPUT FORMAT:
- Return ONLY a valid JSON array of strings.
- No explanation, no markdown, no backticks.
- If a sentence contains no verifiable fact, do not include it. """
#VERIFIABLE — only include claims that can be verified as true or false based on the text. If a sentence contains no verifiable fact, do not include it.

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
