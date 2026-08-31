"""News sentiment, with the contamination boundary enforced in code.

Three scorers, and which one you may use depends on *when* the text was
published, not on preference:

  FinBertScorer   historical and live. Maps text -> polarity, trained on the
                  Financial PhraseBank (human-labelled sentences) and Reuters
                  text. It never saw a price outcome, so it cannot know what
                  happened next.
  VaderScorer     social text, historical and live. Lexicon-based, no model.
  LlmScorer       LIVE ONLY, and it refuses historical text at runtime.

**Why the LLM is restricted and FinBERT is not.** An LLM has read the internet,
including "NVDA soared after...". Ask it to judge a 2019 headline and its answer
may be shaped by knowing what followed. That leakage lives in the weights, is
undetectable by auditing the data path, and would silently invalidate every
result computed afterwards -- there is no test that would reveal it. FinBERT's
training signal was human polarity labels, not returns.

The rule is therefore **"no model that could know the outcome"**, not "no neural
networks". `LlmScorer.score` raises on any article older than
`LLM_FORWARD_ONLY_DAYS`, so the boundary is enforced by the code rather than by
whoever is holding the keyboard.

**Why FinBERT rather than the finance dictionary everyone reaches for.**
Measured comparisons put FinBERT near 81.7% accuracy against Loughran-McDonald's
54.0% -- the dictionary is barely better than a coin flip on headlines. It was
built for 10-K filings and does not transfer to news. General-purpose lexicons
are worse still on financial text, where "liability", "cost" and "tax" are
neutral terms that every everyday lexicon scores negative.

**On speed.** The measured reaction to news is fast: roughly 0.30% to computer
traders inside five seconds, 0.23% to humans inside sixty. We cannot compete
there and should not try -- entry here is the next open. What we can use is the
slower half: fresh news moves price 39bp on day one against 23bp for stale, and
takes about four days to be fully incorporated. That drift is what the 5- and
20-day horizons are for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterator, Protocol, Sequence

import requests

from .eventstore import Event

ALPACA_NEWS_REST = "https://data.alpaca.markets/v1beta1/news"
ALPACA_NEWS_STREAM = "wss://stream.data.alpaca.markets/v1beta1/news"

SOURCE_NEWS = "alpaca_news"
KIND_SENTIMENT = "news_sentiment"

#: An LLM may only score text published within this many days of now. Anything
#: older is refused: the model may already know how it turned out.
LLM_FORWARD_ONLY_DAYS = 2

#: Content is off by default on the REST endpoint and must be asked for. Missing
#: this returns headlines and short summaries only, which is easy not to notice.
INCLUDE_CONTENT = "true"


class SentimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Article:
    id: int
    headline: str
    summary: str
    content: str
    symbols: tuple[str, ...]
    #: When the article was published. This is `observed_at`: point-in-time by
    #: construction, since an article cannot be read before it exists.
    created_at: datetime
    source: str = ""
    url: str = ""

    @property
    def text(self) -> str:
        """What a scorer reads.

        Headline and summary rather than the full body. The body is HTML, is
        absent for 20-35% of articles, and buries the claim under boilerplate --
        and every measured accuracy figure for these models is on sentence-level
        input, not on 24,000-character documents.
        """
        return f"{self.headline}. {self.summary}".strip()

    @property
    def age_days(self) -> float:
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() / 86400


@dataclass(frozen=True)
class Score:
    """One article's sentiment, and which scorer produced it."""

    article_id: int
    label: str            # positive | negative | neutral
    #: Signed confidence in [-1, 1]. Positive is bullish.
    polarity: float
    scorer: str
    observed_at: datetime
    symbols: tuple[str, ...] = ()

    def to_event(self) -> Event:
        return Event(
            source=SOURCE_NEWS,
            kind=KIND_SENTIMENT,
            external_id=f"{self.scorer}:{self.article_id}",
            observed_at=self.observed_at,
            occurred_at=self.observed_at,
            payload={
                "label": self.label,
                "polarity": self.polarity,
                # Recorded on every row so a backtest can filter to one scorer.
                # Mixing an LLM score into a historical study is the failure this
                # whole module is shaped around; the column makes it detectable.
                "scorer": self.scorer,
                "symbols": list(self.symbols),
            },
        )


class SentimentScorer(Protocol):
    name: str

    def score(self, article: Article) -> Score: ...


# --- scorers -------------------------------------------------------------------

class FinBertScorer:
    """FinBERT. Safe for historical text, which is the whole point of using it.

    Loaded lazily: torch and transformers are ~2GB and have no business being a
    hard dependency of a package whose other jobs are HTTP and SQLite.
    """

    name = "finbert"
    MODEL = "ProsusAI/finbert"

    def __init__(self, model_name: str = MODEL) -> None:
        self.model_name = model_name
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError as exc:  # pragma: no cover - environment
                raise SentimentError(
                    "FinBERT needs transformers and torch:\n"
                    "    pip install 'transformers>=4.40' torch --index-url "
                    "https://download.pytorch.org/whl/cpu\n"
                    "Deliberately optional: it is ~2GB and only the scoring path "
                    "needs it. Scores are stored once, so nothing downstream "
                    "requires the model."
                ) from exc
            self._pipe = pipeline("sentiment-analysis", model=self.model_name,
                                  truncation=True, max_length=512)
        return self._pipe

    def score(self, article: Article) -> Score:
        out = self._pipeline()(article.text)[0]
        label = str(out["label"]).lower()
        confidence = float(out["score"])
        # FinBERT emits three classes; collapse to a signed scalar so a strategy
        # can threshold it without knowing the label vocabulary.
        polarity = confidence if label == "positive" else (
            -confidence if label == "negative" else 0.0)
        return Score(article.id, label, polarity, self.name,
                     article.created_at, article.symbols)


class VaderScorer:
    """VADER, for social text. Wrong tool for newswire, right one for Reddit.

    Tuned on social media, so it reads emphasis, negation and emoji that formal
    text does not contain -- and it misreads financial vocabulary badly, which is
    why it must not be pointed at filings or news.
    """

    name = "vader"

    def __init__(self) -> None:
        self._analyser = None

    def _get(self):
        if self._analyser is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            except ImportError as exc:  # pragma: no cover - environment
                raise SentimentError(
                    "VADER needs: pip install vaderSentiment"
                ) from exc
            self._analyser = SentimentIntensityAnalyzer()
        return self._analyser

    def score(self, article: Article) -> Score:
        compound = float(self._get().polarity_scores(article.text)["compound"])
        label = ("positive" if compound >= 0.05
                 else "negative" if compound <= -0.05 else "neutral")
        return Score(article.id, label, compound, self.name,
                     article.created_at, article.symbols)


class LlmScorer:
    """An LLM scorer that refuses to score history.

    The refusal is the feature. Everything else here could be written without
    this class; what it adds is that the contamination boundary is enforced at
    runtime instead of relying on nobody making a convenient exception later.

    Live scoring is legitimate -- there is no future to leak from text published
    minutes ago -- and it can return structure a polarity classifier cannot:
    event type, whether the company is the subject or merely mentioned, whether
    the item is material or routine.
    """

    name = "llm"

    def __init__(self, client=None, model: str = "claude-haiku-4-5",
                 max_age_days: float = LLM_FORWARD_ONLY_DAYS) -> None:
        self.client = client
        self.model = model
        self.max_age_days = max_age_days

    def score(self, article: Article) -> Score:
        if article.age_days > self.max_age_days:
            raise SentimentError(
                f"refusing to score an article {article.age_days:.0f} days old "
                f"with an LLM (limit {self.max_age_days:.0f}). The model may "
                "already know how this resolved, and that leakage lives in its "
                "weights where no audit of the data path can find it. Use "
                "FinBertScorer for anything historical."
            )
        if self.client is None:
            raise SentimentError("LlmScorer needs an Anthropic client")
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=64,
            system=(
                "Classify the sentiment of a financial news item toward the "
                "company it concerns. Reply with exactly one word: positive, "
                "negative, or neutral."
            ),
            messages=[{"role": "user", "content": article.text}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip().lower()
        label = text if text in ("positive", "negative", "neutral") else "neutral"
        polarity = 1.0 if label == "positive" else -1.0 if label == "negative" else 0.0
        return Score(article.id, label, polarity, self.name,
                     article.created_at, article.symbols)


def historical_safe(scorer: SentimentScorer) -> bool:
    """Whether this scorer may be used on a backtest population."""
    return getattr(scorer, "name", "") in ("finbert", "vader")


# --- fetching --------------------------------------------------------------------

def _parse(raw: dict) -> Article:
    return Article(
        id=int(raw.get("id", 0)),
        headline=raw.get("headline") or "",
        summary=raw.get("summary") or "",
        content=raw.get("content") or "",
        symbols=tuple(raw.get("symbols") or ()),
        created_at=datetime.fromisoformat(
            (raw.get("created_at") or "").replace("Z", "+00:00")
        ).astimezone(timezone.utc),
        source=raw.get("source") or "",
        url=raw.get("url") or "",
    )


class NewsClient:
    """Alpaca news: REST for history, WebSocket for live.

    History reaches 2016, which makes sentiment the one signal here that can be
    backfilled at all -- ApeWisdom serves only a current snapshot, so Reddit
    sentiment can only ever be accumulated forward.
    """

    def __init__(self, api_key: str | None = None, api_secret: str | None = None,
                 session: requests.Session | None = None) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_PAPER_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_PAPER_API_SECRET", "")
        if not self.api_key or not self.api_secret:
            raise SentimentError(
                "Alpaca credentials missing. Set ALPACA_PAPER_API_KEY and "
                "ALPACA_PAPER_API_SECRET."
            )
        self.session = session or requests.Session()

    @property
    def _headers(self) -> dict[str, str]:
        return {"APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret}

    def history(self, start: date, end: date, symbols: Sequence[str] = (),
                limit_pages: int = 0) -> Iterator[Article]:
        """Articles published in a window, oldest page first.

        `include_content` is set explicitly. The default returns headlines and
        truncated summaries with an empty `content` field, which reads as
        working and quietly discards most of the text.
        """
        page, pages = None, 0
        while True:
            params = {
                "start": start.isoformat(), "end": end.isoformat(),
                "limit": 50, "include_content": INCLUDE_CONTENT,
            }
            if symbols:
                params["symbols"] = ",".join(s.upper() for s in symbols)
            if page:
                params["page_token"] = page
            resp = self.session.get(ALPACA_NEWS_REST, params=params,
                                    headers=self._headers, timeout=45)
            resp.raise_for_status()
            body = resp.json()
            for raw in body.get("news") or []:
                yield _parse(raw)
            page = body.get("next_page_token")
            pages += 1
            if not page or (limit_pages and pages >= limit_pages):
                return

    def stream(self, symbols: Sequence[str] = ("*",),
               timeout: float = 60.0) -> Iterator[Article]:
        """Live articles over the WebSocket.

        Verified working on the free plan: authenticated, subscribed, and the
        first article arrived three seconds later.
        """
        try:
            import json
            import websocket
        except ImportError as exc:  # pragma: no cover - environment
            raise SentimentError(
                "The live stream needs: pip install websocket-client"
            ) from exc

        ws = websocket.create_connection(ALPACA_NEWS_STREAM, timeout=timeout)
        try:
            ws.recv()  # connected
            ws.send(json.dumps({"action": "auth", "key": self.api_key,
                                "secret": self.api_secret}))
            auth = ws.recv()
            if "authenticated" not in auth:
                raise SentimentError(f"news stream refused authentication: {auth[:200]}")
            ws.send(json.dumps({"action": "subscribe",
                                "news": [s.upper() for s in symbols]}))
            ws.recv()  # subscription confirmation
            while True:
                for raw in json.loads(ws.recv()):
                    if raw.get("T") == "n":
                        yield _parse(raw)
        finally:
            ws.close()


def score_all(articles: Sequence[Article], scorer: SentimentScorer,
              *, allow_historical: bool = True) -> list[Score]:
    """Score a batch, refusing an unsafe scorer/population combination.

    The guard is here as well as inside `LlmScorer` because a caller can hold a
    scorer without knowing what it is -- and the failure mode this prevents is
    silent rather than loud.
    """
    if allow_historical and not historical_safe(scorer):
        raise SentimentError(
            f"{getattr(scorer, 'name', scorer)!r} may not score a historical "
            "population. It could already know the outcomes, and no audit of the "
            "data path would reveal it. Pass allow_historical=False only for "
            "live text."
        )
    return [scorer.score(a) for a in articles]
