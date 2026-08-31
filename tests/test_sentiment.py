"""Tests for news sentiment.

The contamination boundary carries the weight. A scorer that could already know
how a story resolved must not be allowed near a historical population, and the
refusal has to live in the code rather than in whoever is holding the keyboard.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tradezbotz.research.sentiment import (
    LLM_FORWARD_ONLY_DAYS,
    Article,
    LlmScorer,
    Score,
    SentimentError,
    historical_safe,
    score_all,
)


def article(headline="Company beats earnings", summary="Revenue up 20%",
            age_days=0.0, symbols=("AAA",)):
    return Article(
        id=1, headline=headline, summary=summary, content="",
        symbols=symbols,
        created_at=datetime.now(UTC) - timedelta(days=age_days),
    )


class FakeScorer:
    def __init__(self, name):
        self.name = name

    def score(self, a):
        return Score(a.id, "positive", 0.9, self.name, a.created_at, a.symbols)


# --- the boundary ------------------------------------------------------------------

def test_an_llm_refuses_historical_text():
    """The feature this module exists for. An LLM has read what happened next."""
    scorer = LlmScorer(client=object())

    with pytest.raises(SentimentError, match="refusing to score"):
        scorer.score(article(age_days=400))


def test_the_refusal_explains_why_it_cannot_be_audited_away():
    scorer = LlmScorer(client=object())

    with pytest.raises(SentimentError) as exc:
        scorer.score(article(age_days=90))

    assert "weights" in str(exc.value)
    assert "FinBertScorer" in str(exc.value), "points at the safe alternative"


def test_an_llm_accepts_fresh_text():
    """Live scoring is legitimate: there is no future to leak from an article
    published minutes ago."""
    scorer = LlmScorer(client=None)

    # Passes the age gate, then fails on the missing client -- which proves the
    # age check let it through rather than short-circuiting.
    with pytest.raises(SentimentError, match="needs an Anthropic client"):
        scorer.score(article(age_days=0))


def test_the_age_limit_is_the_boundary_not_a_suggestion():
    scorer = LlmScorer(client=object(), max_age_days=1.0)

    scorer_ok = article(age_days=0.5)
    assert scorer_ok.age_days < 1.0

    with pytest.raises(SentimentError):
        scorer.score(article(age_days=1.5))


def test_finbert_and_vader_are_historical_safe():
    assert historical_safe(FakeScorer("finbert")) is True
    assert historical_safe(FakeScorer("vader")) is True


def test_an_llm_is_not_historical_safe():
    assert historical_safe(FakeScorer("llm")) is False


def test_an_unknown_scorer_is_not_assumed_safe():
    """Default deny. A new scorer has to be reviewed and added deliberately."""
    assert historical_safe(FakeScorer("something-new")) is False


def test_score_all_refuses_an_unsafe_scorer_on_history():
    with pytest.raises(SentimentError, match="may not score a historical"):
        score_all([article()], FakeScorer("llm"), allow_historical=True)


def test_score_all_permits_an_unsafe_scorer_when_told_it_is_live():
    out = score_all([article()], FakeScorer("llm"), allow_historical=False)

    assert len(out) == 1


def test_score_all_accepts_a_safe_scorer():
    out = score_all([article()], FakeScorer("finbert"))

    assert out[0].scorer == "finbert"


# --- article shape -------------------------------------------------------------------

def test_text_is_headline_plus_summary_not_the_body():
    """Every measured accuracy figure for these models is on sentence-level
    input, and the body is HTML that is absent for a fifth to a third of items."""
    a = Article(id=1, headline="Beat", summary="Strongly", content="<p>" + "x" * 9000,
                symbols=(), created_at=datetime.now(UTC))

    assert a.text == "Beat. Strongly"
    assert "x" not in a.text


def test_age_days_is_measured_from_publication():
    assert article(age_days=10).age_days == pytest.approx(10, abs=0.01)


# --- the stored event ------------------------------------------------------------------

def test_the_score_records_which_scorer_produced_it():
    """So a backtest can filter to one scorer. Mixing an LLM score into a
    historical study is the failure the whole module is shaped around."""
    ev = Score(1, "positive", 0.8, "finbert",
               datetime.now(UTC), ("AAA",)).to_event()

    assert ev.payload["scorer"] == "finbert"
    assert ev.kind == "news_sentiment"


def test_observed_at_is_publication_time():
    """Point-in-time by construction: an article cannot be read before it
    exists, so its publication time IS when the information became knowable."""
    published = datetime(2025, 3, 4, 14, 30, tzinfo=UTC)
    ev = Score(1, "positive", 0.8, "finbert", published).to_event()

    assert ev.observed_at == published
    assert ev.occurred_at <= ev.observed_at


def test_polarity_is_signed():
    pos = Score(1, "positive", 0.9, "finbert", datetime.now(UTC))
    neg = Score(2, "negative", -0.9, "finbert", datetime.now(UTC))

    assert pos.polarity > 0 > neg.polarity


def test_external_id_namespaces_by_scorer():
    """The same article scored twice must not collide in the event store."""
    a = Score(7, "positive", 0.5, "finbert", datetime.now(UTC)).to_event()
    b = Score(7, "negative", -0.5, "llm", datetime.now(UTC)).to_event()

    assert a.external_id != b.external_id
