# %% NATURAL LANGUAGE PROCESSING - FINANCE
# Hugging Face -> FinBERT
import re
import pandas as pd
import feedparser
import yfinance as yf
from transformers import pipeline


class YahooFinBERTSentiment:
    """
    Yahoo Finance RSS sentiment analysis using FinBERT.
    Returns Pandas DataFrames only.

    Keyword automation:
      - fetch company name via yfinance
      - derive a keyword (default: first meaningful word)
    """

    _STOPWORDS = {
        "the", "and", "&",
        "inc", "inc.", "corp", "corp.", "co", "co.", "company",
        "ltd", "ltd.", "plc", "sa", "ag", "nv", "group", "holdings",
        "class", "ordinary", "shares"
    }

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        framework: str = "pt",
        neutral_threshold: float = 0.15,
    ):
        self.pipe = pipeline("text-classification", model=model_name, framework=framework)
        self.threshold = float(neutral_threshold)

    @staticmethod
    def _rss_url(ticker: str) -> str:
        return f"https://finance.yahoo.com/rss/headline?s={ticker}"

    @staticmethod
    def _clean_text(s: str) -> str:
        # keep letters/numbers/spaces only, collapse whitespace
        s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _company_name(self, ticker: str) -> str:
        """
        Best-effort company name via yfinance.
        Falls back to ticker if unavailable.
        """
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        name = info.get("shortName") or info.get("longName") or ticker
        return str(name)

    def _auto_keyword(self, ticker: str) -> str:
        """
        Derive a keyword from company name by taking the first meaningful word.
        """
        name = self._clean_text(self._company_name(ticker)).lower()
        parts = [p for p in name.split(" ") if p and p not in self._STOPWORDS]
        # if everything got filtered out, fallback to first token or ticker
        return (parts[0] if parts else (name.split(" ")[0] if name else ticker.lower()))

    def yf(
        self,
        ticker: str,
        keyword: str | None = None,
        max_articles: int | None = None,
    ) -> pd.DataFrame:
        """
        Sentiment for one ticker.

        If keyword is None, it is auto-derived from the company name.
        """
        ticker = ticker.upper().strip()
        keyword = (keyword or self._auto_keyword(ticker)).lower().strip()

        feed = feedparser.parse(self._rss_url(ticker))
        entries = feed.entries or []

        total_score = 0.0
        used = 0

        for entry in entries:
            summary = getattr(entry, "summary", "") or ""
            if keyword and keyword not in summary.lower():
                continue

            sentiment = self.pipe(summary)[0]
            label = sentiment["label"].lower()
            score = float(sentiment["score"])

            if label == "positive":
                total_score += score
                used += 1
            elif label == "negative":
                total_score -= score
                used += 1

            if max_articles and used >= max_articles:
                break

        final_score = total_score / used if used > 0 else 0.0

        if final_score >= self.threshold:
            final_sentiment = "Positive"
        elif final_score <= -self.threshold:
            final_sentiment = "Negative"
        else:
            final_sentiment = "Neutral"

        return pd.DataFrame(
            {
                "Keyword": [keyword],
                "Sentiment": [final_sentiment],
                "Score [-1;1]": [final_score],
                "Articles Used": [used],
            },
            index=[ticker],
        ).round(3)

    def portfolio(
        self,
        tickers: list[str],
        keywords: list[str] | None = None,
        max_articles: int | None = None,
    ) -> pd.DataFrame:
        """
        Portfolio-level sentiment aggregation.

        If keywords is None, keywords are auto-derived per ticker.
        """
        dfs = []
        for i, ticker in enumerate(tickers):
            kw = keywords[i] if keywords is not None else None
            try:
                dfs.append(self.yf(ticker, keyword=kw, max_articles=max_articles))
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs)
        df.sort_values(by="Score [-1;1]", ascending=False, inplace=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
        return df.round(2)


if __name__ == "__main__":
    scorer = YahooFinBERTSentiment()
    l_portfolio = ["META", "NVDA", "BRK-B"]

    # No keywords provided → auto keywords
    df_portfolio = scorer.portfolio(l_portfolio, keywords=None, max_articles=10)
    print(df_portfolio)

# %%
