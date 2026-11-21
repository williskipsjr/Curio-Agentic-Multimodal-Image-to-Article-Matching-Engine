"""Tests for the data download and loading scripts."""


from curio.scripts.download_articles import (
    MATCH_ARTICLES,
    NO_MATCH_ARTICLES,
)
from curio.scripts.download_images import CATEGORIES, HOLDBACK_COUNT, TOTAL_TARGET


class TestImageDownloadConfig:
    """Test image download script configuration."""

    def test_total_target_above_3000(self) -> None:
        """We should target at least 3,000 corpus images + 200 holdback."""
        assert TOTAL_TARGET >= 3000

    def test_holdback_count(self) -> None:
        assert HOLDBACK_COUNT == 200

    def test_categories_have_confusable_groups(self) -> None:
        """Ensure we have confusable groups for evaluation."""
        groups = {
            info["confusable_group"]
            for info in CATEGORIES.values()
            if info.get("confusable_group")
        }
        assert "canid" in groups, "Missing canid confusable group"
        assert "big_cat" in groups, "Missing big cat confusable group"
        assert "corvid" in groups, "Missing corvid confusable group"
        assert "crocodilian" in groups, "Missing crocodilian confusable group"

    def test_at_least_20_categories(self) -> None:
        assert len(CATEGORIES) >= 20


class TestArticleDownloadConfig:
    """Test article download script configuration."""

    def test_at_least_200_match_articles(self) -> None:
        assert len(MATCH_ARTICLES) >= 200

    def test_at_least_30_no_match_articles(self) -> None:
        assert len(NO_MATCH_ARTICLES) >= 30

    def test_no_match_articles_flagged(self) -> None:
        """Every no-match article should have category 'no_match'."""
        for article in NO_MATCH_ARTICLES:
            assert article["category"] == "no_match"

    def test_match_articles_have_categories(self) -> None:
        """Every matching article should have a non-empty category."""
        for article in MATCH_ARTICLES:
            assert article["category"] != ""
            assert article["category"] != "no_match"

    def test_articles_have_confusable_wildlife(self) -> None:
        """Check that articles include confusable wildlife subjects."""
        titles = {a["title"].lower() for a in MATCH_ARTICLES}
        assert any("fox" in t for t in titles), "Missing fox article"
        assert any("wolf" in t for t in titles), "Missing wolf article"
        assert any("leopard" in t for t in titles), "Missing leopard article"
        assert any("cheetah" in t for t in titles), "Missing cheetah article"
