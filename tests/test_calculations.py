import pytest

from market_research.calculations import calculate_share_cost


def test_calculate_share_cost():
    assert calculate_share_cost(185.50, 15) == 2782.50


def test_zero_shares_are_rejected():
    with pytest.raises(ValueError):
        calculate_share_cost(185.50, 0)


def test_negative_shares_are_rejected():
    with pytest.raises(ValueError):
        calculate_share_cost(185.50, -5)


def test_missing_or_invalid_price_is_rejected():
    with pytest.raises(ValueError):
        calculate_share_cost(0, 15)