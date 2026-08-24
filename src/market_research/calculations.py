from decimal import Decimal


def calculate_share_cost(price: float, shares: int) -> Decimal:
    """Calculate the estimated cost for a number of shares."""
    if price <= 0:
        raise ValueError("Price must be greater than zero.")

    if shares <= 0:
        raise ValueError("Shares must be greater than zero.")

    return Decimal(str(price)) * Decimal(str(shares))