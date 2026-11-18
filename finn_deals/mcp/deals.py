"""Evidence summaries for AI-selected comparisons, without a learned price model."""

import math
import re
from statistics import median
from typing import Any


def review_signals(listing: dict[str, Any]) -> list[str]:
    """Surface seller phrases to review; matches do not establish actual defects."""
    signals = []
    price = listing.get("price_amount")
    if isinstance(price, (int, float)) and 0 < price <= 10:
        signals.append(
            "Very low asking price may be a placeholder; verify the actual price in the description"
        )
    if listing.get("availability") != "active":
        signals.append(
            f"Availability requires review: {listing.get('availability', 'unknown')}"
        )
    text = f"{listing.get('title') or ''}\n{listing.get('description') or ''}"
    for pattern, label in [
        (
            r"\b(defekt|ødelagt|reparasjonsobjekt|delelager|damaged|broken|untested|utestet)\b",
            "Possible defect or untested item",
        ),
        (r"\b(ønskes kjøpt|kjøpes|wanted)\b", "Possible wanted ad"),
        (r"\b(uten|mangler|missing|only|kun)\b", "Check included parts and quantity"),
        (
            r"\b(følger ikke med|ikke inkludert|kastet|skade|skader|slitt|riper|løse)\b",
            "Check condition and completeness in context",
        ),
        (
            r"\b(selges separat|pris per|pr\. stk|se beskrivelse|bud ønskes|høystbydende)\b",
            "Check actual price, bundle quantity or bidding terms",
        ),
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start, end = max(0, match.start() - 45), min(len(text), match.end() + 80)
            signals.append(f"{label}; seller text: {text[start:end]!r}")
    if not listing.get("description"):
        signals.append("No description available")
    if not listing.get("image_urls"):
        signals.append("No photos available")
    if not listing.get("shipping_payment_info"):
        signals.append("Shipping/payment costs unknown; asking price is not total cost")
    return signals


def _eligible(listing: dict[str, Any]) -> bool:
    price = listing.get("price_amount")
    return (
        isinstance(price, (int, float))
        and math.isfinite(price)
        and price > 0
        and listing.get("price_currency") == "NOK"
        and listing.get("availability") == "active"
    )


def compare_listings(listings: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a caller-selected peer set without asserting equivalence."""
    unique = {str(row["ad_id"]): row for row in listings}
    eligible = [row for row in unique.values() if _eligible(row)]
    comparisons = []
    for row in unique.values():
        peers = [peer for peer in eligible if peer["ad_id"] != row["ad_id"]]
        enough = _eligible(row) and len(peers) >= 3
        peer_median = median(peer["price_amount"] for peer in peers) if enough else None
        comparisons.append(
            {
                **row,
                "review_signals": review_signals(row),
                "price_evidence_eligible": _eligible(row),
                "peer_ad_ids": [peer["ad_id"] for peer in peers],
                "peer_count": len(peers),
                "peer_median_asking_price_nok": peer_median,
                "percent_below_peer_median": (
                    round(100 * (1 - row["price_amount"] / peer_median), 1)
                    if enough
                    else None
                ),
                "comparison_status": (
                    "unverified_comparability"
                    if enough
                    else "insufficient_price_evidence"
                ),
            }
        )
    return {
        "listings": comparisons,
        "eligible_price_count": len(eligible),
        "basis": "Current asking prices of explicitly selected ads. Each ad is excluded from its own reference median. At least 3 eligible peers required.",
        "limitations": [
            "The AI must verify product/model, variant, size, quantity, condition and accessories before using a price comparison.",
            "Multiple ads may belong to the same seller; independent sellers are not verified.",
            "Asking prices are not sold prices, fair value, a guarantee of authenticity, or evidence of profit.",
            "Shipping, payment fees, travel and repairs are not included in the price differences.",
        ],
    }
