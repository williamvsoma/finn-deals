import re

def parse_price(price_str):
    if not price_str:
        return None

    # Normalize whitespace (NBSP → space)
    cleaned = price_str.replace("\xa0", " ").lower()

    # Extract digits only
    digits = re.findall(r"\d+", cleaned)
    if not digits:
        return None

    return int("".join(digits))