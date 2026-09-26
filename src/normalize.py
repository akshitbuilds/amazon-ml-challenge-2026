import re
import unicodedata


def normalize_text(value) -> str:

    if value is None:
        return ""

    value = str(value)

    value = unicodedata.normalize("NFKC", value)

    value = value.lower().strip()

    value = re.sub(r"[^\w\s]", " ", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_name(value) -> str:
    return normalize_text(value)


def normalize_address(value) -> str:
    return normalize_text(value)