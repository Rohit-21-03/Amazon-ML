import re
import unicodedata


LEGAL_SUFFIXES = {
    "ag",
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "llp",
    "lp",
    "ltd",
    "pc",
    "plc",
    "private",
    "pvt",
    "sa",
    "sarl",
    "sas",
}


ADDRESS_REPLACEMENTS = {
    "apartment": "apt",
    "avenue": "ave",
    "building": "bldg",
    "boulevard": "blvd",
    "drive": "dr",
    "floor": "fl",
    "highway": "hwy",
    "lane": "ln",
    "mount": "mt",
    "road": "rd",
    "saint": "st",
    "street": "st",
    "suite": "ste",
}


WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALPHANUMERIC_PATTERN = re.compile(
    r"[^\w\s]",
    flags=re.UNICODE,
)


def normalize_unicode(value):
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    return unicodedata.normalize("NFKC", text)


def normalize_common_text(value):
    text = normalize_unicode(value).casefold()

    if not text:
        return ""

    text = text.replace("&", " and ")
    text = NON_ALPHANUMERIC_PATTERN.sub(" ", text)
    text = text.replace("_", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


def normalize_business_name(value):
    return normalize_common_text(value)


def normalize_business_name_core(value):
    normalized_name = normalize_business_name(value)

    if not normalized_name:
        return ""

    tokens = normalized_name.split()

    core_tokens = [
        token
        for token in tokens
        if token not in LEGAL_SUFFIXES
    ]

    if not core_tokens:
        return normalized_name

    return " ".join(core_tokens)


def normalize_business_address(value):
    normalized_address = normalize_common_text(value)

    if not normalized_address:
        return ""

    tokens = normalized_address.split()

    normalized_tokens = [
        ADDRESS_REPLACEMENTS.get(token, token)
        for token in tokens
    ]

    return " ".join(normalized_tokens)


def normalize_country(value):
    return normalize_common_text(value)