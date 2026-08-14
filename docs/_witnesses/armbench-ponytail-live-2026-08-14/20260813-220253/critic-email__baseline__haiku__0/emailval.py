import re


_LOCAL_PART = re.compile(r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*\Z")
_DOMAIN_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")


def is_valid_email(s):
    """Return True if s is a valid email address, else False."""
    if not isinstance(s, str) or not 3 <= len(s) <= 254 or s.count("@") != 1:
        return False

    local, domain = s.rsplit("@", 1)
    if not local or len(local) > 64 or not _LOCAL_PART.fullmatch(local):
        return False
    if not domain or len(domain) > 253:
        return False

    labels = domain.split(".")
    return all(_DOMAIN_LABEL.fullmatch(label) for label in labels)
