import re


_LOCAL_PART_RE = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+\Z")
_DOMAIN_LABEL_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")


def is_valid_email(s):
    """Return True if s is a valid email address, else False."""
    if not isinstance(s, str) or not 3 <= len(s) <= 254 or s.count("@") != 1:
        return False

    local, domain = s.rsplit("@", 1)
    if not local or len(local) > 64 or local.startswith(".") or local.endswith("."):
        return False
    if ".." in local or not _LOCAL_PART_RE.fullmatch(local):
        return False
    if not domain or len(domain) > 253 or domain.endswith("."):
        return False

    return all(_DOMAIN_LABEL_RE.fullmatch(label) for label in domain.split("."))
