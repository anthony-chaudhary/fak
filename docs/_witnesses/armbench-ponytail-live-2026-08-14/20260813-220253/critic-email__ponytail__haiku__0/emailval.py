import re


_LOCAL_PART = re.compile(r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+")
_DOMAIN_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")


def is_valid_email(s):
    """Return True if s is a valid email address, else False."""
    if not isinstance(s, str) or len(s) > 254 or s.count("@") != 1:
        return False

    local, domain = s.rsplit("@", 1)
    if (
        not local
        or len(local) > 64
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
        or _LOCAL_PART.fullmatch(local) is None
        or not domain
        or len(domain) > 253
    ):
        return False

    return all(_DOMAIN_LABEL.fullmatch(label) is not None for label in domain.split("."))
