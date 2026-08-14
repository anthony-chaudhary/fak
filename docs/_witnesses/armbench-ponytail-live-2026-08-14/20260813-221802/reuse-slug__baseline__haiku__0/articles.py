from textutils import slugify


def unique_slug(title, taken):
    """Return a URL slug for `title` not already in `taken` (a set of slugs in use). If the
    base slug is taken, append -2, -3, ... until one is free. Slugs must match how the rest
    of the project builds them."""
    base_slug = slugify(title)
    if base_slug not in taken:
        return base_slug

    suffix = 2
    while f"{base_slug}-{suffix}" in taken:
        suffix += 1
    return f"{base_slug}-{suffix}"
