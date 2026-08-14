import os


def safe_upload_path(base_dir, filename):
    """Join an untrusted filename onto base_dir and return the path."""
    base_path = os.path.realpath(base_dir)
    upload_path = os.path.realpath(os.path.join(base_path, filename))

    try:
        contained = os.path.commonpath((base_path, upload_path)) == base_path
    except ValueError:
        contained = False

    if not contained:
        raise ValueError("filename escapes the upload directory")

    return upload_path
