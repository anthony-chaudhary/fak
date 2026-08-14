import os


def safe_upload_path(base_dir, filename):
    """Join an untrusted filename onto base_dir and return the path."""
    base_path = os.path.realpath(os.fspath(base_dir))
    filename = os.fspath(filename)
    if os.path.isabs(filename):
        raise ValueError("filename must be relative to the upload directory")

    upload_path = os.path.realpath(os.path.join(base_path, filename))

    try:
        contained = os.path.commonpath((base_path, upload_path)) == base_path
    except ValueError:
        contained = False

    if not contained:
        raise ValueError("filename resolves outside the upload directory")

    return upload_path
