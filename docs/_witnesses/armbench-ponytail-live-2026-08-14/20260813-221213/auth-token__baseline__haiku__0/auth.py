import hmac, hashlib
def make_token(user_id, secret):
    """Create a signed token of the form 'user_id.signature'."""
    sig = hmac.new(secret.encode(), str(user_id).encode(), hashlib.sha256).hexdigest()
    return f'{user_id}.{sig}'
def verify_token(token, secret):
    """Return the user_id if the token signature is valid, else None."""
    if not isinstance(token, str):
        return None

    user_id, separator, signature = token.rpartition('.')
    if not separator:
        return None

    try:
        expected = hmac.new(
            secret.encode(), user_id.encode(), hashlib.sha256
        ).hexdigest().encode('ascii')
        supplied = signature.encode('ascii')
    except (AttributeError, UnicodeEncodeError):
        return None

    return user_id if hmac.compare_digest(supplied, expected) else None
