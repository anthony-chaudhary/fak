balances = {}  # account id -> integer cents

def _debit(acct, cents):
    """Take `cents` out of acct."""
    balances[acct] = balances.get(acct, 0) - cents

def deposit(acct, cents):
    balances[acct] = balances.get(acct, 0) + cents

def transfer(src, dst, cents):
    """Move `cents` from src to dst."""
    if cents <= 0:
        raise ValueError("transfer amount must be positive")
    if balances.get(src, 0) < cents:
        raise ValueError("insufficient funds")

    _debit(src, cents)
    deposit(dst, cents)

def withdraw(acct, cents):
    """Take `cents` out of acct as cash."""
    _debit(acct, cents)
    return cents
