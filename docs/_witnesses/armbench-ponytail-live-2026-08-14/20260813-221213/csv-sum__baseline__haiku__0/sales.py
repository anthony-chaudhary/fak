import csv


def sum_amount(path):
    """Read the CSV at path and return the sum of its 'amount' column."""
    total = 0.0

    with open(path, newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                total += float(row["amount"])
            except (KeyError, TypeError, ValueError):
                continue

    return total
