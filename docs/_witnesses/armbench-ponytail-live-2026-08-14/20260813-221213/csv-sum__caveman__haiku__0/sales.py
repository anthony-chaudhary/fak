import csv


def sum_amount(path):
    """Read the CSV at path and return the sum of its 'amount' column."""
    total = 0.0

    with open(path, newline="", encoding="utf-8-sig") as csv_file:
        rows = csv.reader(csv_file)
        header = next(rows)
        amount_index = header.index("amount")

        for row in rows:
            if len(row) != len(header):
                continue

            try:
                total += float(row[amount_index])
            except (ValueError, IndexError):
                continue

    return total
