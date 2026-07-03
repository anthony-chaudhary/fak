package main

func issueSyncScanLimit(limit int) int {
	if limit > 0 {
		return limit
	}
	return 100
}
