// Package reportledger defines the (date, generated-at) ordering keys every
// report ledger row carries, so the "latest prior row" lookup is bound once
// here instead of once per report package. Rows stay package-local types; a
// row opts in by implementing Dated over its own two fields.
package reportledger

import "github.com/anthony-chaudhary/fak/internal/jsonlledger"

// Dated is the two-key ordering contract shared by every report ledger row:
// LedgerDate is the tick date the trend orders by, LedgerGeneratedAt the
// same-run stamp that both breaks same-day ties and identifies a row's own
// prior generation.
type Dated interface {
	LedgerDate() string
	LedgerGeneratedAt() string
}

// LatestBefore returns the row in prior with the greatest (LedgerDate,
// LedgerGeneratedAt) sort key, skipping any row whose non-empty
// LedgerGeneratedAt equals row's — its own prior generation — or (zero, false)
// when none remain. A same-day re-run therefore trends against the earlier
// same-day tick, and re-appending a row is idempotent. It binds the shared
// jsonlledger scan to Dated rows so the per-report accessor closures are not
// re-spelled per package.
func LatestBefore[T Dated](row T, prior []T) (T, bool) {
	return jsonlledger.LatestBefore(row, prior, T.LedgerDate, T.LedgerGeneratedAt)
}
