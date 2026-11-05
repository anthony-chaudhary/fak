package report

import (
	"sort"

	"github.com/anthony-chaudhary/fak/pkg/billing"
)

// TokenSum is one token axis summed across a group of records, keeping the
// reported/unknown distinction intact.
type TokenSum struct {
	// Value is the sum of the REPORTED counts on this axis. When Known is
	// false it is a partial sum over the records that did report, not a
	// trustworthy total.
	Value int64

	// Known is true only when every record in the group reported this
	// axis. False means at least one record left it unknown; Value then
	// carries the flagged partial sum.
	Known bool
}

// RollupRow is one (tenant, model, UTC day) group of usage records.
type RollupRow struct {
	// TenantID is the opaque tenant identifier the group belongs to. The
	// empty string groups records that named no tenant.
	TenantID string

	// Model is the requested model name shared by the group.
	Model string

	// Day is the UTC calendar day ("2006-01-02") the group's requests
	// fall in.
	Day string

	// Records is how many valid records contribute to the group.
	Records int64

	Prompt     TokenSum
	Completion TokenSum
	Cached     TokenSum
}

// DayOf returns the UTC calendar day ("2006-01-02") a record's request
// falls in. A zero RequestedAt yields "0001-01-01"; validate records
// before rolling them up if a real day is required.
func DayOf(rec billing.UsageRecord) string {
	return rec.RequestedAt.UTC().Format("2006-01-02")
}

// Rollup groups records by (tenant, model, UTC day) and sums the token
// axes with unknown-preserving arithmetic: an axis that any contributing
// record left unreported comes back Known=false, so "unknown" can never be
// mistaken for "known zero" after aggregation. Records that fail
// billing.Validate are excluded from the sums and counted in the second
// return value, so a caller can tell "no usage" from "bad input". Rows
// are sorted by (TenantID, Model, Day) for deterministic output.
//
// Rollup is a transform, not a ledger: it applies no dedup, so two records
// sharing an event ID both count. Dedup-by-event-id is the consumer's job.
func Rollup(records []billing.UsageRecord) (rows []RollupRow, skipped int) {
	type groupKey struct {
		tenant string
		model  string
		day    string
	}
	groups := make(map[groupKey]*RollupRow)

	for _, rec := range records {
		if err := rec.Validate(); err != nil {
			skipped++
			continue
		}
		k := groupKey{tenant: rec.TenantID, model: rec.Model, day: DayOf(rec)}
		row, ok := groups[k]
		if !ok {
			row = &RollupRow{
				TenantID:   k.tenant,
				Model:      k.model,
				Day:        k.day,
				Prompt:     TokenSum{Known: true},
				Completion: TokenSum{Known: true},
				Cached:     TokenSum{Known: true},
			}
			groups[k] = row
		}
		row.Records++
		addAxis(&row.Prompt, rec.PromptTokens)
		addAxis(&row.Completion, rec.CompletionTokens)
		addAxis(&row.Cached, rec.CachedTokens)
	}

	rows = make([]RollupRow, 0, len(groups))
	for _, row := range groups {
		rows = append(rows, *row)
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].TenantID != rows[j].TenantID {
			return rows[i].TenantID < rows[j].TenantID
		}
		if rows[i].Model != rows[j].Model {
			return rows[i].Model < rows[j].Model
		}
		return rows[i].Day < rows[j].Day
	})
	return rows, skipped
}

// addAxis folds one record's reported value for an axis into the group
// sum. A nil (unreported) value flags the axis unknown without touching
// the partial sum.
func addAxis(sum *TokenSum, reported *int64) {
	if reported == nil {
		sum.Known = false
		return
	}
	sum.Value += *reported
}
