// Package jsonlledger holds the shared JSONL-ledger row helpers the report
// packages (cadencereport, milestonereport, programreport, …) each used to
// copy-paste: Parse scans a JSONL ledger into typed rows, and LatestBefore finds
// the newest prior row. Each caller keeps its own row type and delegates here so
// the duplicated bodies live in exactly one place.
package jsonlledger

import (
	"bufio"
	"encoding/json"
	"sort"
	"strings"
)

// Parse scans content as JSONL, unmarshaling each non-blank line into a T and
// appending it when keep(row) reports true. Blank and malformed lines are
// skipped. A nil keep accepts every well-formed row. The 1 MiB line buffer
// matches the copies this consolidates, so long ledger lines still parse.
func Parse[T any](content string, keep func(T) bool) []T {
	var rows []T
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row T
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if keep != nil && !keep(row) {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

// LatestBefore returns the row in prior with the greatest (date, tiebreak) sort
// key, skipping any row whose non-empty tiebreak equals the reference row's (its
// own prior generation), or (zero, false) when none remain. date and tiebreak
// extract the primary sort key and the stable-sort tiebreaker from a row. It
// consolidates the identical "find the previous ledger row" scan the report
// packages each carried.
func LatestBefore[T any](row T, prior []T, date, tiebreak func(T) string) (T, bool) {
	self := tiebreak(row)
	cands := make([]T, 0, len(prior))
	for _, p := range prior {
		if tb := tiebreak(p); tb != "" && tb == self {
			continue
		}
		cands = append(cands, p)
	}
	if len(cands) == 0 {
		var zero T
		return zero, false
	}
	sort.SliceStable(cands, func(i, j int) bool {
		if di, dj := date(cands[i]), date(cands[j]); di != dj {
			return di < dj
		}
		return tiebreak(cands[i]) < tiebreak(cands[j])
	})
	return cands[len(cands)-1], true
}
