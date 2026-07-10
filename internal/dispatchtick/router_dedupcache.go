package dispatchtick

// Backlog-hash-keyed memo for the O(n^2) duplicate-risk scan (#4171, epic #4160).
// RouteIssues runs DuplicateRiskIssueNumbers over the full routable backlog on
// every dispatch tick, but the backlog rarely changes between adjacent ticks --
// so the pairwise scan is pure recomputation. A long-lived DuplicateRiskCache
// keys the last result on a content hash of the routable slice (Number + Title +
// normalized Body, in slice order) and only rescans when the hash moves. The
// memo elides recomputation only; it never changes which issues are flagged.

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
	"sync"
)

// DuplicateRiskCache memoizes the last DuplicateRiskIssueNumbers result keyed by
// a stable hash of the routable backlog. The zero value is ready to use. A nil
// handle is valid and means "no memo": Risk falls through to a plain recompute,
// so callers that pass no cache keep today's always-recompute behavior exactly.
type DuplicateRiskCache struct {
	mu         sync.Mutex
	hash       string
	risk       map[int]bool
	recomputes int
	hits       int
}

// Risk returns the duplicate-risk set for the routable backlog. On a hash hit it
// returns the memoized map without rescanning; on a miss it calls
// DuplicateRiskIssueNumbers, stores the fresh result, and bumps the recompute
// counter. The returned map is shared across hits and must be treated as
// read-only (RouteIssues only reads it).
func (c *DuplicateRiskCache) Risk(routable []Issue) map[int]bool {
	if c == nil {
		return DuplicateRiskIssueNumbers(routable)
	}
	key := DuplicateRiskBacklogHash(routable)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.risk != nil && c.hash == key {
		c.hits++
		return c.risk
	}
	c.risk = DuplicateRiskIssueNumbers(routable)
	c.hash = key
	c.recomputes++
	return c.risk
}

// Recomputes reports how many times Risk had to run the full pairwise scan.
func (c *DuplicateRiskCache) Recomputes() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.recomputes
}

// Hits reports how many times Risk answered from the memo without a rescan.
func (c *DuplicateRiskCache) Hits() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits
}

// DuplicateRiskBacklogHash content-addresses a routable backlog for the memo:
// sha256 over each issue's Number, Title, and normalized Body in slice order
// (issues arrive already ordered), with unit/record separators so field
// boundaries cannot alias. Any added/removed issue or edited title/body -- even
// at the same set size -- changes the hash and invalidates the memo. Labels are
// deliberately excluded: DuplicateRiskIssueNumbers never reads them, and label
// changes that affect routability already change the routable slice itself.
func DuplicateRiskBacklogHash(issues []Issue) string {
	h := sha256.New()
	for _, issue := range issues {
		h.Write([]byte(strconv.Itoa(issue.Number)))
		h.Write([]byte{0x1f})
		h.Write([]byte(issue.Title))
		h.Write([]byte{0x1f})
		h.Write([]byte(normalizeDuplicateBacklogBody(issue.Body)))
		h.Write([]byte{0x1e})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// normalizeDuplicateBacklogBody canonicalizes an issue body for hashing only:
// CRLF folds to LF and outer whitespace is trimmed, so a pure line-ending or
// trailing-newline change does not force a rescan. Purely cosmetic to the risk
// scan -- the scan itself tokenizes bodies far more aggressively.
func normalizeDuplicateBacklogBody(body string) string {
	return strings.TrimSpace(strings.ReplaceAll(body, "\r\n", "\n"))
}
