package modver

import (
	"sort"
	"time"
)

// DefaultDormantDays is the staleness window the dormant finder applies when the
// caller passes a non-positive threshold: a module whose last trunk touch is at
// least this many days before "now" counts as dormant. Thirty days is the window
// the super-loop fuel query names — "module untouched 30 days with open issues
// naming its paths".
const DefaultDormantDays = 30

// OpenIssue is one open GitHub issue paired with the repo-relative paths its
// body references — the raw provenance the finder joins against the ledger. Each
// path is folded to its module key via moduleOf (the same mapping the version
// spine uses), so an issue naming internal/modver/dormant.go references module
// internal/modver. The finder never reads issues itself; the shell hands them
// in, which keeps the join a pure, fixture-testable function.
type OpenIssue struct {
	Number int      `json:"number"`
	Title  string   `json:"title,omitempty"`
	Paths  []string `json:"paths"`
}

// IssueRef is one open issue attributed to a dormant module — the fuel pointer
// the dispatcher follows back to actionable work.
type IssueRef struct {
	Number int    `json:"number"`
	Title  string `json:"title,omitempty"`
}

// DormantModule is one dormant candidate: a live, ledger-known module whose last
// trunk touch is at least the threshold window old yet still carries open issues
// naming its paths — a mechanical unit of super-loop fuel.
type DormantModule struct {
	Module     string     `json:"module"`
	Kind       string     `json:"kind"`
	Rev        int        `json:"rev"`
	LastDate   string     `json:"last_date"`
	DaysIdle   int        `json:"days_idle"`
	IssueCount int        `json:"issue_count"`
	Issues     []IssueRef `json:"issues"`
}

// DormantReport is the finder's readout: the "now" it judged against, the
// staleness window it applied, how many ledger-known modules an open issue
// referenced, and the dormant candidates (most-dormant first). Every field is
// JSON-tagged so a dispatcher can consume the report directly.
type DormantReport struct {
	Now        string          `json:"now"`
	Days       int             `json:"threshold_days"`
	Scanned    int             `json:"scanned_modules"`
	Candidates []DormantModule `json:"candidates"`
}

// Dormant joins the module-versions ledger's per-module last-touch dates with
// open-issue path references and lists the dormant candidates: modules untouched
// for at least `days` (<= 0 uses DefaultDormantDays) that still carry at least
// one open issue naming their paths.
//
// The ledger is read exactly as Trend reads it — the newest-timestamped row per
// module wins for last_date/rev/kind, and unparseable lines are skipped, since
// an append-only ledger a fleet writes will have scars. `now` is passed as data
// (never read from the clock) so the finder is deterministic and fixture-
// testable. Issue paths are folded to module keys via moduleOf; a path outside
// the tracked key space, or a module with no ledger row (no last-touch to age),
// contributes nothing — an unknown scope is conservatively dropped rather than
// treated as dormant. Results are sorted most-dormant first (days idle desc),
// ties broken by issue count desc then module name asc, so the order is total
// and stable.
func Dormant(ledger []byte, issues []OpenIssue, now time.Time, days int) DormantReport {
	if days <= 0 {
		days = DefaultDormantDays
	}
	// Newest ledger row per module: its last_date/rev/kind at the freshest stamp.
	type modInfo struct {
		ts, lastDate, kind string
		rev                int
	}
	latest := map[string]modInfo{}
	for _, row := range parseLedgerRows(ledger) {
		if cur, ok := latest[row.Module]; ok && row.TS < cur.ts {
			continue
		}
		latest[row.Module] = modInfo{ts: row.TS, lastDate: row.LastDate, kind: row.Kind, rev: row.Rev}
	}
	// Fold open-issue path references onto ledger-known modules, deduping issues
	// per module (an issue naming several paths in one module counts once).
	refs := map[string][]IssueRef{}
	seen := map[string]map[int]bool{}
	for _, iss := range issues {
		for _, p := range iss.Paths {
			name, _, ok := moduleOf(p)
			if !ok {
				continue
			}
			if _, known := latest[name]; !known {
				continue // no ledger last-date: dormancy is not judgeable
			}
			if seen[name] == nil {
				seen[name] = map[int]bool{}
			}
			if seen[name][iss.Number] {
				continue
			}
			seen[name][iss.Number] = true
			refs[name] = append(refs[name], IssueRef{Number: iss.Number, Title: iss.Title})
		}
	}
	rep := DormantReport{Now: now.UTC().Format(time.RFC3339), Days: days}
	cutoff := now.AddDate(0, 0, -days)
	for name, issueRefs := range refs {
		rep.Scanned++
		info := latest[name]
		last, err := time.Parse(time.RFC3339, info.lastDate)
		if err != nil {
			continue // unparseable last_date: skip rather than mis-age it
		}
		if last.After(cutoff) {
			continue // touched within the window: not dormant
		}
		sort.Slice(issueRefs, func(i, j int) bool { return issueRefs[i].Number < issueRefs[j].Number })
		rep.Candidates = append(rep.Candidates, DormantModule{
			Module:     name,
			Kind:       info.kind,
			Rev:        info.rev,
			LastDate:   info.lastDate,
			DaysIdle:   int(now.Sub(last).Hours() / 24),
			IssueCount: len(issueRefs),
			Issues:     issueRefs,
		})
	}
	sort.Slice(rep.Candidates, func(i, j int) bool {
		a, b := rep.Candidates[i], rep.Candidates[j]
		if a.DaysIdle != b.DaysIdle {
			return a.DaysIdle > b.DaysIdle
		}
		if a.IssueCount != b.IssueCount {
			return a.IssueCount > b.IssueCount
		}
		return a.Module < b.Module
	})
	return rep
}
