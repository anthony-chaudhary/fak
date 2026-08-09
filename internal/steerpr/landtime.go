package steerpr

// landtime.go — #5026 (child of epic #5015): assign a landed commit to its unit
// at LAND time instead of at the next maintenance tick.
//
// The overlay's visibility latency was bounded by the tick cadence (loop.go): a
// commit that landed just after a tick stayed invisible until the next one, and
// every tick of latency is budget the fleet may spend going the wrong way. But
// the stamp that decides a commit's unit — `(fak <leaf>)` plus the subject's #N
// — is fully known AT COMMIT TIME; it is in the message being written. Only the
// BAND needs the witness rung, and that rung may legitimately not exist yet. So
// the membership half is assigned here, immediately, and the band settles on the
// tick.
//
// Three rules keep this a latency optimization rather than a second oracle:
//
//   - It is a CACHE, never the source of truth. Tick reads git and folds; it
//     never reads this ledger, so the tick's fold is bit-identical whether or not
//     the hook ran (TestTickFoldIdenticalWithAndWithoutLandTimeHook). When the
//     two disagree the tick wins — ReconcileLandings reports the drift in that
//     direction only.
//   - It NEVER fails a commit. Every entry point returns an error a caller is
//     expected to drop on the floor; nothing here exits, gates, or refuses, and a
//     broken/unwritable ledger costs one un-cached row that the next tick
//     re-derives from git anyway (TestBrokenLedgerNeverBlocksTheLanding).
//   - It resolves NO band. Landing carries no band or verdict field at all
//     (TestLandingCarriesNoBand), so a land-time row cannot pre-empt the witness
//     rung with a guess — the exact "the hook became the oracle" inversion #5026
//     names as its worst outcome.
//
// The ledger is gitignored runtime state under .fak/ (like the ack and pause
// ledgers), NOT a committed docs/nightrun artifact: a row written on every commit
// into a tracked file would dirty the tree from inside the commit path and race
// every peer on the shared trunk — a cache has no business making the trunk
// harder to commit to. The tick's committed ledger stays the durable record.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// LandingSchema is the machine identifier for one land-time assignment row.
const LandingSchema = "fak.steerpr.landing.v1"

// LandingLatencyBudget is the DECLARED per-commit ceiling for the whole
// land-time seam (parse + ledger append), asserted by
// TestLandingSeamStaysWithinLatencyBudget and reported as
// GATE_LATENCY_REGRESSION when breached. It is deliberately far above the
// measured cost (a string parse plus one small append): the number exists to
// catch a REGRESSION — someone teaching this path to shell git, take a lock, or
// walk the whole ledger — not to grade the current implementation.
const LandingLatencyBudget = 25 * time.Millisecond

// Landing is one appended row: "this SHA joined this unit, at this time".
//
// It carries the MEMBERSHIP half of the fold and nothing else. There is
// deliberately no band, verdict, or attention field: the witness rung behind a
// band may not be resolvable at land time (a test run may still be going), and a
// guessed band that later reads CLEARED would launder unwitnessed work. Band is
// the tick's job, always.
type Landing struct {
	Schema string `json:"schema"`
	At     string `json:"at"`  // RFC3339 UTC: when the commit landed
	SHA    string `json:"sha"` // the landed commit
	// Unit is the unit key the commit joins — the `(fak <leaf>)` ship-stamp.
	// Empty exactly when Orphan is true.
	Unit string `json:"unit,omitempty"`
	// GroupedBy is always GroupedByLeaf here, stated rather than implied (the
	// #5040 rule). A WAVE regrouping needs the cohort plan the committer does not
	// have in hand, so land time assigns the leaf unit — the same fallback
	// FoldUnitsByWave uses — and the tick supersedes it if a wave claims the
	// commit.
	GroupedBy string `json:"grouped_by"`
	// Orphan marks an UNSTAMPED commit: legibility debt, recorded rather than
	// dropped, exactly as FoldUnits surfaces it. Assigning nothing silently would
	// make the cache disagree with the fold's total partition.
	Orphan   bool     `json:"orphan,omitempty"`
	Subject  string   `json:"subject"`
	Type     string   `json:"type,omitempty"`
	Resolves []string `json:"resolves,omitempty"`
}

// LandingLedgerPath is the land-time ledger's location under a repo root:
// gitignored runtime state beside the other overlay ledgers, one JSON row per
// line.
func LandingLedgerPath(root string) string {
	return filepath.Join(root, ".fak", "steer-landings.jsonl")
}

// AssignLanded parses one landed commit's message into its unit membership.
//
// It shares parseCommit with ParseLog, so the unit a commit is assigned at land
// time and the unit the tick folds it into are computed by ONE parser: the two
// cannot drift, because there is no second implementation to drift from.
//
// An empty sha or subject is refused — a row that cannot name its commit is not
// a cache entry, it is noise — but an UNSTAMPED subject is not an error: it
// yields an orphan row, which is what the fold does with it too.
func AssignLanded(sha, subject, body string, at time.Time) (Landing, error) {
	sha = strings.TrimSpace(sha)
	subject = strings.TrimSpace(subject)
	if sha == "" {
		return Landing{}, fmt.Errorf("a landing must name the commit that landed (empty sha)")
	}
	if subject == "" {
		return Landing{}, fmt.Errorf("a landing must carry the commit subject: the unit is parsed from it")
	}
	c := parseCommit(sha, subject, body, nil)
	return Landing{
		Schema:    LandingSchema,
		At:        at.UTC().Format(time.RFC3339),
		SHA:       c.SHA,
		Unit:      c.Leaf,
		GroupedBy: GroupedByLeaf,
		Orphan:    c.Leaf == "",
		Subject:   c.Subject,
		Type:      c.Type,
		Resolves:  c.Resolves,
	}, nil
}

// RecordLanding appends one landing row to the ledger, returning whether it
// wrote. It is IDEMPOTENT against a double-fire of the same commit (the
// realistic duplicate: two triggers racing over one commit), which it detects by
// comparing against the ledger's LAST row — a commit re-announced after other
// commits landed is a genuinely new fact about a re-landed SHA and is appended.
//
// Every failure mode returns an error and writes nothing. NO caller is expected
// to act on that error beyond recording it: a missed row costs one commit's
// visibility until the next tick re-derives it from git, which is strictly less
// bad than anything a commit-time refusal could buy.
func RecordLanding(path string, l Landing) (appended bool, err error) {
	if err := CheckLanding(l); err != nil {
		return false, err
	}
	if last, ok := lastLanding(path); ok && last.SHA == l.SHA && last.Unit == l.Unit {
		return false, nil
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return false, err
		}
	}
	line, err := json.Marshal(l)
	if err != nil {
		return false, err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return false, err
	}
	_, werr := f.Write(append(line, '\n'))
	cerr := f.Close()
	if werr != nil {
		return false, werr
	}
	return cerr == nil, cerr
}

// CheckLanding validates one row from the row alone: the schema tag, a
// parseable timestamp, a named commit, the stated grouping basis, and the
// orphan/unit exclusivity that keeps the cache's partition as total and disjoint
// as the fold's (a row is either assigned to exactly one unit or explicitly an
// orphan — never both, never neither).
func CheckLanding(l Landing) error {
	if l.Schema != LandingSchema {
		return fmt.Errorf("landing schema %q, want %q", l.Schema, LandingSchema)
	}
	if strings.TrimSpace(l.SHA) == "" {
		return fmt.Errorf("landing names no commit")
	}
	if _, err := time.Parse(time.RFC3339, l.At); err != nil {
		return fmt.Errorf("landing at %q is not RFC3339: %w", l.At, err)
	}
	if l.GroupedBy != GroupedByLeaf {
		return fmt.Errorf("landing grouped_by %q, want %q: land time assigns the leaf unit and the tick regroups", l.GroupedBy, GroupedByLeaf)
	}
	if l.Orphan && strings.TrimSpace(l.Unit) != "" {
		return fmt.Errorf("landing is marked orphan yet names unit %q", l.Unit)
	}
	if !l.Orphan && strings.TrimSpace(l.Unit) == "" {
		return fmt.Errorf("landing names no unit and is not marked orphan: an unassigned commit must be surfaced as legibility debt, never dropped")
	}
	return nil
}

// LoadLandings reads the ledger best-effort: a missing or unreadable file is an
// empty ledger, and a torn or foreign line is skipped rather than poisoning the
// rows around it. Failure never invents a landing — the tick re-derives the
// truth from git regardless.
func LoadLandings(path string) []Landing {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []Landing
	for _, line := range strings.Split(string(buf), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var l Landing
		if json.Unmarshal([]byte(line), &l) != nil || l.Schema != LandingSchema {
			continue
		}
		out = append(out, l)
	}
	return out
}

// lastLanding returns the ledger's last parseable schema-matching row.
func lastLanding(path string) (Landing, bool) {
	rows := LoadLandings(path)
	if len(rows) == 0 {
		return Landing{}, false
	}
	return rows[len(rows)-1], true
}

// UnitOfLanding answers the land-time question — "which unit did this commit
// join?" — from the cache alone, without waiting for a tick. The LATEST row for
// a SHA wins, because the ledger is append-only and a later row is a newer fact
// about the same commit.
func UnitOfLanding(landings []Landing, sha string) (unit string, ok bool) {
	sha = strings.TrimSpace(sha)
	if sha == "" {
		return "", false
	}
	for i := len(landings) - 1; i >= 0; i-- {
		if landings[i].SHA == sha {
			return landings[i].Unit, true
		}
	}
	return "", false
}

// LandingDrift is one disagreement between the land-time cache and the tick's
// fold, always stated in the tick's favour: Want is what the fold says, Got is
// what the cache recorded.
type LandingDrift struct {
	SHA  string `json:"sha"`
	Want string `json:"want"` // the unit the tick's fold assigns ("" = orphan)
	Got  string `json:"got"`  // the unit the land-time row recorded ("" = orphan/missing)
	Why  string `json:"why"`
}

// ReconcileLandings reports where the land-time cache disagrees with a tick's
// fold. It is the mechanism behind "the tick is the source of truth": the drift
// is reported ONE WAY — the fold's assignment is Want, the cache's is Got — and
// nothing here rewrites the fold. A hook that never ran simply yields "no
// land-time row", which is drift the tick has already fixed by folding from git.
//
// Only commits the fold saw are compared: a cache row for a commit outside the
// tick's range says nothing about this range and is not drift.
func ReconcileLandings(landings []Landing, units []Unit, unstamped []Commit) []LandingDrift {
	want := map[string]string{}
	var order []string
	for _, u := range units {
		for _, c := range u.Commits {
			want[c.SHA] = u.Leaf
			order = append(order, c.SHA)
		}
	}
	for _, c := range unstamped {
		want[c.SHA] = ""
		order = append(order, c.SHA)
	}
	var drift []LandingDrift
	for _, sha := range order {
		got, cached := UnitOfLanding(landings, sha)
		switch {
		case !cached:
			drift = append(drift, LandingDrift{SHA: sha, Want: want[sha], Got: "",
				Why: "no land-time row: the hook did not run (or could not write); the tick assigned it"})
		case got != want[sha]:
			drift = append(drift, LandingDrift{SHA: sha, Want: want[sha], Got: got,
				Why: "land-time row disagrees with the fold; the tick wins and the row is stale"})
		}
	}
	return drift
}
