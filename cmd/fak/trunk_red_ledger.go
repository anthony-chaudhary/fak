package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// trunk_red_ledger.go — turn a PRE-EXISTING trunk red from a per-clone stderr
// shrug into a best-effort, DEDUPLICATED fleet witness the fleet CONVERGES on.
//
// Both build gates (commit_buildcheck.go and prepush_build.go) already do
// differential attribution: when the touched packages fail to build, they rebuild
// the SAME packages at the base/HEAD trunk, and if the red was ALREADY there they
// admit — never wedge a clean delta behind a peer's break — and print a rich
// advisory that NAMES the shared break. That advisory was the fix for "the gate
// silently shrugs." But it is still PER-CLONE and EPHEMERAL: N agents each
// re-discover the same red, each print their own advisory to their own stderr, and
// nobody converges. "Needs a fix at its source" is stated but nothing makes it
// visible that the whole fleet is stuck on ONE break.
//
// This file is the middle ground between that per-clone advisory and a full
// issue-filing convergence system: a best-effort JSONL witness, appended fail-open,
// keyed by a CONVERGENCE CLASS (base sha + the sorted failing packages) so every
// clone that hits the same shared red folds onto ONE class at read time. `fak
// trunk-red` folds the ledger so an operator (or the next agent) sees the shared
// break ONCE, with how many clones are stuck on it — the signal that says "fix this
// at its source" instead of N parallel work-arounds.
//
// Every write is FAIL-OPEN: the witness never changes the gate's admit decision (the
// commit/push is already allowed by the time we record), and any append error is
// advisory (stderr) only. It mirrors guard_stops.go's ledger discipline — single
// Write per row so concurrent O_APPEND writers interleave at line granularity, a
// `.fak/` gitignored runtime path so active gates never dirty tracked docs, and an
// env override + mode-off switch.

// trunkRedRecordSchema versions the row shape so a reader can evolve.
const trunkRedRecordSchema = "fak.trunk-red.v1"

// trunkRedNow is the injectable clock (mirrors prepushNow) so the recorder stays
// testable — a test pins the timestamp instead of racing the wall clock.
var trunkRedNow = time.Now

// trunkRedRecord is one durable row: a single gate invocation that admitted a commit
// or push over a pre-existing trunk red. It carries the shared break's identity, not
// this clone's delta — the point is that many clones write rows that fold to ONE class.
type trunkRedRecord struct {
	Schema     string   `json:"schema"`
	Ts         string   `json:"ts,omitempty"`
	Gate       string   `json:"gate"`                  // "commit" | "pre-push"
	BaseSha    string   `json:"base_sha,omitempty"`    // the trunk commit the red was proven pre-existing at
	Packages   []string `json:"packages,omitempty"`    // the failing import paths (the shared break)
	FirstBreak string   `json:"first_break,omitempty"` // first undefined symbol, when parseable
	Session    string   `json:"session,omitempty"`     // best-effort session id, so distinct clones are countable
}

// trunkRedClass is the convergence key: the shared break's identity, independent of
// which clone, gate, or turn witnessed it — the base trunk it was proven red at, plus
// the sorted set of failing packages. N clones re-hitting the SAME red fold to one
// class at read time; a genuinely different break (different packages, or a moved base)
// is a different class.
func trunkRedClass(baseSha string, pkgs []string) string {
	sorted := append([]string(nil), pkgs...)
	sort.Strings(sorted)
	base := strings.TrimSpace(baseSha)
	if base == "" {
		base = "unknown"
	}
	return base + " :: " + strings.Join(sorted, " ")
}

// Class returns the convergence key for a record.
func (r trunkRedRecord) Class() string { return trunkRedClass(r.BaseSha, r.Packages) }

// ---- ledger path + append ---------------------------------------------------

const (
	// trunkRedLedgerEnv overrides the ledger path for BOTH the writer (the gates) and
	// the reader (`fak trunk-red`). Empty falls back to the repo-root default.
	trunkRedLedgerEnv = "FAK_TRUNK_RED_LEDGER"
	// trunkRedModeEnv opts the WRITER out (value "off") even when a ledger resolves, so
	// a test or a one-off run never appends. The reader ignores it.
	trunkRedModeEnv = "FAK_TRUNK_RED_MODE"
	// trunkRedLedgerDefaultRel is the repo-root-relative default, in the gitignored
	// runtime-state directory, so an active gate never dirties tracked docs.
	trunkRedLedgerDefaultRel = ".fak/trunk-red.jsonl"
)

// trunkRedLedgerDefaultFor is the absolute default path for a NAMED repo root:
// <root>/.fak/trunk-red.jsonl. Empty when the root is blank (so the witness
// silently no-ops rather than scattering ledgers under a random cwd).
func trunkRedLedgerDefaultFor(root string) string {
	if strings.TrimSpace(root) == "" {
		return ""
	}
	return filepath.Join(root, filepath.FromSlash(trunkRedLedgerDefaultRel))
}

// trunkRedLedgerDefault is the reader's default: the repo the PROCESS is in.
func trunkRedLedgerDefault() string { return trunkRedLedgerDefaultFor(repoRoot()) }

// trunkRedLedgerForWrite is the path the gates write to for the repo they are
// GATING: an explicit env override wins, else that repo's default. "" (mode=off,
// or no resolvable root) disables recording.
//
// The gated root is passed in rather than read from the process cwd on purpose. A
// gate run against another checkout — the build-check gate's own tests drive it
// against a throwaway repo — would otherwise file its synthetic break in THIS
// repo's ledger, where the base sha resolves against nothing and the row can never
// fold out. That is how `fak trunk-red` came to report 28 fleet-wide "shared
// breaks" that were all one test fixture, drowning the real break it exists to
// surface.
func trunkRedLedgerForWrite(root string) string {
	if strings.EqualFold(strings.TrimSpace(os.Getenv(trunkRedModeEnv)), "off") {
		return ""
	}
	if p := strings.TrimSpace(os.Getenv(trunkRedLedgerEnv)); p != "" {
		return p
	}
	return trunkRedLedgerDefaultFor(root)
}

// trunkRedLedgerForRead is the path `fak trunk-red` reads: the env override, else the
// repo-root default. Unlike the writer it ignores the off switch — a reader wants
// whatever is on disk.
func trunkRedLedgerForRead() string {
	if p := strings.TrimSpace(os.Getenv(trunkRedLedgerEnv)); p != "" {
		return p
	}
	return trunkRedLedgerDefault()
}

// trunkRedSession is the best-effort session id stamped on a row so distinct clones
// are countable in the fold. "" when neither convention is set.
func trunkRedSession() string {
	if s := strings.TrimSpace(os.Getenv("FAK_SESSION_ID")); s != "" {
		return s
	}
	return strings.TrimSpace(os.Getenv("CLAUDE_SESSION_ID"))
}

// trunkRedWitness is the honest result of trying to record a pre-existing red: whether
// the row was actually written (so a caller only claims "witnessed" when it is true),
// and — for the convergence line — how many rows now share this class and how many
// distinct sessions are stuck on it.
type trunkRedWitness struct {
	Witnessed   bool
	Occurrences int // rows sharing this class, including the one just written
	Sessions    int // distinct sessions across those rows (min 1 when witnessed)
	Ledger      string
	Class       string
}

// emitTrunkRedWitness records one pre-existing-red admission, FAIL-OPEN. It is a no-op
// (Witnessed=false) when recording is disabled or no failing package was parsed — a
// witness with no named break would not converge on anything. An append error is
// advisory (stderr) only and never changes the gate's already-made admit decision.
// On success it folds the ledger to return this class's occurrence and session counts,
// so the caller can print an honest convergence line. root is the repo being GATED
// — the row lands in that repo's ledger, never in whatever checkout the process
// happens to be sitting in.
func emitTrunkRedWitness(stderr io.Writer, root, gate, baseSha string, pkgs []string, firstBreak string) trunkRedWitness {
	w := trunkRedWitness{Class: trunkRedClass(baseSha, pkgs)}
	if len(pkgs) == 0 {
		return w // nothing nameable to converge on
	}
	ledger := trunkRedLedgerForWrite(root)
	if ledger == "" {
		return w // recording disabled or no resolvable root
	}
	w.Ledger = ledger
	rec := trunkRedRecord{
		Schema:     trunkRedRecordSchema,
		Ts:         trunkRedNow().UTC().Format(time.RFC3339),
		Gate:       gate,
		BaseSha:    strings.TrimSpace(baseSha),
		Packages:   pkgs,
		FirstBreak: strings.TrimSpace(firstBreak),
		Session:    trunkRedSession(),
	}
	if err := appendTrunkRedRecord(ledger, rec); err != nil {
		fmt.Fprintf(stderr, "fak: trunk-red witness append skipped (fail-open): %v\n", err)
		return w
	}
	w.Witnessed = true
	w.Occurrences, w.Sessions = countTrunkRedClass(ledger, w.Class)
	if w.Occurrences < 1 {
		w.Occurrences = 1 // we just wrote a row; a read-back miss must not under-report
	}
	if w.Sessions < 1 {
		w.Sessions = 1
	}
	return w
}

// appendTrunkRedRecord writes rec as a single JSONL line (one Write so concurrent
// O_APPEND writers interleave at line granularity), creating parents as needed.
func appendTrunkRedRecord(path string, rec trunkRedRecord) error {
	if rec.Schema == "" {
		rec.Schema = trunkRedRecordSchema
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	return writeLine(f, b) // shared with guard_stops.go (same package)
}

// countTrunkRedClass folds the ledger and returns how many rows share class, and how
// many DISTINCT sessions those rows came from (an empty session id counts as one
// anonymous clone). A read error yields (0, 0) — the caller treats that as "at least
// the row I just wrote."
func countTrunkRedClass(path, class string) (occurrences, sessions int) {
	content, err := os.ReadFile(path)
	if err != nil {
		return 0, 0
	}
	rows := jsonlledger.Parse(string(content), func(r trunkRedRecord) bool {
		return r.Schema == trunkRedRecordSchema && r.Class() == class
	})
	seen := map[string]struct{}{}
	anon := false
	for _, r := range rows {
		occurrences++
		if s := strings.TrimSpace(r.Session); s != "" {
			seen[s] = struct{}{}
		} else {
			anon = true
		}
	}
	sessions = len(seen)
	if anon {
		sessions++
	}
	return occurrences, sessions
}

// trunkRedWitnessNote renders the convergence line appended to a gate's pre-existing-red
// advisory — ONLY when the row was actually written, so the gate never claims a witness
// it does not have. It makes the "working together" payoff visible: this shared break is
// now recorded once for the fleet, with how many clones are stuck on it, and how to fold it.
func trunkRedWitnessNote(w trunkRedWitness) string {
	if !w.Witnessed {
		return ""
	}
	var b strings.Builder
	if w.Sessions > 1 {
		fmt.Fprintf(&b, "  → witnessed to the shared trunk-red ledger: %d clone(s) across %d session(s) are stuck on THIS break.\n", w.Occurrences, w.Sessions)
	} else {
		fmt.Fprintf(&b, "  → witnessed to the shared trunk-red ledger (recorded once for the fleet).\n")
	}
	b.WriteString("    Fold who's hitting it with `fak trunk-red` and fix it at its source — one fix clears every stuck clone.\n")
	return b.String()
}

// ---- summary (fak trunk-red) ------------------------------------------------

// trunkRedStatus is the EVIDENCE GRADE the fold assigns one break class. It exists
// because "not provably resolved" and "still red" are different claims, and collapsing
// them is how the fold came to assert 37 classes were "ALREADY red on the trunk" when
// not one of them had been checked against HEAD at all. Only three answers are
// admissible, and two of them are proofs.
type trunkRedStatus string

const (
	// trunkRedStatusLive: the failing package's own tree at HEAD still REFERENCES the
	// first-break symbol and declares it NOWHERE — the named break is still there.
	trunkRedStatusLive trunkRedStatus = "still-undefined-at-head"
	// trunkRedStatusUnprovable: neither proof could be made. The class is SURFACED with
	// the reason, never dropped — a hidden live break is far worse than a surfaced stale
	// one — but it is not asserted to be red either.
	trunkRedStatusUnprovable trunkRedStatus = "unprovable"
	// trunkRedStatusResolved: the trunk moved PAST the base AND the first-break symbol
	// is declared at HEAD — the break is fixed, so the class folds out of the view.
	trunkRedStatusResolved trunkRedStatus = "resolved"
)

// trunkRedVerdict is a status plus, when nothing could be proven, the NAMED reason the
// class was surfaced anyway. The reason is the whole point: "unprovable" alone is the
// same undifferentiated wall the fold is trying to stop printing.
type trunkRedVerdict struct {
	Status trunkRedStatus
	Reason string
}

// The reasons a class can be surfaced without any proof about it. Each names a MISSING
// WITNESS, not a guess about the break.
const (
	trunkRedReasonNoProbe           = "no resolve probe available"
	trunkRedReasonNoRoot            = "no repo root to check HEAD against"
	trunkRedReasonNoBase            = "the witness row carries no base sha"
	trunkRedReasonNoSymbol          = "the witness row names no first-break symbol"
	trunkRedReasonQualifiedSymbol   = "the first-break symbol is qualified, so it is declared in another package"
	trunkRedReasonOutOfModule       = "the failing package(s) are not inside this module"
	trunkRedReasonSymbolUncheckable = "git could not search HEAD for the symbol"
	trunkRedReasonSymbolAbsent      = "HEAD neither declares nor references the symbol"
	trunkRedReasonBaseNotMergedPast = "the remote trunk has not provably moved past the base"
	trunkRedReasonUnnamedUnprovable = "the probe proved nothing and named no reason"
)

// trunkRedUnprovable is the surfaced-without-proof verdict.
func trunkRedUnprovable(reason string) trunkRedVerdict {
	if strings.TrimSpace(reason) == "" {
		reason = trunkRedReasonUnnamedUnprovable
	}
	return trunkRedVerdict{Status: trunkRedStatusUnprovable, Reason: reason}
}

// trunkRedClassRollup is one converged shared break: the class, its member rows folded
// to counts and the spread of clones/sessions/gates stuck on it, the first break symbol
// for a one-glance headline, and the evidence grade that says whether the break was
// PROVEN still there or merely could not be cleared.
type trunkRedClassRollup struct {
	Class      string         `json:"class"`
	BaseSha    string         `json:"base_sha,omitempty"`
	Packages   []string       `json:"packages,omitempty"`
	FirstBreak string         `json:"first_break,omitempty"`
	Rows       int            `json:"rows"`
	Sessions   int            `json:"sessions"`
	Gates      []string       `json:"gates,omitempty"`
	FirstTs    string         `json:"first_ts,omitempty"`
	LastTs     string         `json:"last_ts,omitempty"`
	Status     trunkRedStatus `json:"status,omitempty"`
	KeepReason string         `json:"keep_reason,omitempty"` // set only when Status is unprovable
}

// trunkRedSummary is the folded value view: the distinct shared breaks currently
// witnessed, PROVEN-still-broken first. Classes PROVABLY resolved — the trunk moved
// past the base AND the symbol that broke is declared at HEAD — are folded out and only
// counted, so a wall of stale rows never buries the breaks that are still biting. The
// rest are surfaced, but split into what was PROVEN still undefined at HEAD and what
// merely could not be checked, each carrying the reason it could not be.
type trunkRedSummary struct {
	Ledger            string                `json:"ledger"`
	Total             int                   `json:"total"`   // SURFACED witness rows (resolved rows excluded)
	Classes           []trunkRedClassRollup `json:"classes"` // distinct SURFACED shared breaks
	LiveClasses       int                   `json:"live_classes,omitempty"`
	LiveRows          int                   `json:"live_rows,omitempty"`
	UnprovableClasses int                   `json:"unprovable_classes,omitempty"`
	UnprovableRows    int                   `json:"unprovable_rows,omitempty"`
	ResolvedClasses   int                   `json:"resolved_classes,omitempty"`
	ResolvedRows      int                   `json:"resolved_rows,omitempty"`
}

// summarizeTrunkRed folds ledger content into distinct convergence classes. Malformed
// or foreign lines are skipped. Classes PROVEN still undefined at HEAD come first — an
// unprovable class outranking an actionable one is how the actionable one gets missed —
// then by session spread (how much of the fleet is stuck), then row count, then class
// key.
//
// probe is the KEEP-SIDE evidence predicate. It is consulted ONCE PER CLASS, after
// the fold rather than per row, so every row of a class shares one verdict and a class
// can never be half-dropped (rows of one class may disagree about first_break, and a
// class that surfaced with some of its rows silently missing would under-report how
// much of the fleet is stuck). It may answer trunkRedStatusResolved ONLY when the break
// is PROVABLY fixed — see trunkRedGitProbe. A resolved class is folded out of the view
// and counted in ResolvedClasses/ResolvedRows instead. Every class it does not resolve
// is surfaced, tagged with the grade it earned: PROVEN still undefined at HEAD, or
// unprovable with the named reason.
//
// The fold refuses to even ASK about a class carrying no base sha or no first-break
// symbol: with nothing to check ancestry or declaredness against, such a class can never
// be PROVEN anything, so it is kept structurally — the keep-side invariant then holds
// even against a buggy or always-resolved predicate, which is what pins it in a test. A
// nil probe likewise keeps everything. A hidden live break is far worse than a surfaced
// stale one, so every uncertainty resolves toward KEEP.
func summarizeTrunkRed(content string, probe func(trunkRedBreak) trunkRedVerdict) trunkRedSummary {
	rows := jsonlledger.Parse(content, func(r trunkRedRecord) bool {
		return r.Schema == trunkRedRecordSchema
	})
	byClass := map[string]*trunkRedClassRollup{}
	order := []string{}
	sessionsByClass := map[string]map[string]struct{}{}
	anonByClass := map[string]bool{}
	gatesByClass := map[string]map[string]struct{}{}
	sum := trunkRedSummary{}
	for _, r := range rows {
		class := r.Class()
		roll, ok := byClass[class]
		if !ok {
			roll = &trunkRedClassRollup{
				Class:      class,
				BaseSha:    r.BaseSha,
				Packages:   r.Packages,
				FirstBreak: r.FirstBreak,
			}
			byClass[class] = roll
			order = append(order, class)
			sessionsByClass[class] = map[string]struct{}{}
			gatesByClass[class] = map[string]struct{}{}
		}
		roll.Rows++
		if roll.FirstBreak == "" && r.FirstBreak != "" {
			roll.FirstBreak = r.FirstBreak
		}
		if s := strings.TrimSpace(r.Session); s != "" {
			sessionsByClass[class][s] = struct{}{}
		} else {
			anonByClass[class] = true
		}
		if g := strings.TrimSpace(r.Gate); g != "" {
			gatesByClass[class][g] = struct{}{}
		}
		if r.Ts != "" {
			if roll.FirstTs == "" || r.Ts < roll.FirstTs {
				roll.FirstTs = r.Ts
			}
			if r.Ts > roll.LastTs {
				roll.LastTs = r.Ts
			}
		}
	}
	for _, class := range order {
		roll := byClass[class]
		roll.Sessions = len(sessionsByClass[class])
		if anonByClass[class] {
			roll.Sessions++
		}
		gates := make([]string, 0, len(gatesByClass[class]))
		for g := range gatesByClass[class] {
			gates = append(gates, g)
		}
		sort.Strings(gates)
		roll.Gates = gates
		verdict := trunkRedClassVerdict(*roll, probe)
		if verdict.Status == trunkRedStatusResolved {
			sum.ResolvedClasses++
			sum.ResolvedRows += roll.Rows
			continue
		}
		roll.Status = verdict.Status
		roll.KeepReason = verdict.Reason
		if verdict.Status == trunkRedStatusLive {
			sum.LiveClasses++
			sum.LiveRows += roll.Rows
		} else {
			sum.UnprovableClasses++
			sum.UnprovableRows += roll.Rows
		}
		sum.Total += roll.Rows
		sum.Classes = append(sum.Classes, *roll)
	}
	sort.SliceStable(sum.Classes, func(i, j int) bool {
		a, b := sum.Classes[i], sum.Classes[j]
		// A class PROVEN still undefined at HEAD outranks every unprovable one: it is
		// the only kind the reader can act on without re-deriving the evidence.
		if (a.Status == trunkRedStatusLive) != (b.Status == trunkRedStatusLive) {
			return a.Status == trunkRedStatusLive
		}
		if a.Sessions != b.Sessions {
			return a.Sessions > b.Sessions
		}
		if a.Rows != b.Rows {
			return a.Rows > b.Rows
		}
		return a.Class < b.Class
	})
	return sum
}

// trunkRedBreak is everything the fold knows about ONE witnessed break class, handed to
// the resolve predicate. It carries the first-break SYMBOL and the failing PACKAGES, not
// just the base sha, because ancestry alone is NOT evidence of a fix: any unrelated peer
// commit landing on the trunk makes a recorded base an ancestor of the tip, so a
// base-only predicate folds still-LIVE breaks out of the view. Proving a break resolved
// takes both conjuncts — the trunk moved past the base AND the symbol that broke is
// defined at HEAD — which is why the predicate needs the whole class, not one field.
type trunkRedBreak struct {
	BaseSha    string
	FirstBreak string
	Packages   []string
}

// trunkRedClassVerdict is the keep-side guard rail the fold puts AROUND the probe. It
// answers unprovable — KEEP, with a named reason — for every class the probe could not
// possibly prove anything about, and only then asks. Deliberately structural: the
// invariant must not depend on the probe being correct, which is why a test pins it
// against a probe that claims EVERYTHING is resolved.
//
// A probe that answers trunkRedStatusUnprovable without naming a reason is given one, so
// no surfaced class can ever appear in the view with an empty explanation.
func trunkRedClassVerdict(roll trunkRedClassRollup, probe func(trunkRedBreak) trunkRedVerdict) trunkRedVerdict {
	if probe == nil {
		return trunkRedUnprovable(trunkRedReasonNoProbe) // no way to check — KEEP
	}
	base := strings.TrimSpace(roll.BaseSha)
	sym := strings.TrimSpace(roll.FirstBreak)
	if base == "" {
		return trunkRedUnprovable(trunkRedReasonNoBase) // nothing to date — KEEP
	}
	if sym == "" {
		return trunkRedUnprovable(trunkRedReasonNoSymbol) // nothing to look up — KEEP
	}
	v := probe(trunkRedBreak{BaseSha: base, FirstBreak: sym, Packages: roll.Packages})
	switch v.Status {
	case trunkRedStatusResolved, trunkRedStatusLive:
		return trunkRedVerdict{Status: v.Status}
	default:
		return trunkRedUnprovable(v.Reason)
	}
}

// trunkRedGitProbe returns the production evidence probe for summarizeTrunkRed. It
// answers one of three grades, and TWO of them are proofs against HEAD:
//
//   - RESOLVED (fold the class out) requires BOTH conjuncts: the base sha is a STRICT
//     ancestor of the remote trunk tip (`git merge-base --is-ancestor` exits 0 and the
//     base is not the tip itself) — the trunk has moved PAST the commit the red was
//     proven at — AND the first-break symbol is DECLARED at HEAD inside the failing
//     packages' own directories: the thing that actually broke is back.
//   - LIVE (surface it first) requires the mirror-image proof: HEAD's copy of the
//     failing package still REFERENCES the first-break symbol and declares it nowhere,
//     so the break the row names is still there.
//   - UNPROVABLE (surface it, with the reason) is everything else.
//
// Conjunct 1 alone is NOT evidence of a fix, and shipping it alone is a live-break
// hazard rather than a stale-row cleanup: every recorded base becomes an ancestor of the
// tip the moment ANY unrelated peer commit lands, so a base-only predicate silently
// folds out breaks that are still red. Measured on a real 338-row ledger, ancestry alone
// dropped 309 rows (91%) — everything except the bases git could not resolve at all.
// Conjunct 2 is what makes the drop mean "fixed" instead of "old".
//
// The LIVE grade exists because the mirror mistake is just as costly in the other
// direction: a fold whose only two answers are "resolved" and "not resolved" ends up
// PRINTING every survivor as a live fleet blocker, which is the wall of stale-reading
// signal this whole path exists to cut. LIVE is asserted only from the same class of
// evidence a drop takes — a git read of HEAD, never the working tree, because on a
// shared multi-session checkout the tree carries peers' uncommitted work.
//
// EVERY uncertainty is UNPROVABLE and keeps the class surfaced with its reason: no repo
// root, an empty base or symbol, an unresolvable sha, a missing remote trunk ref (fresh
// clone, no remote), a package outside this module, a symbol that is not a bare
// identifier, or any git error at all.
//
// Both probes are memoized — ancestry per base, symbol state per symbol+dirs — so a
// large ledger shells git once per distinct question rather than once per class, and
// ancestry is only asked once the symbol reads as declared.
func trunkRedGitProbe(root string) func(trunkRedBreak) trunkRedVerdict {
	root = strings.TrimSpace(root)
	mergedPast := map[string]bool{}
	stateAtHead := map[string]trunkRedSymbolState{}
	return func(b trunkRedBreak) trunkRedVerdict {
		base := strings.TrimSpace(b.BaseSha)
		sym := strings.TrimSpace(b.FirstBreak)
		if root == "" {
			return trunkRedUnprovable(trunkRedReasonNoRoot)
		}
		if base == "" {
			return trunkRedUnprovable(trunkRedReasonNoBase)
		}
		if sym == "" {
			return trunkRedUnprovable(trunkRedReasonNoSymbol)
		}
		if !trunkRedPlainIdent(sym) {
			return trunkRedUnprovable(trunkRedReasonQualifiedSymbol)
		}
		dirs := trunkRedPackageDirs(root, b.Packages)
		if len(dirs) == 0 {
			return trunkRedUnprovable(trunkRedReasonOutOfModule)
		}
		key := sym + " :: " + strings.Join(dirs, " ")
		state, ok := stateAtHead[key]
		if !ok {
			state = trunkRedSymbolStateAtHead(root, dirs, sym)
			stateAtHead[key] = state
		}
		switch state {
		case trunkRedSymbolDeclared:
			merged, ok := mergedPast[base]
			if !ok {
				merged = trunkRedBaseMergedPast(root, base)
				mergedPast[base] = merged
			}
			if !merged {
				return trunkRedUnprovable(trunkRedReasonBaseNotMergedPast)
			}
			return trunkRedVerdict{Status: trunkRedStatusResolved}
		case trunkRedSymbolUndeclared:
			// Referenced at HEAD in the failing package's own tree, declared nowhere in
			// it, and nothing there even LOOKS like a declaration of it: the break the
			// row names is still in the trunk's own HEAD.
			return trunkRedVerdict{Status: trunkRedStatusLive}
		case trunkRedSymbolAbsent:
			// The reference itself is gone. Something changed, but "the caller was
			// deleted" is not the repair conjunct 2 asks for, so this does not earn a
			// drop — it earns a named surface.
			return trunkRedUnprovable(trunkRedReasonSymbolAbsent)
		default:
			return trunkRedUnprovable(trunkRedReasonSymbolUncheckable)
		}
	}
}

// trunkRedModulePath reads the module path out of <root>/go.mod so import paths can be
// mapped to directories without hard-coding this repo's own module name. "" on any
// read or parse failure, which makes every package unmappable and therefore keeps every
// class — the keep-side direction.
func trunkRedModulePath(root string) string {
	b, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(b), "\n") {
		if fields := strings.Fields(line); len(fields) >= 2 && fields[0] == "module" {
			return fields[1]
		}
	}
	return ""
}

// trunkRedPackageDirs maps a class's failing IMPORT PATHS to the repo-relative
// directories to search at HEAD. It returns nil unless EVERY package maps into this
// module, because a class whose failing surface includes something we cannot inspect is
// a class we cannot clear: a gate's own fixture files its break against a synthetic path
// like "buildcheck.test/p", and a stdlib path like "time" names no tree here. The bare
// module path is treated as unmappable too — it would widen the search to the whole
// worktree, which is both slow and prone to matching an unrelated package's symbol.
func trunkRedPackageDirs(root string, pkgs []string) []string {
	mod := trunkRedModulePath(root)
	if mod == "" || len(pkgs) == 0 {
		return nil
	}
	seen := map[string]struct{}{}
	dirs := make([]string, 0, len(pkgs))
	for _, p := range pkgs {
		p = strings.TrimSpace(p)
		if !strings.HasPrefix(p, mod+"/") {
			return nil // foreign, synthetic, or the module root — nothing to check — KEEP
		}
		rel := strings.TrimPrefix(p, mod+"/")
		if rel == "" || strings.Contains(rel, "..") {
			return nil
		}
		if _, dup := seen[rel]; dup {
			continue
		}
		seen[rel] = struct{}{}
		dirs = append(dirs, rel)
	}
	sort.Strings(dirs)
	return dirs
}

// trunkRedSymbolState is what a git read of HEAD could establish about the first-break
// symbol inside the failing packages' own directories. It is deliberately FOUR-valued:
// "git could not answer" and "the answer is no" are different facts, and a two-valued
// probe that collapses them can only ever support one of the two proofs this fold needs.
type trunkRedSymbolState int

const (
	// trunkRedSymbolUnknown: git could not answer, or the answer is ambiguous enough
	// that neither proof may rest on it.
	trunkRedSymbolUnknown trunkRedSymbolState = iota
	// trunkRedSymbolDeclared: a package-level declaration of sym exists at HEAD.
	trunkRedSymbolDeclared
	// trunkRedSymbolUndeclared: sym is still referenced at HEAD in these dirs and
	// declared nowhere in them.
	trunkRedSymbolUndeclared
	// trunkRedSymbolAbsent: HEAD's copy of these dirs neither declares nor mentions sym.
	trunkRedSymbolAbsent
)

// trunkRedSymbolStateAtHead reads HEAD — never the working tree, because on a shared
// multi-session checkout the tree is full of peers' uncommitted work and a symbol that
// only exists in someone's unstaged edit is not a fix anyone else has — and grades what
// it finds inside the failing packages' own directories.
//
// The order of the three greps is the honesty of the answer:
//
//  1. a package-level declaration => DECLARED (the repair conjunct 2 asks for);
//  2. else an INDENTED line that even looks like a declaration of sym — a grouped
//     `var (...)` / `const (...)` member, a struct field — => UNKNOWN. The column-0
//     pattern cannot see a grouped declaration, so this is the guard that stops a
//     grouped-decl miss from being reported as a positive "still undeclared" proof;
//  3. else a bare word reference => UNDECLARED (referenced but declared nowhere: the
//     named break is still there); no reference at all => ABSENT.
func trunkRedSymbolStateAtHead(root string, dirs []string, sym string) trunkRedSymbolState {
	if strings.TrimSpace(root) == "" || len(dirs) == 0 || !trunkRedPlainIdent(sym) {
		return trunkRedSymbolUnknown
	}
	if hit, ok := trunkRedGrepAtHead(root, dirs, "-E", trunkRedDefinitionPattern(sym)); !ok {
		return trunkRedSymbolUnknown
	} else if hit {
		return trunkRedSymbolDeclared
	}
	if hit, ok := trunkRedGrepAtHead(root, dirs, "-E", trunkRedIndentedDeclPattern(sym)); !ok || hit {
		return trunkRedSymbolUnknown
	}
	hit, ok := trunkRedGrepAtHead(root, dirs, "-F", "-w", "-e", sym)
	if !ok {
		return trunkRedSymbolUnknown
	}
	if hit {
		return trunkRedSymbolUndeclared
	}
	return trunkRedSymbolAbsent
}

// trunkRedSymbolDefinedAtHead reports whether sym has a package-level Go declaration at
// HEAD in one of dirs — resolve conjunct 2. Anything short of a proven declaration is
// false, meaning KEEP.
func trunkRedSymbolDefinedAtHead(root string, dirs []string, sym string) bool {
	return trunkRedSymbolStateAtHead(root, dirs, sym) == trunkRedSymbolDeclared
}

// trunkRedGrepAtHead runs one `git grep` over dirs' Go files at HEAD. It returns whether
// the pattern matched AND whether git could answer at all: `git grep` exits 1 for "no
// match" and >1 for a real failure, so the two are kept apart rather than collapsed. A
// failure read as "no match" would turn a broken git into evidence of absence, which is
// exactly the kind of silent, agreeable answer this fold must never give.
func trunkRedGrepAtHead(root string, dirs []string, pattern ...string) (hit, ok bool) {
	args := append([]string{"grep", "--no-color", "-I", "-l"}, pattern...)
	args = append(args, "HEAD", "--")
	for _, d := range dirs {
		args = append(args, d+"/*.go")
	}
	out, err := gitOut(root, args...)
	if err == nil {
		return strings.TrimSpace(out) != "", true
	}
	var exit *exec.ExitError
	if errors.As(err, &exit) && exit.ExitCode() == 1 {
		return false, true // git ran and found nothing
	}
	return false, false // git could not answer
}

// trunkRedDefinitionPattern is the POSIX-ERE `git grep` pattern matching a PACKAGE-LEVEL
// Go declaration of sym: an unindented func/type/var/const whose name is sym, optionally
// followed by a signature or a generic parameter list. Anchoring at column 0 is what
// keeps the answer honest — an indented match is a struct field, a local variable, or a
// grouped-decl member, none of which proves the package-scope symbol the build reported
// as undefined now exists.
func trunkRedDefinitionPattern(sym string) string {
	return "^(func|type|var|const)[[:space:]]+" + sym + "([[:space:]]|\\(|\\[|$)"
}

// trunkRedIndentedDeclPattern matches an INDENTED line that starts with sym — a grouped
// `var (...)` / `const (...)` member, a struct field, a one-per-line parameter. None of
// these proves a package-scope declaration, so they never earn a RESOLVED drop; what
// they do is make the opposite claim unsafe, because a grouped declaration is a real
// declaration the column-0 pattern cannot see. Matching here therefore downgrades the
// symbol to UNKNOWN: over-matching costs a class its LIVE promotion, never its place in
// the view.
func trunkRedIndentedDeclPattern(sym string) string {
	return "^[[:space:]]+" + sym + "([[:space:]]|,|=|$)"
}

// trunkRedPlainIdent reports whether sym is a bare Go identifier — the only shape this
// file is willing to search for. A qualified break like "metrics.AnchorRefusalMonitor"
// names a symbol in ANOTHER package, which the failing class's own directories cannot
// witness; and anything else risks injecting regex metacharacters into the grep pattern.
// Both are unresolvable rather than unresolved, so both KEEP.
func trunkRedPlainIdent(sym string) bool {
	if sym == "" {
		return false
	}
	for i, r := range sym {
		switch {
		case r == '_', r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
			if i == 0 {
				return false
			}
		default:
			return false
		}
	}
	return true
}

// trunkRedTrunkRef names the remote trunk ref this file compares bases against. The
// branch is never hard-coded: it comes from the same branch-role contract every other
// verb reads (safecommit.ExpectedTrunk over dos.toml's [branch_roles], falling back to
// safecommit.DefaultTrunk when the config cannot be read), remote-qualified as
// "origin/<trunk>" the way sweep.go's origin probe does it.
func trunkRedTrunkRef(root string) string {
	return "origin/" + safecommit.ExpectedTrunk(root, "")
}

// trunkRedBaseMergedPast reports whether base is a STRICT ancestor of the remote trunk
// tip. Any git failure (unresolvable base, no remote trunk ref, command error) is
// NOT-resolved.
func trunkRedBaseMergedPast(root, base string) bool {
	baseOut, err := gitOut(root, "rev-parse", "--verify", base+"^{commit}")
	if err != nil {
		return false
	}
	tipOut, err := gitOut(root, "rev-parse", "--verify", trunkRedTrunkRef(root)+"^{commit}")
	if err != nil {
		return false
	}
	baseFull, tipFull := strings.TrimSpace(baseOut), strings.TrimSpace(tipOut)
	if baseFull == "" || tipFull == "" || baseFull == tipFull {
		return false // strictness: the trunk must have moved PAST the base
	}
	if _, err := gitOut(root, "merge-base", "--is-ancestor", baseFull, tipFull); err != nil {
		return false // not an ancestor, or git failed — either way KEEP
	}
	return true
}

// trunkRedStatusLabel is the one-glance evidence grade printed beside a surfaced class.
// An unprovable class ALWAYS carries its reason, so no line in the view can be read as a
// bare assertion that the break is live.
func trunkRedStatusLabel(c trunkRedClassRollup) string {
	if c.Status == trunkRedStatusLive {
		return "still undefined at HEAD"
	}
	reason := strings.TrimSpace(c.KeepReason)
	if reason == "" {
		reason = trunkRedReasonUnnamedUnprovable
	}
	return "unprovable: " + reason
}

// renderTrunkRed formats the folded view for a human reader.
func renderTrunkRed(sum trunkRedSummary) string {
	var b strings.Builder
	if sum.Total == 0 {
		if sum.ResolvedRows > 0 {
			fmt.Fprintf(&b, "fak trunk-red: no LIVE shared breaks — %d resolved class(es) across %d witness row(s) folded out (base merged past on the remote trunk AND first-break symbol declared at HEAD).\n", sum.ResolvedClasses, sum.ResolvedRows)
			fmt.Fprintf(&b, "  ledger: %s", sum.Ledger)
			return strings.TrimRight(b.String(), "\n")
		}
		fmt.Fprintf(&b, "fak trunk-red: no pre-existing trunk-red admissions recorded.\n")
		fmt.Fprintf(&b, "  ledger: %s\n", sum.Ledger)
		b.WriteString("  A build gate records one row here each time it admits a commit/push over a break it proved was ALREADY red on the trunk (a peer's, not yours).")
		return strings.TrimRight(b.String(), "\n")
	}
	fmt.Fprintf(&b, "fak trunk-red: %d shared break(s) across %d witness row(s) — %d still undefined at HEAD, %d unprovable\n",
		len(sum.Classes), sum.Total, sum.LiveClasses, sum.UnprovableClasses)
	fmt.Fprintf(&b, "  ledger: %s\n", sum.Ledger)
	if sum.LiveClasses > 0 {
		b.WriteString("  [still undefined at HEAD] is the break actually biting: HEAD's own copy of the failing package still references the symbol and declares it nowhere. Fix it at its source; one fix clears every stuck clone.\n")
	}
	if sum.UnprovableClasses > 0 {
		b.WriteString("  [unprovable] is surfaced ON PURPOSE and is NOT a claim the break is live: the fold could not prove it either way, for the reason on its own line. A hidden live break is far worse than a surfaced stale one.\n")
	}
	for _, c := range sum.Classes {
		pkgs := strings.Join(c.Packages, " ")
		if pkgs == "" {
			pkgs = "(unnamed)"
		}
		fmt.Fprintf(&b, "  - [%s] %s\n", trunkRedStatusLabel(c), pkgs)
		fmt.Fprintf(&b, "      %d clone(s) across %d session(s) stuck", c.Rows, c.Sessions)
		if len(c.Gates) > 0 {
			fmt.Fprintf(&b, " (gate: %s)", strings.Join(c.Gates, ","))
		}
		b.WriteString("\n")
		if c.BaseSha != "" {
			fmt.Fprintf(&b, "      base: %s", short(c.BaseSha))
			if c.FirstBreak != "" {
				fmt.Fprintf(&b, "  first break: undefined: %s", c.FirstBreak)
			}
			b.WriteString("\n")
		} else if c.FirstBreak != "" {
			fmt.Fprintf(&b, "      first break: undefined: %s\n", c.FirstBreak)
		}
		if c.FirstTs != "" {
			fmt.Fprintf(&b, "      %s .. %s\n", c.FirstTs, c.LastTs)
		}
	}
	if sum.ResolvedRows > 0 {
		fmt.Fprintf(&b, "  (+ %d resolved class(es) across %d row(s) folded out: base merged past on the remote trunk AND first-break symbol declared at HEAD)\n", sum.ResolvedClasses, sum.ResolvedRows)
	}
	return strings.TrimRight(b.String(), "\n")
}

// cmdTrunkRed is the `fak trunk-red` entry point: fold the pre-existing-red witness
// ledger into the distinct shared breaks the fleet is currently stuck on.
func cmdTrunkRed(argv []string) {
	os.Exit(runTrunkRed(os.Stdout, os.Stderr, argv))
}

func runTrunkRed(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("trunk-red", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak trunk-red [--ledger PATH] [--json]")
		fmt.Fprintln(stderr, "  Fold the pre-existing trunk-red witness ledger: the distinct shared")
		fmt.Fprintln(stderr, "  breaks build gates admitted over, worst (most clones stuck) first.")
		fs.PrintDefaults()
	}
	ledgerFlag := fs.String("ledger", "", "path to the trunk-red JSONL ledger (default: $FAK_TRUNK_RED_LEDGER or <repo>/.fak/trunk-red.jsonl)")
	jsonFlag := fs.Bool("json", false, "emit the summary as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	ledger := strings.TrimSpace(*ledgerFlag)
	if ledger == "" {
		ledger = trunkRedLedgerForRead()
	}
	if ledger == "" {
		fmt.Fprintln(stderr, "fak trunk-red: no ledger path (pass --ledger, set $FAK_TRUNK_RED_LEDGER, or run inside a repo)")
		return 1
	}
	content, err := readTrunkRedLedger(ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak trunk-red: read %s: %v\n", ledger, err)
		return 1
	}
	sum := summarizeTrunkRed(content, trunkRedGitProbe(repoRoot()))
	sum.Ledger = ledger
	if *jsonFlag {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(sum); err != nil {
			fmt.Fprintf(stderr, "fak trunk-red: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintln(stdout, renderTrunkRed(sum))
	return 0
}

// readTrunkRedLedger reads the ledger file. A missing file is a valid empty view (no
// pre-existing red has been witnessed yet), not an error.
func readTrunkRedLedger(path string) (string, error) {
	return readLedgerText(path)
}
