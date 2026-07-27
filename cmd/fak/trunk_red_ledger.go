package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
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

// trunkRedClassRollup is one converged shared break: the class, its member rows folded
// to counts and the spread of clones/sessions/gates stuck on it, plus the first break
// symbol for a one-glance headline.
type trunkRedClassRollup struct {
	Class      string   `json:"class"`
	BaseSha    string   `json:"base_sha,omitempty"`
	Packages   []string `json:"packages,omitempty"`
	FirstBreak string   `json:"first_break,omitempty"`
	Rows       int      `json:"rows"`
	Sessions   int      `json:"sessions"`
	Gates      []string `json:"gates,omitempty"`
	FirstTs    string   `json:"first_ts,omitempty"`
	LastTs     string   `json:"last_ts,omitempty"`
}

// trunkRedSummary is the folded value view: the distinct shared breaks currently
// witnessed, worst (most clones stuck) first. Classes whose base the trunk has
// PROVABLY moved past are folded out of the live view and only counted, so a wall
// of stale rows never buries the breaks that are still biting.
type trunkRedSummary struct {
	Ledger          string                `json:"ledger"`
	Total           int                   `json:"total"`   // LIVE witness rows (resolved rows excluded)
	Classes         []trunkRedClassRollup `json:"classes"` // distinct LIVE shared breaks
	ResolvedClasses int                   `json:"resolved_classes,omitempty"`
	ResolvedRows    int                   `json:"resolved_rows,omitempty"`
}

// summarizeTrunkRed folds ledger content into distinct convergence classes. Malformed
// or foreign lines are skipped. Classes are ordered by session spread (how much of the
// fleet is stuck), then row count, then class key — the worst shared break first.
//
// resolved is the KEEP-SIDE resolve predicate: it must return true ONLY when the
// break recorded against baseSha is PROVABLY resolved (see trunkRedGitResolver).
// A resolved class is folded out of the live view and counted in
// ResolvedClasses/ResolvedRows instead. nil, an empty base, or ANY uncertainty in
// the predicate keeps the row — a hidden live break is far worse than a surfaced
// stale one. The predicate is consulted once per distinct base.
func summarizeTrunkRed(content string, resolved func(baseSha string) bool) trunkRedSummary {
	rows := jsonlledger.Parse(content, func(r trunkRedRecord) bool {
		return r.Schema == trunkRedRecordSchema
	})
	resolvedByBase := map[string]bool{}
	baseResolved := func(baseSha string) bool {
		base := strings.TrimSpace(baseSha)
		if resolved == nil || base == "" {
			return false // nothing checkable — KEEP
		}
		v, ok := resolvedByBase[base]
		if !ok {
			v = resolved(base)
			resolvedByBase[base] = v
		}
		return v
	}
	byClass := map[string]*trunkRedClassRollup{}
	order := []string{}
	sessionsByClass := map[string]map[string]struct{}{}
	anonByClass := map[string]bool{}
	gatesByClass := map[string]map[string]struct{}{}
	resolvedClasses := map[string]struct{}{}
	sum := trunkRedSummary{}
	for _, r := range rows {
		class := r.Class()
		if baseResolved(r.BaseSha) {
			sum.ResolvedRows++
			resolvedClasses[class] = struct{}{}
			continue
		}
		sum.Total++
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
		sum.Classes = append(sum.Classes, *roll)
	}
	sort.SliceStable(sum.Classes, func(i, j int) bool {
		a, b := sum.Classes[i], sum.Classes[j]
		if a.Sessions != b.Sessions {
			return a.Sessions > b.Sessions
		}
		if a.Rows != b.Rows {
			return a.Rows > b.Rows
		}
		return a.Class < b.Class
	})
	sum.ResolvedClasses = len(resolvedClasses)
	return sum
}

// trunkRedGitResolver returns the production resolve predicate for summarizeTrunkRed:
// a base sha is PROVABLY resolved only when it is a STRICT ancestor of the remote
// trunk tip (`git merge-base --is-ancestor` exits 0 and the base is not that tip
// itself) — the trunk has moved PAST the commit the red was proven at, so the break
// was very likely fixed upstream. EVERY uncertainty — no repo root, an unresolvable
// sha, a missing remote trunk ref (fresh clone, no remote), any git error — reports
// NOT resolved, keeping the row surfaced: a hidden live break is far worse than a
// stale one. Results are memoized per base so a large ledger shells git once per
// distinct base.
func trunkRedGitResolver(root string) func(baseSha string) bool {
	cache := map[string]bool{}
	return func(baseSha string) bool {
		base := strings.TrimSpace(baseSha)
		if strings.TrimSpace(root) == "" || base == "" {
			return false // nothing checkable — KEEP
		}
		v, ok := cache[base]
		if !ok {
			v = trunkRedBaseMergedPast(root, base)
			cache[base] = v
		}
		return v
	}
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

// renderTrunkRed formats the folded view for a human reader.
func renderTrunkRed(sum trunkRedSummary) string {
	var b strings.Builder
	if sum.Total == 0 {
		if sum.ResolvedRows > 0 {
			fmt.Fprintf(&b, "fak trunk-red: no LIVE shared breaks — %d resolved class(es) across %d witness row(s) folded out (base already merged past on the remote trunk).\n", sum.ResolvedClasses, sum.ResolvedRows)
			fmt.Fprintf(&b, "  ledger: %s", sum.Ledger)
			return strings.TrimRight(b.String(), "\n")
		}
		fmt.Fprintf(&b, "fak trunk-red: no pre-existing trunk-red admissions recorded.\n")
		fmt.Fprintf(&b, "  ledger: %s\n", sum.Ledger)
		b.WriteString("  A build gate records one row here each time it admits a commit/push over a break it proved was ALREADY red on the trunk (a peer's, not yours).")
		return strings.TrimRight(b.String(), "\n")
	}
	fmt.Fprintf(&b, "fak trunk-red: %d shared break(s) across %d witness row(s)\n", len(sum.Classes), sum.Total)
	fmt.Fprintf(&b, "  ledger: %s\n", sum.Ledger)
	b.WriteString("  Each break below is ALREADY red on the trunk — every clone that touches it inherits it. Fix it at its source; one fix clears every stuck clone.\n")
	for _, c := range sum.Classes {
		pkgs := strings.Join(c.Packages, " ")
		if pkgs == "" {
			pkgs = "(unnamed)"
		}
		fmt.Fprintf(&b, "  - %s\n", pkgs)
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
		fmt.Fprintf(&b, "  (+ %d resolved class(es) across %d row(s) folded out: base already merged past on the remote trunk)\n", sum.ResolvedClasses, sum.ResolvedRows)
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
	sum := summarizeTrunkRed(content, trunkRedGitResolver(repoRoot()))
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
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	return string(b), nil
}
