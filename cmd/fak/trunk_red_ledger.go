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
// witnessed, worst (most clones stuck) first. Classes PROVABLY resolved — the trunk
// moved past the base AND the symbol that broke is defined at HEAD — are folded out
// of the live view and only counted, so a wall of stale rows never buries the breaks
// that are still biting.
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
// resolved is the KEEP-SIDE resolve predicate. It is consulted ONCE PER CLASS, after
// the fold rather than per row, so every row of a class shares one verdict and a class
// can never be half-dropped (rows of one class may disagree about first_break, and a
// class that surfaced with some of its rows silently missing would under-report how
// much of the fleet is stuck). It must return true ONLY when the break is PROVABLY
// resolved — see trunkRedGitResolver. A resolved class is folded out of the live view
// and counted in ResolvedClasses/ResolvedRows instead.
//
// The fold refuses to even ASK about a class carrying no base sha or no first-break
// symbol: with nothing to check ancestry or definedness against, such a class can never
// be PROVEN resolved, so it is kept structurally — the keep-side invariant then holds
// even against a buggy or always-true predicate, which is what pins it in a test. A nil
// predicate likewise keeps everything. A hidden live break is far worse than a surfaced
// stale one, so every uncertainty resolves toward KEEP.
func summarizeTrunkRed(content string, resolved func(trunkRedBreak) bool) trunkRedSummary {
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
		if trunkRedClassResolved(*roll, resolved) {
			sum.ResolvedClasses++
			sum.ResolvedRows += roll.Rows
			continue
		}
		sum.Total += roll.Rows
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

// trunkRedClassResolved is the keep-side guard rail the fold puts AROUND the predicate.
// It answers false — KEEP — for every class the predicate could not possibly prove
// resolved, and only then asks. Deliberately structural: the invariant must not depend
// on the predicate being correct.
func trunkRedClassResolved(roll trunkRedClassRollup, resolved func(trunkRedBreak) bool) bool {
	if resolved == nil {
		return false // no way to check — KEEP
	}
	base := strings.TrimSpace(roll.BaseSha)
	sym := strings.TrimSpace(roll.FirstBreak)
	if base == "" || sym == "" {
		return false // no base to date, or no symbol to look up — KEEP
	}
	return resolved(trunkRedBreak{BaseSha: base, FirstBreak: sym, Packages: roll.Packages})
}

// trunkRedGitResolver returns the production resolve predicate for summarizeTrunkRed. A
// break is PROVABLY resolved only when BOTH conjuncts hold:
//
//  1. its base sha is a STRICT ancestor of the remote trunk tip (`git merge-base
//     --is-ancestor` exits 0 and the base is not the tip itself) — the trunk has moved
//     PAST the commit the red was proven at; AND
//  2. its first-break symbol is DEFINED at HEAD inside the failing packages' own
//     directories — the thing that actually broke is back.
//
// Conjunct 1 alone is NOT evidence of a fix, and shipping it alone is a live-break
// hazard rather than a stale-row cleanup: every recorded base becomes an ancestor of the
// tip the moment ANY unrelated peer commit lands, so a base-only predicate silently
// folds out breaks that are still red. Measured on a real 338-row ledger, ancestry alone
// dropped 309 rows (91%) — everything except the bases git could not resolve at all.
// Conjunct 2 is what makes the drop mean "fixed" instead of "old".
//
// EVERY uncertainty reports NOT resolved, keeping the class surfaced: no repo root, an
// empty base or symbol, an unresolvable sha, a missing remote trunk ref (fresh clone, no
// remote), a package outside this module, a symbol that is not a bare identifier, or any
// git error at all. A hidden live break is far worse than a surfaced stale one.
//
// Both conjuncts are memoized — ancestry per base, definedness per symbol+dirs — so a
// large ledger shells git once per distinct question rather than once per class, and
// conjunct 2 is only asked when conjunct 1 already held.
func trunkRedGitResolver(root string) func(trunkRedBreak) bool {
	mergedPast := map[string]bool{}
	definedAtHead := map[string]bool{}
	return func(b trunkRedBreak) bool {
		base := strings.TrimSpace(b.BaseSha)
		sym := strings.TrimSpace(b.FirstBreak)
		if strings.TrimSpace(root) == "" || base == "" || sym == "" {
			return false // nothing checkable — KEEP
		}
		dirs := trunkRedPackageDirs(root, b.Packages)
		if len(dirs) == 0 {
			return false // no in-module tree to search for the symbol — KEEP
		}
		merged, ok := mergedPast[base]
		if !ok {
			merged = trunkRedBaseMergedPast(root, base)
			mergedPast[base] = merged
		}
		if !merged {
			return false // the trunk has not provably moved past the base — KEEP
		}
		key := sym + " :: " + strings.Join(dirs, " ")
		defined, ok := definedAtHead[key]
		if !ok {
			defined = trunkRedSymbolDefinedAtHead(root, dirs, sym)
			definedAtHead[key] = defined
		}
		return defined
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

// trunkRedSymbolDefinedAtHead reports whether sym has a package-level Go declaration at
// HEAD in one of dirs — resolve conjunct 2. It reads HEAD, never the working tree: on a
// shared multi-session checkout the tree is full of peers' uncommitted work, and a
// symbol that only exists in someone's unstaged edit is not a fix anyone else has.
//
// `git grep` exits non-zero both when it finds NO match and when it fails outright, and
// this collapses the two to the same answer on purpose: false, meaning KEEP.
func trunkRedSymbolDefinedAtHead(root string, dirs []string, sym string) bool {
	if strings.TrimSpace(root) == "" || len(dirs) == 0 || !trunkRedPlainIdent(sym) {
		return false
	}
	args := []string{"grep", "--no-color", "-I", "-l", "-E", trunkRedDefinitionPattern(sym), "HEAD", "--"}
	for _, d := range dirs {
		args = append(args, d+"/*.go")
	}
	out, err := gitOut(root, args...)
	if err != nil {
		return false // no match, or git failed — either way KEEP
	}
	return strings.TrimSpace(out) != ""
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

// renderTrunkRed formats the folded view for a human reader.
func renderTrunkRed(sum trunkRedSummary) string {
	var b strings.Builder
	if sum.Total == 0 {
		if sum.ResolvedRows > 0 {
			fmt.Fprintf(&b, "fak trunk-red: no LIVE shared breaks — %d resolved class(es) across %d witness row(s) folded out (base merged past on the remote trunk AND first-break symbol defined at HEAD).\n", sum.ResolvedClasses, sum.ResolvedRows)
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
		fmt.Fprintf(&b, "  (+ %d resolved class(es) across %d row(s) folded out: base merged past on the remote trunk AND first-break symbol defined at HEAD)\n", sum.ResolvedClasses, sum.ResolvedRows)
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
	return readLedgerText(path)
}
