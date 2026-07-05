package main

// fak knownbad -- the impure shell over the internal/knownbad fold core: the verbs
// the blast-radius containment epic (#2712) spine exposes.
//
//	fak knownbad record --tree internal/foo/** --reason build --note "..."
//	    appends one fak.known-bad.v1 JSONL row to a fleet-visible ledger.
//	fak knownbad match  --tree internal/foo/bar.go [--json]
//	    reports whether the requested tree intersects any LIVE (open, unexpired)
//	    known-bad signature, printing the matching record(s). Exit is non-zero
//	    (with --json, matched:false) when nothing matches, so a worker OR the
//	    dispatcher can short-circuit before burning a cycle.
//	fak knownbad claim <signature> [--by ID] [--ttl S] [--json]
//	    elects EXACTLY ONE fixer for a live signature by acquiring an EXCLUSIVE dos
//	    lease over its broken tree (W5, #2717). The first claimant wins (exit 0) and
//	    is stamped onto the ledger; a second, competing claimant is REFUSED (exit 3,
//	    reason KNOWN_BAD_ALREADY_CLAIMED) and told the current fixer's identity so it
//	    parks behind them instead of racing a second edit to the shared tree.
//	fak knownbad resolve <signature> [--by ID] [--witness tests|verify] [--json]
//	    flips a signature open -> resolved ONLY on an independent witness (W6, #2718):
//	    a green `go test` over the broken tree (--witness tests, default) and/or a
//	    `dos verify` binding the fixer's commit to the signature's tree
//	    (--witness verify). No witness -> stays open, refused (exit 3,
//	    KNOWN_BAD_NOT_WITNESSED) and reported as `not yet` with the missing witness.
//	    On resolve: appends a superseding resolved row (which clears the W4
//	    scope-hold on the next dispatch tick) and releases the fixer's exclusive
//	    lease (W5). The witness is the gate; a self-report never releases the fleet.
//	fak knownbad report [--leases FILE] [--repo-url URL] [--operator-after S] \
//	                    [--dry-run] [--json]
//	    folds the LIVE known-bad signatures into ONE operator blast card (W7, #2719):
//	    "N shared blockers -> M affected, K fixing, M-K parked, witness pending",
//	    posted to the #blockers Slack surface (blockerpost). The affected count per
//	    signature is the live dos leases whose tree intersects the signature's tree
//	    (the direct-intersection floor). Severity: clear when the ledger has no live
//	    signature, operator when an UNCLAIMED signature has gone longer than
//	    --operator-after without a fixer (a human must elect one), else a muted status
//	    line. --dry-run (default posture, like the other feeders) renders the card
//	    without posting.
//
// All impurity lives here: the ledger read/write, the exclusive-lease store (git
// refs, via internal/leaseref), the clock (Unix seconds, injected as `nowUnix` so
// runKnownBad is deterministic under test), the witness exec (go test / dos verify,
// injectable as knownBadWitness so the resolve verb is deterministic under test),
// and flag parsing. The signature derivation, tree intersection, liveness, the
// lease-id derivation, and the resolve-status fold are the pure core in
// internal/knownbad.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blastradius"
	"github.com/anthony-chaudhary/fak/internal/blockerpost"
	"github.com/anthony-chaudhary/fak/internal/knownbad"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func cmdKnownBad(argv []string) {
	os.Exit(runKnownBad(os.Stdout, os.Stderr, argv, time.Now().Unix()))
}

func runKnownBad(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak knownbad: expected a subcommand (record|match|claim|resolve)")
		return 2
	}
	switch argv[0] {
	case "record":
		return runKnownBadRecord(stdout, stderr, argv[1:], nowUnix)
	case "match":
		return runKnownBadMatch(stdout, stderr, argv[1:], nowUnix)
	case "claim":
		return runKnownBadClaim(stdout, stderr, argv[1:], nowUnix)
	case "resolve":
		return runKnownBadResolve(stdout, stderr, argv[1:], nowUnix)
	case "report":
		return runKnownBadReport(stdout, stderr, argv[1:], nowUnix)
	case "-h", "--help", "help":
		fmt.Fprintln(stderr, "fak knownbad: record | match | claim | resolve | report  (fleet-wide known-bad signature ledger)")
		return 0
	default:
		fmt.Fprintf(stderr, "fak knownbad: unknown subcommand %q (want record|match|claim|resolve|report)\n", argv[0])
		return 2
	}
}

// knownBadLedgerPath resolves the ledger to write/read: an explicit --ledger wins,
// otherwise the repo-root-relative fleet default.
func knownBadLedgerPath(override string) string {
	if strings.TrimSpace(override) != "" {
		return override
	}
	return filepath.Join(repoRoot(), knownbad.DefaultLedgerRel)
}

// knownBadDiscoverer resolves the discovered_by id when --by is not given: the
// FAK_AGENT_ID env if present, else the hostname, else "unknown".
func knownBadDiscoverer(by string) string {
	if s := strings.TrimSpace(by); s != "" {
		return s
	}
	if s := strings.TrimSpace(os.Getenv("FAK_AGENT_ID")); s != "" {
		return s
	}
	if h, err := os.Hostname(); err == nil && strings.TrimSpace(h) != "" {
		return h
	}
	return "unknown"
}

func runKnownBadRecord(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	fs := flag.NewFlagSet("knownbad record", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var trees stringList
	fs.Var(&trees, "tree", "repo-relative tree glob the failure covers (repeatable), e.g. internal/foo/**")
	reason := fs.String("reason", "", "failure class (required), e.g. build, test, lint")
	note := fs.String("note", "", "free-text note describing the known-bad")
	by := fs.String("by", "", "discoverer id (default: $FAK_AGENT_ID, else hostname)")
	failureHash := fs.String("failure-hash", "", "optional guardrsi failure hash to fold into the signature")
	ttl := fs.Int64("ttl", 0, "time-to-live in seconds (0 = no expiry)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+knownbad.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit the recorded row as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if len(trees) == 0 {
		fmt.Fprintln(stderr, "fak knownbad record: at least one --tree is required")
		return 2
	}
	if strings.TrimSpace(*reason) == "" {
		fmt.Fprintln(stderr, "fak knownbad record: --reason is required")
		return 2
	}
	rec := knownbad.NewRecord(*reason, []string(trees), *note, knownBadDiscoverer(*by), *failureHash, nowUnix, *ttl)
	if len(rec.TreeGlobs) == 0 {
		fmt.Fprintln(stderr, "fak knownbad record: --tree produced no valid repo-relative globs")
		return 2
	}
	path := knownBadLedgerPath(*ledger)
	if err := appendKnownBadRow(path, rec); err != nil {
		fmt.Fprintf(stderr, "fak knownbad record: %v\n", err)
		return 1
	}
	if *asJSON {
		return knownBadEmitJSON(stdout, stderr, rec)
	}
	fmt.Fprintf(stdout, "recorded known-bad %s reason=%s trees=%s -> %s\n",
		rec.Signature, rec.ReasonClass, strings.Join(rec.TreeGlobs, ","), path)
	return 0
}

func runKnownBadMatch(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	fs := flag.NewFlagSet("knownbad match", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var trees stringList
	fs.Var(&trees, "tree", "repo-relative tree glob (or file) to check (repeatable)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+knownbad.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if len(trees) == 0 {
		fmt.Fprintln(stderr, "fak knownbad match: at least one --tree is required")
		return 2
	}
	path := knownBadLedgerPath(*ledger)
	records, err := readKnownBadLedger(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak knownbad match: %v\n", err)
		return 1
	}
	matches := knownbad.Match(records, knownbad.Query{TreeGlobs: []string(trees)}, nowUnix)

	if *asJSON {
		out := struct {
			Schema  string            `json:"schema"`
			Matched bool              `json:"matched"`
			Query   []string          `json:"query"`
			Count   int               `json:"count"`
			Records []knownbad.Record `json:"records"`
		}{
			Schema:  knownbad.Schema,
			Matched: len(matches) > 0,
			Query:   []string(trees),
			Count:   len(matches),
			Records: matches,
		}
		if code := knownBadEmitJSON(stdout, stderr, out); code != 0 {
			return code
		}
	} else if len(matches) == 0 {
		fmt.Fprintf(stdout, "matched:false  no live known-bad signature intersects %s\n", strings.Join([]string(trees), ","))
	} else {
		fmt.Fprintf(stdout, "matched:true  %d live known-bad signature(s) intersect %s:\n", len(matches), strings.Join([]string(trees), ","))
		for _, m := range matches {
			fmt.Fprintf(stdout, "  %s reason=%s trees=%s by=%s note=%q\n",
				m.Signature, m.ReasonClass, strings.Join(m.TreeGlobs, ","), m.DiscoveredBy, m.Note)
		}
	}

	// Exit non-zero on a match so a worker/dispatcher can short-circuit in shell.
	if len(matches) > 0 {
		return 3
	}
	return 0
}

// reasonKnownBadAlreadyClaimed is the closed-vocabulary refusal a LOSING claim
// carries: a DIFFERENT agent already holds the exclusive lease electing it as this
// signature's sole fixer. Registered in dos.toml [reasons.KNOWN_BAD_ALREADY_CLAIMED]
// so the refusal is `dos check-reason`-verifiable, not free text. The loser is always
// handed the WINNER's identity (a pointer to the fixer), never a bare "refused".
const reasonKnownBadAlreadyClaimed = "KNOWN_BAD_ALREADY_CLAIMED"

// knownBadClaimResult is the claim verb's --json shape. On a win OK is true, Record is
// the stamped claim row, and Fixer is the claimant; on a loss OK is false, Reason is
// KNOWN_BAD_ALREADY_CLAIMED, and Fixer is the WINNER (the pointer the loser needs),
// with the underlying lease verdict carried as evidence.
type knownBadClaimResult struct {
	Schema    string                 `json:"schema"`
	OK        bool                   `json:"ok"`
	Signature string                 `json:"signature"`
	LeaseID   string                 `json:"lease_id"`
	Fixer     string                 `json:"fixer,omitempty"`
	Reason    string                 `json:"reason,omitempty"`
	Detail    string                 `json:"detail,omitempty"`
	Record    *knownbad.Record       `json:"record,omitempty"`
	Lease     *leaseref.FenceVerdict `json:"lease,omitempty"`
}

// runKnownBadClaim elects exactly one fixer for a live known-bad signature (W5,
// #2717). The signature is the sole positional argument (flags first). The exclusive
// lease at refs/fak/locks/knownbad-<sig> IS the exactly-one gate: AcquireFenced
// CAS-creates the ref when free, renews it for the same holder (an idempotent
// re-claim), and refuses a different live holder — so two racing claimants can both
// append-intend, but only one can own the ref. The ledger stamp is bookkeeping the
// winner writes AFTER winning; it never decides the election.
func runKnownBadClaim(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	fs := flag.NewFlagSet("knownbad claim", flag.ContinueOnError)
	fs.SetOutput(stderr)
	by := fs.String("by", "", "claimant id (default: $FAK_AGENT_ID, else hostname)")
	session := fs.String("session", "", "owning session id for lease-liveness reap (a dead claimant's lease is reaped by the session path)")
	ttl := fs.Int64("ttl", 0, "claim lease lifetime in seconds (0 = no expiry)")
	dir := fs.String("dir", "", "repo dir for the exclusive-lease store (default: git discovery from cwd)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+knownbad.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "fak knownbad claim: exactly one <signature> is required (flags first): fak knownbad claim [--by ID] [--ttl S] <signature>")
		return 2
	}
	sig := strings.TrimSpace(fs.Arg(0))
	leaseID := knownbad.LeaseID(sig)
	if leaseID == "" {
		fmt.Fprintf(stderr, "fak knownbad claim: %q is not a usable signature (no ref-safe content)\n", sig)
		return 2
	}

	// Find the signature's live record to learn the broken tree to lease over. A
	// signature with no live row cannot be claimed — never recorded, or already
	// resolved/expired, so there is no live failure to elect a fixer for.
	ledgerPath := knownBadLedgerPath(*ledger)
	records, err := readKnownBadLedger(ledgerPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak knownbad claim: %v\n", err)
		return 1
	}
	sigRec, found := knownbad.FindLatestLive(records, sig, nowUnix)
	if !found {
		fmt.Fprintf(stderr, "fak knownbad claim: no LIVE known-bad signature %s in %s (record it first, or it has been resolved)\n", sig, ledgerPath)
		return 2
	}

	claimant := knownBadDiscoverer(*by)
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	ctx := context.Background()
	req := leaseref.Record{
		ID:         leaseID,
		TreeGlobs:  append([]string(nil), sigRec.TreeGlobs...),
		Holder:     claimant,
		SessionID:  strings.TrimSpace(*session),
		TTLSeconds: *ttl,
	}
	// The exclusive lease is the exactly-one gate. A simultaneous CREATE loses the CAS
	// as LEASE_CONTENDED, which names no winner; retry a bounded number of times so the
	// loser re-reads the now-committed ref and resolves to a definitive LEASE_HELD
	// carrying the winner — turning an ambiguous race into a nameable fixer.
	var verdict leaseref.FenceVerdict
	for attempt := 0; attempt < 5; attempt++ {
		if _, verdict, err = store.AcquireFenced(ctx, req, time.Unix(nowUnix, 0)); err != nil {
			fmt.Fprintf(stderr, "fak knownbad claim: acquire exclusive lease: %v\n", err)
			return 1
		}
		if verdict.OK || verdict.Reason != leaseref.ReasonLeaseContended {
			break
		}
	}

	if !verdict.OK {
		// Lost the election. A different live holder refuses LEASE_HELD carrying the
		// winner; a simultaneous-CREATE race loses the CAS as LEASE_CONTENDED, which
		// carries NO holder — so re-read the ref to recover the winner's identity. The
		// loser must always get a pointer to the fixer, never a bare "refused".
		fixer := verdict.Holder
		if fixer == "" {
			if cur, ok, gerr := store.Get(ctx, leaseID); gerr == nil && ok {
				fixer = cur.Holder
			}
		}
		res := knownBadClaimResult{
			Schema:    knownbad.Schema,
			OK:        false,
			Signature: sig,
			LeaseID:   leaseID,
			Fixer:     fixer,
			Reason:    reasonKnownBadAlreadyClaimed,
			Detail:    verdict.Detail,
			Lease:     &verdict,
		}
		if *asJSON {
			if code := knownBadEmitJSON(stdout, stderr, res); code != 0 {
				return code
			}
		} else {
			fmt.Fprintf(stdout, "refused: known-bad %s already claimed by fixer %q (%s); park behind them — do not open a competing fix\n",
				sig, fixer, reasonKnownBadAlreadyClaimed)
		}
		return leaserefRefused
	}

	// Won the election: stamp the claimant onto the ledger as a superseding row. The
	// lease guarantees we are the sole writer of this claim, so the append is not a
	// race — but if it fails we surface an error rather than claim a silent success.
	claimRec := sigRec.WithClaim(claimant, nowUnix)
	if err := appendKnownBadRow(ledgerPath, claimRec); err != nil {
		fmt.Fprintf(stderr, "fak knownbad claim: won the exclusive lease but could not record the claim: %v\n", err)
		return 1
	}
	res := knownBadClaimResult{
		Schema:    knownbad.Schema,
		OK:        true,
		Signature: sig,
		LeaseID:   leaseID,
		Fixer:     claimant,
		Detail:    verdict.Detail,
		Record:    &claimRec,
		Lease:     &verdict,
	}
	if *asJSON {
		return knownBadEmitJSON(stdout, stderr, res)
	}
	fmt.Fprintf(stdout, "claimed: known-bad %s -> fixer %q owns the fix (exclusive lease %s over %s)\n",
		sig, claimant, leaseID, strings.Join(claimRec.TreeGlobs, ","))
	return 0
}

// reasonKnownBadNotWitnessed is the closed-vocabulary refusal a `resolve` carries when
// the fix has NO independent witness: the green over the broken tree (and/or the dos
// verify) did not pass, so the signature stays open and the parked fleet is NOT released.
// Registered in dos.toml [reasons.KNOWN_BAD_NOT_WITNESSED] so the refusal is
// `dos check-reason`-verifiable, not free text. Clearing a known-bad on a self-report is
// the exact failure W6 exists to forbid.
const reasonKnownBadNotWitnessed = "KNOWN_BAD_NOT_WITNESSED"

// witnessKindTests / witnessKindVerify are the two independent witness kinds the resolve
// gate accepts, matching the knownbad.Record.Witness vocabulary: a green `go test`/`fak
// affected` over the broken tree, or a `dos verify` binding the fixer's commit to the
// signature's tree.
const (
	witnessKindTests  = "tests"
	witnessKindVerify = "verify"
)

// knownBadWitnessResult is one witness run's outcome: whether the fix is proven (green),
// a one-line detail for the report, and the kind that graded it. It is the shape the
// injectable seam returns so the resolve verb never inspects raw exec output.
type knownBadWitnessResult struct {
	OK     bool
	Kind   string
	Detail string
}

// knownBadWitness is the INJECTABLE witness seam (swapped in tests, the same idiom as
// dispatchWitnessCommitAudit): it runs the requested independent witness over the
// signature's broken tree and reports whether the fix is proven. The default runs a real
// `go test` (witness=tests) or `dos verify` (witness=verify); a test stub returns a fixed
// verdict so the resolve verb is deterministic. It is FAIL-CLOSED: any error or non-green
// outcome reports OK=false, because an unproven witness must never release the fleet.
var knownBadWitness = runKnownBadWitness

// runKnownBadResolve flips a live known-bad signature open -> resolved ONLY on an
// independent witness (W6, #2718). The signature is the sole positional argument (flags
// first). It runs the witness FIRST over the signature's broken tree; a red or missing
// witness leaves the signature open and refuses (KNOWN_BAD_NOT_WITNESSED), reported as
// `not yet` with the failing witness. On a green witness it appends a superseding resolved
// row (which clears the W4 scope-hold on the next dispatch tick — Match drops a resolved
// row because it is not Live) AND releases the fixer's exclusive lease (W5), so a
// half-release can never leave the fleet stuck. The witness is the gate; a self-report
// never releases the fleet.
func runKnownBadResolve(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	fs := flag.NewFlagSet("knownbad resolve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	by := fs.String("by", "", "resolver id (default: $FAK_AGENT_ID, else hostname)")
	witnessKind := fs.String("witness", witnessKindTests, "independent witness of the fix: tests (green go test over the broken tree) | verify (dos verify on the fixer commit)")
	commit := fs.String("commit", "", "fixer commit sha the verify witness binds to the signature's tree (--witness verify)")
	dir := fs.String("dir", "", "repo dir for the exclusive-lease store (default: git discovery from cwd)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+knownbad.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "fak knownbad resolve: exactly one <signature> is required (flags first): fak knownbad resolve [--by ID] [--witness tests|verify] <signature>")
		return 2
	}
	sig := strings.TrimSpace(fs.Arg(0))
	kind := strings.TrimSpace(strings.ToLower(*witnessKind))
	if kind != witnessKindTests && kind != witnessKindVerify {
		fmt.Fprintf(stderr, "fak knownbad resolve: --witness %q is not a witness kind (want tests|verify)\n", *witnessKind)
		return 2
	}

	// Find the signature's live record: the tree to witness over, and whether a claim
	// (fixer lease) stands to release. A signature with no live row cannot be resolved —
	// never recorded, or already resolved/expired, so there is no open failure to close.
	ledgerPath := knownBadLedgerPath(*ledger)
	records, err := readKnownBadLedger(ledgerPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak knownbad resolve: %v\n", err)
		return 1
	}
	sigRec, found := knownbad.FindLatestLive(records, sig, nowUnix)
	if !found {
		fmt.Fprintf(stderr, "fak knownbad resolve: no LIVE known-bad signature %s in %s (record it first, or it is already resolved)\n", sig, ledgerPath)
		return 2
	}

	// THE GATE: run the independent witness over the broken tree BEFORE touching the
	// ledger. No green -> the signature stays open and the fleet stays parked. This is the
	// whole point of W6: a resolve is refused unless the fix is proven, never on a claim.
	wr := knownBadWitness(*dir, kind, sigRec.TreeGlobs, strings.TrimSpace(*commit))
	if !wr.OK {
		res := knownBadResolveResult{
			Schema:    knownbad.Schema,
			OK:        false,
			Signature: sig,
			Witness:   kind,
			Reason:    reasonKnownBadNotWitnessed,
			Detail:    wr.Detail,
			Trees:     sigRec.TreeGlobs,
		}
		if *asJSON {
			if code := knownBadEmitJSON(stdout, stderr, res); code != 0 {
				return code
			}
		} else {
			fmt.Fprintf(stdout, "not yet: known-bad %s stays open — %s witness did not pass (%s); land the fix, then resolve again\n",
				sig, kind, wr.Detail)
		}
		return leaserefRefused
	}

	// Witnessed green. Append the superseding resolved row FIRST: the ledger flip is what
	// the dispatcher re-reads each tick to clear the W4 scope-hold (Match drops a resolved
	// row), so a resolve that recorded the release is durable even if the lease drop below
	// hiccups. The resolve carries the witness kind as its evidence stamp.
	resolver := knownBadDiscoverer(*by)
	resolvedRec := sigRec.WithResolve(resolver, nowUnix, kind)
	if err := appendKnownBadRow(ledgerPath, resolvedRec); err != nil {
		fmt.Fprintf(stderr, "fak knownbad resolve: witnessed the fix but could not record the resolve: %v\n", err)
		return 1
	}

	// Release the fixer's exclusive lease (W5) so the tree is free for normal dispatch.
	// Dropping the hold (the ledger flip) WITHOUT dropping the lease would leave the fleet
	// half-stuck, so this is part of the same resolve. The release is best-effort AND
	// reported: an absent lease is an idempotent OK (nothing was claimed), and a live lease
	// held by someone else is surfaced as a warning rather than failing the resolve — the
	// signature is already witnessed-closed, the stuck lease is a separate operator action.
	leaseID := knownbad.LeaseID(sig)
	lease := releaseKnownBadLease(*dir, leaseID, resolver, nowUnix)

	res := knownBadResolveResult{
		Schema:    knownbad.Schema,
		OK:        true,
		Signature: sig,
		Witness:   kind,
		Detail:    wr.Detail,
		Trees:     resolvedRec.TreeGlobs,
		LeaseID:   leaseID,
		Record:    &resolvedRec,
		Lease:     lease,
	}
	if *asJSON {
		return knownBadEmitJSON(stdout, stderr, res)
	}
	fmt.Fprintf(stdout, "resolved: known-bad %s open -> resolved on a witnessed %s (%s); dropped fixer lease %s — held issues route as dispatchable on the next tick\n",
		sig, kind, wr.Detail, leaseID)
	if lease != nil && !lease.OK {
		fmt.Fprintf(stderr, "fak knownbad resolve: note: fixer lease %s not dropped (%s) — %s; the signature is resolved regardless\n",
			leaseID, lease.Reason, lease.Detail)
	}
	return 0
}

// knownBadResolveResult is the resolve verb's --json shape. On a witnessed resolve OK is
// true, Record is the stamped resolved row, and Lease carries the lease-release verdict;
// on a refused resolve OK is false, Reason is KNOWN_BAD_NOT_WITNESSED, and Detail names the
// witness that did not pass.
type knownBadResolveResult struct {
	Schema    string                 `json:"schema"`
	OK        bool                   `json:"ok"`
	Signature string                 `json:"signature"`
	Witness   string                 `json:"witness"`
	Reason    string                 `json:"reason,omitempty"`
	Detail    string                 `json:"detail,omitempty"`
	Trees     []string               `json:"trees,omitempty"`
	LeaseID   string                 `json:"lease_id,omitempty"`
	Record    *knownbad.Record       `json:"record,omitempty"`
	Lease     *leaseref.FenceVerdict `json:"lease,omitempty"`
}

// releaseKnownBadLease drops the fixer's exclusive lease (W5) at refs/fak/locks/<leaseID>
// as the release arm of resolve. It returns the fenced verdict (nil only when the lease id
// is unusable, which the resolve caller already validated upstream so it never is here).
// An absent lease is an idempotent OK; a live lease held by a different holder refuses
// STALE_LEASE (surfaced, not fatal — the signature is witnessed-closed regardless).
func releaseKnownBadLease(dir, leaseID, holder string, nowUnix int64) *leaseref.FenceVerdict {
	if leaseID == "" {
		return nil
	}
	store := leaseref.NewInDir(pathutil.ExpandTilde(dir))
	verdict, err := store.ReleaseFenced(context.Background(), leaseID, holder, 0, time.Unix(nowUnix, 0))
	if err != nil {
		// Infrastructure failure (git not runnable): report it as a non-OK verdict rather
		// than failing the already-witnessed resolve.
		return &leaseref.FenceVerdict{Reason: "LEASE_RELEASE_ERROR", Detail: err.Error()}
	}
	return &verdict
}

// runKnownBadWitness is the DEFAULT witness runner (the production behind knownBadWitness):
// it runs the requested independent witness over the signature's broken tree and reports
// whether the fix is proven. FAIL-CLOSED: any error or non-green outcome is OK=false, so an
// unproven witness cannot release the fleet. `tests` runs `go test` over the tree's
// package globs; `verify` runs `dos verify` binding the fixer commit to the tree.
func runKnownBadWitness(dir, kind string, treeGlobs []string, commit string) knownBadWitnessResult {
	root := strings.TrimSpace(dir)
	if root == "" {
		root = repoRoot()
	}
	switch kind {
	case witnessKindVerify:
		return runKnownBadVerifyWitness(root, treeGlobs, commit)
	default:
		return runKnownBadTestsWitness(root, treeGlobs)
	}
}

// runKnownBadTestsWitness runs `go test` over the broken tree's package globs (each
// normalized tree becomes ./<tree>/...) and reports green iff the run exits 0. An empty
// tree set cannot be witnessed (there is nothing to prove green over) -> not witnessed.
func runKnownBadTestsWitness(root string, treeGlobs []string) knownBadWitnessResult {
	pkgs := knownBadTreePackages(treeGlobs)
	if len(pkgs) == 0 {
		return knownBadWitnessResult{OK: false, Kind: witnessKindTests, Detail: "no repo-relative tree to run go test over"}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	args := append([]string{"test"}, pkgs...)
	cmd := exec.CommandContext(ctx, "go", args...)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		return knownBadWitnessResult{OK: false, Kind: witnessKindTests, Detail: fmt.Sprintf("go test %s not green: %v", strings.Join(pkgs, " "), knownBadFirstLine(out))}
	}
	return knownBadWitnessResult{OK: true, Kind: witnessKindTests, Detail: fmt.Sprintf("green go test over %s", strings.Join(pkgs, " "))}
}

// runKnownBadVerifyWitness runs `dos verify` binding the fixer commit to the signature's
// tree. Without a commit there is nothing to bind, so the witness cannot pass. Green iff
// the command exits 0.
func runKnownBadVerifyWitness(root string, treeGlobs []string, commit string) knownBadWitnessResult {
	if commit == "" {
		return knownBadWitnessResult{OK: false, Kind: witnessKindVerify, Detail: "--witness verify requires --commit <sha> to bind the fix to the tree"}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "dos", "verify", commit, "--workspace", root)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		return knownBadWitnessResult{OK: false, Kind: witnessKindVerify, Detail: fmt.Sprintf("dos verify %s not green: %v", commit, knownBadFirstLine(out))}
	}
	return knownBadWitnessResult{OK: true, Kind: witnessKindVerify, Detail: fmt.Sprintf("dos verify bound %s to the tree", commit)}
}

// knownBadTreePackages turns the normalized broken-tree globs into ./<tree>/... package
// patterns `go test` accepts, deduping. A tree is already normalized (no glob stars) by
// NewRecord, so this only wraps it as a recursive package pattern.
func knownBadTreePackages(treeGlobs []string) []string {
	seen := make(map[string]struct{}, len(treeGlobs))
	out := make([]string, 0, len(treeGlobs))
	for _, t := range knownbad.NormalizeAll(treeGlobs) {
		p := "./" + t + "/..."
		if _, dup := seen[p]; dup {
			continue
		}
		seen[p] = struct{}{}
		out = append(out, p)
	}
	return out
}

// knownBadFirstLine returns the first non-empty line of command output for a compact
// refusal detail (the full log is the operator's to read; the report names the head).
func knownBadFirstLine(out []byte) string {
	for _, line := range strings.Split(string(out), "\n") {
		if s := strings.TrimSpace(line); s != "" {
			return s
		}
	}
	return "no output"
}

// appendKnownBadRow appends one record as a JSONL line, creating the ledger's
// defaultKnownBadOperatorAfter is how long an UNCLAIMED signature may sit without an
// elected fixer before the blast card escalates from a muted status line to a surfaced
// operator page. It is a bounded window, not zero, so a just-discovered signature does
// not page before the fleet has had a dispatch tick to elect a fixer (W5); it is
// overridable with --operator-after for a tighter or looser fleet cadence.
const defaultKnownBadOperatorAfter = 15 * time.Minute

// runKnownBadReport folds the LIVE known-bad signatures into ONE operator blast card
// (W7, #2719) and posts it to the #blockers Slack surface. It is a READ/render surface:
// it reads the ledger + the live dos leases and reflects that state — it never decides a
// hold, elects a fixer, or resolves a signature. The affected count per signature is the
// direct-intersection floor: the live leases whose declared tree intersects the
// signature's tree (knownbad.TreesIntersect, the same containment W1/W3 use). Default
// posture is --dry-run-safe like the other feeders: a status-tier card never pages, only
// an operator-tier one (an unclaimed signature past --operator-after) does.
func runKnownBadReport(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	fs := flag.NewFlagSet("knownbad report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+knownbad.DefaultLedgerRel+")")
	leasesPath := fs.String("leases", "", "read the live lease set from a JSONL fixture instead of the dos lease ledger (for tests / offline render)")
	dir := fs.String("dir", "", "repo dir for the live dos lease read (default: git discovery from cwd)")
	repoURL := fs.String("repo-url", "", "repo base URL for the operator card's do-this-next link, e.g. https://github.com/owner/repo")
	operatorAfter := fs.Duration("operator-after", defaultKnownBadOperatorAfter, "how long an unclaimed signature may go without a fixer before the card escalates to an operator page")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_BLOCKERS_CHANNEL / .env.slack.local / #blockers)")
	token := fs.String("token", "", "override bot token (default: $FAK_BLOCKERS_TOKEN, then the scoreboard token)")
	asJSON := fs.Bool("json", false, "emit the folded signatures as JSON (implies no post)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if !parseFlags(fs, argv) {
		return 2
	}

	records, err := readKnownBadLedger(knownBadLedgerPath(*ledger))
	if err != nil {
		fmt.Fprintf(stderr, "fak knownbad report: %v\n", err)
		return 1
	}
	leases, err := reportLeaseSet(*leasesPath, *dir, nowUnix)
	if err != nil {
		fmt.Fprintf(stderr, "fak knownbad report: %v\n", err)
		return 1
	}

	live := knownbad.LiveRecords(records, nowUnix)
	sigs := make([]blockerpost.Signature, 0, len(live))
	for _, rec := range live {
		affected := countAffectedLeases(rec.TreeGlobs, leases)
		unclaimed := !rec.Claimed()
		overdue := unclaimed && knownBadOverdue(rec.DiscoveredAtUnix, nowUnix, *operatorAfter)
		sigs = append(sigs, blockerpost.Signature{
			ID:             rec.Signature,
			Reason:         rec.ReasonClass,
			Trees:          rec.TreeGlobs,
			Affected:       affected,
			Fixer:          rec.ClaimedBy,
			WitnessPending: !rec.Resolved(),
			NoFixerOverdue: overdue,
		})
	}

	if *asJSON {
		out := struct {
			Schema     string                  `json:"schema"`
			LiveCount  int                     `json:"live_count"`
			Signatures []blockerpost.Signature `json:"signatures"`
		}{Schema: knownbad.Schema, LiveCount: len(sigs), Signatures: sigs}
		return knownBadEmitJSON(stdout, stderr, out)
	}

	card := blockerpost.FoldBlast(sigs, strings.TrimSpace(*repoURL))
	card.Source = resolveBlockerSource(*source)
	return emitBlocker(stdout, stderr, card, *channel, *token, *dryRun)
}

// reportLeaseSet resolves the lease set the affected count folds over: a --leases JSONL
// fixture wins (offline / test render), otherwise the live dos lease ledger at --dir. A
// missing dir falls back to git discovery from cwd (empty string), the same default the
// blast estimator uses.
func reportLeaseSet(leasesPath, dir string, nowUnix int64) ([]blastradius.Lease, error) {
	if strings.TrimSpace(leasesPath) != "" {
		return readBlastLeases(leasesPath)
	}
	return liveBlastLeases(dir, time.Unix(nowUnix, 0))
}

// countAffectedLeases counts the live leases whose declared tree intersects the
// signature's tree — the direct-intersection floor for the blast card's affected count
// (knownbad.TreesIntersect, the same containment W1 matches on). It is deliberately the
// conservative floor, not the full W3 import-graph blast radius: the card is a surface,
// so it counts who DIRECTLY touches the broken tree without paying a `go list` per render.
func countAffectedLeases(sigTrees []string, leases []blastradius.Lease) int {
	n := 0
	for _, l := range leases {
		if knownbad.TreesIntersect(sigTrees, l.TreeGlobs) {
			n++
		}
	}
	return n
}

// knownBadOverdue reports whether an unclaimed signature discovered at discoveredAtUnix
// has gone longer than the operator window without a fixer. A non-positive window means
// "escalate immediately" (any unclaimed signature is overdue); a zero discovery time
// (a malformed row) is treated as not-overdue so a bad row never pages on its own.
func knownBadOverdue(discoveredAtUnix, nowUnix int64, window time.Duration) bool {
	if discoveredAtUnix <= 0 {
		return false
	}
	if window <= 0 {
		return true
	}
	return nowUnix-discoveredAtUnix >= int64(window.Seconds())
}

// parent directory on first write — the same append idiom the other fak ledgers
// use (see appendLedgerRow in cadence.go).
func appendKnownBadRow(path string, rec knownbad.Record) error {
	line, err := knownbad.MarshalLine(rec)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(line + "\n")
	return err
}

// readKnownBadLedger reads the ledger file and folds it to records. A missing
// ledger is not an error — it means nothing has been recorded yet (no match).
func readKnownBadLedger(path string) ([]knownbad.Record, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	return knownbad.ParseLedger(data), nil
}

// knownBadEmitJSON marshals v as indented JSON to stdout; a marshal failure is a
// rc-1 error on stderr.
func knownBadEmitJSON(stdout, stderr io.Writer, v any) int {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak knownbad: json: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(b))
	return 0
}
