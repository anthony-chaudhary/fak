package main

// fak knownbad -- the impure shell over the internal/knownbad fold core: the verbs
// the blast-radius containment epic (#2712) spine exposes.
//
//	fak knownbad record --tree internal/foo/** --reason build --note "..." [--ttl S]
//	    appends one fak.known-bad.v1 JSONL row to a fleet-visible ledger. Every row
//	    carries a BOUNDED default TTL (knownbad.DefaultRecordTTLSeconds) so a signature
//	    auto-EXPIRES (stops matching, hold auto-lifts) even if nobody resolves it — a
//	    live shared bug re-fires and re-records, a phantom just ages out. --ttl 0 opts a
//	    signature out of expiry for a genuinely durable failure (W8, #2720).
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
//	fak knownbad revoke <signature> --reason <why> [--by ID] [--json]
//	    retracts a live signature WITHOUT a witness (W8, #2720): the UNWITNESSED release
//	    valve resolve deliberately withholds. It is for a mis-recorded or stale signature
//	    (a flaky test read as a shared bug, a fix that landed quietly) that would
//	    otherwise scope-hold the fleet until its TTL lapses — revoke lifts the hold NOW.
//	    Appends a superseding revoked row (Match drops it, clearing the W4 hold next tick)
//	    and drops the fixer lease (W5), same release path as resolve; the ONLY difference
//	    is there is no witness gate, so --reason (the human justification) is required
//	    instead. A revoke is "this was never really a shared bug," a resolve is "the
//	    shared bug is proven gone." A claim/resolve/revoke against an already
//	    expired/revoked signature is refused with KNOWN_BAD_EXPIRED_OR_REVOKED.
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
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blastlease"
	"github.com/anthony-chaudhary/fak/internal/blastradius"
	"github.com/anthony-chaudhary/fak/internal/blockerpost"
	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
	"github.com/anthony-chaudhary/fak/internal/knownbad"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func cmdKnownBad(argv []string) {
	os.Exit(runKnownBad(os.Stdout, os.Stderr, argv, time.Now().Unix()))
}

func runKnownBad(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak knownbad: expected a subcommand (record|match|claim|resolve|revoke|report|correlate|compact)")
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
	case "revoke":
		return runKnownBadRevoke(stdout, stderr, argv[1:], nowUnix)
	case "report":
		return runKnownBadReport(stdout, stderr, argv[1:], nowUnix)
	case "correlate":
		return runKnownBadCorrelate(stdout, stderr, argv[1:], nowUnix)
	case "compact":
		return runKnownBadCompact(stdout, stderr, argv[1:], nowUnix)
	case "-h", "--help", "help":
		fmt.Fprintln(stderr, "fak knownbad: record | match | claim | resolve | revoke | report | correlate | compact  (fleet-wide known-bad signature ledger)")
		return 0
	default:
		fmt.Fprintf(stderr, "fak knownbad: unknown subcommand %q (want record|match|claim|resolve|revoke|report|correlate|compact)\n", argv[0])
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
	derivedFrom := fs.String("derived-from", "", "optional parent signature (or candidate id) this attempt was derived from — the genealogy edge that links a rejected attempt to the one it mutated")
	ttl := fs.Int64("ttl", knownbad.DefaultRecordTTLSeconds, "bounded time-to-live in seconds after which the signature auto-expires (stops matching, hold auto-lifts); pass 0 for a durable no-expiry signature")
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
	rec := knownbad.NewRecord(*reason, []string(trees), *note, knownBadDiscoverer(*by), *failureHash, nowUnix, *ttl).
		WithDerivedFrom(*derivedFrom)
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
	derived := ""
	if rec.Derived() {
		derived = " derived_from=" + rec.DerivedFrom
	}
	fmt.Fprintf(stdout, "recorded known-bad %s reason=%s trees=%s%s -> %s\n",
		rec.Signature, rec.ReasonClass, strings.Join(rec.TreeGlobs, ","), derived, path)
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

// reasonKnownBadExpiredOrRevoked is the closed-vocabulary refusal a `claim`/`resolve`
// carries when it acts on a signature that WAS recorded but is no longer live — its
// bounded TTL lapsed (expired), an operator revoked it, or it was already resolved. It
// is the W8 (#2720) safety-valve's structured "you are chasing a phantom" signal:
// distinct from the plain usage error for a signature that was NEVER recorded (a typo'd
// or made-up id), which stays a bare exit-2 because there is nothing to point the caller
// at. Registered in dos.toml [reasons.KNOWN_BAD_EXPIRED_OR_REVOKED] so the refusal is
// `dos check-reason`-verifiable, not free text.
const reasonKnownBadExpiredOrRevoked = "KNOWN_BAD_EXPIRED_OR_REVOKED"

// knownBadNotLiveResult is the --json envelope a `claim`/`resolve` emits when the target
// signature is not actionable: OK=false, the verb, and — when the signature WAS recorded
// but has since been retracted — the structured KNOWN_BAD_EXPIRED_OR_REVOKED reason and
// the terminal state ("expired" | "revoked" | "resolved") so the caller knows it is
// chasing a phantom, not a typo.
type knownBadNotLiveResult struct {
	Schema    string `json:"schema"`
	OK        bool   `json:"ok"`
	Verb      string `json:"verb"`
	Signature string `json:"signature"`
	Reason    string `json:"reason,omitempty"`
	State     string `json:"state,omitempty"`
	Detail    string `json:"detail,omitempty"`
}

// refuseKnownBadNotLive is the shared not-live gate both `claim` and `resolve` run after
// the live lookup misses. It draws the W8 (#2720) line between two failures a
// FindLatestLive miss conflates:
//
//   - the signature WAS recorded but is now expired/revoked/resolved — a structured
//     refuse (KNOWN_BAD_EXPIRED_OR_REVOKED, exit leaserefRefused) so a worker acting on a
//     stale id is told the signature aged out / was killed, not left to guess; and
//   - the signature was NEVER recorded (a typo, a made-up id) — a plain usage error
//     (exit 2), because there is nothing to point the caller at.
//
// It returns (exitCode, handled): handled=false means the signature is live and the caller
// proceeds; handled=true means this printed the outcome and the caller returns exitCode.
func refuseKnownBadNotLive(stdout, stderr io.Writer, verb, sig, ledgerPath string, records []knownbad.Record, nowUnix int64, asJSON bool) (int, bool) {
	if _, found := knownbad.FindLatestLive(records, sig, nowUnix); found {
		return 0, false
	}
	_, seen, state := knownbad.LatestState(records, sig, nowUnix)
	if !seen {
		fmt.Fprintf(stderr, "fak knownbad %s: no known-bad signature %s in %s (record it first, or check the signature)\n", verb, sig, ledgerPath)
		return 2, true
	}
	// Seen but not live: the signature aged out (TTL), an operator revoked it, or it was
	// already resolved. Refuse with the structured reason so the caller stops chasing it.
	detail := fmt.Sprintf("known-bad %s is %s — it no longer holds the fleet; there is nothing to %s", sig, state, verb)
	if asJSON {
		code := knownBadEmitJSON(stdout, stderr, knownBadNotLiveResult{
			Schema:    knownbad.Schema,
			OK:        false,
			Verb:      verb,
			Signature: sig,
			Reason:    reasonKnownBadExpiredOrRevoked,
			State:     state,
			Detail:    detail,
		})
		if code != 0 {
			return code, true
		}
	} else {
		fmt.Fprintf(stdout, "refused: %s (%s) — do not %s an aged-out or revoked signature; if the failure is live it re-fires and re-records\n",
			detail, reasonKnownBadExpiredOrRevoked, verb)
	}
	return leaserefRefused, true
}

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
	if code, handled := refuseKnownBadNotLive(stdout, stderr, "claim", sig, ledgerPath, records, nowUnix, *asJSON); handled {
		return code
	}
	sigRec, _ := knownbad.FindLatestLive(records, sig, nowUnix)

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
	if code, handled := refuseKnownBadNotLive(stdout, stderr, "resolve", sig, ledgerPath, records, nowUnix, *asJSON); handled {
		return code
	}
	sigRec, _ := knownbad.FindLatestLive(records, sig, nowUnix)

	// THE GATE: run the independent witness over the broken tree BEFORE touching the
	// ledger. No green -> the signature stays open and the fleet stays parked. This is the
	// whole point of W6: a resolve is refused unless the fix is proven, never on a claim.
	repoDir := pathutil.ExpandTilde(*dir)
	wr := knownBadWitness(repoDir, kind, sigRec.TreeGlobs, strings.TrimSpace(*commit))
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
	lease := releaseKnownBadLease(repoDir, leaseID, resolver, nowUnix)

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

// knownBadRevokeResult is the revoke verb's --json shape. On success OK is true, Record is
// the stamped revoked row, and Lease carries the fixer-lease release verdict; the reason is
// echoed as the human justification the retraction recorded.
type knownBadRevokeResult struct {
	Schema    string                 `json:"schema"`
	OK        bool                   `json:"ok"`
	Signature string                 `json:"signature"`
	Reason    string                 `json:"reason"`
	Trees     []string               `json:"trees,omitempty"`
	LeaseID   string                 `json:"lease_id,omitempty"`
	Record    *knownbad.Record       `json:"record,omitempty"`
	Lease     *leaseref.FenceVerdict `json:"lease,omitempty"`
}

// runKnownBadRevoke retracts a live known-bad signature WITHOUT a witness (W8, #2720) — the
// unwitnessed release valve resolve deliberately does not provide. The signature is the sole
// positional argument (flags first); --reason is the REQUIRED human justification ("was
// flaky, not shared", "wrong tree", "fix landed without a green"), because a revoke is a
// judgement the audit trail must be able to read. It is the escape hatch for a phantom
// signature: a mis-attribution, or a fix that landed quietly, would otherwise scope-hold the
// fleet until the TTL lapsed — revoke lifts the hold NOW. Like resolve it (1) appends a
// superseding revoked row (which Match drops because it is not Live, clearing the W4
// scope-hold on the next tick) and (2) releases the fixer's exclusive lease (W5), so a
// half-release can never leave the fleet stuck. The only difference from resolve is there is
// NO witness gate — that is the whole point: a revoke is "this was never really a shared
// bug," not "the shared bug is proven gone."
func runKnownBadRevoke(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	fs := flag.NewFlagSet("knownbad revoke", flag.ContinueOnError)
	fs.SetOutput(stderr)
	by := fs.String("by", "", "revoker id (default: $FAK_AGENT_ID, else hostname)")
	reason := fs.String("reason", "", "human justification for the retraction (required), e.g. \"flaky not shared\", \"wrong tree\"")
	dir := fs.String("dir", "", "repo dir for the exclusive-lease store (default: git discovery from cwd)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+knownbad.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "fak knownbad revoke: exactly one <signature> is required (flags first): fak knownbad revoke [--by ID] --reason <why> <signature>")
		return 2
	}
	if strings.TrimSpace(*reason) == "" {
		fmt.Fprintln(stderr, "fak knownbad revoke: --reason is required (a revoke is a judgement; the audit trail must record why)")
		return 2
	}
	sig := strings.TrimSpace(fs.Arg(0))

	ledgerPath := knownBadLedgerPath(*ledger)
	records, err := readKnownBadLedger(ledgerPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak knownbad revoke: %v\n", err)
		return 1
	}
	// Same not-live gate as claim/resolve: a signature that was never recorded is a plain
	// usage error; one already expired/revoked/resolved is a structured refuse (there is
	// nothing left to retract — it already stopped matching).
	if code, handled := refuseKnownBadNotLive(stdout, stderr, "revoke", sig, ledgerPath, records, nowUnix, *asJSON); handled {
		return code
	}
	sigRec, _ := knownbad.FindLatestLive(records, sig, nowUnix)

	// Append the superseding revoked row FIRST: the ledger flip is what the dispatcher
	// re-reads each tick to clear the W4 scope-hold (Match drops a revoked row), so a revoke
	// that recorded the release is durable even if the lease drop below hiccups. NO witness
	// runs — the revoke's authority is the operator's judgement, stamped as the reason.
	revoker := knownBadDiscoverer(*by)
	revokedRec := sigRec.WithRevoke(revoker, nowUnix, *reason)
	if err := appendKnownBadRow(ledgerPath, revokedRec); err != nil {
		fmt.Fprintf(stderr, "fak knownbad revoke: could not record the revoke: %v\n", err)
		return 1
	}

	// Release the fixer's exclusive lease (W5) so the tree is free for normal dispatch. Unlike
	// resolve — where the FIXER releases their OWN claim, so the caller IS the holder — a revoke
	// is an operator OVERRIDE that dissolves whoever else's stale claim held the signature. So
	// the release is fenced to the RECORDED holder (the claimed row's ClaimedBy), not the
	// revoker: the operator authoritatively hands the fixer's region back. An unclaimed
	// signature has no live lease, so ReleaseFenced returns the idempotent absent-OK. The revoke
	// is durable via the ledger flip above regardless of this best-effort lease drop.
	holder := strings.TrimSpace(sigRec.ClaimedBy)
	if holder == "" {
		holder = revoker
	}
	leaseID := knownbad.LeaseID(sig)
	repoDir := pathutil.ExpandTilde(*dir)
	lease := releaseKnownBadLease(repoDir, leaseID, holder, nowUnix)

	res := knownBadRevokeResult{
		Schema:    knownbad.Schema,
		OK:        true,
		Signature: sig,
		Reason:    strings.TrimSpace(*reason),
		Trees:     revokedRec.TreeGlobs,
		LeaseID:   leaseID,
		Record:    &revokedRec,
		Lease:     lease,
	}
	if *asJSON {
		return knownBadEmitJSON(stdout, stderr, res)
	}
	fmt.Fprintf(stdout, "revoked: known-bad %s open -> revoked (%s); dropped fixer lease %s — held issues route as dispatchable on the next tick\n",
		sig, strings.TrimSpace(*reason), leaseID)
	if lease != nil && !lease.OK {
		fmt.Fprintf(stderr, "fak knownbad revoke: note: fixer lease %s not dropped (%s) — %s; the signature is revoked regardless\n",
			leaseID, lease.Reason, lease.Detail)
	}
	return 0
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
	cmd := windowgate.CommandContext(ctx, "go", args...)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	// `go test` forks the toolchain + the compiled test binary — a descendant tree.
	// The dispatch helper only hides the window; it is not a tree kill. So on the
	// 5-minute deadline, bare ctx-cancel reaps only the `go` launcher and orphans the
	// test-binary subtree (#3106 defect class). Tree-kill on cancel + bound the reap.
	procguard.ConfigureProcessTreeCancel(cmd)
	cmd.WaitDelay = 10 * time.Second
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
	cmd := windowgate.CommandContext(ctx, "dos", "verify", commit, "--workspace", root)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
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
	leases, err := reportLeaseSet(*leasesPath, pathutil.ExpandTilde(*dir), nowUnix)
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
		return blastlease.Read(leasesPath)
	}
	return blastlease.Live(dir, time.Unix(nowUnix, 0))
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

// knownBadCompactResult is the JSON shape `fak knownbad compact --json` emits: the
// CompactStats reduction, the resolved ledger path, and whether the ledger was actually
// rewritten. Wrote is false for an already-minimal ledger (a clean no-op re-compact) and
// for a --dry-run, so a caller can tell a real GC from an inspection.
type knownBadCompactResult struct {
	Ledger string                `json:"ledger"`
	DryRun bool                  `json:"dry_run"`
	Wrote  bool                  `json:"wrote"`
	Stats  knownbad.CompactStats `json:"stats"`
}

// runKnownBadCompact folds the append-to-supersede ledger down to its minimal current
// state (#3471): it drops superseded and expired rows, keeps every live signature's latest
// row, and keeps a bounded tail of resolved/revoked history. It rewrites the ledger ONLY
// when the fold actually reduced it — an already-minimal ledger (or --dry-run) is left
// byte-untouched, so a re-compact is a clean no-op.
func runKnownBadCompact(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	fs := flag.NewFlagSet("knownbad compact", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+knownbad.DefaultLedgerRel+")")
	keepTerminal := fs.Int("keep-terminal", -1, "resolved/revoked signatures to retain, most-recently-retracted first (-1 = keep all audit history, 0 = live-only)")
	dryRun := fs.Bool("dry-run", false, "report the reduction without rewriting the ledger")
	asJSON := fs.Bool("json", false, "emit the compaction result as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak knownbad compact: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	path := knownBadLedgerPath(*ledger)
	records, err := readKnownBadLedger(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak knownbad compact: %v\n", err)
		return 1
	}
	kept, stats := knownbad.Compact(records, nowUnix, *keepTerminal)
	// The fold emits kept rows in original append order, so KeptRows == InputRows means
	// nothing was dropped and the rewrite would be byte-identical: skip it (clean no-op).
	res := knownBadCompactResult{Ledger: path, DryRun: *dryRun, Stats: stats}
	if stats.KeptRows != stats.InputRows && !*dryRun {
		if err := writeKnownBadLedger(path, kept); err != nil {
			fmt.Fprintf(stderr, "fak knownbad compact: %v\n", err)
			return 1
		}
		res.Wrote = true
	}
	if *asJSON {
		return knownBadEmitJSON(stdout, stderr, res)
	}
	note := ""
	if *dryRun {
		note = " (dry-run: ledger unchanged)"
	} else if !res.Wrote {
		note = " (already minimal: ledger unchanged)"
	}
	fmt.Fprintf(stdout, "compacted known-bad ledger %s: %d -> %d rows (superseded=%d expired=%d terminal_dropped=%d; live=%d terminal=%d kept)%s\n",
		path, stats.InputRows, stats.KeptRows, stats.SupersededDropped, stats.ExpiredDropped,
		stats.TerminalDropped, stats.LiveKept, stats.TerminalKept, note)
	return 0
}

// writeKnownBadLedger atomically rewrites the ledger with exactly records, one
// knownbad.MarshalLine row per line — the whole-file replacement the compact GC needs (the
// other verbs only ever append). An empty record set writes an empty file, collapsing a
// ledger with nothing worth keeping.
func writeKnownBadLedger(path string, records []knownbad.Record) error {
	var b strings.Builder
	for _, rec := range records {
		line, err := knownbad.MarshalLine(rec)
		if err != nil {
			return err
		}
		b.WriteString(line)
		b.WriteByte('\n')
	}
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return writeFileAtomic(path, []byte(b.String()), 0o644)
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

// knownBadFleetObsRel is the repo-relative fleet-observation feed the gateway appends to
// (via FAK_FLEET_OBS_PATH) and `correlate` folds by default — the docs/nightrun/*.jsonl
// idiom the ledger itself uses. It is the CROSS-trace input the per-trace LIVELOCK journal
// row cannot be: one line per real result-side trip, keyed by content-free failure hash.
const knownBadFleetObsRel = "docs/nightrun/fleet-observations.jsonl"

// readFleetObservations reads the fleet-observation JSONL feed into the guardrsi shape the
// correlator folds. A missing feed is not an error — it is simply zero observations (the
// nightrun has not run yet), so correlate reports no candidates rather than failing. A
// malformed line is skipped, not fatal, so one bad append never blinds the whole fold.
func readFleetObservations(path string) ([]guardrsi.FleetObservation, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []guardrsi.FleetObservation
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var o guardrsi.FleetObservation
		if err := json.Unmarshal([]byte(line), &o); err != nil {
			continue
		}
		out = append(out, o)
	}
	return out, nil
}

// runKnownBadCorrelate folds the cross-trace fleet-observation feed into known-bad
// candidates: one per failure hash that at least --k DISTINCT traces hit inside the --window
// (guardrsi.Correlate — a per-trace loop, k repeats from ONE trace, deliberately does not
// promote). It is READ-ONLY by default (fold + report); the two write toggles are explicit:
//
//	--record       appends one bounded-TTL known-bad ledger row per NEW candidate signature,
//	               skipping any signature already live in the ledger (the signature is a
//	               content hash, so re-correlating the same shared failure does not duplicate
//	               the row; a live shared bug re-fires and the existing row's TTL keeps it
//	               live, a phantom ages out).
//	--file-issues  creates or updates ONE deduped gh issue per candidate signature (Layer E),
//	               bumping an occurrence count in place rather than opening a duplicate.
//
// The clock is injected (nowUnix) so the fold, the liveness check, and the recorded rows are
// deterministic under test.
func runKnownBadCorrelate(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	fs := flag.NewFlagSet("knownbad correlate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	obsPath := fs.String("observations", "", "fleet-observation JSONL feed to fold (default: <root>/"+knownBadFleetObsRel+")")
	k := fs.Int("k", guardrsi.DefaultCorrelateK, "distinct-trace threshold at/above which a shared failure promotes to a candidate")
	window := fs.Int64("window", guardrsi.DefaultCorrelateWindowSecs, "correlation window in seconds; observations older than this at now do not count toward k")
	record := fs.Bool("record", false, "append a bounded-TTL known-bad ledger row for each NEW candidate signature (skips signatures already live)")
	ttl := fs.Int64("ttl", knownbad.DefaultRecordTTLSeconds, "TTL in seconds for recorded rows (0 = durable no-expiry)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+knownbad.DefaultLedgerRel+")")
	fileIssues := fs.Bool("file-issues", false, "create/update one deduped gh issue per candidate signature")
	repo := fs.String("repo", "", "gh repo (owner/name) for --file-issues; empty uses the gh-detected repo")
	limit := fs.Int("limit", 300, "max existing issues to fetch when deduping --file-issues")
	by := fs.String("by", "", "discoverer id stamped on recorded rows (default: $FAK_AGENT_ID, else hostname)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if !parseFlags(fs, argv) {
		return 2
	}

	path := strings.TrimSpace(*obsPath)
	if path == "" {
		path = filepath.Join(repoRoot(), knownBadFleetObsRel)
	}
	obs, err := readFleetObservations(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak knownbad correlate: %v\n", err)
		return 1
	}
	candidates := guardrsi.Correlate(obs, *k, *window, nowUnix)

	ledgerPath := knownBadLedgerPath(*ledger)
	var existingLedger []knownbad.Record
	if *record {
		existingLedger, err = readKnownBadLedger(ledgerPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak knownbad correlate: %v\n", err)
			return 1
		}
	}
	var existingIssues []dogfoodissues.Issue
	if *fileIssues {
		existingIssues, err = dogfoodissues.FetchExistingIssues(*repo, *limit)
		if err != nil {
			fmt.Fprintf(stderr, "fak knownbad correlate: fetch issues: %v\n", err)
			return 1
		}
	}
	discoverer := knownBadDiscoverer(*by)

	type outcome struct {
		Signature   string   `json:"signature"`
		FailureHash string   `json:"failure_hash"`
		ReasonClass string   `json:"reason_class"`
		TreeGlobs   []string `json:"tree_globs"`
		Distinct    int      `json:"distinct_traces"`
		Recorded    bool     `json:"recorded"`
		AlreadyLive bool     `json:"already_live"`
		IssueAction string   `json:"issue_action,omitempty"`
		IssueURL    string   `json:"issue_url,omitempty"`
	}
	outcomes := make([]outcome, 0, len(candidates))

	for _, cand := range candidates {
		note := fmt.Sprintf("fleet-correlated: %d distinct traces share this failure within %ds",
			cand.DistinctTraces, cand.WindowSecs)
		rec := knownbad.NewRecord(cand.ReasonClass, cand.TreeGlobs, note, discoverer, cand.FailureHash, nowUnix, *ttl)
		oc := outcome{
			Signature:   rec.Signature,
			FailureHash: cand.FailureHash,
			ReasonClass: rec.ReasonClass,
			TreeGlobs:   rec.TreeGlobs,
			Distinct:    cand.DistinctTraces,
		}
		if *record {
			if _, live := knownbad.FindLatestLive(existingLedger, rec.Signature, nowUnix); live {
				oc.AlreadyLive = true
			} else if err := appendKnownBadRow(ledgerPath, rec); err != nil {
				fmt.Fprintf(stderr, "fak knownbad correlate: record %s: %v\n", rec.Signature, err)
				return 1
			} else {
				oc.Recorded = true
				// Fold the just-appended row back in so a duplicate candidate this batch dedups too.
				existingLedger = append(existingLedger, rec)
			}
		}
		if *fileIssues {
			plan := buildKnownBadIssuePlan(rec, cand, existingIssues)
			oc.IssueAction = plan.Action
			rows := dogfoodissues.Sync([]dogfoodissues.PlanRow{plan}, *repo, []string{knownBadIssueLabel}, nil)
			if len(rows) == 1 {
				oc.IssueURL = rows[0].URL
			}
		}
		outcomes = append(outcomes, oc)
	}

	if *asJSON {
		out := struct {
			Schema       string    `json:"schema"`
			Observations int       `json:"observations"`
			K            int       `json:"k"`
			WindowSecs   int64     `json:"window_secs"`
			Count        int       `json:"count"`
			Candidates   []outcome `json:"candidates"`
		}{
			Schema:       knownbad.Schema,
			Observations: len(obs),
			K:            *k,
			WindowSecs:   *window,
			Count:        len(outcomes),
			Candidates:   outcomes,
		}
		return knownBadEmitJSON(stdout, stderr, out)
	}

	if len(outcomes) == 0 {
		fmt.Fprintf(stdout, "no fleet-correlated known-bad candidates (%d observation(s), k=%d, window=%ds)\n",
			len(obs), *k, *window)
		return 0
	}
	fmt.Fprintf(stdout, "%d fleet-correlated known-bad candidate(s) from %d observation(s):\n", len(outcomes), len(obs))
	for _, oc := range outcomes {
		fmt.Fprintf(stdout, "  %s reason=%s trees=%s distinct=%d",
			oc.Signature, oc.ReasonClass, strings.Join(oc.TreeGlobs, ","), oc.Distinct)
		switch {
		case oc.AlreadyLive:
			fmt.Fprint(stdout, " [already live]")
		case oc.Recorded:
			fmt.Fprint(stdout, " [recorded]")
		}
		if oc.IssueAction != "" {
			fmt.Fprintf(stdout, " issue:%s", oc.IssueAction)
			if oc.IssueURL != "" {
				fmt.Fprintf(stdout, " %s", oc.IssueURL)
			}
		}
		fmt.Fprintln(stdout)
	}
	return 0
}
