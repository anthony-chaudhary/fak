package main

// fak knownbad -- the impure shell over the internal/knownbad fold core: the two
// verbs the blast-radius containment epic (#2712) spine exposes.
//
//   fak knownbad record --tree internal/foo/** --reason build --note "..."
//       appends one fak.known-bad.v1 JSONL row to a fleet-visible ledger.
//   fak knownbad match  --tree internal/foo/bar.go [--json]
//       reports whether the requested tree intersects any LIVE (open, unexpired)
//       known-bad signature, printing the matching record(s). Exit is non-zero
//       (with --json, matched:false) when nothing matches, so a worker OR the
//       dispatcher can short-circuit before burning a cycle.
//   fak knownbad claim <signature> [--by ID] [--ttl S] [--json]
//       elects EXACTLY ONE fixer for a live signature by acquiring an EXCLUSIVE dos
//       lease over its broken tree (W5, #2717). The first claimant wins (exit 0) and
//       is stamped onto the ledger; a second, competing claimant is REFUSED (exit 3,
//       reason KNOWN_BAD_ALREADY_CLAIMED) and told the current fixer's identity so it
//       parks behind them instead of racing a second edit to the shared tree.
//
// All impurity lives here: the ledger read/write, the exclusive-lease store (git
// refs, via internal/leaseref), the clock (Unix seconds, injected as `nowUnix` so
// runKnownBad is deterministic under test), and flag parsing. The signature
// derivation, tree intersection, liveness, and the lease-id derivation are the pure
// core in internal/knownbad.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func cmdKnownBad(argv []string) {
	os.Exit(runKnownBad(os.Stdout, os.Stderr, argv, time.Now().Unix()))
}

func runKnownBad(stdout, stderr io.Writer, argv []string, nowUnix int64) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak knownbad: expected a subcommand (record|match|claim)")
		return 2
	}
	switch argv[0] {
	case "record":
		return runKnownBadRecord(stdout, stderr, argv[1:], nowUnix)
	case "match":
		return runKnownBadMatch(stdout, stderr, argv[1:], nowUnix)
	case "claim":
		return runKnownBadClaim(stdout, stderr, argv[1:], nowUnix)
	case "-h", "--help", "help":
		fmt.Fprintln(stderr, "fak knownbad: record | match | claim  (fleet-wide known-bad signature ledger)")
		return 0
	default:
		fmt.Fprintf(stderr, "fak knownbad: unknown subcommand %q (want record|match|claim)\n", argv[0])
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

// appendKnownBadRow appends one record as a JSONL line, creating the ledger's
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
