package main

// fak leaseref -- the operator-facing CLI surface over internal/leaseref, the
// CROSS-MACHINE LEASE VISIBILITY substrate (#825). internal/leaseref persists a
// lease record under refs/fak/locks/<id> so lease state rides ordinary git
// fetch/push between clones; this verb is the READ side that lets that state feed
// an admission decision:
//
//   fak leaseref live [--dir DIR]            -> JSON [{lane,lane_kind,tree}, ...]
//   fak leaseref list [--json] [--dir DIR]   -> the records under refs/fak/locks/*
//   fak leaseref reap [--dir DIR]            -> delete the expired (reapable) records
//
// `live` is the headline: it emits the non-expired records projected into the
// exact live_leases shape a dos_arbitrate-style admission kernel consumes, so an
// arbiter on machine B can SEE a lease machine A pushed (after an ordinary fetch)
// instead of being blind to it. The wiring an operator runs is, e.g.:
//
//   git fetch origin 'refs/fak/locks/*:refs/fak/locks/*'
//   dos arbitrate --lane <l> --tree <t> --leases "$(fak leaseref live)"
//
// HONEST BOUNDARY (kept in lockstep with the package doc): this is DISTRIBUTION /
// VISIBILITY, not atomic acquisition — it lets the arbiter see a cross-machine
// conflict, it does not arbitrate a same-fetch-window race. Documented in
// docs/cli-reference.md.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func cmdLeaseref(argv []string) { os.Exit(runLeaseref(os.Stdout, os.Stderr, argv)) }

func runLeaseref(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, leaserefUsage)
		return 2
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "live":
		return runLeaserefLive(stdout, stderr, rest)
	case "list":
		return runLeaserefList(stdout, stderr, rest)
	case "reap":
		return runLeaserefReap(stdout, stderr, rest)
	case "audit":
		return runLeaserefAudit(stdout, stderr, rest)
	case "acquire":
		return runLeaserefAcquire(stdout, stderr, rest)
	case "fence":
		return runLeaserefFence(stdout, stderr, rest)
	case "renew":
		return runLeaserefRenew(stdout, stderr, rest)
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, leaserefUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak leaseref: unknown subcommand %q\n%s\n", sub, leaserefUsage)
		return 2
	}
}

// leaserefRefused is the exit code for a STRUCTURED fence refusal (STALE_LEASE / LEASE_HELD /
// LEASE_CONTENDED / NO_LEASE) — distinct from 1 (a git/store failure) and 2 (a usage error),
// so a shell caller can branch `fak leaseref fence ... || halt-and-reacquire` while still
// telling a refusal apart from a broken git. The verdict JSON is emitted on stdout either way.
const leaserefRefused = 3

// repeatedString is a flag.Value that accumulates each `--tree GLOB` into a slice, so a lease
// can cover several trees without comma-splitting a glob that may itself contain a comma.
type repeatedString []string

func (r *repeatedString) String() string { return fmt.Sprint([]string(*r)) }
func (r *repeatedString) Set(v string) error {
	*r = append(*r, v)
	return nil
}

// fencedResult is the acquire/renew JSON shape: the deny-as-value verdict plus, on admit, the
// WRITTEN record (so the caller learns its assigned Generation — the fencing token it must
// present on every later write/fence). On a refusal Record is omitted.
type fencedResult struct {
	Verdict leaseref.FenceVerdict `json:"verdict"`
	Record  *leaseref.Record      `json:"record,omitempty"`
}

const leaserefUsage = `fak leaseref - cross-machine lease visibility (over internal/leaseref, #825)

  fak leaseref live [--dir DIR]
      Read the NON-EXPIRED records under refs/fak/locks/* and emit them as the
      dos_arbitrate live_leases JSON array [{lane,lane_kind,tree}, ...]. This is
      the source that makes a peer's lease (fetched into the local ref store)
      visible at admission. Pipe it: dos arbitrate ... --leases "$(fak leaseref live)".

  fak leaseref list [--json] [--dir DIR]
      List every record under refs/fak/locks/* (incl. expired), one per line with
      its LIVE/EXPIRED status; --json emits the raw records.

  fak leaseref reap [--dir DIR]
      Delete the expired (reapable) records — BOTH expired lock leases and expired
      guard-session descriptors under refs/fak/locks/*. A crashed holder's lapsed
      lease (or a crashed node's lapsed session) is bounded, not a permanent ghost.
      The delete is an ordinary ref delete that converges across clones the same way
      acquisition does.

  fak leaseref audit [--dir DIR]
      READ-ONLY staleness report over refs/fak/locks/*: list every lease, classify
      live-vs-expired against now, and emit the garden control-pane envelope
      (ok/verdict/reason) plus would_reap[] dry-run evidence: owner, lane, tree,
      age, TTL threshold, and the exact expiry comparison that selected the stale
      lease. Reaps NOTHING — verdict ACTION when an expired lease lingers is the
      signal to run 'fak leaseref reap'. This is the member 'fak garden' folds.

  fak leaseref acquire --id ID --holder H [--tree GLOB ...] [--ttl SEC] [--dir DIR]
      FENCED acquire (#906-C1): take the lease with a monotonic fencing token.
      Fresh -> generation 1; reaping an EXPIRED holder -> generation bumps (a
      transition); the SAME holder reacquiring a live lease -> a renew (generation
      kept). A DIFFERENT live holder is refused LEASE_HELD. Emits {verdict, record};
      the record carries the assigned 'generation' you must present to 'fence'.

  fak leaseref fence --id ID --holder H --generation N [--dir DIR]
      The GATE an agent runs BEFORE a write: is the lease you hold still current?
      Emits the fence verdict. STALE_LEASE means a newer holder was admitted while
      you were paused/dormant — halt and reacquire, never resume:
        fak leaseref fence --id L --holder $ME --generation $G || reacquire

  fak leaseref renew --id ID --holder H [--ttl SEC] [--dir DIR]
      Heartbeat: extend YOUR live lease's window WITHOUT bumping the generation. A
      lease taken over by a peer is refused STALE_LEASE; a lapsed/absent lease NO_LEASE.

This is VISIBILITY, not atomic acquisition across machines: it lets an arbiter SEE a
cross-machine conflict and does not arbitrate a same-fetch-window race. The fenced
acquire/renew DO enforce real SAME-HOST atomicity via an update-ref compare-and-swap.
Exit: 0 ok, 2 usage/parse error, 1 a git/store failure, 3 a structured fence refusal.`

func runLeaserefLive(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref live", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	leases, err := store.LiveLeases(context.Background(), time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref live: %v\n", err)
		return 1
	}
	return emitLeaserefJSON(stdout, stderr, leases, "live")
}

func runLeaserefList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	asJSON := fs.Bool("json", false, "emit the raw records as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	recs, err := store.List(context.Background())
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref list: %v\n", err)
		return 1
	}
	if *asJSON {
		if recs == nil {
			recs = []leaseref.Record{}
		}
		return emitLeaserefJSON(stdout, stderr, recs, "list")
	}
	now := time.Now()
	if len(recs) == 0 {
		fmt.Fprintln(stdout, "no leases under refs/fak/locks/*")
		return 0
	}
	for _, r := range recs {
		status := "LIVE"
		if r.Expired(now) {
			status = "EXPIRED"
		}
		fmt.Fprintf(stdout, "%-24s holder=%s tree=%v ttl=%ds %s\n", r.ID, r.Holder, r.TreeGlobs, r.TTLSeconds, status)
	}
	return 0
}

func runLeaserefReap(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref reap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	ctx := context.Background()
	now := time.Now()
	rc := 0

	// Reap BOTH ref kinds under refs/fak/locks/*: expired lock leases and expired session
	// descriptors. Each sweep is independent and best-effort — a failure on one kind is
	// reported but never suppresses the other. Store.Reap / ReapSessions delete only their
	// own kind (the namespace split), so a session is never mistaken for a lock lease.
	leases, lerr := store.Reap(ctx, now)
	if lerr != nil {
		fmt.Fprintf(stderr, "fak leaseref reap: leases: %v\n", lerr)
		rc = 1
	}
	sessions, serr := store.ReapSessions(ctx, now)
	if serr != nil {
		fmt.Fprintf(stderr, "fak leaseref reap: sessions: %v\n", serr)
		rc = 1
	}
	fmt.Fprintf(stdout, "reaped %d expired lease(s), %d expired session(s)\n", len(leases), len(sessions))
	return rc
}

// runLeaserefAudit is the READ-ONLY staleness reporter over refs/fak/locks/*: it lists every
// lease, classifies live-vs-expired against now, and emits the garden control-pane envelope
// (ok/verdict/reason) so the `fak garden` bundle can fold it. It REAPS NOTHING — deleting an
// expired lease stays the explicit `fak leaseref reap` verb, kept separate from this audit so a
// read-only garden tick never mutates the cross-machine lock state. ok is always true (reporting
// is the pass working); verdict is ACTION only when an expired lease lingers, the signal to reap.
func runLeaserefAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	asControlPane := fs.Bool("control-pane", false, "emit the garden control-pane envelope (the default for this verb)")
	_ = asControlPane // the audit verb only speaks the control-pane envelope; the flag is accepted for symmetry
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	recs, err := store.List(context.Background())
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref audit: %v\n", err)
		return 1
	}
	now := time.Now()
	var liveIDs, expiredIDs []string
	var liveRows, expiredRows []map[string]any
	for _, r := range recs {
		if r.Expired(now) {
			expiredIDs = append(expiredIDs, r.ID)
			expiredRows = append(expiredRows, leaserefAuditLeaseRow(r, now, true))
		} else {
			liveIDs = append(liveIDs, r.ID)
			liveRows = append(liveRows, leaserefAuditLeaseRow(r, now, false))
		}
	}
	verdict := "OK"
	reason := fmt.Sprintf("%d live lease(s), 0 expired under refs/fak/locks/*", len(liveIDs))
	if len(expiredIDs) > 0 {
		verdict = "ACTION"
		reason = fmt.Sprintf("%d live, %d EXPIRED lease(s) under refs/fak/locks/* (%s) — run `fak leaseref reap`",
			len(liveIDs), len(expiredIDs), strings.Join(expiredIDs, ", "))
	}
	env := map[string]any{
		"schema":        "fak.leaseref-audit-control-pane.v1",
		"ok":            true,
		"verdict":       verdict,
		"reason":        reason,
		"live_count":    len(liveIDs),
		"expired_count": len(expiredIDs),
		"expired_ids":   expiredIDs,
		"live":          liveRows,
		"would_reap":    expiredRows,
	}
	return emitLeaserefJSON(stdout, stderr, env, "audit")
}

func leaserefAuditLeaseRow(r leaseref.Record, now time.Time, expired bool) map[string]any {
	active := r.AcquiredAt
	if r.RenewedAt > active {
		active = r.RenewedAt
	}
	age := int64(0)
	if active > 0 {
		age = now.Unix() - active
		if age < 0 {
			age = 0
		}
	}
	expiresAt := int64(0)
	if r.TTLSeconds > 0 && active > 0 {
		expiresAt = active + r.TTLSeconds
	}
	reason := "TTL_LIVE"
	evidence := "ttl_seconds<=0 so the lease has no expiry threshold"
	if r.TTLSeconds > 0 {
		evidence = fmt.Sprintf("now_unix=%d >= active_unix=%d + ttl_seconds=%d (expires_at_unix=%d)",
			now.Unix(), active, r.TTLSeconds, expiresAt)
		if !expired {
			evidence = fmt.Sprintf("now_unix=%d < active_unix=%d + ttl_seconds=%d (expires_at_unix=%d)",
				now.Unix(), active, r.TTLSeconds, expiresAt)
		}
	}
	if expired {
		reason = "TTL_EXPIRED"
	}
	return map[string]any{
		"id":                    r.ID,
		"lane":                  leaserefAuditLane(r.ID),
		"owner":                 r.Holder,
		"holder":                r.Holder,
		"tree":                  append([]string(nil), r.TreeGlobs...),
		"age_seconds":           age,
		"age_threshold_seconds": r.TTLSeconds,
		"ttl_seconds":           r.TTLSeconds,
		"active_unix":           active,
		"acquired_unix":         r.AcquiredAt,
		"renewed_unix":          r.RenewedAt,
		"expires_at_unix":       expiresAt,
		"stale":                 expired,
		"reason":                reason,
		"evidence":              evidence,
	}
}

func leaserefAuditLane(id string) string {
	id = strings.TrimSpace(id)
	if !strings.HasPrefix(id, "resolve-") {
		return id
	}
	lane := strings.TrimPrefix(id, "resolve-")
	if i := strings.LastIndex(lane, "-"); i > 0 {
		if leaserefAuditAllDigits(lane[i+1:]) {
			lane = lane[:i]
		}
	}
	if lane == "" {
		return id
	}
	return lane
}

func leaserefAuditAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func runLeaserefAcquire(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref acquire", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	id := fs.String("id", "", "lease id (one safe ref segment under refs/fak/locks/)")
	holder := fs.String("holder", "", "holder identity (machine/session); required to fence a write")
	ttl := fs.Int64("ttl", 0, "lease lifetime in seconds (0 = no expiry)")
	var trees repeatedString
	fs.Var(&trees, "tree", "repo-relative tree glob this lease covers (repeatable)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *id == "" {
		fmt.Fprintln(stderr, "fak leaseref acquire: --id is required")
		return 2
	}
	store := leaseref.NewInDir(*dir)
	rec, v, err := store.AcquireFenced(context.Background(), leaseref.Record{
		ID:         *id,
		TreeGlobs:  trees,
		Holder:     *holder,
		TTLSeconds: *ttl,
	}, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref acquire: %v\n", err)
		return 1
	}
	out := fencedResult{Verdict: v}
	if v.OK {
		out.Record = &rec
	}
	if code := emitLeaserefJSON(stdout, stderr, out, "acquire"); code != 0 {
		return code
	}
	if !v.OK {
		return leaserefRefused
	}
	return 0
}

func runLeaserefFence(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref fence", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	id := fs.String("id", "", "lease id to fence against")
	holder := fs.String("holder", "", "the holder identity you hold the lease as")
	gen := fs.Int64("generation", 0, "the fencing token (generation) you were granted at acquire")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *id == "" {
		fmt.Fprintln(stderr, "fak leaseref fence: --id is required")
		return 2
	}
	store := leaseref.NewInDir(*dir)
	v, err := store.Fence(context.Background(), leaseref.Record{ID: *id, Holder: *holder, Generation: *gen}, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref fence: %v\n", err)
		return 1
	}
	if code := emitLeaserefJSON(stdout, stderr, v, "fence"); code != 0 {
		return code
	}
	if !v.OK {
		return leaserefRefused
	}
	return 0
}

func runLeaserefRenew(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref renew", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	id := fs.String("id", "", "lease id to renew")
	holder := fs.String("holder", "", "the holder identity that owns the lease")
	ttl := fs.Int64("ttl", 0, "new lifetime in seconds (0 = keep the lease's existing TTL)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *id == "" || *holder == "" {
		fmt.Fprintln(stderr, "fak leaseref renew: --id and --holder are required")
		return 2
	}
	store := leaseref.NewInDir(*dir)
	rec, v, err := store.Renew(context.Background(), *id, *holder, *ttl, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref renew: %v\n", err)
		return 1
	}
	out := fencedResult{Verdict: v}
	if v.OK {
		out.Record = &rec
	}
	if code := emitLeaserefJSON(stdout, stderr, out, "renew"); code != 0 {
		return code
	}
	if !v.OK {
		return leaserefRefused
	}
	return 0
}

func emitLeaserefJSON(stdout, stderr io.Writer, v any, sub string) int {
	if err := writeIndentedJSON(stdout, v); err != nil {
		fmt.Fprintf(stderr, "fak leaseref %s: encode json: %v\n", sub, err)
		return 1
	}
	return 0
}
