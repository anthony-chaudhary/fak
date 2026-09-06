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
	"github.com/anthony-chaudhary/fak/internal/loopdrive"
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
	case "liveness":
		return runLeaserefLiveness(stdout, stderr, rest)
	case "session-publish":
		return runLeaserefSessionPublish(stdout, stderr, rest)
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
	case "release":
		return runLeaserefRelease(stdout, stderr, rest)
	case "announce":
		return runLeaserefAnnounce(stdout, stderr, rest)
	case "announce-view":
		return runLeaserefAnnounceView(stdout, stderr, rest)
	case "sync":
		return runLeaserefSync(stdout, stderr, rest)
	case "drain":
		return runLeaserefDrain(stdout, stderr, rest)
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

  fak leaseref liveness [--session ME] [--summary] [--dir DIR]
      Classify each LIVE lease by its OWNING SESSION's liveness (#2164):
      self | peer-live | peer-dead | peer-unknown, keyed on the session descriptor
      heartbeat at refs/fak/locks/session-<id> — never the ephemeral acquiring pid.
      Emits the JSON ARRAY [{...record, node, liveness, reclaimable, evidence,
      evidence_kind}, ...]. evidence_kind is the machine-routable companion to the
      evidence sentence — no-session-binding | self-session | no-descriptor |
      terminal-stopped | heartbeat-lapsed | heartbeating — so a loop that wants to
      REPAIR picks a remedy without pattern-matching prose. FAIL-CLOSED:
      only a POSITIVELY dead session (lapsed heartbeat or terminal STOPPED) is
      reclaimable; a heartbeating peer's lane is never stolen, and a lease with no
      session binding (or no descriptor) is peer-unknown, not reclaimable.
      --session ME tags your own leases self.
      THE ARRAY IS THE DEFAULT and its rows keep their old keys, because the array
      is a live contract (tools/issue_dispatch.py's cross-machine lane gate iterates
      it and fails OPEN on any other top-level shape). The AGGREGATE (#5485) rides
      two compatible channels instead: a one-line liveness_coverage banner on
      STDERR on every default run, and --summary, which emits
      {schema, summary, leases} on stdout — summary = {total, by_class,
      by_evidence_kind, positive_evidence, liveness_coverage}, both histograms
      zero-filled over their closed vocabularies.
      HOW TO READ liveness_coverage: it is the fraction of live leases whose class
      rests on an OBSERVED input rather than on an absence. 0.0 with total > 0 is
      the case the field exists for — every row is an absence of evidence, which is
      far more often a WIRING DEFECT IN THIS OBSERVER (nothing on the write path
      publishes what the classification consumes, so every downstream verdict is
      uninformative) than a fleet that genuinely went unclassifiable at once.
      by_evidence_kind then names the remedy: all no-session-binding means acquirers
      are not passing --session, all no-descriptor means 'fak leaseref
      session-publish' is down. An EMPTY live set also reports 0.0 and is NOT that
      signal — read total first.

  fak leaseref session-publish --session S [--host H] [--state RUNNING] [--ttl SEC] [--dir DIR]
      Publish/refresh a lightweight session descriptor at refs/fak/locks/session-S
      so leases acquired with --session S have a heartbeat for 'liveness'. This is
      a side-ref update only, never a branch/HEAD mutation.

  fak leaseref list [--json] [--dir DIR]
      List every record under refs/fak/locks/* (incl. expired), one per line with
      its LIVE/EXPIRED status; --json emits the raw records.

  fak leaseref reap [--dir DIR]
      Delete the expired (reapable) records — expired lock leases, expired
      guard-session descriptors, and lapsed intent claims (see 'fak intent') under
      refs/fak/locks/*. A crashed holder's lapsed lease (or a crashed node's lapsed
      session) is bounded, not a permanent ghost.
      The delete is an ordinary ref delete that converges across clones the same way
      acquisition does.

  fak leaseref audit [--dir DIR]
      READ-ONLY staleness report over refs/fak/locks/*: list every lease, classify
      live-vs-expired against now, and emit the garden control-pane envelope
      (ok/verdict/reason) plus would_reap[] dry-run evidence: owner, lane, tree,
      age, the threshold that age was judged against, and the exact comparison
      that selected the stale lease. A lease carrying NO TTL (--ttl 0, the
      acquire default) cannot expire and so can never be reaped; those are
      judged by AGE instead — never by the holder's pid, which names a
      per-invocation CLI child and is dead even for a healthy lease — and
      reported under age_stale[] / age_stale_ids, whose remedy is
      'fak leaseref release ID', not the reaper. Reaps NOTHING — verdict ACTION
      is the signal to act. This is the member 'fak garden' folds.

  fak leaseref acquire --id ID --holder H [--session S] [--tree GLOB ...] [--ttl SEC] [--dir DIR] [--announce on|off|offline --announce-issue N --announce-repo OWNER/REPO]
      FENCED acquire (#906-C1): take the lease with a monotonic fencing token.
      Fresh -> generation 1; reaping an EXPIRED holder -> generation bumps (a
      transition); the SAME holder reacquiring a live lease -> a renew (generation
      kept). A DIFFERENT live holder is refused LEASE_HELD. Emits {verdict, record};
      the record carries the assigned 'generation' you must present to 'fence'.
      --session S binds the lease to its owning session descriptor so 'liveness'
      can classify it by heartbeat (#2164); a renew adopts a binding a legacy
      record lacked but never rebinds an existing one.
      When --holder is omitted and --session is given, the holder is MINTED as
      <node-id>/<session-id> (#2304): node-id is this machine's stable identity,
      hostname-keyed to the hardware catalog (experiments/benchmark/catalog.json).
      list/liveness/audit surface that node component; a legacy free-form holder
      classifies node-unknown, never an error.

  fak leaseref fence --id ID --holder H --generation N [--dir DIR]
      The GATE an agent runs BEFORE a write: is the lease you hold still current?
      Emits the fence verdict. STALE_LEASE means a newer holder was admitted while
      you were paused/dormant — halt and reacquire, never resume:
        fak leaseref fence --id L --holder $ME --generation $G || reacquire

  fak leaseref renew --id ID --holder H [--ttl SEC] [--dir DIR] [--announce on|off|offline --announce-issue N --announce-repo OWNER/REPO]
      Heartbeat: extend YOUR live lease's window WITHOUT bumping the generation. A
      lease taken over by a peer is refused STALE_LEASE; a lapsed/absent lease NO_LEASE.

  fak leaseref release --id ID --holder H [--generation N] [--force] [--dir DIR] [--announce on|off|offline --announce-issue N --announce-repo OWNER/REPO]
      The release twin of 'acquire': delete YOUR lease the moment the work is
      done instead of waiting out the TTL (a finished exclusive-lane lease stops
      stalling the fleet). Holder-checked and CAS-deleted: a live lease held by
      a DIFFERENT holder is refused STALE_LEASE, a wrong --generation likewise,
      and a ref that advanced under the delete is LEASE_CONTENDED. An absent
      lease is an idempotent OK; an EXPIRED record is releasable by anyone
      (single-id reap). --force skips the holder check (operator override).

  fak leaseref announce --issue N --id ID --holder H --action acquire|renew|release [--generation N] [--tree GLOB ...] [--ttl SEC] [--repo OWNER/REPO] [--dry-run]
      PLANE 2 (#2300): post a structured one-line announcement of a lease transition
      to a coordination issue's comments — the durable, human-visible backup channel
      for when a node can reach neither the git remote (plane 0) nor the dev server
      (plane 1). The comment carries a human summary line plus one fenced JSON line
      'announce-view' folds back. NEVER BLOCKS the underlying lease op: a gh failure is
      a WARNING and a 0 exit, never a refusal. --dry-run prints the body without posting.
      A comment is EVIDENCE, never a lock.

  fak leaseref announce-view --issue N [--repo OWNER/REPO] [--dir DIR]
      Read a coordination issue's comments and FOLD the announcements into the advisory
      held-set view: the latest announced record per lease id (a release drops it out),
      emitted as JSON [{lease_id,holder,generation,tree,ttl_seconds,action}, ...]. This
      is ADVISORY visibility only — never an admission input on its own.

  fak leaseref sync [--remote R] [--push-only|--fetch-only] [--dir DIR]
      CONVERGE the refs/fak/locks/* namespace with a remote (default origin): push
      the local records, then fetch the remote's — the manual refspec the docs used
      to quote, now a verb a multi-node loop can run every tick. Push runs FIRST
      and a failed push STOPS the sync, so the force-fetch never regresses a
      just-acquired local lease the remote has not seen. Deletions do not ride a
      refspec: a released/reaped lease converges on peers via TTL expiry plus their
      own 'reap'. Side refs only — no branch, HEAD, or tag ever moves.

  fak leaseref drain [--apply] [--remote R] [--dir DIR]
      DRAIN the expired guard-session descriptors on a remote (default origin) that
      the no-prune 'sync' deliberately leaves behind: for each PROVEN-EXPIRED
      refs/fak/locks/session-<id> it delete-pushes the one-sided :ref refspec — the
      targeted per-id convergence, never a blanket prune — so origin's expired
      backlog actually drains instead of being re-materialized on every fetch (#5358).
      DRY-RUN BY DEFAULT: with no --apply it reports which ids WOULD be delete-pushed
      and mutates nothing. --apply performs the live drain (opting into the pre-push
      bulk-deletion gate the hook reserves for this drainer) and reaps each drained id
      locally too, so this clone's own later sync push cannot resurrect it. A LIVE
      descriptor is never a target; the live drain of the real fleet remote is an
      explicit operator step, not an automatic sweep.
      Targets are proven expired from the LOCAL ref view, so to drain a remote's
      backlog first import it: 'fak leaseref sync --fetch-only' (or the glob fetch),
      then 'fak leaseref drain' (report) and 'fak leaseref drain --apply' (drain).

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

// leaserefLivenessSchema names the --summary envelope, in the same
// fak.<verb>.<vN> shape as the audit verb's control-pane schema.
const leaserefLivenessSchema = "fak.leaseref-liveness.v1"

// leaserefLivenessReport is the OPT-IN (--summary) shape: the aggregate beside the very
// rows it was folded from, so a reader never has to run the verb twice or trust that two
// invocations saw the same ref namespace.
type leaserefLivenessReport struct {
	Schema  string                     `json:"schema"`
	Summary leaseref.LivenessSummary   `json:"summary"`
	Leases  []leaseref.ClassifiedLease `json:"leases"`
}

// runLeaserefLiveness is the #2164 witness: each LIVE lease tagged by its owning
// session's liveness (self | peer-live | peer-dead | peer-unknown), keyed on the session
// descriptor heartbeat — never the ephemeral acquiring pid. reclaimable is true ONLY on
// peer-dead (a positively-dead session); everything else fails closed.
//
// #5485 wires the AGGREGATE (leaseref.SummarizeLiveness) onto this verb, and the shape
// choice is load-bearing. The per-row array has a blind spot the rows themselves cannot
// close: when nothing on the write path publishes the input this classification consumes,
// every row comes back peer-unknown and the output is a complete, well-formed,
// correctly-computed array in which every row is an ABSENCE of evidence — indistinguishable
// from a healthy read, though one is a fleet state and the other is a wiring defect in THIS
// observer that silently invalidates every verdict downstream. liveness_coverage is the
// field that tells them apart.
//
// It is NOT promoted to a top-level object by default, because the array is a live
// contract: tools/issue_dispatch.py's lease_ref_busy_lanes runs exactly this verb,
// json.loads its stdout and rejects any non-list with "unexpected leaseref liveness shape"
// — returning an EMPTY busy-lane set. That failure is silent and fails OPEN: the wave
// planner would stop seeing cross-machine leases and schedule onto lanes a live peer holds.
// A diagnostic must never be able to cause a lane collision, so the aggregate takes two
// channels that cannot: --summary (opt-in, stdout, the full envelope) and, on the default
// path where stdout is spoken for, a one-line banner on STDERR — which callers capture
// separately and, in issue_dispatch.py's case, read only on a non-zero exit.
func runLeaserefLiveness(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref liveness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	session := fs.String("session", "", "this agent's own session id (its leases classify 'self')")
	summary := fs.Bool("summary", false, "emit {schema, summary, leases} instead of the bare row array")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	rows, err := store.ClassifyLive(context.Background(), *session, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref liveness: %v\n", err)
		return 1
	}
	// One fold over the rows just classified — never a second read of the ref namespace,
	// so the aggregate can never disagree with the rows printed beside it.
	sum := leaseref.SummarizeLiveness(rows)
	if *summary {
		return emitLeaserefJSON(stdout, stderr, leaserefLivenessReport{
			Schema:  leaserefLivenessSchema,
			Summary: sum,
			Leases:  rows,
		}, "liveness")
	}
	code := emitLeaserefJSON(stdout, stderr, rows, "liveness")
	fmt.Fprintln(stderr, leaserefCoverageBanner(sum))
	return code
}

// leaserefCoverageBanner renders the one-line stderr companion to the default array: the
// coverage number, and — in the one case that matters — what it means and who fixes it.
//
// It is a pure function of the summary so a test can assert the sentence without a repo,
// and it says the remedy rather than only the ratio because the two absence kinds route to
// DIFFERENT owners: no-session-binding is the acquirer (waiting never helps), no-descriptor
// is the publisher. An empty live set is called out explicitly, since its 0.0 is an
// undefined ratio reported as the zero value and NOT the wiring signal.
func leaserefCoverageBanner(s leaseref.LivenessSummary) string {
	const prefix = "fak leaseref liveness: "
	if s.Total == 0 {
		return prefix + "0 live lease(s); liveness_coverage is undefined for an empty live set (reported 0.0) and is NOT the wiring signal. --summary emits the aggregate as JSON."
	}
	head := fmt.Sprintf("%sliveness_coverage=%.2f (%d/%d live lease(s) classified on OBSERVED evidence); --summary emits the aggregate as JSON",
		prefix, s.Coverage, s.PositiveEvidence, s.Total)
	if s.PositiveEvidence > 0 {
		return head + "."
	}
	var remedies []string
	if n := s.ByEvidenceKind[leaseref.EvidenceNoBinding]; n > 0 {
		remedies = append(remedies, fmt.Sprintf("%d %s (the ACQUIRER never bound a session — pass --session at acquire; waiting never helps)",
			n, leaseref.EvidenceNoBinding))
	}
	if n := s.ByEvidenceKind[leaseref.EvidenceNoDescriptor]; n > 0 {
		remedies = append(remedies, fmt.Sprintf("%d %s (the PUBLISHER is absent or died — start/repair `fak leaseref session-publish`)",
			n, leaseref.EvidenceNoDescriptor))
	}
	warn := head + ". WARNING: every live lease rests on an ABSENCE of evidence — far more often a WIRING DEFECT IN THIS OBSERVER, which leaves every verdict above uninformative, than a fleet that genuinely went unclassifiable at once"
	if len(remedies) > 0 {
		warn += "; by_evidence_kind: " + strings.Join(remedies, ", ")
	}
	return warn + "."
}

func runLeaserefSessionPublish(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref session-publish", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	session := fs.String("session", "", "session id to publish under refs/fak/locks/session-<id>")
	host := fs.String("host", "", "host/node label (default: os hostname)")
	state := fs.String("state", "RUNNING", "session PCB state")
	ttl := fs.Int64("ttl", 0, "descriptor lifetime in seconds (0 = no expiry)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *session == "" {
		fmt.Fprintln(stderr, "fak leaseref session-publish: --session is required")
		return 2
	}
	if *host == "" {
		if h, err := os.Hostname(); err == nil {
			*host = h
		}
	}
	now := time.Now()
	desc := leaseref.SessionDescriptor{
		ID:        *session,
		Host:      *host,
		PCBState:  strings.TrimSpace(*state),
		UpdatedAt: now.Unix(),
		TTLSecs:   *ttl,
	}
	if desc.PCBState == "" {
		desc.PCBState = "RUNNING"
	}
	store := leaseref.NewInDir(*dir)
	if _, err := store.PublishSession(context.Background(), desc); err != nil {
		fmt.Fprintf(stderr, "fak leaseref session-publish: %v\n", err)
		return 1
	}
	return emitLeaserefJSON(stdout, stderr, desc, "session-publish")
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
		// Surface the node component when the holder carries the
		// <node-id>/<session-id> convention (#2304); a legacy free-form
		// holder shows unchanged.
		node := ""
		if hi := leaseref.ParseHolder(r.Holder); hi.Structured() {
			node = " node=" + hi.Node
		}
		fmt.Fprintf(stdout, "%-24s holder=%s%s tree=%v ttl=%ds %s\n", r.ID, r.Holder, node, r.TreeGlobs, r.TTLSeconds, status)
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

	// Reap ALL THREE ref kinds under refs/fak/locks/*: expired lock leases, expired session
	// descriptors, and lapsed intent claims (#2155). Each sweep is independent and
	// best-effort — a failure on one kind is reported but never suppresses the others.
	// Store.Reap / ReapSessions / ReapIntents delete only their own kind (the namespace
	// split), so one kind is never mistaken for another.
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
	intents, ierr := store.ReapIntents(ctx, now)
	if ierr != nil {
		fmt.Fprintf(stderr, "fak leaseref reap: intents: %v\n", ierr)
		rc = 1
	}
	// Fourth sweep (#5348): the filesystem-side orphan .lock reaper, completing the
	// ref-side Reap/ReapSessions/ReapIntents trio. A holder killed mid-CAS leaves a
	// ghost <git-common-dir>/refs/fak/locks/*.lock that git never removes; ReapLockFiles
	// deletes only a .lock older than the lease TTL it guards (maxAge 0 => the 2400 s
	// DefaultLockFileMaxAge), keeps a fresh (possibly-live) CAS, and fails closed on a
	// future-dated mtime. Best-effort like the sweeps above: a failure is reported but
	// never suppresses the others.
	orphanLocks, keptLocks, kerr := store.ReapLockFiles(ctx, now, 0)
	if kerr != nil {
		fmt.Fprintf(stderr, "fak leaseref reap: locks: %v\n", kerr)
		rc = 1
	}
	fmt.Fprintf(stdout, "reaped %d expired lease(s), %d expired session(s), %d lapsed intent(s), %d orphan lock(s) (%d live lock(s) kept)\n",
		len(leases), len(sessions), len(intents), len(orphanLocks), len(keptLocks))
	return rc
}

// runLeaserefAudit is the READ-ONLY staleness reporter over refs/fak/locks/*: it lists every
// lease, classifies live-vs-expired against now, and emits the garden control-pane envelope
// (ok/verdict/reason) so the `fak garden` bundle can fold it. It REAPS NOTHING — deleting an
// expired lease stays the explicit `fak leaseref reap` verb, kept separate from this audit so a
// read-only garden tick never mutates the cross-machine lock state. ok is always true (reporting
// is the pass working); verdict is ACTION when there is something to act on.
//
// THREE CONDITIONS TRIP ACTION, and they do not share a remedy. A TTL-expired lease and an
// expired session descriptor are both collected by `fak leaseref reap`. The third —
// a TTL-LESS lease past the age floor (leaserefNoTTLStaleAgeS) — is not: Record.Expired
// short-circuits false at ttl<=0, so it never enters Live's expired partition and Reap can
// never delete it. Before this rung such a lease was reported `stale=false, TTL_LIVE` forever
// while the lane it names stayed permanently refused, which is why the fleet accumulated
// 18 wedged lanes with nothing anywhere reporting them. It is now counted and named on its
// own keys (age_stale_count / age_stale_ids / age_stale) with the remedy that works,
// `fak leaseref release <id>`, spelled out in the reason.
//
// THE RESIDUAL, stated rather than hidden: internal/gardenbundle maps this member to ActReap,
// so an ACTION driven only by age-stale ghosts still makes the garden tick run the reaper,
// which will collect none of them and report reaping 0. The reason string says so in words.
// Making them collectable is a lease-DELETING change to the reaper and is deliberately not
// bundled with a read-only detector.
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
	ctx := context.Background()
	recs, err := store.List(ctx)
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref audit: %v\n", err)
		return 1
	}
	now := time.Now()
	// store.List EXCLUDES session descriptors (refs/fak/locks/session-*) by the
	// leaseref namespace split (internal/leaseref/session.go), so a leases-only
	// audit is structurally blind to a stale SESSION: it would report OK while the
	// fleet's session refs are TTL-expired, so the garden tick's stale_leases member
	// never sets Perform=true and performGardenTick's ActReap case — which already
	// calls store.ReapSessions — is never reached. Fold expired session descriptors
	// in so the verdict trips on them too. A read failure is handled the same way as
	// the lease read above: report to stderr and return 1.
	liveDescriptors, expiredDescriptorIDs, serr := store.LiveSessions(ctx, now)
	if serr != nil {
		fmt.Fprintf(stderr, "fak leaseref audit: %v\n", serr)
		return 1
	}
	var liveIDs, expiredIDs, ageStaleIDs []string
	var liveRows, expiredRows, ageStaleRows []map[string]any
	for _, r := range recs {
		if r.Expired(now) {
			expiredIDs = append(expiredIDs, r.ID)
			expiredRows = append(expiredRows, leaserefAuditLeaseRow(r, now, true))
			continue
		}
		row := leaserefAuditLeaseRow(r, now, false)
		liveIDs = append(liveIDs, r.ID)
		liveRows = append(liveRows, row)
		// A TTL-less lease past the age floor is un-expired by the TTL rule and therefore
		// still counted live — the live/expired partition keeps meaning exactly what it
		// always meant — but it is now ALSO reported as stale by age. It is deliberately
		// kept OUT of would_reap: `fak leaseref reap` provably cannot collect it (Reap
		// deletes only Live's expired partition, which a ttl<=0 record can never enter),
		// and a dry-run that promised a deletion the reaper will not perform would be a
		// worse lie than the silence this rung replaces.
		if s, _ := row["stale"].(bool); s {
			ageStaleIDs = append(ageStaleIDs, r.ID)
			ageStaleRows = append(ageStaleRows, row)
		}
	}
	verdict := "OK"
	reason := fmt.Sprintf("%d live lease(s), 0 expired under refs/fak/locks/*", len(liveIDs))
	if len(expiredIDs) > 0 || len(expiredDescriptorIDs) > 0 || len(ageStaleIDs) > 0 {
		verdict = "ACTION"
		parts := []string{fmt.Sprintf("%d live, %d EXPIRED lease(s)", len(liveIDs), len(expiredIDs))}
		if len(expiredIDs) > 0 {
			parts[0] += fmt.Sprintf(" (%s)", strings.Join(expiredIDs, ", "))
		}
		if len(expiredDescriptorIDs) > 0 {
			parts = append(parts, fmt.Sprintf("%d EXPIRED session descriptor(s)", len(expiredDescriptorIDs)))
		}
		reason = strings.Join(parts, ", ") + " under refs/fak/locks/* — run `fak leaseref reap`"
		if len(ageStaleIDs) > 0 {
			// Named separately from the reapable set, with the remedy that actually works.
			// The reaper is not it, and saying so here is the whole point of the rung.
			reason += fmt.Sprintf("; %d TTL-less lease(s) stale by age >= %ds (%s) — `reap` cannot collect these, release each with `fak leaseref release <id>`",
				len(ageStaleIDs), leaserefNoTTLStaleAgeS, strings.Join(ageStaleIDs, ", "))
		}
	}
	env := map[string]any{
		"schema":                   "fak.leaseref-audit-control-pane.v1",
		"ok":                       true,
		"verdict":                  verdict,
		"reason":                   reason,
		"live_count":               len(liveIDs),
		"expired_count":            len(expiredIDs),
		"expired_ids":              expiredIDs,
		"live":                     liveRows,
		"would_reap":               expiredRows,
		"live_descriptor_count":    len(liveDescriptors),
		"expired_descriptor_count": len(expiredDescriptorIDs),
		"expired_descriptor_ids":   expiredDescriptorIDs,
		// The age rung's own readout, strictly additive: every key above keeps its
		// pre-existing meaning and value.
		"age_stale_count":          len(ageStaleIDs),
		"age_stale_ids":            ageStaleIDs,
		"age_stale":                ageStaleRows,
		"no_ttl_stale_age_seconds": leaserefNoTTLStaleAgeS,
	}
	return emitLeaserefJSON(stdout, stderr, env, "audit")
}

// leaserefNoTTLStaleAgeS is the AGE bound the audit judges a TTL-LESS lease by, in
// seconds. It is the second half of a promise this row already made and did not keep:
// the row has always emitted an `age_seconds` and an `age_threshold_seconds`, but the
// threshold was just an echo of Record.TTLSeconds — which is 0 on every lease taken
// through `fak leaseref acquire` (its --ttl default is "0 = no expiry"). So the one
// comparison the field names could never fire, and a TTL-less lease reported
// `stale=false, reason=TTL_LIVE` no matter how old it was. That is not a cosmetic gap:
// Record.Expired short-circuits false at ttl<=0, Live therefore never puts such a record
// in its expired partition, and Reap — which deletes only what Live called expired —
// can never collect it, so the lane it names is refused for the life of the repository
// with nothing anywhere reporting it. Measured on this workspace's own lane journal:
// ttl null on 60 of 60 acquires, 60 ACQUIRE against 42 RELEASE, so 18 lanes wedged, the
// newest of them over a week old (docs/notes/DOWNSTREAM-REVIEW-VERIFIED-2026-08-05.md §4).
//
// WHY AGE, AND EXPLICITLY NOT PID. A lease record carries no pid at all, and that is
// deliberate: the acquiring process is a per-invocation CLI child that dies almost
// immediately, so "its pid is dead" is true of a perfectly healthy lease and a pid probe
// would report every live lane in the fleet as abandoned (internal/leaseref/liveness.go
// states the same rule for the session classifier). Age against a floor is the
// conservative proxy that has no such failure mode — the identical argument
// internal/leaseref/lockfile.go already makes for orphaned .lock FILES, applied here to
// the lease RECORD it never covered.
//
// WHY 24 HOURS. It must sit far above every legitimate un-renewed hold. The longest one
// in the tree is a dispatch worker's lane lease at WorkerTimeoutS + LeaseTTLMarginS
// (~40 min), and anything with a real TTL is judged by that TTL and never reaches this
// rung at all. A day is ~36x the longest legitimate hold, well under the observed
// week-plus wedge, and a holder still working after a day without a single renew has
// nothing left that can distinguish it from a ghost.
const leaserefNoTTLStaleAgeS = int64(24 * 60 * 60)

// The closed reason vocabulary of an audit row — the `--json` contract a control pane
// routes on, in the same shape as leaseref's Reason*/Liveness* families. TTL_* are the
// pre-existing pair (unchanged, byte for byte, for any lease carrying a TTL); NO_TTL_*
// are the age rung, reached only when ttl_seconds <= 0.
const (
	leaserefReasonTTLLive    = "TTL_LIVE"            // has a TTL, not yet lapsed
	leaserefReasonTTLExpired = "TTL_EXPIRED"         // has a TTL and it lapsed: reapable
	leaserefReasonNoTTLYoung = "NO_TTL_YOUNG"        // no TTL, but too young (or undatable) to call abandoned
	leaserefReasonNoTTLStale = "NO_TTL_AGE_EXCEEDED" // no TTL and older than the age floor: abandoned
)

// leaserefAuditLeaseRow renders one lease as an audit row: who holds it, how much age it
// has accumulated, the threshold that age was judged against, and the closed
// reason/evidence pair naming exactly which comparison decided.
//
// TWO RULES, because a lease need not carry a TTL. `expired` is the caller's TTL verdict
// (leaseref.Record.Expired) and owns every lease with ttl_seconds > 0 — those rows are
// unchanged. A lease with ttl_seconds <= 0 has no expiry at all, so it falls to the age
// rung: stale once its age reaches leaserefNoTTLStaleAgeS, and NEVER on the pid or the
// holder string (see that const for why a pid probe is the wrong test).
//
// FAILS CLOSED on unknown age: a record with no acquired/renewed stamp at all has no age
// to compare, so it stays not-stale rather than being called abandoned on absent evidence
// — the same posture ClassifyLiveness takes toward a missing session descriptor.
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
	stale := expired
	threshold := r.TTLSeconds
	reason := leaserefReasonTTLLive
	evidence := ""
	switch {
	case r.TTLSeconds > 0:
		evidence = fmt.Sprintf("now_unix=%d < active_unix=%d + ttl_seconds=%d (expires_at_unix=%d)",
			now.Unix(), active, r.TTLSeconds, expiresAt)
		if expired {
			reason = leaserefReasonTTLExpired
			evidence = fmt.Sprintf("now_unix=%d >= active_unix=%d + ttl_seconds=%d (expires_at_unix=%d)",
				now.Unix(), active, r.TTLSeconds, expiresAt)
		}
	case active <= 0:
		threshold = leaserefNoTTLStaleAgeS
		reason = leaserefReasonNoTTLYoung
		evidence = "ttl_seconds<=0 (no expiry) and the record carries no acquired/renewed stamp, so its age is unknown — fails closed to not stale"
	default:
		threshold = leaserefNoTTLStaleAgeS
		if age >= threshold {
			stale = true
			reason = leaserefReasonNoTTLStale
			evidence = fmt.Sprintf("ttl_seconds<=0 (no expiry) and age_seconds=%d >= age_threshold_seconds=%d (now_unix=%d, active_unix=%d) — judged by age, never by the holder's pid",
				age, threshold, now.Unix(), active)
			break
		}
		reason = leaserefReasonNoTTLYoung
		evidence = fmt.Sprintf("ttl_seconds<=0 (no expiry) and age_seconds=%d < age_threshold_seconds=%d (now_unix=%d, active_unix=%d) — too young to call abandoned",
			age, threshold, now.Unix(), active)
	}
	return map[string]any{
		"id":                    r.ID,
		"lane":                  leaserefAuditLane(r.ID),
		"owner":                 r.Holder,
		"holder":                r.Holder,
		"node":                  r.HolderNode(),
		"tree":                  append([]string(nil), r.TreeGlobs...),
		"age_seconds":           age,
		"age_threshold_seconds": threshold,
		"ttl_seconds":           r.TTLSeconds,
		"active_unix":           active,
		"acquired_unix":         r.AcquiredAt,
		"renewed_unix":          r.RenewedAt,
		"expires_at_unix":       expiresAt,
		"stale":                 stale,
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
	session := fs.String("session", "", "owning session id (the descriptor at refs/fak/locks/session-<id>) for liveness classification")
	ttl := fs.Int64("ttl", 0, "lease lifetime in seconds (0 = no expiry)")
	announce := fs.String("announce", "", "public-safe lifecycle announcement: on, off, or offline")
	announceIssue := fs.Int("announce-issue", 0, "coordination issue number for --announce=on")
	announceRepo := fs.String("announce-repo", "", "owner/repo for --announce=on")
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
	// Writer adoption of the node-identity convention (#2304): when the caller
	// names its session but no holder, mint holder = <node-id>/<session-id> from
	// this machine's stable node id (hostname keyed to the hardware catalog).
	// An explicit --holder is always honored verbatim.
	if *holder == "" && *session != "" {
		*holder = leaseref.MintHolder(leaseref.LocalNodeID(*dir), *session)
	}
	store := leaseref.NewInDir(*dir)
	ambientLeaseRefSync(loopdrive.LeaseRefSyncSurfaceLeaserefAcquire, store, "", false)
	rec, v, err := store.AcquireFenced(context.Background(), leaseref.Record{
		ID:         *id,
		TreeGlobs:  trees,
		Holder:     *holder,
		SessionID:  *session,
		TTLSeconds: *ttl,
	}, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref acquire: %v\n", err)
		return 1
	}
	out := fencedResult{Verdict: v}
	if v.OK {
		out.Record = &rec
		ambientLeaseRefSync(loopdrive.LeaseRefSyncSurfaceLeaserefAcquire, store, "", true)
		postVerdict, postErr := store.Fence(context.Background(), rec, time.Now())
		if postErr != nil {
			fmt.Fprintf(stderr, "fak leaseref acquire: post-sync witness error: %v\n", postErr)
			return 1
		}
		if !postVerdict.OK {
			out.Record = nil
			out.Verdict = postVerdict
			return emitLeaserefOutcome(stdout, stderr, out, false, "acquire")
		}
		ambientLeaserefAnnounce(stderr, *dir, leaseref.AnnounceAcquire, rec, resolveAmbientLeaserefConfig(*announce, *announceIssue, *announceRepo))
	}
	return emitLeaserefOutcome(stdout, stderr, out, v.OK, "acquire")
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
	return emitLeaserefResult(stdout, stderr, v, err, "fak leaseref fence", "fence", func(v leaseref.FenceVerdict) bool { return v.OK })
}

func runLeaserefRenew(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref renew", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	id := fs.String("id", "", "lease id to renew")
	holder := fs.String("holder", "", "the holder identity that owns the lease")
	ttl := fs.Int64("ttl", 0, "new lifetime in seconds (0 = keep the lease's existing TTL)")
	announce := fs.String("announce", "", "public-safe lifecycle announcement: on, off, or offline")
	announceIssue := fs.Int("announce-issue", 0, "coordination issue number for --announce=on")
	announceRepo := fs.String("announce-repo", "", "owner/repo for --announce=on")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *id == "" || *holder == "" {
		fmt.Fprintln(stderr, "fak leaseref renew: --id and --holder are required")
		return 2
	}
	store := leaseref.NewInDir(*dir)
	ambientLeaseRefSync(loopdrive.LeaseRefSyncSurfaceLeaserefRenew, store, "", false)
	rec, v, err := store.Renew(context.Background(), *id, *holder, *ttl, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref renew: %v\n", err)
		return 1
	}
	out := fencedResult{Verdict: v}
	if v.OK {
		out.Record = &rec
		ambientLeaseRefSync(loopdrive.LeaseRefSyncSurfaceLeaserefRenew, store, "", true)
		ambientLeaserefAnnounce(stderr, *dir, leaseref.AnnounceRenew, rec, resolveAmbientLeaserefConfig(*announce, *announceIssue, *announceRepo))
	}
	return emitLeaserefOutcome(stdout, stderr, out, v.OK, "renew")
}

// runLeaserefSync is the convergence verb: move the refs/fak/locks/* namespace
// between this clone and a remote so every node's arbiter sees every node's leases.
// Directions default to BOTH (push then fetch — the order internal/leaseref/sync.go
// documents as load-bearing); --push-only / --fetch-only select one, and asking for
// both selectors at once is a usage error rather than a guess.
func runLeaserefSync(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref sync", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	remote := fs.String("remote", "origin", "the git remote (name or URL) to converge with")
	pushOnly := fs.Bool("push-only", false, "publish local lease refs only (skip the fetch)")
	fetchOnly := fs.Bool("fetch-only", false, "import remote lease refs only (skip the push)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *pushOnly && *fetchOnly {
		fmt.Fprintln(stderr, "fak leaseref sync: --push-only and --fetch-only are mutually exclusive")
		return 2
	}
	store := leaseref.NewInDir(*dir)
	res, err := store.Sync(context.Background(), *remote, !*fetchOnly, !*pushOnly)
	if err != nil {
		fmt.Fprintf(stderr, "fak leaseref sync: %v\n", err)
		return 1
	}
	return emitLeaserefJSON(stdout, stderr, res, "sync")
}

// runLeaserefDrain is the sanctioned session-descriptor drainer (#5358): it delete-pushes the
// PROVEN-EXPIRED refs/fak/locks/session-* descriptors to origin — the targeted per-id
// convergence the no-prune sync deliberately omits, so origin's expired-descriptor backlog
// (the ~5882-ref treadmill) can actually drain instead of being re-materialized on every fetch.
//
// DRY-RUN BY DEFAULT, like every retention collector: with no --apply it reports exactly which
// ids WOULD be delete-pushed and deletes NOTHING. --apply performs the live drain: it opts into
// the pre-push bulk-deletion gate (FLEET_ALLOW_REF_PRUNE=1 — the escape the hook reserves for
// THIS drainer) and removes only the proven-expired descriptors on origin, reaping each locally
// too so this clone's own later glob sync push cannot resurrect them. A live descriptor is
// never a target. The live drain of the real fleet remote is thus an explicit operator step,
// never an automatic sweep.
func runLeaserefDrain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak leaseref drain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	remote := fs.String("remote", "origin", "the git remote (name or URL) to drain expired descriptors from")
	apply := fs.Bool("apply", false, "PERFORM the delete-push (default: dry-run, report the target set and mutate nothing)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *apply {
		// Opt into the pre-push bulk-deletion gate (tools/githooks/pre-push): this verb IS the
		// sanctioned expired-ref drainer the hook reserves that escape for. Set on the one-shot CLI
		// process so the push child inherits it; a dry-run never touches it.
		os.Setenv("FLEET_ALLOW_REF_PRUNE", "1")
	}
	store := leaseref.NewInDir(*dir)
	res, err := store.ConvergeDescriptorDrain(context.Background(), *remote, time.Now(), *apply)
	if err != nil {
		// A partial best-effort failure still carries an accurate report + counts; surface both.
		fmt.Fprintf(stderr, "fak leaseref drain: %v\n", err)
		emitLeaserefJSON(stdout, stderr, res, "drain")
		return 1
	}
	return emitLeaserefJSON(stdout, stderr, res, "drain")
}

func emitLeaserefJSON(stdout, stderr io.Writer, v any, sub string) int {
	return encodeJSONOrFail(stdout, stderr, v, "fak leaseref "+sub)
}

// emitLeaserefOutcome is the shared tail of every fenced-lease verb that can be REFUSED:
// emit the payload under the verb's own label, then map a denied verdict to leaserefRefused
// (3) instead of 0, so a caller can tell a structured refusal from a broken store. The
// verdict JSON is emitted either way — a refusal is a value, not an error.
//
// payload and ok are separate arguments because the verbs genuinely diverge in what they
// print: acquire, renew and `intent claim` emit a wrapper carrying the stamped record on a
// win, while fence and release emit the bare verdict. Only the OK bit is shared, so only
// the OK bit is passed; nothing here reaches into the payload to re-derive it.
func emitLeaserefOutcome(stdout, stderr io.Writer, payload any, ok bool, sub string) int {
	if code := emitLeaserefJSON(stdout, stderr, payload, sub); code != 0 {
		return code
	}
	if !ok {
		return leaserefRefused
	}
	return 0
}
