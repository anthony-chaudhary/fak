package main

// wip_reconcile.go — the `fak wip reconcile` seam: checkpoint reconciliation and the
// RECLAIM recovery worklist (the ranked queue that arrived with #5480), together with
// the git-witnessed facts those verdicts rest on (liveness, base drift, clean-apply).
//
// Split out of wip.go verbatim — no renames, no signature or logic changes — because
// wip.go had grown past the god-file ceiling the tree enforces on any tracked non-test
// .go file (internal/hooks/gate_godfile.go, godFileMaxLines = 1500). Same package, same
// symbols; only the file boundary moved. The reconcile surface is the natural cut: it is
// a self-contained subcommand whose helpers (wipReclaimWorklist, wipBaseDistance,
// wipLiveSessions, wipDeltaApplies) serve nothing else in wip.go.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/wiprecon"
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// ---- reconcile (#3875) ----

// wipReconcileResult is the JSON/plain result of a reconciliation pass.
type wipReconcileResult struct {
	Decisions []wiprecon.Decision `json:"decisions"`
	// Reclaim is the RECOVERY worklist (#5480): the RECLAIM rows, ranked
	// most-decayed-first, each carrying the base-drift facts that say how much life
	// the verdict has left. Always present (empty when nothing is reclaimable) so a
	// scripted consumer never has to distinguish absent from none.
	Reclaim []wiprecon.ReclaimRow `json:"reclaim"`
}

// runWipReconcile is advisory: it prints the per-checkpoint decision and mutates
// nothing (no ref delete, no restore). Acting on a decision is a later, explicit cut.
//
// --reclaim narrows the listing to the recovery worklist and exits 3 when it is
// non-empty, the same "there is a lever here" contract `wip blocked --landable` and
// `wip attribute --orphans` use — so a hook or a scheduler can branch on the queue
// being non-empty without parsing output. It still mutates nothing: the driver #5480
// asks for is a worklist, not an auto-lander, because materializing a crashed
// stranger's delta into a SHARED working tree would land on peers' live edits.
func runWipReconcile(stdout, stderr io.Writer, argv []string) int {
	// The ADOPTION seam (#5998). `fak wip reconcile adopt|resume|receipt <session>` are
	// sub-verbs of reconcile rather than a peer of `wip land`, because they are the
	// consumer of THIS verb's worklist: the queue names them, and keeping the naming and
	// the code in one place is what stops the printed command from drifting away from the
	// command that exists. A positional token is safe to intercept here because the
	// reconcile flag parse rejects positional arguments outright.
	if subcommand, args, ok := splitLeadingSubcommand(argv); ok {
		switch subcommand {
		case "adopt", "resume", "receipt":
			return runWipReclaim(stdout, stderr, subcommand, args)
		}
	}

	fs := flag.NewFlagSet("wip reconcile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	asJSON := fs.Bool("json", false, "emit the reconciliation decisions as JSON")
	reclaimOnly := fs.Bool("reclaim", false, "print only the RECLAIM rows — the recovery worklist, ranked most-decayed-first by base drift; exit 3 if any exist")
	dispatch := fs.Bool("dispatch", false, "with --reclaim, ADOPT the head unclaimed row (opt-in): it claims, materializes to an isolated target, and never lands, quarantines, or deletes anything")
	fileTicket := fs.Bool("file-ticket", false, "on a QUARANTINE verdict, bind the orphan to ONE idempotent GitHub tracking ticket (keyed by session+start-SHA)")
	dryRun := fs.Bool("dry-run", false, "with --file-ticket, print the exact ticket that would be filed instead of filing it (also the automatic behavior when gh is unavailable)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *dispatch && !*reclaimOnly {
		fmt.Fprintln(stderr, "fak wip reconcile: --dispatch requires --reclaim (it acts on the recovery worklist, and only on it)")
		return 2
	}
	ctx := context.Background()
	res, err := wipReconcile(ctx, *repo)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip reconcile: %v\n", err)
		return 1
	}
	switch {
	case *asJSON:
		out := res
		if *reclaimOnly {
			// Narrow the decision list to match the printed queue, mirroring how
			// `wip blocked --landable` narrows Rows: the JSON and the human listing
			// must never disagree about which rows the caller was shown.
			out.Decisions = wiprecon.Reclaimable(res.Decisions)
		}
		if code := encodeJSONOrFail(stdout, stderr, out, "fak wip reconcile"); code != 0 {
			return code
		}
	case *reclaimOnly:
		wipReclaimRender(stdout, res)
	case len(res.Decisions) == 0:
		fmt.Fprintln(stdout, "no checkpoints to reconcile")
	default:
		for _, d := range res.Decisions {
			fmt.Fprintf(stdout, "%s\t%s\tclass=%s repl=%s\t%s\n", d.Action, d.Session, firstNonEmpty(d.CheckpointClass, "unknown"), firstNonEmpty(d.Replication, "unknown"), d.Reason)
			if d.NextCommand != "" {
				fmt.Fprintf(stdout, "  next: %s\n", d.NextCommand)
			}
			for _, command := range d.ReviewCommands {
				fmt.Fprintf(stdout, "  review: %s\n", command)
			}
		}
	}
	// Opt-in follow-up: file (or dry-run print) one idempotent ticket per QUARANTINE
	// orphan. Advisory only — it never changes the reconcile exit code. In --json mode
	// its human lines go to stderr so stdout stays pure JSON.
	if *fileTicket {
		tout := stdout
		if *asJSON {
			tout = stderr
		}
		wipReconcileFileTickets(ctx, tout, stderr, *repo, res.Decisions, *dryRun, newWipTicketGH())
	}
	// The opt-in dispatcher. It iterates ONLY res.Reclaim, so it is structurally incapable
	// of touching a QUARANTINE checkpoint — the exclusion is a property of the worklist,
	// not a check that could be forgotten. It never lands (that stays an explicit --land)
	// and never deletes a ref, so its worst case is a claimed checkpoint and a patch file.
	if *dispatch {
		dout := stdout
		if *asJSON {
			dout = stderr
		}
		wipReconcileDispatch(ctx, dout, *repo, res.Reclaim)
	}
	if *reclaimOnly && len(res.Reclaim) > 0 {
		return 3
	}
	return 0
}

// wipReconcileDispatch adopts the head UNCLAIMED row, if there is one. Advisory in the
// same sense the rest of this verb is: it never changes the reconcile exit code, because a
// scheduler's branch is on "is there recoverable work", and whether one row got claimed
// this tick does not change that answer.
func wipReconcileDispatch(ctx context.Context, stdout io.Writer, repo string, rows []wiprecon.ReclaimRow) {
	free := wiprecon.UnownedReclaim(rows)
	if len(free) == 0 {
		fmt.Fprintln(stdout, "dispatch: no unclaimed RECLAIM row to adopt")
		return
	}
	head := free[0]
	succ := wipAdoptSuccessorDefault()
	if succ == "" {
		fmt.Fprintf(stdout, "dispatch: %s is adoptable but this process has no session id (set $CLAUDE_CODE_SESSION_ID) — run `fak %s`\n",
			head.Session, strings.Join(head.Argv, " "))
		return
	}
	host, _ := os.Hostname()
	res, code, err := wipAdoptRun(ctx, repo, wipAdoptOptions{
		Session: head.Session, Successor: succ, Host: host, Now: time.Now(),
	})
	if err != nil {
		fmt.Fprintf(stdout, "dispatch: %s: %v\n", head.Session, err)
		return
	}
	fmt.Fprintf(stdout, "dispatch: %s -> %s (rc=%d) %s\n", head.Session, res.Verdict, code, res.Reason)
	if res.Target != "" {
		fmt.Fprintf(stdout, "  materialized %d verified file(s) to %s; the checkpoint ref is untouched\n", res.Verified, res.Target)
	}
}

// wipReclaimRender prints the recovery worklist head-first: the top row is the one a
// successor should take, which since #5998 means UNCLAIMED first and then closest to
// decaying out of RECLAIM. Each row names the EXACT command that advances it — or says
// plainly that no command may, because someone else holds the claim.
//
// The OWNER and REPL columns are not decoration. Without OWNER a fleet reads one queue and
// races itself; without REPL a successor cannot tell a checkpoint that survives this
// machine from one that does not, which is the difference between "recover it when
// convenient" and "recover it now".
func wipReclaimRender(stdout io.Writer, res wipReconcileResult) {
	if len(res.Reclaim) == 0 {
		fmt.Fprintf(stdout, "no reclaimable checkpoints: of %d reconciled, none is an unlanded delta that still applies cleanly\n", len(res.Decisions))
		return
	}
	fmt.Fprintln(stdout, "DRIFT\tAGE_H\tREPL\tOWNER\tSESSION\tLEAVES")
	for _, r := range res.Reclaim {
		drift := "?"
		if r.TrunkDistance != wiprecon.DriftUnknown {
			drift = fmt.Sprintf("%d", r.TrunkDistance)
		}
		leaves := strings.Join(r.Leaves, ",")
		if leaves == "" {
			leaves = "-"
		}
		fmt.Fprintf(stdout, "%s\t%.1f\t%s\t%s\t%s\t%s\n",
			drift, r.AgeHours, firstNonEmpty(r.Replication, "-"), wipReclaimOwnerCell(r), r.Session, leaves)
	}
	free := 0
	for _, r := range res.Reclaim {
		if len(r.Argv) == 0 {
			fmt.Fprintf(stdout, "  %s: %s — held by %s at phase %s (attempt %d); wait for that claim to lapse rather than racing it\n",
				r.Session, r.Reason, r.AdoptedBy, r.AdoptPhase, r.Attempts)
			continue
		}
		free++
		// Attempt history rides on the ACTIONABLE rows too, not just the held ones: a row
		// that has already been claimed and dropped four times is the row an operator wants
		// to see before a fifth automatic try, and it is precisely the row a dispatcher
		// would otherwise pick up silently forever.
		tried := ""
		if r.Attempts > 0 {
			tried = fmt.Sprintf(" — %d prior attempt(s)", r.Attempts)
		}
		fmt.Fprintf(stdout, "  %s: %s%s — recover with `fak %s`\n", r.Session, r.Reason, tried, strings.Join(r.Argv, " "))
	}
	fmt.Fprintf(stdout, "%d reclaimable of %d reconciled, %d unclaimed · DRIFT is commits HEAD has advanced past the checkpoint's base (? = base unresolvable); a higher drift is closer to decaying into QUARANTINE · mirror %s\n",
		len(res.Reclaim), len(res.Decisions), free, firstNonEmpty(res.Reclaim[0].MirrorFreshness, "unknown"))
}

// wipReclaimOwnerCell renders one row's adoption ownership in a single column: "-" for
// unclaimed, the successor for a live claim, and a trailing "!" when the claim has lapsed
// and its holder is provably gone — the takeover-eligible state, marked rather than
// silently treated as free.
func wipReclaimOwnerCell(r wiprecon.ReclaimRow) string {
	if r.AdoptedBy == "" {
		return "-"
	}
	cell := r.AdoptedBy
	if r.AdoptedMine {
		cell = "self:" + cell
	}
	if r.AdoptExpired {
		cell += "!"
	}
	return cell
}

// wipReconcile classifies every WIP checkpoint into a reconciliation action from three
// git-witnessed facts: liveness (does the owning session still hold a lease under
// refs/fak/locks/*?), landing (wipOwnerState — is the delta in HEAD?), and clean-apply
// (git apply --check — does the delta still apply?). A live owner's checkpoint is SKIP;
// only a crashed owner's checkpoint is DISCARD_WITNESSED / RECLAIM / QUARANTINE.
func wipReconcile(ctx context.Context, repo string) (wipReconcileResult, error) {
	return wipReconcileAt(ctx, repo, time.Now())
}

// wipReconcileAt is wipReconcile with the clock injected, so the recovery worklist's
// age column is testable without sleeping.
func wipReconcileAt(ctx context.Context, repo string, now time.Time) (wipReconcileResult, error) {
	recs, err := wipListRecords(ctx, repo)
	if err != nil {
		return wipReconcileResult{}, err
	}
	status, err := wipStatus(ctx, repo, now.Unix())
	if err != nil {
		return wipReconcileResult{}, err
	}
	replication := make(map[string]string, len(status.Sessions))
	for _, row := range status.Sessions {
		replication[row.Session] = string(row.Replication)
	}
	live, err := wipLiveSessions(ctx, repo)
	if err != nil {
		return wipReconcileResult{}, err
	}
	cands := make([]wiprecon.Candidate, 0, len(recs))
	bySession := make(map[string]wipref.RefRecord, len(recs))
	payloads := make(map[string]wipref.PayloadCensus, len(recs))
	classes := make(map[string]wipref.CensusClass, len(recs))
	for _, r := range recs {
		session := wipSessionOf(r)
		bySession[session] = r
		c := wiprecon.Candidate{Session: session, Owner: wiprecon.OwnerCrashed}
		if live[session] {
			c.Owner = wiprecon.OwnerLive
		}
		st, oerr := wipOwnerState(ctx, repo, r)
		if oerr != nil {
			return wipReconcileResult{}, oerr
		}
		c.Landed = st == wipref.OwnerLanded
		payload := wipref.BuildPayloadCensus(wipPayloadReading(ctx, repo, r))
		if !payload.Read {
			return wipReconcileResult{}, fmt.Errorf("measure checkpoint payload %s: %s", r.Ref, payload.Unreadable)
		}
		payloads[session] = payload
		if !c.Landed && c.Owner != wiprecon.OwnerLive {
			c.DivergedPaths = len(payload.DivergedPaths)
			if c.DivergedPaths == 0 {
				c.Applies = wipDeltaApplies(ctx, repo, r)
			}
		}
		read, files, absent, diverged := payload.Facts()
		classes[session] = wipref.Classify(wipref.CensusFacts{
			Live: c.Owner == wiprecon.OwnerLive, Landed: c.Landed, Resolved: true,
			PayloadRead: read, PayloadFiles: files, PayloadAbsent: absent, PayloadDiverged: diverged,
		})
		cands = append(cands, c)
	}
	decisions := wiprecon.Reconcile(cands)
	for i := range decisions {
		d := &decisions[i]
		r := bySession[d.Session]
		payload := payloads[d.Session]
		d.CheckpointClass = string(classes[d.Session])
		d.Replication = replication[d.Session]
		d.AbsentPaths = len(payload.AbsentPaths)
		d.DivergedPaths = len(payload.DivergedPaths)
		d.LandedPaths = payload.Landed
		for _, path := range payload.AbsentPaths {
			command, _ := wipref.PayloadRemedy(wipref.PayloadAbsent, r.Ref, path)
			d.ReviewCommands = append(d.ReviewCommands, command)
		}
		for _, path := range payload.DivergedPaths {
			command, _ := wipref.PayloadRemedy(wipref.PayloadDiverged, r.Ref, path)
			d.ReviewCommands = append(d.ReviewCommands, command)
		}
		switch d.Action {
		case wiprecon.ActSkip:
			d.NextCommand = "fak wip status"
			d.Reason += "; checkpoint is crash/restart safety while its owner is live, not the normal landing path"
		case wiprecon.ActReclaim:
			d.NextCommand = fmt.Sprintf("fak wip reconcile --reclaim --session %s", d.Session)
			d.Reason += "; adopt the whole clean delta instead of copying ref objects by hand"
		case wiprecon.ActQuarantine:
			if len(d.ReviewCommands) > 0 {
				d.NextCommand = "run review_commands; salvage absent paths directly and merge divergent paths before a normal commit"
			} else {
				d.NextCommand = "fak wip reap --census --json"
			}
			d.Reason += "; retained because automatic landing could overwrite newer HEAD content"
		}
		if d.Replication == string(wipref.ReplicationLocalOnly) {
			d.Reason += "; LOCAL_ONLY survives session loss but not loss of this clone (`fak wip sync` replicates it)"
		}
	}
	return wipReconcileResult{
		Decisions: decisions,
		Reclaim:   wipReclaimWorklist(ctx, repo, decisions, bySession, live, now),
	}, nil
}

// wipReclaimRemote is the remote the recovery queue grades replication and mirror
// freshness against. `origin` is the only remote the fleet actually syncs checkpoints to
// (internal/wipref/sync.go), and reading the mirror is a local ref sweep — no network — so
// a clone with no origin simply reports NEVER_SYNCED rather than failing.
const wipReclaimRemote = "origin"

// wipReclaimWorklist resolves the base-drift facts for the RECLAIM decisions and ranks
// them most-decayed-first (#5480). It costs at most ONE extra git spawn per RECLAIM row
// and none at all for the other verdicts — deliberately, because RECLAIM is the rare
// verdict (the reporter's fleet repo read 2 RECLAIM to 131 QUARANTINE), so the default
// reconcile path's read cost is unchanged for every other row.
func wipReclaimWorklist(ctx context.Context, repo string, decisions []wiprecon.Decision, bySession map[string]wipref.RefRecord, live map[string]bool, now time.Time) []wiprecon.ReclaimRow {
	reclaimable := wiprecon.Reclaimable(decisions)
	if len(reclaimable) == 0 {
		return []wiprecon.ReclaimRow{}
	}
	// The durability and ownership facts are resolved ONCE for the whole queue, and only
	// when the queue is non-empty: RECLAIM is the rare verdict, so the common reconcile
	// pass still pays nothing for columns it will not print.
	me := wipAdoptSuccessorDefault()
	mirror, merr := wipMirrorIndex(ctx, repo, wipReclaimRemote)
	freshness := ""
	if merr == nil {
		if view, verr := wipMirrorView(ctx, repo, wipReclaimRemote, len(mirror), now.Unix(), wipref.DefaultMirrorMaxAgeSeconds); verr == nil {
			freshness = string(view.Freshness)
		}
	}

	rows := make([]wiprecon.ReclaimRow, 0, len(reclaimable))
	for _, d := range reclaimable {
		rec := bySession[d.Session]
		row := wiprecon.ReclaimRow{
			Session:         d.Session,
			Object:          rec.Object,
			StartSHA:        rec.Stamp.StartSHA,
			TrunkDistance:   wipBaseDistance(ctx, repo, rec.Stamp.StartSHA),
			Leaves:          rec.Stamp.Leaves,
			Reason:          d.Reason,
			MirrorFreshness: freshness,
		}
		if row.Leaves == nil {
			row.Leaves = []string{}
		}
		// Replication may never OVERSTATE durability, so an unreadable mirror grades the
		// row LOCAL_ONLY exactly as an empty one does (ClassifyReplication's contract).
		state, _ := wipref.ClassifyReplication(rec, mirror)
		row.Replication = string(state)
		// An unstamped or future capture time yields age 0 rather than a fabricated
		// one: the queue is sorted by urgency, and an unmeasurable row must not be
		// promoted by an artifact of the zero value.
		if at := rec.Stamp.CheckpointedAt; at > 0 {
			if age := now.Sub(time.Unix(at, 0)); age > 0 {
				row.AgeHours = age.Hours()
			}
		}
		wipReclaimAnnotateAdoption(ctx, repo, &row, me, live, now.Unix())
		row.Argv = wiprecon.AdoptArgv(row)
		rows = append(rows, row)
	}
	return wiprecon.RankReclaim(rows)
}

// wipReclaimAnnotateAdoption folds one row's adoption receipt onto it. An UNREADABLE
// receipt is reported as an anonymous, non-expired claim rather than as "unclaimed": the
// queue's job is to stop two successors from racing, and a claim it cannot parse is still
// a claim someone wrote. Reading it as free is the one error that produces the collision.
func wipReclaimAnnotateAdoption(ctx context.Context, repo string, row *wiprecon.ReclaimRow, me string, live map[string]bool, nowUnix int64) {
	rec, _, has, err := wipReadReceipt(ctx, repo, row.Session)
	if err != nil {
		row.AdoptedBy, row.AdoptPhase = "?", "UNREADABLE"
		return
	}
	if !has {
		return
	}
	row.AdoptedBy = rec.Successor
	row.AdoptedMine = me != "" && rec.Successor == me
	row.AdoptPhase = string(rec.Phase)
	row.Attempts = rec.Attempt
	// Expired is the TAKEOVER precondition reported, not applied: both legs — the claim
	// lapsed AND its holder holds no live lease — exactly as wiprecon.DecideAdopt requires.
	row.AdoptExpired = !live[rec.Successor] && rec.Expired(nowUnix)
}

// wipBaseDistance reports how many commits HEAD has advanced past the base the
// checkpoint was captured on (Stamp.StartSHA) — the decay clock behind RECLAIM. An
// empty, unparseable or no-longer-present base yields wiprecon.DriftUnknown rather than
// a fabricated 0, so "I cannot measure this" is never rendered as "captured on today's
// HEAD".
func wipBaseDistance(ctx context.Context, repo, startSHA string) int {
	startSHA = strings.TrimSpace(startSHA)
	if startSHA == "" {
		return wiprecon.DriftUnknown
	}
	out, _, code, err := gitWip(ctx, repo, nil, "rev-list", "--count", startSHA+"..HEAD")
	if err != nil || code != 0 {
		return wiprecon.DriftUnknown
	}
	n, cerr := strconv.Atoi(strings.TrimSpace(out))
	if cerr != nil || n < 0 {
		return wiprecon.DriftUnknown
	}
	return n
}

// wipLiveSessions returns the set of session ids that currently hold a live lease under
// refs/fak/locks/* — the liveness signal that distinguishes a crashed owner (no live
// lease) from one still working. Read-only over the lease namespace.
func wipLiveSessions(ctx context.Context, repo string) (map[string]bool, error) {
	store := leaseref.NewInDir(repo)
	now := time.Now()
	recs, _, err := store.LiveRegistrations(ctx, now)
	if err != nil {
		return nil, fmt.Errorf("read live leases: %w", err)
	}
	live := make(map[string]bool, len(recs))
	for _, r := range recs {
		if r.SessionID != "" {
			live[r.SessionID] = true
		}
	}
	// #5343: a wip checkpoint stamps the STABLE Claude session UUID (wipSessionOf), but a
	// lock lease's SessionID is the VOLATILE agent-claude-<pid> trace id, so the UUID could
	// never hit the set built above — every ref read non-LIVE, even a live one. ALSO index the
	// LIVE guard-session descriptors by the Claude UUID they now carry (AgentUUID), so a
	// checkpoint stamped with that UUID resolves LIVE. This is STRICTLY ADDITIVE: it only ever
	// ADDS a live match, so a currently-kept ref can never become newly reclaimable — the same
	// fail-toward-keeping rule the lease cascade (internal/leaseref/liveness.go) obeys.
	sessions, _, serr := store.LiveSessions(ctx, now)
	if serr != nil {
		return nil, fmt.Errorf("read live sessions: %w", serr)
	}
	for _, d := range sessions {
		if d.AgentUUID != "" {
			live[d.AgentUUID] = true
		}
	}
	return live, nil
}

// wipDeltaApplies reports whether the checkpoint's recorded delta applies cleanly to the
// current working tree (`git apply --check`), the RECLAIM-vs-QUARANTINE discriminator.
// The RAW (untrimmed) diff is fed so the patch's trailing newline survives for apply.
func wipDeltaApplies(ctx context.Context, repo string, rec wipref.RefRecord) bool {
	diff, _, code, err := gitWip(ctx, repo, nil, "diff", rec.Object+"^", rec.Object)
	if err != nil || code != 0 || strings.TrimSpace(diff) == "" {
		return false
	}
	_, _, acode, aerr := gitWipStdin(ctx, repo, diff, "apply", "--check", "-")
	return aerr == nil && acode == 0
}
