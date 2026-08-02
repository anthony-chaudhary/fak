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
	fs := flag.NewFlagSet("wip reconcile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	asJSON := fs.Bool("json", false, "emit the reconciliation decisions as JSON")
	reclaimOnly := fs.Bool("reclaim", false, "print only the RECLAIM rows — the recovery worklist, ranked most-decayed-first by base drift; exit 3 if any exist")
	fileTicket := fs.Bool("file-ticket", false, "on a QUARANTINE verdict, bind the orphan to ONE idempotent GitHub tracking ticket (keyed by session+start-SHA)")
	dryRun := fs.Bool("dry-run", false, "with --file-ticket, print the exact ticket that would be filed instead of filing it (also the automatic behavior when gh is unavailable)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
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
			fmt.Fprintf(stdout, "%s\t%s\t%s\n", d.Action, d.Session, d.Reason)
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
	if *reclaimOnly && len(res.Reclaim) > 0 {
		return 3
	}
	return 0
}

// wipReclaimRender prints the recovery worklist head-first: the top row is the one
// closest to decaying out of RECLAIM. Each row names the exact recovery command, since
// the whole point of the queue is that RECLAIM is actionable and QUARANTINE is not.
func wipReclaimRender(stdout io.Writer, res wipReconcileResult) {
	if len(res.Reclaim) == 0 {
		fmt.Fprintf(stdout, "no reclaimable checkpoints: of %d reconciled, none is an unlanded delta that still applies cleanly\n", len(res.Decisions))
		return
	}
	fmt.Fprintln(stdout, "DRIFT\tAGE_H\tSESSION\tLEAVES")
	for _, r := range res.Reclaim {
		drift := "?"
		if r.TrunkDistance != wiprecon.DriftUnknown {
			drift = fmt.Sprintf("%d", r.TrunkDistance)
		}
		leaves := strings.Join(r.Leaves, ",")
		if leaves == "" {
			leaves = "-"
		}
		fmt.Fprintf(stdout, "%s\t%.1f\t%s\t%s\n", drift, r.AgeHours, r.Session, leaves)
	}
	for _, r := range res.Reclaim {
		fmt.Fprintf(stdout, "  %s: %s — recover with `fak wip land %s`\n", r.Session, r.Reason, r.Session)
	}
	fmt.Fprintf(stdout, "%d reclaimable of %d reconciled · DRIFT is commits HEAD has advanced past the checkpoint's base (? = base unresolvable); a higher drift is closer to decaying into QUARANTINE\n",
		len(res.Reclaim), len(res.Decisions))
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
	live, err := wipLiveSessions(ctx, repo)
	if err != nil {
		return wipReconcileResult{}, err
	}
	cands := make([]wiprecon.Candidate, 0, len(recs))
	bySession := make(map[string]wipref.RefRecord, len(recs))
	for _, r := range recs {
		session := wipSessionOf(r)
		bySession[session] = r
		c := wiprecon.Candidate{Session: session, Owner: wiprecon.OwnerCrashed}
		if live[session] {
			c.Owner = wiprecon.OwnerLive
			cands = append(cands, c)
			continue
		}
		st, oerr := wipOwnerState(ctx, repo, r)
		if oerr != nil {
			return wipReconcileResult{}, oerr
		}
		c.Landed = st == wipref.OwnerLanded
		if !c.Landed {
			c.Applies = wipDeltaApplies(ctx, repo, r)
		}
		cands = append(cands, c)
	}
	decisions := wiprecon.Reconcile(cands)
	return wipReconcileResult{
		Decisions: decisions,
		Reclaim:   wipReclaimWorklist(ctx, repo, decisions, bySession, now),
	}, nil
}

// wipReclaimWorklist resolves the base-drift facts for the RECLAIM decisions and ranks
// them most-decayed-first (#5480). It costs at most ONE extra git spawn per RECLAIM row
// and none at all for the other verdicts — deliberately, because RECLAIM is the rare
// verdict (the reporter's fleet repo read 2 RECLAIM to 131 QUARANTINE), so the default
// reconcile path's read cost is unchanged for every other row.
func wipReclaimWorklist(ctx context.Context, repo string, decisions []wiprecon.Decision, bySession map[string]wipref.RefRecord, now time.Time) []wiprecon.ReclaimRow {
	reclaimable := wiprecon.Reclaimable(decisions)
	rows := make([]wiprecon.ReclaimRow, 0, len(reclaimable))
	for _, d := range reclaimable {
		rec := bySession[d.Session]
		row := wiprecon.ReclaimRow{
			Session:       d.Session,
			Object:        rec.Object,
			StartSHA:      rec.Stamp.StartSHA,
			TrunkDistance: wipBaseDistance(ctx, repo, rec.Stamp.StartSHA),
			Leaves:        rec.Stamp.Leaves,
			Reason:        d.Reason,
		}
		if row.Leaves == nil {
			row.Leaves = []string{}
		}
		// An unstamped or future capture time yields age 0 rather than a fabricated
		// one: the queue is sorted by urgency, and an unmeasurable row must not be
		// promoted by an artifact of the zero value.
		if at := rec.Stamp.CheckpointedAt; at > 0 {
			if age := now.Sub(time.Unix(at, 0)); age > 0 {
				row.AgeHours = age.Hours()
			}
		}
		rows = append(rows, row)
	}
	return wiprecon.RankReclaim(rows)
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
