package main

// wip_owner.go — `fak wip owner`, the creation-time ownership surface.
//
// `fak wip attribute` answers "who owns this dirty hunk" from `git diff HEAD`, which is
// silent about a file that does not exist in HEAD — so a path a session has just CREATED
// is invisible to it, to `wip sweep-guard`, and to every other attribution surface here.
// This verb closes that gap using evidence the fleet already mints: a checkpoint capture
// is `read-tree HEAD` + `add -A`, so a checkpoint's own delta records every untracked path
// alive at capture time as an ADDITION. Reading those additions back tells us which
// sessions have a live claim on each created path, and how long ago they last checked in.
//
// All git I/O lives here; the classification is the pure wipref.BuildOwnerReport fold.
// Read-only throughout — this verb never writes a ref, stages, moves, or removes anything.
// An expired claim is a CHECK-IN OVERDUE, never a licence to reap.
//
// THE TTL IS ALSO THE COST BOUND. Only checkpoints inside the claim window (or belonging
// to a live session) can hold a claim, so the pass diffs those refs and no others. On this
// fleet that is ~31 of 1216 refs at the 90-minute default — the difference between a verb
// you can run at first mutation and one you run overnight.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// wipOwnerResult is the JSON/plain result of an ownership pass. Skipped names the paths
// the caller asked about that are NOT untracked: checkpoint-addition evidence cannot
// speak about a tracked file, and silently grading one UNCLAIMED would be a lie, so they
// are reported out of band and pointed at the verb that can answer for them.
type wipOwnerResult struct {
	wipref.OwnerReport
	Skipped []string `json:"skipped,omitempty"`
}

// runWipOwner exits 3 (in --unclaimed mode) when any created path carries no fresh claim
// — the one-bit signal a start-of-task check or a CI gate keys on, mirroring the exit-3
// contract of `wip attribute --orphans`. Listing without --unclaimed is informational.
func runWipOwner(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip owner", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	asJSON := fs.Bool("json", false, "emit the ownership report as JSON")
	unclaimedOnly := fs.Bool("unclaimed", false, "print only UNCLAIMED paths; exit 3 if any exist")
	ttl := fs.Duration("ttl", wipref.DefaultClaimTTL, "how long a checkpoint keeps claiming what it captured")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	res, err := wipOwner(context.Background(), *repo, fs.Args(), *ttl)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip owner: %v\n", err)
		return 1
	}

	rows := res.Paths
	if *unclaimedOnly {
		rows = wipref.Unclaimed(rows)
	}
	if *asJSON {
		out := res
		out.Paths = rows
		if code := encodeJSONOrFail(stdout, stderr, out, "fak wip owner"); code != 0 {
			return code
		}
	} else {
		wipOwnerRender(stdout, res, rows, *unclaimedOnly)
	}
	if *unclaimedOnly && res.Unclaimed > 0 {
		return 3
	}
	return 0
}

// wipOwnerRender prints the plain listing: the counts headline first (the whole point is
// the shape of the tree, which a 300-row dump buries), then one row per path.
func wipOwnerRender(stdout io.Writer, res wipOwnerResult, rows []wipref.Ownership, unclaimedOnly bool) {
	fmt.Fprintf(stdout, "created-path ownership over %d untracked path(s), claim TTL %s:\n",
		len(res.Paths), (time.Duration(res.ClaimTTLSeconds) * time.Second))
	fmt.Fprintf(stdout, "  CLAIMED_LIVE     %5d  a session is working here\n", res.Live)
	fmt.Fprintf(stdout, "  AMBIGUOUS        %5d  several fresh capturers; tree-wide capture cannot name an author\n", res.Ambiguous)
	fmt.Fprintf(stdout, "  CLAIMED_EXPIRED  %5d  named owner, check-in overdue (NOT a reap licence)\n", res.Expired)
	fmt.Fprintf(stdout, "  UNCLAIMED        %5d  no fresh checkpoint records this creation — at risk from any broad add/clean\n", res.Unclaimed)
	if len(res.Skipped) > 0 {
		fmt.Fprintf(stdout, "  skipped %d tracked path(s) — created-path evidence cannot speak about them; use `fak wip attribute`\n", len(res.Skipped))
	}
	switch {
	case len(rows) == 0 && unclaimedOnly:
		fmt.Fprintln(stdout, "no unclaimed creations: every created path carries a fresh claim")
	case len(rows) == 0:
		fmt.Fprintln(stdout, "no untracked paths to grade")
	default:
		for _, o := range rows {
			fmt.Fprintf(stdout, "%s\t%s\t%s\t%s\n", o.State, o.Path, wipOwnerLabel(o), o.Reason)
		}
	}
}

// wipOwnerLabel renders the owner column: the sole owner, every claimant on a tie, or a
// dash when nobody holds the path.
func wipOwnerLabel(o wipref.Ownership) string {
	switch o.State {
	case wipref.OwnAmbiguous:
		return strings.Join(o.Owners, ",")
	case wipref.OwnUnclaimed:
		return "-"
	default:
		return o.Owner
	}
}

// wipOwner gathers the git facts and folds them. want is the caller's explicit path list;
// empty means "every untracked path in the tree".
func wipOwner(ctx context.Context, repo string, want []string, ttl time.Duration) (wipOwnerResult, error) {
	if ttl <= 0 {
		ttl = wipref.DefaultClaimTTL
	}
	untracked, err := wipUntrackedPaths(ctx, repo)
	if err != nil {
		return wipOwnerResult{}, err
	}

	targets, skipped := untracked, []string(nil)
	if len(want) > 0 {
		inTree := make(map[string]bool, len(untracked))
		for _, p := range untracked {
			inTree[p] = true
		}
		targets = nil
		for _, p := range want {
			if q := strings.TrimSpace(strings.ReplaceAll(p, `\`, "/")); inTree[q] {
				targets = append(targets, q)
			} else if q != "" {
				skipped = append(skipped, q)
			}
		}
		sort.Strings(skipped)
	}

	claims, err := wipOwnerClaims(ctx, repo, targets, ttl)
	if err != nil {
		return wipOwnerResult{}, err
	}
	rep := wipref.BuildOwnerReport(targets, claims, time.Now().Unix(), ttl)
	return wipOwnerResult{OwnerReport: rep, Skipped: skipped}, nil
}

// wipUntrackedPaths lists the tree's untracked, non-ignored paths — the set for which
// checkpoint-addition evidence is meaningful.
func wipUntrackedPaths(ctx context.Context, repo string) ([]string, error) {
	out, err := gitWipOut(ctx, repo, nil, "ls-files", "--others", "--exclude-standard")
	if err != nil {
		return nil, fmt.Errorf("list untracked paths: %w", err)
	}
	var paths []string
	for _, line := range strings.Split(out, "\n") {
		if p := strings.TrimSpace(strings.TrimRight(line, "\r")); p != "" {
			paths = append(paths, p)
		}
	}
	return paths, nil
}

// wipOwnerClaims reads, for every checkpoint that can still hold a claim, which of the
// target paths that checkpoint recorded as an ADDITION. Every retained checkpoint is
// examined because an old single claimant is evidence for CLAIMED_EXPIRED; dropping old
// refs would incorrectly grade that path UNCLAIMED. A ref whose delta cannot be read (a root-parent or
// a damaged object) contributes no claim — the same fail-toward-unclaimed posture
// wipBuildAttributions takes, so an unreadable ref can never manufacture an owner.
func wipOwnerClaims(ctx context.Context, repo string, targets []string, ttl time.Duration) (map[string][]wipref.Claim, error) {
	claims := map[string][]wipref.Claim{}
	if len(targets) == 0 {
		return claims, nil
	}
	recs, err := wipListRecords(ctx, repo)
	if err != nil {
		return nil, err
	}
	live, err := wipLiveSessions(ctx, repo)
	if err != nil {
		return nil, err
	}

	interesting := make(map[string]bool, len(targets))
	for _, p := range targets {
		interesting[p] = true
	}
	for _, r := range recs {
		session := wipSessionOf(r)
		if session == "" {
			continue
		}
		isLive := live[session]
		added, derr := gitWipOut(ctx, repo, nil,
			"diff", "--diff-filter=A", "--name-only", r.Object+"^", r.Object, "--")
		if derr != nil {
			continue
		}
		c := wipref.Claim{Session: session, CheckpointedAt: r.Stamp.CheckpointedAt, Live: isLive}
		for _, line := range strings.Split(added, "\n") {
			p := strings.TrimSpace(strings.TrimRight(line, "\r"))
			if p != "" && interesting[p] {
				claims[p] = append(claims[p], c)
			}
		}
	}
	return claims, nil
}
