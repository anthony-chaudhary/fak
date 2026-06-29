package marketing

import (
	"context"
	"time"
)

// tick.go — THE single idempotent entrypoint every trigger funnels through. The serve
// bgloop calls Tick in-process; the git post-commit hook and cron shell `fak marketing tick`,
// which calls Tick. One code path means the high-water-mark + dedupe logic lives in exactly
// one place, so no two triggers can double-announce the same commit.
//
// Ordering is the load-bearing part: read the mark, gather the genuinely-new ships, gate on
// emptiness, then ADVANCE THE MARK (the compare-and-swap) BEFORE posting. The CAS is the
// claim on the window — whoever advances first owns it; a loser bails without posting. A
// dry-run NEVER advances (a preview must be repeatable and must not consume the ships).

// Poster posts a rendered artifact and returns the message ts (or "" if a dedupe-aware
// poster skipped an unchanged repost). It is the seam to internal/scoreboard's client, kept
// as an interface so tier-1 internal/marketing does not import the transport wiring directly
// and so Tick is unit-testable with a fake poster.
type Poster interface {
	PostArtifact(ctx context.Context, a Artifact) (ts string, err error)
}

// Opts configures one Tick. Root is the repo; Source labels who fired (serve|hook|cron|cli|ci);
// Bootstrap is how far back to look on the FIRST run (no mark yet), e.g. "HEAD~30" — capped so
// a fresh repo doesn't announce all of history at once. DryRun renders without posting or
// advancing the mark. When Poster is nil the tick is render-only (the dry path).
type Opts struct {
	Root      string
	Source    string
	Bootstrap string // first-run range start, default "HEAD~20"; "" disables bootstrap (whole history)
	DryRun    bool
	Poster    Poster
}

// Result is the structured outcome of a Tick: what it did and why, so a loop/CLI can report
// honestly (and so a test can assert the decision without a real git/Slack).
type Result struct {
	Status   string   // "posted" | "skipped:no-new-ships" | "skipped:raced" | "dry-run" | "skipped:no-poster"
	Artifact Artifact // the built artifact (empty Kind if nothing was built)
	NewShips int      // marketable ships in the window
	PostedTS string   // the Slack ts on a real post
	OldMark  string   // the mark before this tick
	NewMark  string   // the mark after a successful advance ("" if not advanced)
}

// Tick runs one idempotent marketing pass. It is safe to call from any trigger concurrently:
// the high-water CAS serializes same-host racers to one winner.
func (o Opts) Tick(ctx context.Context, now time.Time) (Result, error) {
	root := o.Root
	if root == "" {
		root = "."
	}
	old := ReadHighWater(root)
	revRange := tickRange(old, o.Bootstrap)

	col, err := Gather(root, revRange)
	if err != nil {
		return Result{}, err
	}
	res := Result{NewShips: len(col.Ships), OldMark: old}

	// Genuinely-new gate: nothing witnessed since the mark -> do not post, do not advance.
	if len(col.Ships) == 0 {
		res.Status = "skipped:no-new-ships"
		return res, nil
	}

	art := col.DigestFrom(now)
	art.Source = o.Source
	res.Artifact = art

	if o.DryRun {
		res.Status = "dry-run"
		return res, nil // never advance the mark on a preview
	}
	if o.Poster == nil {
		res.Status = "skipped:no-poster"
		return res, nil
	}

	// Claim the window via the CAS BEFORE posting. If we lose, another trigger already
	// owns it and will post — we must not double-announce.
	head := headSHA(root)
	if head == "" || !AdvanceHighWater(root, head, old) {
		res.Status = "skipped:raced"
		return res, nil
	}
	res.NewMark = head

	ts, err := o.Poster.PostArtifact(ctx, art)
	if err != nil {
		// The mark is already advanced; a transient post failure should not re-announce the
		// whole window next tick. The error is surfaced so the caller can retry the SAME
		// artifact out-of-band (its DedupeKey makes a retry idempotent).
		return res, err
	}
	res.PostedTS = ts
	res.Status = "posted"
	return res, nil
}

// tickRange builds the rev-range for a tick: <mark>..HEAD when a mark exists, else the
// bootstrap window (default HEAD~20) for the first run, else "" (whole history) when bootstrap
// is explicitly disabled.
func tickRange(mark, bootstrap string) string {
	if mark != "" {
		return mark + "..HEAD"
	}
	if bootstrap == "" {
		bootstrap = "HEAD~20"
	}
	if bootstrap == "ALL" {
		return "" // explicit opt-in to the whole history
	}
	return bootstrap + "..HEAD"
}
