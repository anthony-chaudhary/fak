package main

// superloop_followon.go — the impure shell half of the ORPHANED-FOLLOWON witness
// (issue #4957): resolve a loop's emitted follow-on refs (relay.ArtifactIssue baton
// pointers "#N" / the issue an a2achan.WorkerStatus names) against LIVE GitHub
// issue state, fold them into a superloop.FollowonRead, and hand the pure
// superloop.ClassifyFollowon the durable evidence. This mirrors memberProgress
// (superloop.go, #4956) exactly: the verdict is assembled ONLY from re-verifiable
// issue/artifact state — never a member's self-narrated field — and every
// unreadable edge fails closed (an orphan is never fabricated from an absence).
//
// GATED (gen/next). The live join costs one `gh issue view` per emitted ref, so the
// default `fak superloop walk` stays offline and fast: the witness reads the axis
// only when FAK_SUPERLOOP_FOLLOWON is set. Gate OFF is not "clean" — it is the axis
// UNREAD (""), surface-only, weighed by nothing, exactly like a member with no
// emissions. Turning the gate on is the dogfood path; promotion to on-by-default
// wants a run where the orphan count tracks a real backlog an operator agrees is
// stuck (see the issue's promotion evidence).

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ghexec"
	"github.com/anthony-chaudhary/fak/internal/loopfleet"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// followonWitnessEnv is the explicit opt-in for the LIVE issue join. Unset (or a
// falsey value) leaves the follow-on axis unread — no network, no verdict.
const followonWitnessEnv = "FAK_SUPERLOOP_FOLLOWON"

// defaultFollowonCadence is the FLOOR of the advance window an emission gets before
// it counts as orphaned. Generous on purpose: the dispatch loop ticks hourly, and
// judging emitted work orphaned because nobody touched it within one tick would
// fabricate orphans wholesale. The fail-closed direction for this window is to
// under-report orphans, never to slander live work with a window that was too tight.
const defaultFollowonCadence = 24 * time.Hour

// followonWitnessOn reports whether the gated live join is armed.
func followonWitnessOn() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(followonWitnessEnv))) {
	case "", "0", "false", "no", "off":
		return false
	}
	return true
}

// memberFollowon reads one loop member's emitted follow-on refs (#4957) and folds
// them into the closed follow-on verdict plus its closed reason token. The join is
// emitted refs → live issue state (`gh issue view --json state,updatedAt`, the same
// live ground truth the dispatch tick reads) and the fold is
// superloop.ClassifyFollowon. Fail-closed at every edge: the gate off or a loop with
// no bound emitted refs reads no axis at all — surface-only, never weighed — and any
// unreadable/unparsable ref resolves UNRESOLVED, which the pure fold turns into
// FollowonUnknown, never a fabricated orphan.
func (c *superloopCollector) memberFollowon(m superloop.Member, lh loopfleet.LoopHealth) (superloop.MemberFollowon, string) {
	if !followonWitnessOn() {
		return "", ""
	}
	refs := followonEmissionRefs(c.root, lh.Kind)
	if len(refs) == 0 {
		return "", ""
	}
	window := followonWindow(lh.CadenceSeconds)
	now := time.Now()
	read := superloop.FollowonRead{}
	for _, ref := range refs {
		read.Emissions = append(read.Emissions, resolveFollowonEmission(ref, window, now))
	}
	return superloop.ClassifyFollowon(m, read)
}

// followonWindow is the advance window one emission gets before it counts as
// orphaned: the loop's own declared cadence, widened to at least
// defaultFollowonCadence so a fast-ticking loop cannot fabricate an orphan from a
// window narrower than a work day.
func followonWindow(cadenceSeconds int64) time.Duration {
	w := time.Duration(cadenceSeconds) * time.Second
	if w < defaultFollowonCadence {
		return defaultFollowonCadence
	}
	return w
}

// resolveFollowonEmission joins ONE emitted ref against live issue state. The ref is
// the relay artifact form ("#1234", relay.ArtifactIssue) or a bare issue number (the
// a2achan.WorkerStatus.Issue form). Fail-closed: an unparsable ref or a failed live
// read stays UNRESOLVED (Resolved false), which ClassifyFollowon folds to
// FollowonUnknown — never a fabricated orphan. A closed issue counts as carried; an
// open issue counts as carried only when its durable updatedAt falls within the
// window ending at now.
func resolveFollowonEmission(ref string, window time.Duration, now time.Time) superloop.FollowonEmission {
	em := superloop.FollowonEmission{Ref: ref}
	n, err := strconv.Atoi(strings.TrimPrefix(strings.TrimSpace(ref), "#"))
	if err != nil || n <= 0 {
		return em // not an issue ref this reader can join — unresolved, fail closed
	}
	state, updatedAt, err := followonIssueState(n)
	if err != nil {
		return em // live state unreadable — unresolved, fail closed
	}
	em.Resolved = true
	em.Open = strings.EqualFold(state, "OPEN")
	em.Advanced = !em.Open || (!updatedAt.IsZero() && now.Sub(updatedAt) <= window)
	return em
}

// followonIssueState resolves one issue number to its durable live (state,
// updatedAt) via `gh issue view N --json state,updatedAt` — issue ground truth,
// never a worker log. A var so tests bind a hermetic resolver and drive the
// ORPHANED path end to end without the network.
var followonIssueState = func(issue int) (state string, updatedAt time.Time, err error) {
	cmd, cancel := ghexec.CommandTimeout(context.Background(), ghexec.DefaultTimeout,
		"issue", "view", strconv.Itoa(issue), "--json", "state,updatedAt")
	defer cancel()
	out, errOut, err := runBufferedCommand(cmd)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("gh issue view %d: %w: %s", issue, err, strings.TrimSpace(errOut))
	}
	var v struct {
		State     string    `json:"state"`
		UpdatedAt time.Time `json:"updatedAt"`
	}
	if err := json.Unmarshal([]byte(out), &v); err != nil {
		return "", time.Time{}, fmt.Errorf("decode gh issue view %d: %w", issue, err)
	}
	return v.State, v.UpdatedAt, nil
}

// followonEmissionRefs maps a fleet loop's stable identity to the follow-on refs its
// ticks emitted, in relay.ArtifactIssue pointer form ("#N"). Only the dispatch loop
// keeps such a record on this host (its own tick ledger, dispatchEmittedRefs); every
// other loop returns nil, which ClassifyFollowon folds to the zero verdict — the axis
// unread, surface-only — never a fabricated orphan. A var so tests bind hermetic refs
// and so a later leaf can widen the binding (a baton Artifact store / the
// a2achan.WorkerStatus join) without touching the fold.
var followonEmissionRefs = func(root, kind string) []string {
	if kind != "dispatch" {
		return nil
	}
	return dispatchEmittedRefs(root)
}

// dispatchEmittedRefs reads the dispatch loop's LATEST durable tick row and returns
// the downstream issue refs that tick emitted as still-open follow-on work. The
// ledger is the loop's OWN tick journal (dispatch_progress.go writes it, and this
// reader reuses the writer's schema/path constants so the two cannot drift): each
// row's `witnessed_numbers` are the issues that tick witnessed as worked yet still
// OPEN — emitted follow-on work in relay.ArtifactIssue pointer form. Only the REFS
// come from here; whether each ref advanced is re-read from live issue state, so no
// upbeat ledger self-report can make the verdict read clean.
//
// Fail closed everywhere: no ledger on this host, an unparsable tail, a foreign
// schema, or a tick that emitted nothing all yield no refs (the axis simply goes
// unread) rather than inventing an emission.
func dispatchEmittedRefs(root string) []string {
	data, err := os.ReadFile(filepath.Join(root, dispatchProgressRunsDir, dispatchProgressLogName))
	if err != nil {
		return nil
	}
	lines := strings.Split(string(data), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			continue
		}
		var row struct {
			Schema    string `json:"schema"`
			Witnessed []int  `json:"witnessed_numbers"`
		}
		if json.Unmarshal([]byte(line), &row) != nil || row.Schema != dispatchProgressSchema {
			continue
		}
		refs := make([]string, 0, len(row.Witnessed))
		for _, n := range row.Witnessed {
			if n > 0 {
				refs = append(refs, "#"+strconv.Itoa(n))
			}
		}
		return refs
	}
	return nil
}
