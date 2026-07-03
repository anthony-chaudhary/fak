package dispatchpost

import (
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

// Run cards (#2263, epic #2259): the dispatch consumer migrated from one
// terminal chat.postMessage to a thread-per-run live card. The card posts once
// when the dispatch starts, is finalized in place with the witness fold when it
// ends, and the full result body (the per-commit lines) rides in the card's
// thread — one channel line per run instead of a terminal-only message. All
// sends go through the durable slackoutbox spool, so a crash between the run
// and its report never loses the card, and a restarted process resumes editing
// the same message.

// stampRE grabs the `(fak <leaf>)` ship-stamp trailer from a commit subject —
// the same trailer `dos verify`'s referee binds on.
var stampRE = regexp.MustCompile(`\((fak [a-z0-9-]+)\)\s*$`)

// CardWitness folds the git-witnessed Result into the card's final-edit
// witness. Provenance stays honest: the commit SHA and ship-stamp come from the
// run's `git log before..after` delta (the witness the result card already
// rests on), never from the child's self-report. VerifySource is "grep" only
// when a ship-stamp trailer was actually grepped out of a landed commit
// subject; a run with no landed commit — or a landed commit with no bindable
// stamp — carries "none" and therefore folds to NOT_SHIPPED, exactly the
// exit-0-but-committed-nothing honesty Result.Shipped already enforces.
func CardWitness(res Result) slackoutbox.Witness {
	w := slackoutbox.Witness{VerifySource: slackoutbox.VerifyNone, ExitCode: res.ExitCode}
	if res.RunID != "" {
		w.Artifact = "run " + res.RunID
	}
	if !res.Shipped() {
		return w
	}
	// The newest landed commit is the first `git log --oneline` row: "sha subject".
	sha, subject, _ := strings.Cut(res.Commits[0], " ")
	w.CommitSHA = sha
	if m := stampRE.FindStringSubmatch(subject); m != nil {
		w.ShipStamp = "(" + m[1] + ")"
		w.VerifySource = slackoutbox.VerifyGrep
	}
	return w
}

// RunCard is one dispatch run's live card over the durable outbox.
type RunCard struct {
	Outbox *slackoutbox.Outbox
	Card   *slackoutbox.Card
}

// OpenRunCard opens (or, after a restart, resumes) the run's card: the outbox
// spool at outboxDir, the card's durable identity under
// <outboxDir>/cards/<loop>-<run>.json.
func OpenRunCard(outboxDir, loopID, runID string) (*RunCard, error) {
	o, err := slackoutbox.Open(outboxDir)
	if err != nil {
		return nil, err
	}
	cardsDir := filepath.Join(outboxDir, "cards")
	if err := pathutil.EnsureDir(cardsDir); err != nil {
		return nil, err
	}
	c, err := slackoutbox.OpenCard(o, filepath.Join(cardsDir, safeName(loopID)+"-"+safeName(runID)+".json"))
	if err != nil {
		return nil, err
	}
	return &RunCard{Outbox: o, Card: c}, nil
}

// Start enqueues the card's start post — the run banner the final edit will
// replace. Idempotent across restarts: a card that already posted is left
// alone (the resumed process keeps editing it).
func (rc *RunCard) Start(channel string, res Result) error {
	label := cardLabel(res)
	text := fmt.Sprintf(":hourglass_flowing_sand: %s — running `%s`", label, res.Command)
	if res.Command == "" {
		text = fmt.Sprintf(":hourglass_flowing_sand: %s — running", label)
	}
	err := rc.Card.Start(channel, label, text, nil)
	if errors.Is(err, slackoutbox.ErrCardAlreadyStarted) {
		return nil
	}
	return err
}

// Finalize enqueues the card's final witness edit plus the full result body as
// a threaded reply — the channel keeps one line per run, the detail (the
// per-commit witness lines) lands under it.
func (rc *RunCard) Finalize(res Result) error {
	if err := rc.Card.Finalize(CardWitness(res)); err != nil {
		return err
	}
	return rc.Card.Thread(res.Text(), res.Blocks())
}

// cardLabel is the run's identity line, frozen into the card at Start.
func cardLabel(res Result) string {
	loop := res.LoopID
	if loop == "" {
		loop = "dispatch"
	}
	if res.RunID == "" {
		return "dispatch " + loop
	}
	return fmt.Sprintf("dispatch %s · run `%s`", loop, res.RunID)
}

// safeName maps a loop/run id onto a filesystem-safe state-file stem.
func safeName(s string) string {
	if s == "" {
		return "run"
	}
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			return r
		}
		return '_'
	}, s)
}
