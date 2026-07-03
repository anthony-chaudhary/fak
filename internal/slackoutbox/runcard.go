package slackoutbox

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
)

// Run cards (#2263, epic #2259): one Slack message per run, updated in place.
// A Card wraps the outbox with the card lifecycle — post once, chat.update for
// progress, a final edit whose text is RENDERED FROM a Witness (commit SHA,
// ship-stamp, `dos verify` source, exit code, artifact pointer), never from the
// worker's self-report. The card's durable identity (post nonce, channel, ts)
// lives in a run-state file, not in memory, so a process restart resumes
// updating the same card. Results are read from run state and posted out-of-band
// — never parsed back from channel history.

// cardSource marks outbox rows a Card enqueued, for status/dead reporting.
const cardSource = "runcard"

// Card lifecycle errors, distinguishable with errors.Is.
var (
	// ErrCardAlreadyStarted means Start was called on a card whose post row is
	// already spooled — a card posts exactly once.
	ErrCardAlreadyStarted = errors.New("slackoutbox: card already started")
	// ErrCardNotStarted means Update/Finalize/Thread was called before Start.
	ErrCardNotStarted = errors.New("slackoutbox: card not started")
	// ErrCardNotPosted means the card's post has not drained yet, so no ts exists
	// to update — drain the outbox first, then retry.
	ErrCardNotPosted = errors.New("slackoutbox: card post not drained yet (no ts)")
	// ErrCardFinal means the card already carries its final witness edit; further
	// progress updates would overwrite the verdict and are refused.
	ErrCardFinal = errors.New("slackoutbox: card is final")
)

// VerifySource values — the closed set of places a ship claim can be
// corroborated from, mirroring `dos verify`'s answer. Anything else is refused
// at Finalize (fail closed, never post an unverifiable verdict).
const (
	VerifyRegistry = "registry"
	VerifyGrep     = "grep"
	VerifyNone     = "none"
)

// Witness is the evidence a final card edit is rendered from. There is no
// free-text field and no verdict field: the verdict is DERIVED from the
// evidence (Verdict), so a worker cannot narrate "shipped" — it can only
// present a commit, a stamp, and where `dos verify` corroborated them.
type Witness struct {
	CommitSHA    string `json:"commit_sha,omitempty"`
	ShipStamp    string `json:"ship_stamp,omitempty"` // the (fak <leaf>) trailer bound to the commit
	VerifySource string `json:"verify_source"`        // registry | grep | none
	ExitCode     int    `json:"exit_code"`            // the run's exit code
	Artifact     string `json:"artifact,omitempty"`   // pointer to the proof artifact
}

// valid reports whether the witness draws its verify source from the closed set.
func (w Witness) valid() error {
	switch w.VerifySource {
	case VerifyRegistry, VerifyGrep, VerifyNone:
		return nil
	}
	return fmt.Errorf("slackoutbox: witness verify_source %q is not in the closed set (registry|grep|none)", w.VerifySource)
}

// Verdict derives SHIPPED/NOT_SHIPPED from the evidence: a commit, a bindable
// ship-stamp, and a corroborating verify source. An un-corroborated claim
// (source "none", or a missing SHA/stamp) is NOT_SHIPPED — the same discipline
// `fak commit`'s referee applies.
func (w Witness) Verdict() string {
	if w.CommitSHA != "" && w.ShipStamp != "" &&
		(w.VerifySource == VerifyRegistry || w.VerifySource == VerifyGrep) {
		return "SHIPPED"
	}
	return "NOT_SHIPPED"
}

// FinalText is the deterministic fold of the witness into the final card line —
// the ONLY text Finalize will post.
func (w Witness) FinalText() string {
	parts := []string{w.Verdict()}
	if w.CommitSHA != "" {
		parts = append(parts, "commit "+w.CommitSHA)
	}
	if w.ShipStamp != "" {
		parts = append(parts, w.ShipStamp)
	}
	parts = append(parts, "verify="+w.VerifySource, fmt.Sprintf("exit=%d", w.ExitCode))
	if w.Artifact != "" {
		parts = append(parts, "artifact="+w.Artifact)
	}
	return strings.Join(parts, " · ")
}

// CardState is the durable identity of one run card — persisted in the run's
// state file so a restarted process resumes editing the same message.
type CardState struct {
	Channel   string   `json:"channel,omitempty"`
	PostNonce string   `json:"post_nonce,omitempty"`
	TS        string   `json:"ts,omitempty"` // resolved from the outbox after the post drains
	Final     bool     `json:"final,omitempty"`
	Witness   *Witness `json:"witness,omitempty"`
}

// Card is one run's live Slack card bound to an outbox and a state file. It is
// a producer: every send is an enqueued intent; the outbox drainer owns the
// wire (pacing, leak fence, idempotency probe).
type Card struct {
	o    *Outbox
	path string
	st   CardState
}

// OpenCard loads (or initializes) the card state at path, bound to outbox o.
// A missing file is a fresh, un-started card.
func OpenCard(o *Outbox, path string) (*Card, error) {
	if path == "" {
		return nil, fmt.Errorf("slackoutbox: card state path is required")
	}
	c := &Card{o: o, path: path}
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return c, nil
		}
		return nil, err
	}
	if err := json.Unmarshal(b, &c.st); err != nil {
		return nil, fmt.Errorf("slackoutbox: card state %s: %w", path, err)
	}
	return c, nil
}

// State returns a copy of the persisted card state (for diagnostics).
func (c *Card) State() CardState { return c.st }

// save persists the state via write-temp-then-rename so a crash never leaves a
// torn state file.
func (c *Card) save() error {
	b, err := json.Marshal(c.st)
	if err != nil {
		return err
	}
	tmp := c.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, c.path)
}

// Start enqueues the card's one post. The nonce is persisted to run state
// BEFORE the spool append, so a crash between the two re-enqueues under the
// SAME nonce on the next Start instead of minting a second card; once the spool
// row exists, Start refuses (post once).
func (c *Card) Start(channel, text string, blocks []any) error {
	if c.st.PostNonce != "" {
		snap, err := c.o.Load()
		if err != nil {
			return err
		}
		for _, r := range snap.Rows {
			if r.Nonce == c.st.PostNonce {
				return ErrCardAlreadyStarted
			}
		}
		// Crash window: state persisted, spool append never landed — finish it.
		_, err = c.o.Enqueue(Row{Nonce: c.st.PostNonce, Channel: c.st.Channel, Text: text, Blocks: blocks, Source: cardSource})
		return err
	}
	if channel == "" {
		return fmt.Errorf("slackoutbox: card start: channel is required")
	}
	c.st.Channel = channel
	c.st.PostNonce = NewNonce()
	if err := c.save(); err != nil {
		return err
	}
	_, err := c.o.Enqueue(Row{Nonce: c.st.PostNonce, Channel: channel, Text: text, Blocks: blocks, Source: cardSource})
	return err
}

// ensureTS resolves (and persists) the card's message ts from the outbox state
// — the post's drain outcome, never channel history.
func (c *Card) ensureTS() (string, error) {
	if c.st.TS != "" {
		return c.st.TS, nil
	}
	if c.st.PostNonce == "" {
		return "", ErrCardNotStarted
	}
	snap, err := c.o.Load()
	if err != nil {
		return "", err
	}
	s := snap.state(c.st.PostNonce)
	if s.State != statePosted || s.TS == "" {
		return "", ErrCardNotPosted
	}
	c.st.TS = s.TS
	if err := c.save(); err != nil {
		return "", err
	}
	return c.st.TS, nil
}

// Update enqueues an in-place progress edit of the card. Consecutive updates
// share the card key, so the drainer coalesces to the newest state. Refused
// after Finalize — progress must never overwrite the verdict.
func (c *Card) Update(text string, blocks []any) error {
	if c.st.Final {
		return ErrCardFinal
	}
	ts, err := c.ensureTS()
	if err != nil {
		return err
	}
	_, err = c.o.Enqueue(Row{Channel: c.st.Channel, Text: text, Blocks: blocks, UpdateTS: ts, Source: cardSource})
	return err
}

// Finalize enqueues the card's final edit, rendered from the witness alone.
// There is no text parameter: the caller supplies evidence, the fold decides
// what the channel reads. The witness rides into run state for audit.
func (c *Card) Finalize(w Witness) error {
	if c.st.Final {
		return ErrCardFinal
	}
	if err := w.valid(); err != nil {
		return err
	}
	ts, err := c.ensureTS()
	if err != nil {
		return err
	}
	if _, err := c.o.Enqueue(Row{Channel: c.st.Channel, Text: w.FinalText(), UpdateTS: ts, Source: cardSource + ":final"}); err != nil {
		return err
	}
	c.st.Final = true
	c.st.Witness = &w
	return c.save()
}

// Thread enqueues overflow detail (long logs, refusal reasons) as a reply in
// the card's thread, keeping the channel one line per run. Allowed after
// Finalize — late detail never edits the card itself.
func (c *Card) Thread(text string, blocks []any) error {
	ts, err := c.ensureTS()
	if err != nil {
		return err
	}
	_, err = c.o.Enqueue(Row{Channel: c.st.Channel, Text: text, Blocks: blocks, ThreadTS: ts, Source: cardSource})
	return err
}
