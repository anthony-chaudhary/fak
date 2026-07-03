package slackoutbox

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

// cardWire is an in-memory Wire that records the full send parameters —
// including thread_ts and update targets — so card-lifecycle tests can witness
// WHERE each message landed, not just that it landed.
type cardWire struct {
	sends  []string // "post channel ts thread=<ts> text" / "update channel ts text"
	nextTS int
	byTS   map[string]string // ts -> current text (the in-place card state)
}

func newCardWire() *cardWire { return &cardWire{byTS: map[string]string{}} }

func (f *cardWire) PostMessageIdem(ctx context.Context, channel, text string, blocks []any, threadTS, nonce string) (string, error) {
	f.nextTS++
	ts := fmt.Sprintf("%d.0", f.nextTS)
	f.sends = append(f.sends, fmt.Sprintf("post %s %s thread=%s %s", channel, ts, threadTS, text))
	f.byTS[ts] = text
	return ts, nil
}

func (f *cardWire) UpdateMessage(ctx context.Context, channel, ts, text string, blocks []any) error {
	f.sends = append(f.sends, fmt.Sprintf("update %s %s %s", channel, ts, text))
	f.byTS[ts] = text
	return nil
}

func (f *cardWire) History(ctx context.Context, channel, oldestTS string, limit int) ([]slackwire.Message, error) {
	return nil, nil
}

func testCard(t *testing.T, o *Outbox) *Card {
	t.Helper()
	c, err := OpenCard(o, filepath.Join(t.TempDir(), "card.json"))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// TestCardLifecyclePostUpdateFinal is the DoD lifecycle witness: post once,
// N in-place updates coalescing to the newest state, and a final edit whose
// text is the witness fold — all through the durable outbox against a fake wire.
func TestCardLifecyclePostUpdateFinal(t *testing.T) {
	o := testOutbox(t)
	c := testCard(t, o)
	w := newCardWire()

	if err := c.Start("C1", "run r-1", "run r-1 started", nil); err != nil {
		t.Fatal(err)
	}
	if err := c.Start("C1", "run r-1", "run r-1 started", nil); !errors.Is(err, ErrCardAlreadyStarted) {
		t.Fatalf("second Start must refuse, got %v", err)
	}
	if _, err := o.Drain(context.Background(), w, drainOpts(nil)); err != nil {
		t.Fatal(err)
	}

	// Two progress updates before the next drain: only the newest reaches the wire.
	if err := c.Update("run r-1: 50%", nil); err != nil {
		t.Fatal(err)
	}
	if err := c.Update("run r-1: 90%", nil); err != nil {
		t.Fatal(err)
	}
	rep, err := o.Drain(context.Background(), w, drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Updated != 1 || rep.Superseded != 1 {
		t.Fatalf("updates must coalesce: %+v", rep)
	}

	wit := Witness{CommitSHA: "abc1234", ShipStamp: "(fak slackoutbox)", VerifySource: VerifyRegistry, ExitCode: 0, Artifact: "docs/x.md"}
	if err := c.Finalize(wit); err != nil {
		t.Fatal(err)
	}
	if _, err := o.Drain(context.Background(), w, drainOpts(nil)); err != nil {
		t.Fatal(err)
	}

	// Exactly ONE post ever; every edit targeted the same ts; the card's final
	// text is the start-frozen label plus the deterministic witness fold.
	joined := strings.Join(w.sends, "\n")
	if strings.Count(joined, "post ") != 1 {
		t.Fatalf("card must post exactly once:\n%s", joined)
	}
	if want := "run r-1 — " + wit.FinalText(); w.byTS["1.0"] != want {
		t.Fatalf("final card text = %q, want label + witness fold %q", w.byTS["1.0"], want)
	}
	if !strings.HasPrefix(wit.FinalText(), "SHIPPED · commit abc1234") {
		t.Fatalf("witness fold wrong: %q", wit.FinalText())
	}

	// The verdict is frozen: further progress updates are refused.
	if err := c.Update("late narration", nil); !errors.Is(err, ErrCardFinal) {
		t.Fatalf("update after final must refuse, got %v", err)
	}
	if err := c.Finalize(wit); !errors.Is(err, ErrCardFinal) {
		t.Fatalf("double finalize must refuse, got %v", err)
	}
}

// TestCardRestartResumesSameCard is the DoD restart witness: a brand-new
// process (fresh Outbox + fresh Card over the same files) keeps editing the
// SAME message — ts comes from run state / the outbox fold, never memory.
func TestCardRestartResumesSameCard(t *testing.T) {
	o := testOutbox(t)
	statePath := filepath.Join(t.TempDir(), "card.json")
	c, err := OpenCard(o, statePath)
	if err != nil {
		t.Fatal(err)
	}
	w := newCardWire()
	if err := c.Start("C1", "", "started", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := o.Drain(context.Background(), w, drainOpts(nil)); err != nil {
		t.Fatal(err)
	}

	// "Restart": everything rebuilt from disk.
	o2, err := Open(o.Dir())
	if err != nil {
		t.Fatal(err)
	}
	c2, err := OpenCard(o2, statePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := c2.Start("C1", "", "started", nil); !errors.Is(err, ErrCardAlreadyStarted) {
		t.Fatalf("restarted process must not post a second card, got %v", err)
	}
	if err := c2.Update("recovered, resuming", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := o2.Drain(context.Background(), w, drainOpts(nil)); err != nil {
		t.Fatal(err)
	}
	last := w.sends[len(w.sends)-1]
	if last != "update C1 1.0 recovered, resuming" {
		t.Fatalf("restart did not edit the original card: %v", w.sends)
	}
	if c2.State().TS != "1.0" {
		t.Fatalf("ts not persisted in run state: %+v", c2.State())
	}
}

// TestCardUpdateBeforePostDrainedRefuses: no ts exists until the post drains,
// so an early update fails typed (the caller retries after the next drain)
// instead of silently posting a duplicate message.
func TestCardUpdateBeforePostDrainedRefuses(t *testing.T) {
	o := testOutbox(t)
	c := testCard(t, o)
	if err := c.Update("too early", nil); !errors.Is(err, ErrCardNotStarted) {
		t.Fatalf("update before start must refuse, got %v", err)
	}
	if err := c.Start("C1", "", "started", nil); err != nil {
		t.Fatal(err)
	}
	if err := c.Update("still too early", nil); !errors.Is(err, ErrCardNotPosted) {
		t.Fatalf("update before the post drained must refuse, got %v", err)
	}
}

// TestCardFinalTextIsWitnessSourced proves the final text cannot be worker
// narration: Finalize takes only evidence fields, the verdict is derived
// (uncorroborated evidence folds to NOT_SHIPPED), and a verify source outside
// the closed vocabulary is refused outright.
func TestCardFinalTextIsWitnessSourced(t *testing.T) {
	unverified := Witness{CommitSHA: "abc1234", ShipStamp: "(fak slackoutbox)", VerifySource: VerifyNone, ExitCode: 1}
	if unverified.Verdict() != "NOT_SHIPPED" {
		t.Fatalf("verify=none must fold to NOT_SHIPPED, got %s", unverified.Verdict())
	}
	noStamp := Witness{CommitSHA: "abc1234", VerifySource: VerifyRegistry}
	if noStamp.Verdict() != "NOT_SHIPPED" {
		t.Fatalf("a bare un-stamped commit must stay NOT_SHIPPED, got %s", noStamp.Verdict())
	}

	o := testOutbox(t)
	c := testCard(t, o)
	w := newCardWire()
	if err := c.Start("C1", "", "started", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := o.Drain(context.Background(), w, drainOpts(nil)); err != nil {
		t.Fatal(err)
	}
	if err := c.Finalize(Witness{VerifySource: "trust-me"}); err == nil || !strings.Contains(err.Error(), "closed set") {
		t.Fatalf("out-of-vocabulary verify source must refuse, got %v", err)
	}
	if err := c.Finalize(unverified); err != nil {
		t.Fatal(err)
	}
	if _, err := o.Drain(context.Background(), w, drainOpts(nil)); err != nil {
		t.Fatal(err)
	}
	if got := w.byTS["1.0"]; got != unverified.FinalText() || !strings.Contains(got, "NOT_SHIPPED") {
		t.Fatalf("final text must be the witness fold: %q", got)
	}
	// The witness rides into run state for audit.
	if st := c.State(); st.Witness == nil || st.Witness.VerifySource != VerifyNone {
		t.Fatalf("witness not persisted: %+v", c.State())
	}
}

// TestCardThreadCarriesOverflowDetail: late detail posts as a threaded reply
// under the card (thread_ts = card ts), allowed even after the final edit.
func TestCardThreadCarriesOverflowDetail(t *testing.T) {
	o := testOutbox(t)
	c := testCard(t, o)
	w := newCardWire()
	if err := c.Start("C1", "", "started", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := o.Drain(context.Background(), w, drainOpts(nil)); err != nil {
		t.Fatal(err)
	}
	if err := c.Finalize(Witness{CommitSHA: "abc1234", ShipStamp: "(fak slackoutbox)", VerifySource: VerifyGrep}); err != nil {
		t.Fatal(err)
	}
	if err := c.Thread("full refusal log: ...", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := o.Drain(context.Background(), w, drainOpts(nil)); err != nil {
		t.Fatal(err)
	}
	want := "post C1 2.0 thread=1.0 full refusal log: ..."
	if last := w.sends[len(w.sends)-1]; last != want {
		t.Fatalf("thread reply wrong: got %q want %q", last, want)
	}
}
