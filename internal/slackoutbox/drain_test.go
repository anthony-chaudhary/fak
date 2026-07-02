package slackoutbox

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

// fakeWire is an in-memory Wire: posts append to a per-channel history (with the idem
// nonce riding in metadata, like live Slack), scripted errors inject failures. deliver
// controls whether a failing post still lands in history — the half-succeeded case the
// nonce probe exists for.
type fakeWire struct {
	posts   []string // "channel/text" in send order, posts and updates alike
	history map[string][]slackwire.Message
	nextTS  int

	postErrs     []error // scripted per-post errors, consumed in order (nil = success)
	deliverOnErr bool    // a failing post STILL lands in history (ambiguous half-success)
	updateErrs   []error
	historyErr   error
}

func newFakeWire() *fakeWire {
	return &fakeWire{history: map[string][]slackwire.Message{}}
}

func (f *fakeWire) nextPostErr() error {
	if len(f.postErrs) == 0 {
		return nil
	}
	err := f.postErrs[0]
	f.postErrs = f.postErrs[1:]
	return err
}

func (f *fakeWire) PostMessageIdem(ctx context.Context, channel, text string, blocks []any, threadTS, nonce string) (string, error) {
	err := f.nextPostErr()
	if err != nil && !f.deliverOnErr {
		return "", err
	}
	f.nextTS++
	ts := fmt.Sprintf("%d.0", f.nextTS)
	f.posts = append(f.posts, channel+"/"+text)
	f.history[channel] = append(f.history[channel], slackwire.Message{
		Type: "message", TS: ts, Text: text,
		Metadata: &slackwire.MessageMetadata{
			EventType:    slackwire.IdemEventType,
			EventPayload: map[string]any{"nonce": nonce},
		},
	})
	if err != nil {
		return "", err // delivered, then the response was lost — the ambiguous window
	}
	return ts, nil
}

func (f *fakeWire) UpdateMessage(ctx context.Context, channel, ts, text string, blocks []any) error {
	if len(f.updateErrs) > 0 {
		err := f.updateErrs[0]
		f.updateErrs = f.updateErrs[1:]
		if err != nil {
			return err
		}
	}
	f.posts = append(f.posts, channel+"/update:"+ts+":"+text)
	return nil
}

func (f *fakeWire) History(ctx context.Context, channel, oldestTS string, limit int) ([]slackwire.Message, error) {
	if f.historyErr != nil {
		return nil, f.historyErr
	}
	return f.history[channel], nil
}

// drainOpts returns test options: no real sleeping, waits recorded.
func drainOpts(waits *[]time.Duration) DrainOpts {
	return DrainOpts{
		Sleep: func(ctx context.Context, d time.Duration) error {
			if waits != nil {
				*waits = append(*waits, d)
			}
			return ctx.Err()
		},
	}
}

func TestDrainPostsPendingRowsFIFOAndPacesPerChannel(t *testing.T) {
	o := testOutbox(t)
	for _, text := range []string{"one", "two", "three"} {
		if _, err := o.Enqueue(Row{Channel: "C1", Text: text}); err != nil {
			t.Fatal(err)
		}
	}
	w := newFakeWire()
	var waits []time.Duration
	rep, err := o.Drain(context.Background(), w, drainOpts(&waits))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Posted != 3 || rep.Remaining != 0 {
		t.Fatalf("report wrong: %+v", rep)
	}
	if strings.Join(w.posts, ",") != "C1/one,C1/two,C1/three" {
		t.Fatalf("order wrong: %v", w.posts)
	}
	// pacing: a gap before the 2nd and 3rd send into the same channel, none before the 1st
	if len(waits) != 2 || waits[0] != time.Second {
		t.Fatalf("pacing waits = %v, want two 1s gaps", waits)
	}
	// second drain is a no-op: everything posted
	rep2, err := o.Drain(context.Background(), w, drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if rep2.Posted != 0 || len(w.posts) != 3 {
		t.Fatalf("posted rows were re-sent: %+v posts=%v", rep2, w.posts)
	}
}

func TestDrainSurvivesRestartWithoutDoublePost(t *testing.T) {
	o := testOutbox(t)
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "durable"}); err != nil {
		t.Fatal(err)
	}
	w := newFakeWire()
	if _, err := o.Drain(context.Background(), w, drainOpts(nil)); err != nil {
		t.Fatal(err)
	}
	// A brand-new process over the same dir: state replays, nothing re-sends.
	o2, err := Open(o.Dir())
	if err != nil {
		t.Fatal(err)
	}
	rep, err := o2.Drain(context.Background(), w, drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Posted != 0 || len(w.posts) != 1 {
		t.Fatalf("restart re-posted: %+v posts=%v", rep, w.posts)
	}
}

// TestDrainCrashBetweenPostAndRecordDoesNotDoublePost is the DoD crash-resume witness:
// the process dies AFTER the wire delivered but BEFORE the posted record hit disk. The
// sending intent-marker plus the nonce probe recover the truth from history — exactly
// one message ever reaches the channel.
func TestDrainCrashBetweenPostAndRecordDoesNotDoublePost(t *testing.T) {
	o := testOutbox(t)
	nonce, err := o.Enqueue(Row{Channel: "C1", Text: "exactly once"})
	if err != nil {
		t.Fatal(err)
	}

	w := newFakeWire()
	die := errors.New("simulated crash at record time")
	o.appendStateSeam = func(tr transition) error {
		if tr.State == statePosted {
			return die // the post landed; recording it "kills" the process
		}
		return appendJSONL(o.Dir()+"/state.jsonl", tr)
	}
	if _, err := o.Drain(context.Background(), w, drainOpts(nil)); !errors.Is(err, die) {
		t.Fatalf("want the simulated crash, got %v", err)
	}
	if len(w.posts) != 1 {
		t.Fatalf("the message must have left before the crash: %v", w.posts)
	}

	// Restart: fresh Outbox, healthy state appends. The row's last state is `sending`,
	// so the drain probes history for the nonce instead of re-sending.
	o2, err := Open(o.Dir())
	if err != nil {
		t.Fatal(err)
	}
	rep, err := o2.Drain(context.Background(), w, drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Recovered != 1 || rep.Posted != 0 {
		t.Fatalf("want 1 recovered / 0 posted, got %+v", rep)
	}
	if len(w.posts) != 1 {
		t.Fatalf("double post on nonce %s: %v", nonce, w.posts)
	}
	snap, _ := o2.Load()
	if got := snap.state(nonce); got.State != statePosted || got.TS == "" {
		t.Fatalf("row not recovered to posted: %+v", got)
	}
}

func TestDrainAmbiguousTransportFailureRecoversFromProbe(t *testing.T) {
	o := testOutbox(t)
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "half sent"}); err != nil {
		t.Fatal(err)
	}
	w := newFakeWire()
	w.postErrs = []error{errors.New("connection reset")} // transport error, NOT *APIError
	w.deliverOnErr = true                                // ...but the request landed

	rep, err := o.Drain(context.Background(), w, drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Failed != 1 {
		t.Fatalf("first pass must record a transient failure: %+v", rep)
	}
	rep2, err := o.Drain(context.Background(), w, drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if rep2.Recovered != 1 || len(w.posts) != 1 {
		t.Fatalf("probe did not recover the half-sent post: %+v posts=%v", rep2, w.posts)
	}
}

func TestDrainAPIErrorIsNotAmbiguousAndDeadLettersAtBudget(t *testing.T) {
	o := testOutbox(t)
	nonce, err := o.Enqueue(Row{Channel: "C1", Text: "doomed"})
	if err != nil {
		t.Fatal(err)
	}
	w := newFakeWire()
	// Every attempt gets a definitive Slack "no" (ok:false envelope — not ambiguous).
	fail := &slackwire.APIError{Method: "chat.postMessage", Status: 200, Code: "channel_not_found"}
	w.postErrs = []error{fail, fail, fail}

	opts := drainOpts(nil)
	opts.MaxAttempts = 3
	for i := 0; i < 2; i++ {
		rep, err := o.Drain(context.Background(), w, opts)
		if err != nil {
			t.Fatal(err)
		}
		if rep.Failed != 1 || rep.Dead != 0 {
			t.Fatalf("pass %d: %+v", i, rep)
		}
	}
	rep, err := o.Drain(context.Background(), w, opts)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Dead != 1 {
		t.Fatalf("third failure must dead-letter: %+v", rep)
	}
	st, _ := o.Status(time.Now())
	if st.Dead != 1 || len(st.DeadRows) != 1 || !strings.Contains(st.DeadRows[0].Reason, "channel_not_found") {
		t.Fatalf("dead row not surfaced: %+v", st)
	}
	// deliverOnErr=false: nothing ever reached the channel, and the dead row stays dead
	// on further drains (no re-send without an explicit retry).
	if _, err := o.Drain(context.Background(), w, opts); err != nil {
		t.Fatal(err)
	}
	if len(w.posts) != 0 {
		t.Fatalf("dead row was sent: %v", w.posts)
	}
	// Operator re-arm: retry works and the (now healthy) wire delivers.
	if _, err := o.Retry(nonce); err != nil {
		t.Fatal(err)
	}
	rep, err = o.Drain(context.Background(), w, opts)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Posted != 1 || len(w.posts) != 1 {
		t.Fatalf("retried row did not post: %+v posts=%v", rep, w.posts)
	}
}

func TestDrainRatelimitedStopsChannelButNotOthers(t *testing.T) {
	o := testOutbox(t)
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "head"}); err != nil {
		t.Fatal(err)
	}
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "behind the head"}); err != nil {
		t.Fatal(err)
	}
	if _, err := o.Enqueue(Row{Channel: "C2", Text: "other channel"}); err != nil {
		t.Fatal(err)
	}
	w := newFakeWire()
	w.postErrs = []error{&slackwire.APIError{Method: "chat.postMessage", Status: 429, Code: "ratelimited"}}

	rep, err := o.Drain(context.Background(), w, drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	// C1's head fails transiently; C1's second row must NOT jump the queue; C2 posts.
	if rep.Failed != 1 || rep.Posted != 1 {
		t.Fatalf("report wrong: %+v", rep)
	}
	if len(w.posts) != 1 || w.posts[0] != "C2/other channel" {
		t.Fatalf("FIFO violated or other channel blocked: %v", w.posts)
	}
	if rep.Remaining != 2 { // the failed head + the row behind it
		t.Fatalf("remaining = %d, want 2", rep.Remaining)
	}
}

func TestDrainCoalescesUpdatesToNewestState(t *testing.T) {
	o := testOutbox(t)
	for _, v := range []string{"10%", "50%", "90%"} {
		if _, err := o.Enqueue(Row{Channel: "C1", Text: v, UpdateTS: "7.7"}); err != nil {
			t.Fatal(err)
		}
	}
	w := newFakeWire()
	rep, err := o.Drain(context.Background(), w, drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Updated != 1 || rep.Superseded != 2 {
		t.Fatalf("coalescing wrong: %+v", rep)
	}
	if len(w.posts) != 1 || w.posts[0] != "C1/update:7.7:90%" {
		t.Fatalf("must ship only the newest card state: %v", w.posts)
	}
}

func TestDrainLeakFenceRefusesRowTerminally(t *testing.T) {
	o := testOutbox(t)
	needle := "node-" + "windows-a" // base PUBLIC_LEAK needle, assembled at runtime
	nonce, err := o.Enqueue(Row{Channel: "C1", Text: "ran on " + needle + " overnight"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "clean sibling"}); err != nil {
		t.Fatal(err)
	}
	w := newFakeWire()
	rep, err := o.Drain(context.Background(), w, drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Refused != 1 || rep.Posted != 1 {
		t.Fatalf("report wrong: %+v", rep)
	}
	if len(w.posts) != 1 || w.posts[0] != "C1/clean sibling" {
		t.Fatalf("leaky row must not reach the wire: %v", w.posts)
	}
	snap, _ := o.Load()
	got := snap.state(nonce)
	if got.State != stateRefused || !strings.Contains(got.Reason, "PUBLIC_LEAK") {
		t.Fatalf("refusal not structured: %+v", got)
	}
	// Refused is terminal-forever: retry must not resurrect it.
	if _, err := o.Retry(nonce); err == nil {
		t.Fatal("retrying a refused row must refuse")
	}
}

func TestDrainFencesBlocksPayloadToo(t *testing.T) {
	o := testOutbox(t)
	needle := "node-" + "windows-a"
	if _, err := o.Enqueue(Row{
		Channel: "C1", Text: "clean fallback",
		Blocks: []any{map[string]any{"type": "section", "text": "from " + needle}},
	}); err != nil {
		t.Fatal(err)
	}
	w := newFakeWire()
	rep, err := o.Drain(context.Background(), w, drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Refused != 1 || len(w.posts) != 0 {
		t.Fatalf("needle inside blocks not fenced: %+v posts=%v", rep, w.posts)
	}
}

func TestDrainSerializesViaLock(t *testing.T) {
	o := testOutbox(t)
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "x"}); err != nil {
		t.Fatal(err)
	}
	w := newFakeWire()
	release := make(chan struct{})
	entered := make(chan struct{})
	opts := drainOpts(nil)
	seamHit := false
	o.appendStateSeam = func(tr transition) error {
		if !seamHit {
			seamHit = true
			close(entered)
			<-release // hold the lock mid-drain
		}
		return appendJSONL(o.Dir()+"/state.jsonl", tr)
	}
	done := make(chan error, 1)
	go func() {
		_, err := o.Drain(context.Background(), w, opts)
		done <- err
	}()
	<-entered

	o2, err := Open(o.Dir())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := o2.Drain(context.Background(), w, drainOpts(nil)); !errors.Is(err, ErrDrainBusy) {
		t.Fatalf("second drainer must see ErrDrainBusy, got %v", err)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
