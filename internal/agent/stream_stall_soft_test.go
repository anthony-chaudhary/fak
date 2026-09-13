package agent

// stream_stall_soft_test.go — #10638: the SOFT, NON-TERMINAL no-progress diagnostic that must
// fire BEFORE the destructive progress deadline ends a turn. The sglang soft-watchdog pattern:
// observe an alive-but-stalled stream (elapsed silence + retry attempt) without killing it, so
// a slow turn still has every chance to finish, while the hard deadline stays the only thing
// that terminates. These drive the stallReader directly with a controllable blocking body, so
// they pin the strike/reset/hard-ceiling contract without a live upstream.

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"
	"time"
)

// softProbeBody is a ReadCloser whose Read parks until the test closes it, modelling a stream
// that opened and then went completely silent. Close unblocks the parked Read with io.EOF so
// the reader goroutine can return once a deadline trips.
type softProbeBody struct {
	once sync.Once
	done chan struct{}
	mu   sync.Mutex
	n    int
}

func newSoftProbeBody() *softProbeBody { return &softProbeBody{done: make(chan struct{})} }

func (b *softProbeBody) Read(p []byte) (int, error) {
	<-b.done
	return 0, io.EOF
}

func (b *softProbeBody) Close() error {
	b.once.Do(func() { close(b.done) })
	return nil
}

func (b *softProbeBody) reads() int { b.mu.Lock(); defer b.mu.Unlock(); return b.n }

// TestStallReaderSoftStrikeIsNonTerminalAndPrecedesHard is the core #10638 witness: with a soft
// window strictly below the hard progress window, the soft observer fires FIRST, the body is
// still OPEN at that instant (the turn was not killed), and only the later hard deadline closes
// it and surfaces ErrUpstreamStalled with the no-progress cause.
func TestStallReaderSoftStrikeIsNonTerminalAndPrecedesHard(t *testing.T) {
	body := newSoftProbeBody()
	hard := 900 * time.Millisecond
	soft := 250 * time.Millisecond
	sr := newStallReader(body, 0, hard, 0) // idle disabled: only the progress deadlines can fire
	defer sr.Close()

	var mu sync.Mutex
	var strikes []SoftProgressStall
	closedAtStrike := true
	sr.armSoftProgress(soft, func() int { return 3 }, func(st SoftProgressStall) {
		mu.Lock()
		strikes = append(strikes, st)
		select {
		case <-body.done:
			closedAtStrike = true
		default:
			closedAtStrike = false
		}
		mu.Unlock()
	})

	readErr := make(chan error, 1)
	start := time.Now()
	go func() {
		buf := make([]byte, 8)
		_, err := sr.Read(buf)
		readErr <- err
	}()

	// Wait for the soft strike (well before the hard window).
	deadline := time.After(hard)
	for {
		mu.Lock()
		n := len(strikes)
		mu.Unlock()
		if n > 0 {
			break
		}
		select {
		case <-deadline:
			t.Fatal("soft strike never fired — the diagnostic deadline is dead (#10638)")
		case <-time.After(10 * time.Millisecond):
		}
	}
	elapsedAtStrike := time.Since(start)

	mu.Lock()
	got := strikes[0]
	wasClosed := closedAtStrike
	mu.Unlock()
	if wasClosed {
		t.Fatalf("body was already CLOSED at the soft strike — the diagnostic killed the turn (#10638)")
	}
	if got.ElapsedSinceProgress != soft || got.Window != soft {
		t.Fatalf("strike = %+v, want elapsed=window=%s", got, soft)
	}
	if got.RetryAttempt != 3 {
		t.Fatalf("RetryAttempt = %d, want the 3 reported by the accessor", got.RetryAttempt)
	}
	if elapsedAtStrike >= hard {
		t.Fatalf("soft strike at %s, not before the %s hard window — no lead time", elapsedAtStrike, hard)
	}

	// The hard deadline must still end the silent turn, after the soft diagnostic.
	select {
	case err := <-readErr:
		if err == nil {
			t.Fatal("Read returned nil after a fully silent stream — hard ceiling did not fire")
		}
		if err != ErrUpstreamStalled {
			t.Fatalf("Read err = %v, want ErrUpstreamStalled", err)
		}
		kind, window := sr.stallCause()
		if kind != stallKindNoProgress || window != hard {
			t.Fatalf("stallCause = (%q, %s), want (%q, %s)", kind, window, stallKindNoProgress, hard)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("hard deadline never ended the turn — a silent stream rode past its ceiling")
	}
}

// TestStallReaderSoftStrikeOncePerSilence pins the anti-storm rule: a single long silence
// produces exactly ONE diagnostic, not a ticker. The soft timer is not re-armed on its own
// fire; only noteProgress re-arms it.
func TestStallReaderSoftStrikeOncePerSilence(t *testing.T) {
	body := newSoftProbeBody()
	sr := newStallReader(body, 0, 5*time.Second, 0)
	defer sr.Close()

	var count int
	var mu sync.Mutex
	sr.armSoftProgress(200*time.Millisecond, nil, func(SoftProgressStall) {
		mu.Lock()
		count++
		mu.Unlock()
	})
	time.Sleep(700 * time.Millisecond)
	mu.Lock()
	got := count
	mu.Unlock()
	if got != 1 {
		t.Fatalf("soft strike fired %d times in one silent interval, want exactly 1", got)
	}
}

// TestStallReaderSoftStrikeResetsOnProgress is the other half of the contract: a turn that IS
// advancing resets the soft interval, so the diagnostic does not fire while progress keeps
// arriving. The silence after the last progress still strikes, on a fresh window.
func TestStallReaderSoftStrikeResetsOnProgress(t *testing.T) {
	body := newSoftProbeBody()
	sr := newStallReader(body, 0, 10*time.Second, 0)
	defer sr.Close()

	var strikes int
	var mu sync.Mutex
	sr.armSoftProgress(400*time.Millisecond, nil, func(SoftProgressStall) {
		mu.Lock()
		strikes++
		mu.Unlock()
	})
	// Progress every 200ms for 1s: strictly inside the 400ms soft window, so it must never fire.
	for i := 0; i < 5; i++ {
		time.Sleep(200 * time.Millisecond)
		sr.noteProgress()
	}
	mu.Lock()
	mid := strikes
	mu.Unlock()
	if mid != 0 {
		t.Fatalf("soft strike fired %d times while progress was arriving, want 0", mid)
	}
	// Then go quiet: the interval restarts, and one strike lands on the fresh window.
	select {
	case <-time.After(2 * time.Second):
	}
	deadline := time.After(2 * time.Second)
	for {
		mu.Lock()
		n := strikes
		mu.Unlock()
		if n >= 1 {
			break
		}
		select {
		case <-deadline:
			t.Fatal("soft strike never fired after progress stopped — reset left it disarmed")
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// TestStallReaderSoftProgressDisabled pins the off switches: a non-positive soft window or a
// nil observer leaves the diagnostic disabled, so prior behavior is preserved byte-for-byte.
func TestStallReaderSoftProgressDisabled(t *testing.T) {
	for _, c := range []struct {
		name   string
		window time.Duration
		obs    func(SoftProgressStall)
	}{
		{"zero window", 0, func(SoftProgressStall) {}},
		{"negative window", -time.Second, func(SoftProgressStall) {}},
		{"nil observer", time.Second, nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			body := newSoftProbeBody()
			sr := newStallReader(body, 0, 0, 0)
			defer sr.Close()
			sr.armSoftProgress(c.window, nil, c.obs)
			if sr.softProgressTimer != nil {
				t.Fatalf("softProgressTimer armed for %s, want disabled", c.name)
			}
		})
	}
}

// TestStreamSoftProgressWindowResolvesTheConfigField pins the resolver: zero derives from the
// hard window, a negative value is the explicit off switch, an in-band value is honored, and a
// value at/above the hard window falls back to the derived default — a soft deadline with no
// lead time is refused.
func TestStreamSoftProgressWindowResolvesTheConfigField(t *testing.T) {
	const hard = 300 * time.Second
	cases := []struct {
		name string
		set  time.Duration
		hard time.Duration
		want time.Duration
	}{
		{"unset derives hard/3", 0, hard, hard / DefaultStreamSoftProgressRatio},
		{"negative is the off switch", -1, hard, 0},
		{"in-band is honored", 45 * time.Second, hard, 45 * time.Second},
		{"at the hard window falls back", hard, hard, hard / DefaultStreamSoftProgressRatio},
		{"above the hard window falls back", 400 * time.Second, hard, hard / DefaultStreamSoftProgressRatio},
		{"below the floor falls back", 2 * time.Second, hard, hard / DefaultStreamSoftProgressRatio},
		{"no hard window disables", 45 * time.Second, 0, 0},
	}
	for _, c := range cases {
		p := &HTTPPlanner{StreamSoftProgressTimeout: c.set}
		if got := p.streamSoftProgressWindow(c.hard); got != c.want {
			t.Errorf("%s: streamSoftProgressWindow(%s) with set=%s = %s, want %s", c.name, c.hard, c.set, got, c.want)
		}
	}
}

// TestCompleteStreamSoftDiagnosticPrecedesHardStall is the end-to-end #10638 witness on the real
// OpenAI-wire path: a keepalive-warm upstream that never advances the turn fires the soft
// observer FIRST (with the configured soft window and the attempt the stream opened on), leaves
// the turn running, and only then trips the hard progress deadline. The soft strike must be
// observed strictly before the terminal error.
func TestCompleteStreamSoftDiagnosticPrecedesHardStall(t *testing.T) {
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")
	const prefix = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"
	const keepalive = ": keepalive\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":null}]}\n\n"
	srv := keepaliveOnlySSEServer(t, "text/event-stream", prefix, keepalive, 200*time.Millisecond)

	p := NewHTTPPlanner(srv.URL, "m", "")
	p.StreamProgressTimeout = 6 * time.Second // the hard ceiling (>= the 5s progress floor)
	p.StreamSoftProgressTimeout = 5 * time.Second
	strikes := make(chan SoftProgressStall, 4)
	p.SoftStallNotify = func(st SoftProgressStall) { strikes <- st }

	done := make(chan error, 1)
	go func() {
		_, err := p.CompleteStream(context.Background(), func(string) error { return nil },
			[]Message{{Role: RoleUser, Content: "hi"}}, nil)
		done <- err
	}()

	var strike SoftProgressStall
	select {
	case strike = <-strikes:
	case <-time.After(9 * time.Second):
		t.Fatal("soft diagnostic never fired before the hard stall (#10638)")
	}
	if strike.Window != 5*time.Second || strike.ElapsedSinceProgress != 5*time.Second {
		t.Fatalf("strike = %+v, want window/elapsed 5s", strike)
	}

	select {
	case err := <-done:
		var stalled *UpstreamStalledError
		if !errors.As(err, &stalled) {
			t.Fatalf("err = %v, want *UpstreamStalledError (hard ceiling)", err)
		}
		if stalled.Kind != stallKindNoProgress {
			t.Fatalf("Kind = %q, want %q", stalled.Kind, stallKindNoProgress)
		}
		if stalled.Idle != 6*time.Second {
			t.Fatalf("hard window = %s, want 6s", stalled.Idle)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("hard ceiling never ended the silent turn (#10638)")
	}
}
