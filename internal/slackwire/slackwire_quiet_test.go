package slackwire

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// TestQuietBrakesEveryWriteWithoutHTTP is the core contract: while the brake is
// engaged, all three write methods return ErrQuiet and NOT ONE request reaches the
// server. The request counter is the witness — a brake that returned ErrQuiet after
// sending would still be a leak.
func TestQuietBrakesEveryWriteWithoutHTTP(t *testing.T) {
	var hits int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&hits, 1)
		_, _ = w.Write([]byte(`{"ok":true,"ts":"1.2"}`))
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv, WithQuiet(func() bool { return true }))
	ctx := context.Background()

	if _, err := c.PostMessage(ctx, "C1", "hi", nil, ""); !errors.Is(err, ErrQuiet) {
		t.Errorf("PostMessage err = %v, want ErrQuiet", err)
	}
	if _, err := c.PostMessageIdem(ctx, "C1", "hi", nil, "", "nonce"); !errors.Is(err, ErrQuiet) {
		t.Errorf("PostMessageIdem err = %v, want ErrQuiet", err)
	}
	if err := c.UpdateMessage(ctx, "C1", "1.2", "edit", nil); !errors.Is(err, ErrQuiet) {
		t.Errorf("UpdateMessage err = %v, want ErrQuiet", err)
	}
	if err := c.DeleteMessage(ctx, "C1", "1.2"); !errors.Is(err, ErrQuiet) {
		t.Errorf("DeleteMessage err = %v, want ErrQuiet", err)
	}
	if got := atomic.LoadInt64(&hits); got != 0 {
		t.Fatalf("server saw %d request(s) while quiet; the brake must refuse before HTTP", got)
	}
}

// TestQuietLeavesReadsLive is the other half: a health probe must still be able to
// observe the workspace while the fleet is muted, so history/replies/auth.test are
// NOT braked. Without this, engaging the switch would blind the very watchdog that
// reports on it.
func TestQuietLeavesReadsLive(t *testing.T) {
	var hits int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&hits, 1)
		switch r.URL.Path {
		case "/auth.test":
			_, _ = w.Write([]byte(`{"ok":true,"user_id":"U1","team_id":"T1"}`))
		default:
			_, _ = w.Write([]byte(`{"ok":true,"messages":[]}`))
		}
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv, WithQuiet(func() bool { return true }))
	ctx := context.Background()

	if _, err := c.History(ctx, "C1", "", 10); err != nil {
		t.Errorf("History err = %v, want nil (reads stay live while quiet)", err)
	}
	if _, err := c.Replies(ctx, "C1", "1.2", 10); err != nil {
		t.Errorf("Replies err = %v, want nil (reads stay live while quiet)", err)
	}
	if _, err := c.AuthTest(ctx); err != nil {
		t.Errorf("AuthTest err = %v, want nil (reads stay live while quiet)", err)
	}
	if got := atomic.LoadInt64(&hits); got != 3 {
		t.Fatalf("server saw %d read request(s), want 3", got)
	}
}

// TestQuietDisengagedWritesNormally pins the default: a brake that reports false must
// not change behavior at all, so an unset switch is byte-for-byte the old transport.
func TestQuietDisengagedWritesNormally(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"ok":true,"ts":"111.222"}`))
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv, WithQuiet(func() bool { return false }))
	ts, err := c.PostMessage(context.Background(), "C1", "hi", nil, "")
	if err != nil {
		t.Fatalf("PostMessage err = %v, want nil when the brake is disengaged", err)
	}
	if ts != "111.222" {
		t.Errorf("ts = %q, want %q", ts, "111.222")
	}
}

// TestNilQuietIsUnbraked keeps the fail-safe direction: a Client built without the
// option (every existing caller) must post exactly as before.
func TestNilQuietIsUnbraked(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"ok":true,"ts":"111.222"}`))
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv)
	if _, err := c.PostMessage(context.Background(), "C1", "hi", nil, ""); err != nil {
		t.Fatalf("PostMessage err = %v, want nil for a Client with no quiet option", err)
	}
}
