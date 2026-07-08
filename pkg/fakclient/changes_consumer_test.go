package fakclient

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
)

// serveChanges stands up a fake GET /v1/fak/changes that returns whatever respond
// produces for the parsed `since` cursor, and hands back a Client pointed at it.
func serveChanges(t *testing.T, respond func(since uint64) ChangesResponse) *Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var since uint64
		if s := r.URL.Query().Get("since"); s != "" {
			since, _ = strconv.ParseUint(s, 10, 64)
		}
		_ = json.NewEncoder(w).Encode(respond(since))
	}))
	t.Cleanup(srv.Close)
	return New(srv.URL)
}

func collect(sink *[]uint64) ChangeHandler {
	return func(_ context.Context, ev ChangeEvent) error {
		*sink = append(*sink, ev.Seq)
		return nil
	}
}

// TestConsumerDrainsAndAdvancesCursor: a drain delivers every event in Seq order,
// advances the cursor to head, and a caught-up re-drain delivers nothing.
func TestConsumerDrainsAndAdvancesCursor(t *testing.T) {
	client := serveChanges(t, func(since uint64) ChangesResponse {
		if since >= 3 {
			return ChangesResponse{Events: nil, Cursor: 3}
		}
		return ChangesResponse{Events: []ChangeEvent{{Seq: 1}, {Seq: 2}, {Seq: 3}}, Cursor: 3}
	})
	cs := NewConsumer(client)

	var seen []uint64
	n, err := cs.Drain(context.Background(), collect(&seen))
	if err != nil || n != 3 {
		t.Fatalf("first drain: n=%d err=%v, want 3,nil", n, err)
	}
	if cs.Cursor() != 3 {
		t.Errorf("cursor=%d after drain, want 3", cs.Cursor())
	}

	n, _ = cs.Drain(context.Background(), collect(&seen))
	if n != 0 {
		t.Errorf("caught-up re-drain delivered %d, want 0", n)
	}
}

// TestConsumerResumesFromPersistedCursor: NewConsumerAt pages forward from a saved
// cursor rather than re-reading the window.
func TestConsumerResumesFromPersistedCursor(t *testing.T) {
	client := serveChanges(t, func(since uint64) ChangesResponse {
		// The server only ever hands back events after `since`.
		if since >= 5 {
			return ChangesResponse{Events: []ChangeEvent{{Seq: 6}, {Seq: 7}}, Cursor: 7}
		}
		return ChangesResponse{Events: []ChangeEvent{{Seq: 6}, {Seq: 7}}, Cursor: 7}
	})
	cs := NewConsumerAt(client, 5)

	var seen []uint64
	n, _ := cs.Drain(context.Background(), collect(&seen))
	if n != 2 || len(seen) != 2 || seen[0] != 6 {
		t.Fatalf("resumed drain: n=%d seen=%v, want 2 [6 7]", n, seen)
	}
}

// TestConsumerDedupesAtLeastOnce: delivery is at-least-once, so a re-presented Seq
// at or below the cursor is skipped — the handler sees each advance exactly once.
func TestConsumerDedupesAtLeastOnce(t *testing.T) {
	// The server re-presents seq 2 (already consumed) alongside the new seq 3.
	client := serveChanges(t, func(since uint64) ChangesResponse {
		if since == 0 {
			return ChangesResponse{Events: []ChangeEvent{{Seq: 1}, {Seq: 2}}, Cursor: 2}
		}
		return ChangesResponse{Events: []ChangeEvent{{Seq: 2}, {Seq: 3}}, Cursor: 3}
	})
	cs := NewConsumer(client)

	var seen []uint64
	cs.Drain(context.Background(), collect(&seen)) // consumes 1,2
	n, _ := cs.Drain(context.Background(), collect(&seen))
	if n != 1 {
		t.Fatalf("second drain delivered %d, want 1 (seq 2 deduped)", n)
	}
	want := []uint64{1, 2, 3}
	if len(seen) != 3 || seen[2] != 3 {
		t.Errorf("seen=%v, want %v (no duplicate seq 2)", seen, want)
	}
}

// TestConsumerReportsRetentionGap: resuming below the retained window (first event
// Seq far above since+1) raises Gapped(), which then clears.
func TestConsumerReportsRetentionGap(t *testing.T) {
	client := serveChanges(t, func(since uint64) ChangesResponse {
		return ChangesResponse{Events: []ChangeEvent{{Seq: 9}, {Seq: 10}}, Cursor: 10}
	})
	cs := NewConsumerAt(client, 5) // window has moved on to seq 9

	var seen []uint64
	cs.Drain(context.Background(), collect(&seen))
	if !cs.Gapped() {
		t.Fatal("expected Gapped()=true after resuming below the retained window")
	}
	if cs.Gapped() {
		t.Error("Gapped() should clear after being read")
	}
}

// TestConsumerFailsInPlace: a handler error stops the drain with the cursor left
// BEFORE the failing event, so the next drain retries it (at-least-once, no skip).
func TestConsumerFailsInPlace(t *testing.T) {
	client := serveChanges(t, func(since uint64) ChangesResponse {
		return ChangesResponse{Events: []ChangeEvent{{Seq: 1}, {Seq: 2}, {Seq: 3}}, Cursor: 3}
	})
	cs := NewConsumer(client)

	boom := errors.New("handler failed")
	var seen []uint64
	n, err := cs.Drain(context.Background(), func(_ context.Context, ev ChangeEvent) error {
		if ev.Seq == 2 {
			return boom
		}
		seen = append(seen, ev.Seq)
		return nil
	})
	if !errors.Is(err, boom) || n != 1 {
		t.Fatalf("drain: n=%d err=%v, want 1,boom", n, err)
	}
	if cs.Cursor() != 1 {
		t.Fatalf("cursor=%d after failing on seq 2, want 1 (left before the failure)", cs.Cursor())
	}

	// Next drain retries from seq 2 — the failing event is re-presented, not skipped.
	n, err = cs.Drain(context.Background(), collect(&seen))
	if err != nil || n != 2 {
		t.Fatalf("retry drain: n=%d err=%v, want 2,nil", n, err)
	}
	if len(seen) != 3 || seen[1] != 2 {
		t.Errorf("seen=%v, want seq 2 retried after the failure", seen)
	}
}

// TestConsumerResyncsToHeadOnEmptyWindow: when the window has fully elapsed (no
// events but a head cursor ahead), the consumer adopts head so it stops
// re-requesting the elapsed span.
func TestConsumerResyncsToHeadOnEmptyWindow(t *testing.T) {
	client := serveChanges(t, func(since uint64) ChangesResponse {
		return ChangesResponse{Events: nil, Cursor: 100}
	})
	cs := NewConsumerAt(client, 3)

	n, err := cs.Drain(context.Background(), collect(new([]uint64)))
	if err != nil || n != 0 {
		t.Fatalf("drain: n=%d err=%v, want 0,nil", n, err)
	}
	if cs.Cursor() != 100 {
		t.Errorf("cursor=%d, want 100 (re-synced to head)", cs.Cursor())
	}
}
