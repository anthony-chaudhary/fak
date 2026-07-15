package gateway

import (
	"errors"
	"testing"
)

func TestQueryNotChatPinStableAcrossSwap(t *testing.T) {
	r := NewQueryNotChatRegistry()
	task := []byte("resolve customer case CASE-417")
	opened, err := r.Open("session-1", task, "error=timeout")
	if err != nil {
		t.Fatal(err)
	}
	swapped, err := r.Swap("session-1", opened.PinnedOriginatingTask, "next=search_kb")
	if err != nil {
		t.Fatal(err)
	}
	if swapped.PinnedOriginatingTask != opened.PinnedOriginatingTask || swapped.WorkingState != "next=search_kb" || swapped.Swaps != 1 {
		t.Fatalf("opened=%+v swapped=%+v", opened, swapped)
	}
}

func TestQueryNotChatRejectsPinMutation(t *testing.T) {
	r := NewQueryNotChatRegistry()
	opened, err := r.Open("session-1", []byte("task A"), "state")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Open("session-1", []byte("task B"), "state"); !errors.Is(err, ErrQueryNotChatViolation) {
		t.Fatalf("pin replacement err=%v", err)
	}
	if _, err := r.Swap("session-1", "different-pin", "state"); !errors.Is(err, ErrQueryNotChatViolation) {
		t.Fatalf("mutated expected pin err=%v opened=%+v", err, opened)
	}
}

func TestQueryNotChatAppendObservedByDefault(t *testing.T) {
	t.Setenv(queryNotChatEnforceEnv, "")
	r := NewQueryNotChatRegistry()
	s, err := r.Open("session-1", []byte("task"), "current")
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.ObserveAppend("session-1", s.PinnedOriginatingTask, "another error turn")
	if err != nil || !got.Observed || !got.Allowed {
		t.Fatalf("default soak got=%+v err=%v", got, err)
	}
	observed, rejected := r.Metrics()
	if observed != 1 || rejected != 0 {
		t.Fatalf("metrics observed=%d rejected=%d", observed, rejected)
	}
}

func TestQueryNotChatAppendRejectedBehindFlag(t *testing.T) {
	t.Setenv(queryNotChatEnforceEnv, "true")
	r := NewQueryNotChatRegistry()
	s, err := r.Open("session-1", []byte("task"), "current")
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.ObserveAppend("session-1", s.PinnedOriginatingTask, "another error turn")
	if !errors.Is(err, ErrQueryNotChatViolation) || !got.Observed || got.Allowed {
		t.Fatalf("enforced got=%+v err=%v", got, err)
	}
	observed, rejected := r.Metrics()
	if observed != 1 || rejected != 1 {
		t.Fatalf("metrics observed=%d rejected=%d", observed, rejected)
	}
}
