package compute

import (
	"errors"
	"testing"
)

func TestMetalOwnerLifecycleTerminalStates(t *testing.T) {
	var s metalOwnerLifecycle
	if err := s.finish(); !errors.Is(err, errMetalOwnerEmpty) {
		t.Fatalf("empty finish = %v", err)
	}
	if err := s.encode(); err != nil {
		t.Fatal(err)
	}
	if err := s.encode(); err != nil {
		t.Fatal(err)
	}
	if err := s.finish(); err != nil {
		t.Fatal(err)
	}
	if s.encoders != 2 || s.state != metalOwnerSubmitted {
		t.Fatalf("state=%v encoders=%d", s.state, s.encoders)
	}
	if err := s.encode(); !errors.Is(err, errMetalOwnerTerminal) {
		t.Fatalf("post-finish encode=%v", err)
	}
	if err := s.abort(); !errors.Is(err, errMetalOwnerTerminal) {
		t.Fatalf("post-finish abort=%v", err)
	}
}

func TestMetalOwnerLifecycleAbortIsTerminal(t *testing.T) {
	var s metalOwnerLifecycle
	if err := s.abort(); err != nil {
		t.Fatal(err)
	}
	if err := s.encode(); !errors.Is(err, errMetalOwnerTerminal) {
		t.Fatalf("post-abort encode=%v", err)
	}
	if err := s.finish(); !errors.Is(err, errMetalOwnerTerminal) {
		t.Fatalf("post-abort finish=%v", err)
	}
}
