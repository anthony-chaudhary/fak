package util

import (
	"testing"
	"time"
)

func TestLogGate_Allow(t *testing.T) {
	g := NewLogGate(50 * time.Millisecond)

	// First call should be allowed
	if !g.Allow() {
		t.Fatal("first Allow() should return true")
	}

	// Immediate second call should be suppressed
	if g.Allow() {
		t.Fatal("immediate second Allow() should return false")
	}

	// Wait for interval to elapse
	time.Sleep(60 * time.Millisecond)

	// Should be allowed again
	if !g.Allow() {
		t.Fatal("Allow() after interval should return true")
	}

	// And suppressed again immediately
	if g.Allow() {
		t.Fatal("Allow() immediately after should return false")
	}
}
