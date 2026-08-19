package processstart

import (
	"os"
	"testing"
	"time"
)

func TestStartDistinguishesCurrentAndMissingProcess(t *testing.T) {
	started, ok := Start(os.Getpid())
	if !ok || started.IsZero() {
		t.Fatalf("Start(self) = %v, %v", started, ok)
	}
	if started.After(time.Now().Add(time.Second)) {
		t.Fatalf("Start(self) is in future: %v", started)
	}
	if _, ok := Start(-1); ok {
		t.Fatal("Start(-1) unexpectedly succeeded")
	}
}
