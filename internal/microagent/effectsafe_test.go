package microagent_test

import (
	"context"
	"errors"
	"github.com/anthony-chaudhary/fak/internal/idempotency"
	"github.com/anthony-chaudhary/fak/internal/microagent"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestEffectCoordinatorSerializesConflictAndDedupesRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "effects.jsonl")
	s, err := idempotency.Open(path, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	c := microagent.NewEffectCoordinator(s)
	started := make(chan struct{})
	release := make(chan struct{})
	var applied atomic.Int32
	intent := microagent.EffectIntent{ContextID: "a", Capability: "write", Resource: "row:1", Operation: "set", IdempotencyToken: "k1"}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, _ = c.Run(context.Background(), intent, []string{"write"}, func() (string, error) { applied.Add(1); close(started); <-release; return "v1", nil }, func(context.Context, microagent.EffectIntent, string) error { return nil })
	}()
	<-started
	other := intent
	other.ContextID = "b"
	if _, err = c.Run(context.Background(), other, []string{"write"}, func() (string, error) { return "bad", nil }, func(context.Context, microagent.EffectIntent, string) error { return nil }); !errors.Is(err, microagent.ErrEffectConflict) {
		t.Fatalf("conflict err=%v", err)
	}
	close(release)
	wg.Wait()
	s2, err := idempotency.Open(path, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	c2 := microagent.NewEffectCoordinator(s2)
	out, err := c2.Run(context.Background(), intent, []string{"write"}, func() (string, error) { applied.Add(1); return "bad", nil }, func(_ context.Context, _ microagent.EffectIntent, result string) error {
		if result != "v1" {
			return errors.New("wrong readback")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Replayed || applied.Load() != 1 {
		t.Fatalf("out=%+v applied=%d", out, applied.Load())
	}
}
func TestEffectCoordinatorCapabilityAndIndependentReadback(t *testing.T) {
	s, _ := idempotency.Open(filepath.Join(t.TempDir(), "e"), time.Hour)
	c := microagent.NewEffectCoordinator(s)
	in := microagent.EffectIntent{ContextID: "x", Capability: "write", Resource: "r", Operation: "set", IdempotencyToken: "k"}
	if _, err := c.Run(context.Background(), in, []string{"read"}, func() (string, error) { return "x", nil }, func(context.Context, microagent.EffectIntent, string) error { return nil }); !errors.Is(err, microagent.ErrAuthorityRefused) {
		t.Fatal(err)
	}
	if _, err := c.Run(context.Background(), in, []string{"write"}, func() (string, error) { return "x", nil }, func(context.Context, microagent.EffectIntent, string) error { return errors.New("not observed") }); err == nil {
		t.Fatal("expected verification refusal")
	}
}
