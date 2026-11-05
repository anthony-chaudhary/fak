package agent

import (
	"bytes"
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3kv"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// gatedRecordWriter blocks every Put until release is closed, then delegates to
// the real store so publication stays crash-safe. It records how many writes
// started so tests can pin the complete-only boundary.
type gatedRecordWriter struct {
	inner   *l3kv.DiskStore
	release chan struct{}
	started atomic.Int32 // Puts that began, counted before the gate so tests can pin pickup
}

func (g *gatedRecordWriter) Put(ctx context.Context, key string, payload []byte) error {
	g.started.Add(1)
	select {
	case <-g.release:
	case <-ctx.Done():
		return ctx.Err()
	}
	return g.inner.Put(ctx, key, payload)
}

// convStateAwaitPublish polls the real store until key is published or the
// deadline expires.
func convStateAwaitPublish(t *testing.T, store *l3kv.DiskStore, key string) []byte {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		got, found, err := store.GetBounded(context.Background(), key, 1<<20)
		if err == nil && found {
			return got
		}
		if time.Now().After(deadline) {
			t.Fatalf("record %s never published (found=%v err=%v)", key, found, err)
		}
		time.Sleep(2 * time.Millisecond)
	}
}

// convStateAbsent asserts the record is not yet visible in the real store.
func convStateAbsent(t *testing.T, store *l3kv.DiskStore, key string) {
	t.Helper()
	_, found, err := store.GetBounded(context.Background(), key, 1<<20)
	if err != nil || found {
		t.Fatalf("record %s visible before its full write synced (found=%v err=%v)", key, found, err)
	}
}

// convStatePattern builds a deterministic payload of size n.
func convStatePattern(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i % 251)
	}
	return b
}

// fak-test:runtime fast est=3s lane=default
// TestConversationSnapshotCompleteOnly pins the adapted TensorFold
// --learn-conversations contract on the opt-in conversation-state seam: an
// eligible owned retained state is staged into a bounded I/O queue, written and
// fsync'd by one writer, reusable only after all files are written and synced;
// saves that cannot fit are skipped; a planned stop drains to the deadline then
// cancels; and a killed server loses unfinished saves without publishing them.
func TestConversationSnapshotCompleteOnly(t *testing.T) {
	ctx := context.Background()

	t.Run("record is reusable only after the full write is synced", func(t *testing.T) {
		store, err := l3kv.NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		gate := make(chan struct{})
		w := &gatedRecordWriter{inner: store, release: gate}
		s := newWarmDiskStager(w, 1<<20, 4)
		payload := convStatePattern(4096)
		key := "aa11bb22cc33dd44ee55ff66aa77bb88cc99dd00ee11ff22aa33bb44cc55dd66"
		if _, err := s.stage(ctx, key, payload); err != nil {
			t.Fatalf("stage: %v", err)
		}
		convStateAbsent(t, store, key) // nothing half-written is ever published
		close(gate)
		if got := convStateAwaitPublish(t, store, key); !bytes.Equal(got, payload) {
			t.Fatalf("published bytes differ: got %d want %d bytes", len(got), len(payload))
		}
		if got := w.started.Load(); got != 1 {
			t.Fatalf("writer started %d puts, want 1", got)
		}
	})

	t.Run("save that exceeds the permit alone is skipped without writing", func(t *testing.T) {
		store, err := l3kv.NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		s := newWarmDiskStager(store, 100, 4)
		key := "bb22cc33dd44ee55ff66aa77bb88cc99dd00ee11ff22aa33bb44cc55dd66aa11"
		if _, err := s.stage(ctx, key, convStatePattern(150)); !errors.Is(err, errConversationStateCapacityFull) {
			t.Fatalf("stage error = %v, want capacity-full sentinel", err)
		}
		convStateAbsent(t, store, key) // skipped, never written
		if published, cancelled := s.drain(50 * time.Millisecond); published != 0 || cancelled != 0 {
			t.Fatalf("drain = (%d,%d), want (0,0)", published, cancelled)
		}
		if _, err := s.stage(ctx, key, convStatePattern(10)); !errors.Is(err, errConversationStateClosed) {
			t.Fatalf("post-drain stage error = %v, want closed sentinel", err)
		}
	})

	t.Run("save that fits alone waits for a completion when capacity is full", func(t *testing.T) {
		store, err := l3kv.NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		gate := make(chan struct{})
		w := &gatedRecordWriter{inner: store, release: gate}
		s := newWarmDiskStager(w, 100, 4)
		k1 := "cc33dd44ee55ff66aa77bb88cc99dd00ee11ff22aa33bb44cc55dd66aa11bb22"
		k2 := "dd44ee55ff66aa77bb88cc99dd00ee11ff22aa33bb44cc55dd66aa11bb22cc33"
		wantPayload := convStatePattern(60)
		if _, err := s.stage(ctx, k1, bytes.Clone(wantPayload)); err != nil {
			t.Fatalf("stage first: %v", err)
		}
		// The first save holds 60 of the 100-byte permit; the second would
		// exceed it and must wait for the first write to finish.
		second := make(chan error, 1)
		go func() {
			_, err := s.stage(ctx, k2, bytes.Clone(wantPayload))
			second <- err
		}()
		deadline := time.Now().Add(5 * time.Second)
		for w.started.Load() == 0 {
			if time.Now().After(deadline) {
				t.Fatal("writer never picked up the first save")
			}
			time.Sleep(2 * time.Millisecond)
		}
		close(gate)
		select {
		case err := <-second:
			if err != nil {
				t.Fatalf("stage second after completion: %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("second save never enqueued after the first write finished")
		}
		if got := convStateAwaitPublish(t, store, k1); !bytes.Equal(got, wantPayload) {
			t.Fatalf("first record bytes differ: got %d bytes, want %d bytes", len(got), len(wantPayload))
		}
		if got := convStateAwaitPublish(t, store, k2); !bytes.Equal(got, wantPayload) {
			t.Fatalf("second record bytes differ: got %d bytes, want %d bytes", len(got), len(wantPayload))
		}
	})

	t.Run("planned stop drains to the deadline then cancels the rest", func(t *testing.T) {
		store, err := l3kv.NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		gate := make(chan struct{}) // never released: the in-flight write cannot finish
		w := &gatedRecordWriter{inner: store, release: gate}
		s := newWarmDiskStager(w, 1<<20, 8)
		k1 := "ee55ff66aa77bb88cc99dd00ee11ff22aa33bb44cc55dd66aa11bb22cc33dd44"
		k2 := "ff66aa77bb88cc99dd00ee11ff22aa33bb44cc55dd66aa11bb22cc33dd44ee55"
		if _, err := s.stage(ctx, k1, convStatePattern(60)); err != nil {
			t.Fatalf("stage first: %v", err)
		}
		deadline := time.Now().Add(5 * time.Second)
		for w.started.Load() == 0 {
			if time.Now().After(deadline) {
				t.Fatal("writer never picked up the first save")
			}
			time.Sleep(2 * time.Millisecond)
		}
		second, err := s.stage(ctx, k2, convStatePattern(60))
		if err != nil {
			t.Fatalf("stage second: %v", err)
		}
		published, cancelled := s.drain(60 * time.Millisecond)
		if published != 0 || cancelled != 1 {
			t.Fatalf("drain = (%d,%d), want (0,1)", published, cancelled)
		}
		select {
		case err := <-second.done:
			if !errors.Is(err, errConversationStateCancelled) {
				t.Fatalf("cancelled save delivered %v, want cancellation sentinel", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("cancelled save never reported")
		}
		// A killed server loses unfinished saves (best effort): neither record
		// is published, and no torn state is visible.
		convStateAbsent(t, store, k1)
		convStateAbsent(t, store, k2)
	})

	t.Run("opt-in flag stages through persistWarmPrefixDisk and restart restores", func(t *testing.T) {
		dir := t.TempDir()
		in := warmFixtureInputs()

		newPlanner := func(dir string, convState bool) *InKernelPlanner {
			p := NewInKernelPlanner(model.NewSynthetic(warmCfg()), loadProbeTok(t), "conv-state-e2e", false, nil, false)
			p.quant = false
			cfg := WarmDiskConfig{Dir: dir, MaxEntries: 8, MaxBytes: 8 << 20}
			cfg.ConversationState = convState
			p.SetWarmDiskConfig(cfg)
			p.SetWarmDiskModelDigest("sha256:conv-state-e2e")
			return p
		}
		derive := func(p *InKernelPlanner) WarmPrefixSpec {
			spec, err := p.DeriveWarmPrefix("tenant-conv", "agent-conv", in)
			if err != nil {
				t.Fatalf("DeriveWarmPrefix: %v", err)
			}
			p.SetWarmPrefixInputs(in)
			return spec
		}

		first := newPlanner(dir, true)
		spec := derive(first)
		receipt, err := first.WarmPrefix(WithPrefixCacheIdentity(ctx, spec.Scope.Tenant, spec.Scope.Agent), spec)
		if err != nil {
			t.Fatalf("WarmPrefix: %v", err)
		}
		if receipt.Disk == nil || receipt.Disk.Outcome != WarmDiskStaged {
			t.Fatalf("opt-in persist outcome = %+v, want staged", receipt.Disk)
		}
		// The real store may already have published before the drain runs; the
		// drain must then cancel nothing, and the restart below witnesses the
		// published record.
		published, cancelled := first.DrainConversationState(ConversationStateStopDrain)
		if cancelled != 0 {
			t.Fatalf("drain = (%d,%d), want cancelled=0", published, cancelled)
		}

		// Restart witness: a fresh planner reusing the same directory and digest
		// restores the staged record — it was reusable because every file was
		// written and synced before publication.
		second := newPlanner(dir, true)
		restored, err := second.WarmPrefix(WithPrefixCacheIdentity(ctx, "tenant-conv", "agent-conv"), derive(second))
		if err != nil {
			t.Fatalf("restart WarmPrefix: %v", err)
		}
		if restored.Disk == nil || restored.Disk.Outcome != WarmDiskRestored {
			t.Fatalf("restart outcome = %+v, want restored", restored.Disk)
		}

		// Default off keeps the historical synchronous persist unchanged.
		off := newPlanner(t.TempDir(), false)
		offSpec := derive(off)
		offReceipt, err := off.WarmPrefix(WithPrefixCacheIdentity(ctx, offSpec.Scope.Tenant, offSpec.Scope.Agent), offSpec)
		if err != nil {
			t.Fatalf("default-off WarmPrefix: %v", err)
		}
		if offReceipt.Disk == nil || offReceipt.Disk.Outcome != WarmDiskPersisted {
			t.Fatalf("default-off outcome = %+v, want persisted", offReceipt.Disk)
		}
	})
}
