package l3kv

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Three valid span-digest-shaped keys (hex, <=128 chars) for a three-span prefix.
const (
	digestP0 = "1111111111111111111111111111111111111111111111111111111111111111"
	digestP1 = "2222222222222222222222222222222222222222222222222222222222222222"
	digestP2 = "3333333333333333333333333333333333333333333333333333333333333333"
)

// threeSpanBackend wires a stagerKV (with byte-sources for three spans) through the
// durable l3kv tier, plus the spans and the store so a test can reap or fault a record.
func threeSpanBackend() (abi.KVBackend, *memStore, []Span) {
	spans := []Span{
		{Digest: digestP0, From: 0, Positions: 100},
		{Digest: digestP1, From: 100, Positions: 60},
		{Digest: digestP2, From: 160, Positions: 40},
	}
	stager := &stagerKV{spans: map[[2]int][]byte{
		{0, 100}:  spanBytes(4096),
		{100, 60}: spanBytes(2048),
		{160, 40}: spanBytes(1024),
	}, n: 200}
	store := newMemStore()
	return New(stager, store), store, spans
}

func assertSumsToTotal(t *testing.T, r ResumeReport) {
	t.Helper()
	if got := r.RestoredPositions + r.MissedPositions + r.FaultedPositions; got != r.TotalPositions {
		t.Fatalf("outcome buckets sum to %d, want TotalPositions=%d (restored=%d missed=%d faulted=%d)",
			got, r.TotalPositions, r.RestoredPositions, r.MissedPositions, r.FaultedPositions)
	}
}

// TestWarmResumeBeatsColdBaseline is the #2853 witness: after a going-idle PersistPrefix
// and a wake RestorePrefix over the durable tier, the first resumed turn's cache-read
// fraction is materially higher than a genuinely-cold baseline (no persisted prefix).
func TestWarmResumeBeatsColdBaseline(t *testing.T) {
	ctx := context.Background()
	backend, _, spans := threeSpanBackend()

	// GOING IDLE: persist the whole warm prefix. All three spans have a byte-source, so
	// all stage OK and the manifest can page every position back.
	m, err := PersistPrefix(ctx, backend, spans)
	if err != nil {
		t.Fatalf("PersistPrefix: %v", err)
	}
	if len(m.Spans) != 3 {
		t.Fatalf("manifest staged %d spans, want 3", len(m.Spans))
	}
	if m.TotalPositions != 200 {
		t.Fatalf("manifest TotalPositions = %d, want 200", m.TotalPositions)
	}

	// WAKE: page the prefix back. Every span is warm in the durable tier.
	warm := RestorePrefix(ctx, backend, m)
	assertSumsToTotal(t, warm)
	if warm.RestoredPositions != 200 {
		t.Fatalf("warm RestoredPositions = %d, want 200 (whole prefix)", warm.RestoredPositions)
	}
	if got := warm.CacheReadFraction(); got != 1.0 {
		t.Fatalf("warm CacheReadFraction = %.3f, want 1.000", got)
	}

	cmp := CompareToCold(warm)
	if got := cmp.Cold.CacheReadFraction(); got != 0.0 {
		t.Fatalf("cold-baseline CacheReadFraction = %.3f, want 0.000", got)
	}
	if !cmp.MateriallyWarmer(0.5) {
		t.Fatalf("warm resume not materially warmer than cold: warmer-by %.3f", cmp.WarmerByFraction())
	}
	if got := cmp.WarmerByFraction(); got != 1.0 {
		t.Fatalf("WarmerByFraction = %.3f, want 1.000 (warm 1.0 - cold 0.0)", got)
	}
}

// TestUnstageablePrefixStaysCold is the refute guard: wrapping a backend with NO span
// byte-source (bareKV) makes every PersistPrefix stage a FAULT, so the manifest holds no
// pageable spans and the wake fraction is 0 — an un-stageable prefix can never fake
// warmth. TotalPositions still counts the whole prefix so the loss is visible.
func TestUnstageablePrefixStaysCold(t *testing.T) {
	ctx := context.Background()
	backend := New(bareKV{}, newMemStore())
	spans := []Span{{Digest: digestP0, From: 0, Positions: 100}, {Digest: digestP1, From: 100, Positions: 60}}

	m, err := PersistPrefix(ctx, backend, spans)
	if err != nil {
		t.Fatalf("PersistPrefix: %v", err)
	}
	if len(m.Spans) != 0 {
		t.Fatalf("manifest staged %d spans over a byte-source-less backend, want 0", len(m.Spans))
	}
	if m.TotalPositions != 160 {
		t.Fatalf("manifest TotalPositions = %d, want 160", m.TotalPositions)
	}

	warm := RestorePrefix(ctx, backend, m)
	assertSumsToTotal(t, warm)
	if warm.RestoredPositions != 0 || warm.MissedPositions != 160 {
		t.Fatalf("un-stageable prefix: restored=%d missed=%d, want restored=0 missed=160", warm.RestoredPositions, warm.MissedPositions)
	}
	if got := warm.CacheReadFraction(); got != 0.0 {
		t.Fatalf("un-stageable CacheReadFraction = %.3f, want 0.000", got)
	}
	if CompareToCold(warm).MateriallyWarmer(0.01) {
		t.Fatal("un-stageable prefix reported as materially warmer than cold — fake warmth")
	}
}

// TestPartialRestoreFractionIsHonest proves a prefix the durable tier only PARTLY holds
// on wake reports the true partial fraction, not a rounded-to-full hit. The middle span
// is reaped from the store between persist and wake, so it MISSes; the two survivors are
// warm. The fraction lands strictly between cold (0) and full (1).
func TestPartialRestoreFractionIsHonest(t *testing.T) {
	ctx := context.Background()
	backend, store, spans := threeSpanBackend()

	m, err := PersistPrefix(ctx, backend, spans)
	if err != nil {
		t.Fatalf("PersistPrefix: %v", err)
	}
	// The tier reaps the middle span (60 positions) while the session was idle.
	delete(store.m, digestP1)

	warm := RestorePrefix(ctx, backend, m)
	assertSumsToTotal(t, warm)
	if warm.RestoredPositions != 140 { // 100 + 40 survive; 60 reaped
		t.Fatalf("partial RestoredPositions = %d, want 140", warm.RestoredPositions)
	}
	if warm.MissedPositions != 60 {
		t.Fatalf("partial MissedPositions = %d, want 60", warm.MissedPositions)
	}
	if got := warm.CacheReadFraction(); got != 0.7 { // 140/200
		t.Fatalf("partial CacheReadFraction = %.3f, want 0.700", got)
	}
	cmp := CompareToCold(warm)
	if !cmp.MateriallyWarmer(0.5) {
		t.Fatalf("partial resume not materially warmer than cold: warmer-by %.3f", cmp.WarmerByFraction())
	}
}

// faultyStore wraps a memStore and returns a typed FAULT (a non-nil error) on Get for
// one chosen key — a store/integrity failure the wake must count as cold, never a hit.
type faultyStore struct {
	inner    *memStore
	faultKey string
}

func (s *faultyStore) Put(ctx context.Context, key string, payload []byte) error {
	return s.inner.Put(ctx, key, payload)
}

func (s *faultyStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	if key == s.faultKey {
		return nil, false, errors.New("l3kv: simulated integrity fault")
	}
	return s.inner.Get(ctx, key)
}

// TestRestoreFaultCountsCold proves a store/integrity FAULT on wake re-prefills cold
// (FaultedPositions), never a silent wrong hit inflating the cache-read fraction.
func TestRestoreFaultCountsCold(t *testing.T) {
	ctx := context.Background()
	spans := []Span{
		{Digest: digestP0, From: 0, Positions: 100},
		{Digest: digestP1, From: 100, Positions: 60},
	}
	stager := &stagerKV{spans: map[[2]int][]byte{
		{0, 100}:  spanBytes(4096),
		{100, 60}: spanBytes(2048),
	}, n: 160}
	store := &faultyStore{inner: newMemStore(), faultKey: digestP1}
	backend := New(stager, store)

	m, err := PersistPrefix(ctx, backend, spans)
	if err != nil {
		t.Fatalf("PersistPrefix: %v", err)
	}
	warm := RestorePrefix(ctx, backend, m)
	assertSumsToTotal(t, warm)
	if warm.RestoredPositions != 100 || warm.FaultedPositions != 60 {
		t.Fatalf("fault wake: restored=%d faulted=%d, want restored=100 faulted=60", warm.RestoredPositions, warm.FaultedPositions)
	}
	if got := warm.CacheReadFraction(); got != 0.625 { // 100/160 — the faulted span is NOT counted warm
		t.Fatalf("fault-wake CacheReadFraction = %.3f, want 0.625", got)
	}
}

// TestPersistNilBackendErrors proves the one hard error: a nil backend has nothing to
// stage through, so PersistPrefix refuses rather than returning a phantom manifest.
func TestPersistNilBackendErrors(t *testing.T) {
	if _, err := PersistPrefix(context.Background(), nil, []Span{{Digest: digestP0, Positions: 10}}); err == nil {
		t.Fatal("PersistPrefix(nil backend) returned no error")
	}
}
