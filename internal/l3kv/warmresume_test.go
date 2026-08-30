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
type compliantRestoreBackend struct{ abi.KVBackend }

func (b compliantRestoreBackend) RestoreSpans(ctx context.Context, requests []abi.KVResidencyRequest) []abi.KVResidency {
	out := make([]abi.KVResidency, len(requests))
	for i, req := range requests {
		out[i], _ = b.KVBackend.RestoreSpan(ctx, req.Digest)
		if out[i].Outcome == abi.KVResidencyOK {
			out[i].Positions = req.Positions
		}
	}
	return out
}

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
	return compliantRestoreBackend{KVBackend: New(stager, store)}, store, spans
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
	backend := compliantRestoreBackend{KVBackend: New(stager, store)}

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

type warmBatchBackend struct {
	stageBatch    []abi.KVResidency
	restoreBatch  []abi.KVResidency
	stageCalls    int
	restoreCalls  int
	legacyStage   int
	legacyRestore int
}

func (*warmBatchBackend) Len() int                { return 0 }
func (*warmBatchBackend) Prefill([]int) []float32 { return nil }
func (*warmBatchBackend) Evict(int, int) int      { return 0 }
func (*warmBatchBackend) ModelID() string         { return "warm-batch-test" }
func (b *warmBatchBackend) StageSpan(context.Context, string, int, int) (abi.KVResidency, error) {
	b.legacyStage++
	return abi.KVResidency{}, errors.New("legacy stage must not be called")
}
func (b *warmBatchBackend) RestoreSpan(context.Context, string) (abi.KVResidency, error) {
	b.legacyRestore++
	return abi.KVResidency{}, errors.New("legacy restore must not be called")
}
func (b *warmBatchBackend) StageSpans(_ context.Context, _ []abi.KVResidencyRequest) []abi.KVResidency {
	b.stageCalls++
	return b.stageBatch
}
func (b *warmBatchBackend) RestoreSpans(_ context.Context, _ []abi.KVResidencyRequest) []abi.KVResidency {
	b.restoreCalls++
	return b.restoreBatch
}

type warmSerialBackend struct {
	stage   map[string]abi.KVResidency
	restore map[string]abi.KVResidency
}

func (*warmSerialBackend) Len() int                { return 0 }
func (*warmSerialBackend) Prefill([]int) []float32 { return nil }
func (*warmSerialBackend) Evict(int, int) int      { return 0 }
func (*warmSerialBackend) ModelID() string         { return "warm-serial-test" }
func (b *warmSerialBackend) StageSpan(_ context.Context, digest string, _, _ int) (abi.KVResidency, error) {
	return b.stage[digest], nil
}
func (b *warmSerialBackend) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	return b.restore[digest], nil
}

func warmReceipt(sp Span, outcome abi.KVResidencyOutcome) abi.KVResidency {
	return abi.KVResidency{Outcome: outcome, Digest: sp.Digest, Positions: sp.Positions}
}

func TestPersistPrefixNativeBatchFiltersMixedReceiptsInOrder(t *testing.T) {
	spans := []Span{{Digest: digestP0, From: 0, Positions: 100}, {Digest: digestP1, From: 100, Positions: 60}, {Digest: digestP2, From: 160, Positions: 40}}
	b := &warmBatchBackend{stageBatch: []abi.KVResidency{
		warmReceipt(spans[0], abi.KVResidencyOK), warmReceipt(spans[1], abi.KVResidencyFault), warmReceipt(spans[2], abi.KVResidencyOK),
	}}
	m, err := PersistPrefix(context.Background(), b, spans)
	if err != nil {
		t.Fatal(err)
	}
	if b.stageCalls != 1 || b.legacyStage != 0 {
		t.Fatalf("batch/legacy calls = %d/%d, want 1/0", b.stageCalls, b.legacyStage)
	}
	if m.TotalPositions != 200 || len(m.Spans) != 2 || m.Spans[0].Digest != digestP0 || m.Spans[1].Digest != digestP2 {
		t.Fatalf("manifest = %+v, want ordered OK spans and total 200", m)
	}
}

func TestRestorePrefixNativeBatchAccountsMixedReceipts(t *testing.T) {
	m := PrefixManifest{TotalPositions: 240, Spans: []ManifestSpan{
		{Digest: digestP0, Positions: 100}, {Digest: digestP1, Positions: 60}, {Digest: digestP2, Positions: 40},
	}}
	b := &warmBatchBackend{restoreBatch: []abi.KVResidency{
		{Outcome: abi.KVResidencyOK, Digest: digestP0, Positions: 100},
		{Outcome: abi.KVResidencyMiss, Digest: digestP1, Positions: 60},
		{Outcome: abi.KVResidencyFault, Digest: digestP2, Positions: 40},
	}}
	got := RestorePrefix(context.Background(), b, m)
	assertSumsToTotal(t, got)
	if b.restoreCalls != 1 || b.legacyRestore != 0 {
		t.Fatalf("batch/legacy calls = %d/%d, want 1/0", b.restoreCalls, b.legacyRestore)
	}
	if got.RestoredPositions != 100 || got.MissedPositions != 100 || got.FaultedPositions != 40 {
		t.Fatalf("report = %+v, want restored/missed/faulted 100/100/40", got)
	}
}

func TestWarmResumeSerialAndBatchAreStructurallyEquivalent(t *testing.T) {
	spans := []Span{{Digest: digestP0, From: 0, Positions: 100}, {Digest: digestP1, From: 100, Positions: 60}}
	stage := []abi.KVResidency{warmReceipt(spans[0], abi.KVResidencyOK), warmReceipt(spans[1], abi.KVResidencyMiss)}
	restore := []abi.KVResidency{warmReceipt(spans[0], abi.KVResidencyOK)}
	serial := &warmSerialBackend{
		stage:   map[string]abi.KVResidency{digestP0: stage[0], digestP1: stage[1]},
		restore: map[string]abi.KVResidency{digestP0: restore[0]},
	}
	batch := &warmBatchBackend{stageBatch: stage, restoreBatch: restore}
	serialManifest, _ := PersistPrefix(context.Background(), serial, spans)
	batchManifest, _ := PersistPrefix(context.Background(), batch, spans)
	if len(serialManifest.Spans) != len(batchManifest.Spans) || serialManifest.TotalPositions != batchManifest.TotalPositions || serialManifest.Spans[0] != batchManifest.Spans[0] {
		t.Fatalf("serial manifest %+v != batch manifest %+v", serialManifest, batchManifest)
	}
	if serialReport, batchReport := RestorePrefix(context.Background(), serial, serialManifest), RestorePrefix(context.Background(), batch, batchManifest); serialReport != batchReport {
		t.Fatalf("serial report %+v != batch report %+v", serialReport, batchReport)
	}
}

func TestRestorePrefixMalformedBatchCannotInflateRestoredPositions(t *testing.T) {
	m := PrefixManifest{TotalPositions: 160, Spans: []ManifestSpan{{Digest: digestP0, Positions: 100}, {Digest: digestP1, Positions: 60}}}
	b := &warmBatchBackend{restoreBatch: []abi.KVResidency{
		{Outcome: abi.KVResidencyOK, Digest: digestP0, Positions: 1000},
		{Outcome: abi.KVResidencyOK, Digest: digestP1, Positions: 60},
	}}
	got := RestorePrefix(context.Background(), b, m)
	assertSumsToTotal(t, got)
	if got.RestoredPositions != 60 || got.FaultedPositions != 100 {
		t.Fatalf("report = %+v, malformed receipt inflated restoration", got)
	}
}

func TestWarmResumeCancellationAccountsEveryPosition(t *testing.T) {
	spans := []Span{{Digest: digestP0, From: 0, Positions: 100}, {Digest: digestP1, From: 100, Positions: 60}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	b := &warmSerialBackend{}
	m, err := PersistPrefix(ctx, b, spans)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Spans) != 0 || m.TotalPositions != 160 {
		t.Fatalf("canceled persist manifest = %+v", m)
	}
	report := RestorePrefix(ctx, b, PrefixManifest{TotalPositions: 160, Spans: []ManifestSpan{{Digest: digestP0, Positions: 100}, {Digest: digestP1, Positions: 60}}})
	assertSumsToTotal(t, report)
	if report.FaultedPositions != 160 {
		t.Fatalf("canceled restore report = %+v, want all faulted", report)
	}
}
