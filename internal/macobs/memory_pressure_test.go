package macobs

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// mockOpenCodePageManager implements OpenCodePageManager for testing.
type mockOpenCodePageManager struct {
	mu             sync.Mutex
	totalPages     int
	activeSessions int
	reclaimCalls   int
	reclaimedBytes uint64
	pageOutCalls   int
	pagedOutBlocks int
	pagedOutBytes  uint64
	bytesPerPage   uint64
	failReclaim    bool
	failPageOut    bool
}

func newMockOpenCodePageManager(totalPages, activeSessions int) *mockOpenCodePageManager {
	return &mockOpenCodePageManager{
		totalPages:     totalPages,
		activeSessions: activeSessions,
		bytesPerPage:   16384, // 16 KB pages on Apple Silicon
	}
}

func (m *mockOpenCodePageManager) ReclaimUnpinnedPages(targetBytes uint64) (uint64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.reclaimCalls++
	if m.failReclaim {
		return 0, context.DeadlineExceeded
	}
	reclaimed := targetBytes
	maxPossible := uint64(m.totalPages) * m.bytesPerPage
	if reclaimed > maxPossible {
		reclaimed = maxPossible
	}
	m.reclaimedBytes += reclaimed
	return reclaimed, nil
}

func (m *mockOpenCodePageManager) PageOutColdPrefixes(maxBlocks int) (int, uint64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pageOutCalls++
	if m.failPageOut {
		return 0, 0, context.DeadlineExceeded
	}
	paged := maxBlocks
	if paged > m.totalPages {
		paged = m.totalPages
	}
	reclaimed := uint64(paged) * m.bytesPerPage
	m.pagedOutBlocks += paged
	m.pagedOutBytes += reclaimed
	return paged, reclaimed, nil
}

func (m *mockOpenCodePageManager) TotalPages() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.totalPages
}

func (m *mockOpenCodePageManager) ActiveSessions() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.activeSessions
}

// TestMemoryPressureNotification verifies the real-time Apple Silicon unified memory pressure governor
// contract specified in Issue #12305:
// 1. Subscribes to Darwin native dispatch memory pressure notifications via DISPATCH_SOURCE_TYPE_MEMORYPRESSURE.
// 2. On DISPATCH_MEMORYPRESSURE_WARN:
//   - Triggers dynamic KV cache context compaction and pages out cold session prefixes to disk.
//   - Throttles speculative MTP candidate depth to preserve DRAM headroom.
//
// 3. On DISPATCH_MEMORYPRESSURE_CRITICAL:
//   - Gracefully pauses new batch admissions and returns structured backpressure (429 CAPACITY_BACKOFF) to OpenCode.
//
// 4. On recovery to NORMAL:
//   - Unpauses admissions and restores nominal MTP candidate depth.
func TestMemoryPressureNotification(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// 1. Initialize memory pressure subscriber (native platform or simulated).
	sub := NewPlatformMemoryPressureSubscriber()
	if err := sub.Start(ctx); err != nil {
		t.Fatalf("sub.Start failed: %v", err)
	}
	defer func() { _ = sub.Stop() }()

	pm := newMockOpenCodePageManager(2048, 4)
	mtpGov := NewDefaultMTPDepthGovernor(3, 1, 0) // nominal=3, throttled=1, critical=0

	logBuf := &bytes.Buffer{}
	cfg := DefaultUnifiedMemoryGovernorConfig()
	cfg.NominalMTPDepth = 3
	cfg.ThrottledMTPDepth = 1
	cfg.CriticalMTPDepth = 0
	cfg.WarnReclaimBytes = 256 * 1024 * 1024 // 256 MB
	cfg.WarnPrefixPageOutMax = 32
	cfg.CritReclaimBytes = 1024 * 1024 * 1024 // 1 GB
	cfg.CritPrefixPageOutMax = 128
	cfg.BackpressureRetrySec = 2
	cfg.LogEmitter = logBuf

	gov := NewUnifiedMemoryPressureGovernor(sub, pm, mtpGov, cfg)
	if err := gov.Start(ctx); err != nil {
		t.Fatalf("gov.Start failed: %v", err)
	}
	defer func() { _ = gov.Stop() }()

	// Baseline state checks
	if gov.CurrentLevel() != PressureNormal {
		t.Errorf("expected initial pressure level NORMAL, got %s", gov.CurrentLevel())
	}
	if gov.AdmissionState() != AdmissionNormal {
		t.Errorf("expected initial admission state NORMAL, got %s", gov.AdmissionState())
	}
	if mtpGov.CurrentDepth() != 3 {
		t.Errorf("expected initial MTP depth 3, got %d", mtpGov.CurrentDepth())
	}

	// Normal admission verification
	reqNormal := OpenCodeBatchRequest{
		SessionID: "opencode-session-1",
		Tokens:    128,
	}
	admNormal := gov.EvaluateBatchAdmission(reqNormal)
	if !admNormal.Admitted {
		t.Fatalf("expected batch admitted under NORMAL, got rejected: %s", admNormal.Reason)
	}
	if admNormal.StatusCode != http.StatusOK {
		t.Errorf("expected status 200 under NORMAL, got %d", admNormal.StatusCode)
	}

	// --- Phase A: DISPATCH_MEMORYPRESSURE_WARN ---
	t.Log("Testing DISPATCH_MEMORYPRESSURE_WARN transition...")
	warnStart := time.Now()
	sub.SimulatePressure(PressureWarn)
	warnDuration := time.Since(warnStart)

	if warnDuration > 50*time.Millisecond {
		t.Errorf("WARN reaction took %v, exceeded 50ms latency target", warnDuration)
	}

	if gov.CurrentLevel() != PressureWarn {
		t.Errorf("expected governor pressure WARN, got %s", gov.CurrentLevel())
	}
	if gov.AdmissionState() != AdmissionThrottled {
		t.Errorf("expected admission state THROTTLED under WARN, got %s", gov.AdmissionState())
	}

	// Verification 2a: KV cache context compaction and cold prefix page-out triggered
	pm.mu.Lock()
	recCalls := pm.reclaimCalls
	pageCalls := pm.pageOutCalls
	pagedBlocks := pm.pagedOutBlocks
	pm.mu.Unlock()

	if recCalls == 0 {
		t.Errorf("expected unpinned page reclamation called under WARN, got 0")
	}
	if pageCalls == 0 || pagedBlocks == 0 {
		t.Errorf("expected cold prefix page-out under WARN, got calls=%d, blocks=%d", pageCalls, pagedBlocks)
	}

	// Verification 2b: Speculative MTP candidate depth throttled to preserve DRAM headroom
	if mtpGov.CurrentDepth() != 1 {
		t.Errorf("expected MTP depth throttled to 1 under WARN, got %d", mtpGov.CurrentDepth())
	}

	// Batch admitted with throttled MTP notice under WARN
	reqWarn := OpenCodeBatchRequest{
		SessionID: "opencode-session-2",
		Tokens:    64,
	}
	admWarn := gov.EvaluateBatchAdmission(reqWarn)
	if !admWarn.Admitted {
		t.Errorf("expected batch still admitted under WARN, got rejected: %s", admWarn.Reason)
	}
	if admWarn.StatusCode != http.StatusOK {
		t.Errorf("expected status 200 under WARN, got %d", admWarn.StatusCode)
	}

	// --- Phase B: DISPATCH_MEMORYPRESSURE_CRITICAL ---
	t.Log("Testing DISPATCH_MEMORYPRESSURE_CRITICAL transition...")
	critStart := time.Now()
	sub.SimulatePressure(PressureCritical)
	critDuration := time.Since(critStart)

	if critDuration > 50*time.Millisecond {
		t.Errorf("CRITICAL reaction took %v, exceeded 50ms latency target", critDuration)
	}

	if gov.CurrentLevel() != PressureCritical {
		t.Errorf("expected governor pressure CRITICAL, got %s", gov.CurrentLevel())
	}
	if gov.AdmissionState() != AdmissionPaused {
		t.Errorf("expected admission state PAUSED under CRITICAL, got %s", gov.AdmissionState())
	}

	// Speculative MTP depth throttled to 0 (disabled) under CRITICAL
	if mtpGov.CurrentDepth() != 0 {
		t.Errorf("expected MTP depth 0 under CRITICAL, got %d", mtpGov.CurrentDepth())
	}

	// Verification 3: Gracefully pause batch admissions and return structured backpressure (429 CAPACITY_BACKOFF)
	reqCrit := OpenCodeBatchRequest{
		SessionID: "opencode-session-3",
		Tokens:    256,
	}
	admCrit := gov.EvaluateBatchAdmission(reqCrit)
	if admCrit.Admitted {
		t.Fatalf("expected batch REJECTED under CRITICAL, but was admitted")
	}
	if admCrit.StatusCode != StatusCapacityBackoff {
		t.Errorf("expected status code %d (429), got %d", StatusCapacityBackoff, admCrit.StatusCode)
	}
	if admCrit.Reason != ReasonCapacityBackoff {
		t.Errorf("expected reason %s, got %s", ReasonCapacityBackoff, admCrit.Reason)
	}
	if admCrit.Backpressure == nil {
		t.Fatalf("expected non-nil BackpressureResponse under CRITICAL")
	}
	if admCrit.Backpressure.StatusCode != 429 {
		t.Errorf("expected Backpressure.StatusCode 429, got %d", admCrit.Backpressure.StatusCode)
	}
	if admCrit.Backpressure.ErrorCode != CodeCapacityBackoff {
		t.Errorf("expected Backpressure.ErrorCode %s, got %s", CodeCapacityBackoff, admCrit.Backpressure.ErrorCode)
	}
	if admCrit.Backpressure.PressureLevel != PressureCritical {
		t.Errorf("expected Backpressure.PressureLevel CRITICAL, got %s", admCrit.Backpressure.PressureLevel)
	}
	if admCrit.Backpressure.RetryAfterSec != 2 {
		t.Errorf("expected Backpressure.RetryAfterSec 2, got %d", admCrit.Backpressure.RetryAfterSec)
	}
	if admCrit.Backpressure.Message == "" {
		t.Errorf("expected non-empty Backpressure.Message")
	}

	// Verify structured JSON serialization of Backpressure response for OpenCode wire format
	bpJSON, err := json.Marshal(admCrit.Backpressure)
	if err != nil {
		t.Fatalf("failed to marshal BackpressureResponse: %v", err)
	}
	var decoded BackpressureResponse
	if err := json.Unmarshal(bpJSON, &decoded); err != nil {
		t.Fatalf("failed to unmarshal BackpressureResponse JSON: %v", err)
	}
	if decoded.StatusCode != 429 || decoded.ErrorCode != CodeCapacityBackoff {
		t.Errorf("decoded backpressure mismatch: %+v", decoded)
	}

	// --- Phase C: Recovery to DISPATCH_MEMORYPRESSURE_NORMAL ---
	t.Log("Testing recovery to DISPATCH_MEMORYPRESSURE_NORMAL...")
	sub.SimulatePressure(PressureNormal)

	if gov.CurrentLevel() != PressureNormal {
		t.Errorf("expected recovered pressure NORMAL, got %s", gov.CurrentLevel())
	}
	if gov.AdmissionState() != AdmissionNormal {
		t.Errorf("expected recovered admission state NORMAL, got %s", gov.AdmissionState())
	}
	if mtpGov.CurrentDepth() != 3 {
		t.Errorf("expected restored MTP depth 3, got %d", mtpGov.CurrentDepth())
	}

	// Admissions unpaused and functioning normally
	reqRecovered := OpenCodeBatchRequest{
		SessionID: "opencode-session-4",
		Tokens:    512,
	}
	admRecovered := gov.EvaluateBatchAdmission(reqRecovered)
	if !admRecovered.Admitted {
		t.Fatalf("expected batch admitted after recovery, got rejected: %s", admRecovered.Reason)
	}
	if admRecovered.StatusCode != http.StatusOK {
		t.Errorf("expected status 200 after recovery, got %d", admRecovered.StatusCode)
	}

	// Telemetry verification
	telem := gov.Telemetry()
	if telem.PressureLevel != PressureNormal {
		t.Errorf("expected telemetry pressure NORMAL, got %s", telem.PressureLevel)
	}
	if telem.TotalBackpressure429s == 0 {
		t.Errorf("expected non-zero 429 counter in telemetry, got %d", telem.TotalBackpressure429s)
	}
	if telem.TotalReclaimedBytes == 0 {
		t.Errorf("expected non-zero reclaimed bytes in telemetry, got %d", telem.TotalReclaimedBytes)
	}
	if telem.TotalPrefixesPagedOut == 0 {
		t.Errorf("expected non-zero paged out prefixes in telemetry, got %d", telem.TotalPrefixesPagedOut)
	}
}

// TestUnifiedMemoryPressureGovernor_ConcurrentAdmissions verifies thread safety under concurrent requests.
func TestUnifiedMemoryPressureGovernor_ConcurrentAdmissions(t *testing.T) {
	sub := NewSimulatedSubscriber()
	pm := newMockOpenCodePageManager(1024, 2)
	mtpGov := NewDefaultMTPDepthGovernor(4, 2, 0)
	cfg := DefaultUnifiedMemoryGovernorConfig()

	gov := NewUnifiedMemoryPressureGovernor(sub, pm, mtpGov, cfg)
	if err := gov.Start(context.Background()); err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	defer func() { _ = gov.Stop() }()

	var wg sync.WaitGroup
	var backpressureTally uint64
	var admittedTally uint64

	// Concurrently simulate pressure changes and batch admissions
	pressureLevels := []MemoryPressureLevel{PressureNormal, PressureWarn, PressureCritical, PressureNormal}

	for i := 0; i < 30; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			// Alternate pressure
			sub.SimulatePressure(pressureLevels[idx%len(pressureLevels)])

			// Evaluate admissions
			adm := gov.EvaluateBatchAdmission(OpenCodeBatchRequest{
				SessionID: "concurrent-session",
				Tokens:    64,
			})
			if adm.Admitted {
				atomic.AddUint64(&admittedTally, 1)
			} else if adm.StatusCode == StatusCapacityBackoff {
				atomic.AddUint64(&backpressureTally, 1)
			}
		}(i)
	}

	wg.Wait()

	total := atomic.LoadUint64(&admittedTally) + atomic.LoadUint64(&backpressureTally)
	if total != 30 {
		t.Errorf("expected 30 total processed admissions, got %d", total)
	}
}

// TestMTPDepthGovernorBounds ensures MTP candidate depth constraints are strictly observed.
func TestMTPDepthGovernorBounds(t *testing.T) {
	gov := NewDefaultMTPDepthGovernor(4, 1, 0)
	if gov.CurrentDepth() != 4 {
		t.Errorf("expected initial depth 4, got %d", gov.CurrentDepth())
	}

	gov.SetDepth(2)
	if gov.CurrentDepth() != 2 {
		t.Errorf("expected depth 2, got %d", gov.CurrentDepth())
	}

	// Clamping tests
	gov.SetDepth(-5)
	if gov.CurrentDepth() != 0 {
		t.Errorf("expected clamped depth 0, got %d", gov.CurrentDepth())
	}

	gov.SetDepth(10)
	if gov.CurrentDepth() != 4 {
		t.Errorf("expected clamped depth 4, got %d", gov.CurrentDepth())
	}
}
