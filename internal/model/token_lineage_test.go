package model

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func tokenLineageDenseQwen38Config() Config {
	cfg := qwen35HybridTestCfg()
	cfg.ModelType = "qwen3_8"
	cfg.LayerTypes = nil
	cfg.LinearConvKernelDim = 0
	cfg.LinearKeyHeadDim = 0
	cfg.LinearNumKeyHeads = 0
	cfg.LinearValueHeadDim = 0
	cfg.LinearNumValueHeads = 0
	cfg.AttnOutputGate = false
	cfg.FullAttentionInterval = 0
	return cfg
}

func requireTokenLineage(t *testing.T, s *Session, want []int) TokenLineageVerification {
	t.Helper()
	got, err := s.VerifyTokenLineage(want)
	if err != nil {
		t.Fatalf("VerifyTokenLineage(%v): %v", want, err)
	}
	if got.Positions != len(want) || got.MetadataBytes != int64(len(want)*tokenLineageBytesPerPosition) {
		t.Fatalf("lineage report=%+v, want positions=%d metadata_bytes=%d", got, len(want), len(want)*tokenLineageBytesPerPosition)
	}
	return got
}

func TestTokenLineageLegacyWriteShiftRollbackAndPrefixReuse(t *testing.T) {
	m := NewSynthetic(tokenLineageDenseQwen38Config())
	prompt := []int{3, 7, 11, 5, 17}
	s := m.NewSession()
	defer s.Close()
	s.Prefill(prompt)
	requireTokenLineage(t, s, prompt) // legacy host write

	if removed := s.evictKV(1, 2); removed != 2 {
		t.Fatalf("shift removed=%d, want 2", removed)
	}
	shifted := []int{3, 5, 17}
	requireTokenLineage(t, s, shifted)

	s.Step(19)
	requireTokenLineage(t, s, []int{3, 5, 17, 19})
	if removed := s.evictKV(3, 1); removed != 1 {
		t.Fatalf("rollback removed=%d, want 1", removed)
	}
	requireTokenLineage(t, s, shifted)

	reused := m.SessionFromPrefix(s.Cache)
	defer reused.Close()
	requireTokenLineage(t, reused, shifted)
}

func TestTokenLineagePagedEvictionReusesPhysicalKVWithoutOldIdentity(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "1")
	t.Setenv("FAK_PAGED_KV_BLOCK_TOKENS", "2")
	m := NewSynthetic(tokenLineageDenseQwen38Config())
	s, err := m.NewBackendSessionChecked(compute.Default())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	prompt := []int{3, 7, 11, 5}
	s.Prefill(prompt)
	requireTokenLineage(t, s, prompt)

	paged, ok := s.halKV.(*pagedHALKV)
	if !ok {
		t.Fatalf("HAL KV=%T, want *pagedHALKV", s.halKV)
	}
	oldBlocks := append([]int(nil), paged.seq.table...)
	if removed := s.evictKV(0, len(prompt)); removed != len(prompt) {
		t.Fatalf("eviction removed=%d, want %d", removed, len(prompt))
	}
	requireTokenLineage(t, s, nil)

	s.Step(23)
	requireTokenLineage(t, s, []int{23})
	if len(paged.seq.table) != 1 || !containsInt(oldBlocks, paged.seq.table[0]) {
		t.Fatalf("physical KV block was not reused: old=%v new=%v", oldBlocks, paged.seq.table)
	}
}

func TestTokenLineageQwen38SequencePrefillSnapshotAndMetadataReceipt(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	cfg.ModelType = "qwen3_8"
	m := NewSynthetic(cfg)
	be := newSequencePrefillBackend(m)
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	prompt := []int{3, 7, 11, 5}
	s.Prefill(prompt)
	report := requireTokenLineage(t, s, prompt)
	if be.calls != 1 {
		t.Fatalf("sequence-prefill calls=%d, want 1", be.calls)
	}

	snapshot, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	if got := snapshot.TokenLineageMetadataBytes(); got != report.MetadataBytes {
		t.Fatalf("snapshot lineage metadata=%d, want %d", got, report.MetadataBytes)
	}
	oldProfilePath := prefixProfile.path
	t.Cleanup(func() { prefixProfile.path = oldProfilePath })
	prefixProfile.path = filepath.Join(t.TempDir(), "prefix.jsonl")
	emitPrefixProfile(time.Now(), "device_clone", "complete", snapshot, nil)
	data, err := os.ReadFile(prefixProfile.path)
	if err != nil {
		t.Fatal(err)
	}
	var receipt PrefixProfileEvent
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.MetadataBytes != report.MetadataBytes {
		t.Fatalf("prefix profile metadata_bytes=%d, want %d", receipt.MetadataBytes, report.MetadataBytes)
	}
	clone, err := snapshot.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	fresh, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer fresh.Close()
	if err := clone.Restore(fresh); err != nil {
		t.Fatal(err)
	}
	requireTokenLineage(t, fresh, prompt)
}

func TestTokenLineageVerificationDetectsMismatchAndCorruption(t *testing.T) {
	m := NewSynthetic(tokenLineageDenseQwen38Config())
	s := m.NewSession()
	defer s.Close()
	s.Prefill([]int{3, 7, 11})

	if _, err := s.VerifyTokenLineage([]int{3, 8, 11}); !errors.Is(err, ErrTokenLineageMismatch) {
		t.Fatalf("token mismatch error=%v, want ErrTokenLineageMismatch", err)
	}
	s.Cache.lineage.ids[1] ^= 1
	if _, err := s.VerifyTokenLineage([]int{3, 7, 11}); !errors.Is(err, ErrTokenLineageMismatch) {
		t.Fatalf("corruption error=%v, want ErrTokenLineageMismatch", err)
	}
	s.Cache.lineage.ids = s.Cache.lineage.ids[:2]
	if _, err := s.VerifyTokenLineage([]int{3, 7, 11}); !errors.Is(err, ErrTokenLineageMismatch) {
		t.Fatalf("structural corruption error=%v, want ErrTokenLineageMismatch", err)
	}
}

func TestRestoreTokenLineagePublishesOnlyGeometryMatchedHistory(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	history := []int{3, 7, 11, 5}
	source := m.NewSession()
	source.Prefill(history)
	defer source.Close()

	blob, err := QwenHybridKVCacheToHost(source.Cache, 2)
	if err != nil {
		t.Fatalf("swap to host: %v", err)
	}
	cache, err := QwenHybridKVCacheFromHost(m.Cfg, blob)
	if err != nil {
		t.Fatalf("swap from host: %v", err)
	}
	restored := &Session{M: m, Cache: cache}
	defer restored.Close()

	if report, err := restored.RestoreTokenLineage(history[:len(history)-1]); !errors.Is(err, ErrTokenLineageMismatch) {
		t.Fatalf("short history restore report=%+v error=%v, want ErrTokenLineageMismatch", report, err)
	}
	if got := restored.TokenLineageMetadataBytes(); got != 0 {
		t.Fatalf("refused restore published %d lineage bytes, want 0", got)
	}
	if _, err := restored.VerifyTokenLineage(history); !errors.Is(err, ErrTokenLineageMismatch) {
		t.Fatalf("refused restore verification error=%v, want unpublished lineage", err)
	}
	restored.Cache.pos[2] = 9
	if report, err := restored.RestoreTokenLineage(history); !errors.Is(err, ErrTokenLineageMismatch) {
		t.Fatalf("non-contiguous restore report=%+v error=%v, want ErrTokenLineageMismatch", report, err)
	}
	if got := restored.TokenLineageMetadataBytes(); got != 0 {
		t.Fatalf("non-contiguous restore published %d lineage bytes, want 0", got)
	}
	restored.Cache.pos[2] = 2

	report, err := restored.RestoreTokenLineage(history)
	if err != nil {
		t.Fatalf("matched history restore: %v", err)
	}
	if report.Positions != len(history) || report.MetadataBytes != int64(len(history)*tokenLineageBytesPerPosition) {
		t.Fatalf("restore report=%+v, want positions=%d metadata_bytes=%d", report, len(history), len(history)*tokenLineageBytesPerPosition)
	}
	requireTokenLineage(t, restored, history)
	if _, err := restored.RestoreTokenLineage(history[:len(history)-1]); !errors.Is(err, ErrTokenLineageMismatch) {
		t.Fatalf("second short restore error=%v, want ErrTokenLineageMismatch", err)
	}
	requireTokenLineage(t, restored, history)
}

func containsInt(values []int, want int) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
