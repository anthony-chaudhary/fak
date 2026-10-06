// Default suites: go test ./cmd/fak and go test ./... (also under -race).
// Darwin host suite: go test ./cmd/fak -run '^TestServeResidentMappedQ4K' -count=1.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/metrics"
	"sort"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

func mappedQ4KServeCPUEnv(t *testing.T) {
	t.Helper()
	for _, key := range []string{"FAK_STREAM_Q4K", "FAK_METAL_STREAM_Q4K"} {
		t.Setenv(key, "")
	}
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_Q4K_FREE_CPU", "0")
	original := serveMetalAvailable
	serveMetalAvailable = func() bool { return false }
	t.Cleanup(func() { serveMetalAvailable = original })
}

func mappedQ4KServeCPULogits(t *testing.T, m *fakmodel.Model) [][]float32 {
	t.Helper()
	s := m.NewSession()
	defer s.Close()
	s.Q4K = true
	var out [][]float32
	for _, token := range []int{2, 7, 3} {
		logits := append([]float32(nil), s.Step(token)...)
		if len(logits) != m.Cfg.VocabSize {
			t.Fatalf("full-fixture logits=%d, want vocab %d", len(logits), m.Cfg.VocabSize)
		}
		var nonzero bool
		for i, value := range logits {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("CPU logit %d is nonfinite: %v", i, value)
			}
			nonzero = nonzero || value != 0
		}
		if !nonzero {
			t.Fatal("full nonzero GGUF fixture produced only zero logits")
		}
		out = append(out, logits)
	}
	return out
}

const mappedQ4KServeCaseEnv = "FAK_TEST_MAPPED_CPU_LOAD_CASE"
const mappedQ4KServeReceiptMarker = "FAK_MAPPED_CPU_RECEIPT "
const mappedQ4KServeArtifactEnv = "FAK_TEST_MAPPED_CPU_ARTIFACT"
const mappedQ4KServeHeapDiagnosticEnv = "FAK_TEST_MAPPED_CPU_HEAP_DIAGNOSTIC"

type mappedQ4KServeArtifactReceipt struct {
	MMap                 string                        `json:"mmap"`
	ArtifactSHA256       string                        `json:"artifact_sha256"`
	ArtifactBytes        int64                         `json:"artifact_bytes"`
	Resident             fakmodel.ResidentReport       `json:"resident"`
	MappedQ4KBytes       int64                         `json:"mapped_q4k_bytes"`
	OwnedQ4KBytes        int64                         `json:"owned_q4k_bytes"`
	UnprovenAlignedBytes int64                         `json:"unproven_aligned_q4k_bytes"`
	MappedTensors        int                           `json:"mapped_q4k_tensors"`
	HeapObjectsBytes     uint64                        `json:"heap_objects_bytes"`
	HeapAllocBytes       uint64                        `json:"heap_alloc_bytes"`
	HeapInuseBytes       uint64                        `json:"heap_inuse_bytes"`
	HeapSysBytes         uint64                        `json:"heap_sys_bytes"`
	HeapReleasedBytes    uint64                        `json:"heap_released_bytes"`
	HeapDiagnostic       *mappedQ4KServeHeapDiagnostic `json:"heap_diagnostic,omitempty"`
	PromptTokens         []int                         `json:"prompt_tokens"`
	Vocab                int                           `json:"vocab"`
	LogitBitsSHA256      string                        `json:"logit_bits_sha256"`
	ArgmaxTokens         []int                         `json:"argmax_tokens"`
}

type mappedQ4KServeHeapSnapshot struct {
	HeapObjectsBytes  uint64 `json:"heap_objects_bytes"`
	HeapAllocBytes    uint64 `json:"heap_alloc_bytes"`
	HeapInuseBytes    uint64 `json:"heap_inuse_bytes"`
	HeapSysBytes      uint64 `json:"heap_sys_bytes"`
	HeapReleasedBytes uint64 `json:"heap_released_bytes"`
	HeapObjects       uint64 `json:"heap_objects"`
	NumGC             uint32 `json:"num_gc"`
}

const maxLargeInUseAllocationStacks = 8

type mappedQ4KServeHeapStack struct {
	SampledInUseBytes   int64    `json:"sampled_in_use_bytes"`
	SampledInUseObjects int64    `json:"sampled_in_use_objects"`
	Functions           []string `json:"functions"`
}

type mappedQ4KServeHeapDiagnostic struct {
	FirstGCNum                 uint32                     `json:"first_gc_num"`
	FirstGCHeapObjects         uint64                     `json:"first_gc_heap_objects"`
	SecondGC                   mappedQ4KServeHeapSnapshot `json:"second_gc"`
	MemProfileRate             int                        `json:"mem_profile_rate"`
	ProfileMayLagGCCycles      int                        `json:"profile_may_lag_gc_cycles"`
	ProfileComplete            bool                       `json:"profile_complete"`
	ProfileRecords             int                        `json:"profile_records"`
	LargeInUseAllocationStacks []mappedQ4KServeHeapStack  `json:"large_in_use_allocation_stacks"`
}

func captureMappedQ4KServeHeapDiagnostic(t *testing.T, m *fakmodel.Model, first runtime.MemStats, samples []metrics.Sample) *mappedQ4KServeHeapDiagnostic {
	t.Helper()
	// The first-GC snapshot remains the acceptance measurement. This optional
	// second snapshot diagnoses reclaimable loader temporaries before any CPU
	// session allocation; it never replaces the first-GC assertion inputs.
	runtime.GC()
	var second runtime.MemStats
	runtime.ReadMemStats(&second)
	metrics.Read(samples)
	runtime.KeepAlive(m)
	if samples[0].Value.Kind() != metrics.KindUint64 {
		t.Fatal("diagnostic runtime heap objects metric unavailable")
	}
	diagnostic := &mappedQ4KServeHeapDiagnostic{
		FirstGCNum: first.NumGC, FirstGCHeapObjects: first.HeapObjects,
		SecondGC: mappedQ4KServeHeapSnapshot{
			HeapObjectsBytes: samples[0].Value.Uint64(), HeapAllocBytes: second.HeapAlloc,
			HeapInuseBytes: second.HeapInuse, HeapSysBytes: second.HeapSys,
			HeapReleasedBytes: second.HeapReleased, HeapObjects: second.HeapObjects,
			NumGC: second.NumGC,
		},
		MemProfileRate: runtime.MemProfileRate, ProfileMayLagGCCycles: 2,
	}
	// MemProfile is sampled and may lag by two GC cycles. Keep its diagnostic
	// allocations after the snapshot and emit symbols only, never source paths.
	var records []runtime.MemProfileRecord
	for attempt := 0; attempt < 3; attempt++ {
		n, _ := runtime.MemProfile(nil, false)
		records = make([]runtime.MemProfileRecord, n+16)
		n, complete := runtime.MemProfile(records, false)
		if complete {
			records = records[:n]
			diagnostic.ProfileComplete = true
			diagnostic.ProfileRecords = n
			break
		}
		records = nil
	}
	sort.Slice(records, func(i, j int) bool { return records[i].InUseBytes() > records[j].InUseBytes() })
	for i := range records {
		if records[i].InUseBytes() < 1<<20 || len(diagnostic.LargeInUseAllocationStacks) == maxLargeInUseAllocationStacks {
			break
		}
		stack := mappedQ4KServeHeapStack{
			SampledInUseBytes: records[i].InUseBytes(), SampledInUseObjects: records[i].InUseObjects(),
		}
		frames := runtime.CallersFrames(records[i].Stack())
		for len(stack.Functions) < 8 {
			frame, more := frames.Next()
			if frame.Function != "" {
				stack.Functions = append(stack.Functions, fmt.Sprintf("%s:%d", frame.Function, frame.Line))
			}
			if !more {
				break
			}
		}
		diagnostic.LargeInUseAllocationStacks = append(diagnostic.LargeInUseAllocationStacks, stack)
	}
	runtime.KeepAlive(m)
	return diagnostic
}

func captureMappedQ4KServeArtifactReceipt(t *testing.T) mappedQ4KServeArtifactReceipt {
	t.Helper()
	mappedQ4KServeCPUEnv(t)
	path := os.Getenv(mappedQ4KServeArtifactEnv)
	f, err := os.Open(path)
	if err != nil {
		t.Fatal("open opt-in audit artifact failed")
	}
	defer f.Close()
	stat, err := f.Stat()
	const expectedBytes = 532517120
	const expectedSHA256 = "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"
	if err != nil || stat.Size() != expectedBytes {
		t.Fatal("opt-in artifact does not match the audited byte length")
	}
	hash := sha256.New()
	if _, err := io.Copy(hash, f); err != nil {
		t.Fatal("hash opt-in audit artifact failed")
	}
	artifactHash := fmt.Sprintf("%x", hash.Sum(nil))
	if artifactHash != expectedSHA256 {
		t.Fatalf("opt-in artifact SHA256=%s, want audited SHA256=%s", artifactHash, expectedSHA256)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatal("rewind opt-in audit artifact failed")
	}
	fixture, err := ggufload.Read(f)
	if err != nil {
		t.Fatal("parse opt-in audit artifact header failed")
	}
	if err := f.Close(); err != nil {
		t.Fatal("close opt-in audit artifact header reader failed")
	}
	m, q4k, profile, phase := loadServeInKernelModel(path, nil, false, 0, nil, 16, nil)
	if m == nil || !q4k || profile == nil || profile.Mode != "gguf-resident-q4k" || phase.Name != "model-load" {
		t.Fatal("opt-in artifact did not load through the resident CPU serve entrypoint")
	}
	t.Cleanup(func() { _ = m.CloseWeights() })
	receipt := mappedQ4KServeArtifactReceipt{
		MMap: os.Getenv("FAK_GGUF_MMAP"), ArtifactSHA256: artifactHash,
		ArtifactBytes: stat.Size(), Resident: *m.ResidentReport(),
		PromptTokens: []int{2, 7, 3}, Vocab: m.Cfg.VocabSize,
	}
	page := os.Getpagesize()
	seen := make(map[string]bool)
	for _, tensor := range fixture.Tensors {
		canonical, ok := ggufload.CanonicalTensorNameArch(tensor.Name, m.Cfg.ModelType)
		if !ok || seen[canonical] {
			continue
		}
		raw, found := m.Q4KRaw(canonical)
		if !found || len(raw) == 0 {
			continue
		}
		seen[canonical] = true
		fileOffset := int(tensor.FileOffset % int64(page))
		rawOffset := int(uintptr(unsafe.Pointer(&raw[0])) % uintptr(page))
		lazy := m.Q4KLazy(canonical)
		switch {
		case lazy && fileOffset != 0 && rawOffset == fileOffset:
			// A nonzero file-relative alias cannot be the aligned owned fallback.
			receipt.MappedQ4KBytes += int64(len(raw))
			receipt.MappedTensors++
		case lazy && fileOffset == 0 && rawOffset == 0:
			// These APIs cannot distinguish an aligned mapping from aligned
			// fallback ownership. Do not count descriptor presence as proof.
			receipt.UnprovenAlignedBytes += int64(len(raw))
		case rawOffset == 0:
			receipt.OwnedQ4KBytes += int64(len(raw))
		default:
			t.Fatalf("Q4_K backing offset does not match mapping or aligned ownership for %s", canonical)
		}
	}
	if len(seen) != receipt.Resident.Q4KTensors || receipt.MappedQ4KBytes+receipt.OwnedQ4KBytes+receipt.UnprovenAlignedBytes != receipt.Resident.Q4KBytes {
		t.Fatal("public Q4_K backing enumeration does not cover the complete resident Q4_K store")
	}
	// Drop the test's parsed header before measuring retained model heap.
	fixture = nil
	seen = nil
	samples := []metrics.Sample{{Name: "/memory/classes/heap/objects:bytes"}}
	runtime.GC()
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	metrics.Read(samples)
	runtime.KeepAlive(m)
	if samples[0].Value.Kind() != metrics.KindUint64 {
		t.Fatal("runtime heap objects metric unavailable")
	}
	receipt.HeapObjectsBytes = samples[0].Value.Uint64()
	receipt.HeapAllocBytes = memory.HeapAlloc
	receipt.HeapInuseBytes = memory.HeapInuse
	receipt.HeapSysBytes = memory.HeapSys
	receipt.HeapReleasedBytes = memory.HeapReleased
	if os.Getenv(mappedQ4KServeHeapDiagnosticEnv) == "1" {
		receipt.HeapDiagnostic = captureMappedQ4KServeHeapDiagnostic(t, m, memory, samples)
	}

	s := m.NewSession()
	t.Cleanup(s.Close)
	s.Q4K = true
	logitHash := sha256.New()
	var encoded [4]byte
	for _, token := range receipt.PromptTokens {
		if token < 0 || token >= receipt.Vocab {
			t.Fatal("audit prompt token outside artifact vocabulary")
		}
		logits := s.Step(token)
		if len(logits) != receipt.Vocab {
			t.Fatal("audit CPU output vocabulary mismatch")
		}
		var nonzero bool
		best := 0
		for i, value := range logits {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("audit CPU output contains a nonfinite logit")
			}
			nonzero = nonzero || value != 0
			if value > logits[best] {
				best = i
			}
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(value))
			_, _ = logitHash.Write(encoded[:])
		}
		if !nonzero {
			t.Fatal("audit CPU output contains only zero logits")
		}
		receipt.ArgmaxTokens = append(receipt.ArgmaxTokens, best)
	}
	s.Close()
	receipt.LogitBitsSHA256 = fmt.Sprintf("%x", logitHash.Sum(nil))
	runtime.KeepAlive(m)
	if err := m.CloseWeights(); err != nil {
		t.Fatal("audit model checkpoint teardown failed")
	}
	return receipt
}

func runMappedQ4KServeArtifactCase(t *testing.T, value, path string) mappedQ4KServeArtifactReceipt {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeResidentMappedQ4KSelectionAndCPULogitParity$", "-test.count=1", "-test.v", "-test.timeout=90s")
	for _, entry := range os.Environ() {
		if !strings.HasPrefix(entry, "FAK_GGUF_MMAP=") && !strings.HasPrefix(entry, mappedQ4KServeCaseEnv+"=") && !strings.HasPrefix(entry, mappedQ4KServeArtifactEnv+"=") {
			cmd.Env = append(cmd.Env, entry)
		}
	}
	cmd.Env = append(cmd.Env, "FAK_GGUF_MMAP="+value, mappedQ4KServeCaseEnv+"=artifact", mappedQ4KServeArtifactEnv+"="+path)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("isolated audit-artifact mmap=%q receipt: %v\n%s", value, err, out)
	}
	var receipt mappedQ4KServeArtifactReceipt
	var found int
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, mappedQ4KServeReceiptMarker) {
			found++
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, mappedQ4KServeReceiptMarker)), &receipt); err != nil {
				t.Fatalf("decode audit CPU receipt: %v", err)
			}
		}
	}
	if found != 1 || receipt.MMap != value || receipt.Resident.Q4KBytes <= 0 || len(receipt.ArgmaxTokens) != 3 || receipt.LogitBitsSHA256 == "" {
		t.Fatal("isolated audit CPU receipt is incomplete")
	}
	return receipt
}

func testMappedQ4KServeArtifact(t *testing.T, path string) {
	t.Helper()
	control := runMappedQ4KServeArtifactCase(t, "", path)
	candidate := runMappedQ4KServeArtifactCase(t, "1", path)
	encoded, err := json.Marshal(struct {
		Control   mappedQ4KServeArtifactReceipt `json:"control"`
		Candidate mappedQ4KServeArtifactReceipt `json:"candidate"`
	}{Control: control, Candidate: candidate})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("audit artifact retained-heap and CPU-parity receipt: %s", encoded)
	if control.Resident != candidate.Resident || control.ArtifactSHA256 != candidate.ArtifactSHA256 || control.Vocab != candidate.Vocab || control.LogitBitsSHA256 != candidate.LogitBitsSHA256 {
		t.Fatal("audit A/B resident accounting or complete CPU output bits differ")
	}
	if control.MappedQ4KBytes != 0 || control.UnprovenAlignedBytes != 0 || control.OwnedQ4KBytes != control.Resident.Q4KBytes || candidate.MappedTensors == 0 {
		t.Fatal("audit A/B did not witness owned control and actual mapped CPU backing")
	}
	const tolerance = 8 << 20
	if candidate.MappedQ4KBytes <= 2*tolerance {
		t.Fatal("audit mapped-byte witness is too small to distinguish retained heap from fixed bookkeeping tolerance")
	}
	for _, axis := range []struct {
		name      string
		control   uint64
		candidate uint64
	}{
		{name: "heap objects metric", control: control.HeapObjectsBytes, candidate: candidate.HeapObjectsBytes},
		{name: "MemStats HeapAlloc", control: control.HeapAllocBytes, candidate: candidate.HeapAllocBytes},
	} {
		if axis.control <= axis.candidate {
			t.Fatalf("audit %s control=%d candidate=%d, want lower retained mapped heap", axis.name, axis.control, axis.candidate)
		}
		saved := axis.control - axis.candidate
		if saved+tolerance < uint64(candidate.MappedQ4KBytes) || saved > uint64(candidate.Resident.Q4KBytes)+tolerance {
			t.Fatalf("audit %s retained heap saving=%d outside mapped-byte lower bound=%d and eligible-byte upper bound=%d +/- %d", axis.name, saved, candidate.MappedQ4KBytes, candidate.Resident.Q4KBytes, tolerance)
		}
	}
}

type mappedQ4KServeBackingReceipt struct {
	Name           string `json:"name"`
	FilePageOffset int    `json:"file_page_offset"`
	RawPageOffset  int    `json:"raw_page_offset"`
	Lazy           bool   `json:"lazy"`
	Bytes          int    `json:"bytes"`
	SHA256         string `json:"sha256"`
}

type mappedQ4KServeReceipt struct {
	MMap       string                         `json:"mmap"`
	Q4KTensors int                            `json:"q4k_tensors"`
	Backing    []mappedQ4KServeBackingReceipt `json:"backing"`
	LogitBits  [][]uint32                     `json:"logit_bits"`
}

func captureMappedQ4KServeReceipt(t *testing.T) mappedQ4KServeReceipt {
	t.Helper()
	// FAK_GGUF_MMAP arrived in this fresh process's environment before the
	// loader's immutable gate could be initialized. Keep that setting intact.
	mappedQ4KServeCPUEnv(t)
	value := os.Getenv("FAK_GGUF_MMAP")
	path := createMappedQ4KServeCompleteGGUF(t)
	blob, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	fixture, err := ggufload.Read(bytes.NewReader(blob))
	if err != nil {
		t.Fatal(err)
	}
	m, q4k, profile, phase := loadServeInKernelModel(path, nil, false, 0, nil, 16, nil)
	if m == nil || !q4k || profile == nil || profile.Mode != "gguf-resident-q4k" || phase.Name != "model-load" {
		t.Fatalf("resident CPU load: model=%v Q4K=%v profile=%+v phase=%+v", m != nil, q4k, profile, phase)
	}
	t.Cleanup(func() { _ = m.CloseWeights() })
	receipt := mappedQ4KServeReceipt{MMap: value, Q4KTensors: m.Q4KCount()}
	page := os.Getpagesize()
	for _, names := range [][2]string{
		{"blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight"},
		{"output.weight", "lm_head.weight"},
	} {
		fileOffset := -1
		for _, tensor := range fixture.Tensors {
			if tensor.Name == names[0] {
				fileOffset = int(tensor.FileOffset % int64(page))
			}
		}
		if fileOffset <= 0 {
			t.Fatalf("fixture %s page offset=%d, need a nonzero offset that distinguishes mapping from aligned ownership", names[0], fileOffset)
		}
		raw, found := m.Q4KRaw(names[1])
		if !found || len(raw) == 0 {
			t.Fatalf("resident CPU bytes missing for %s", names[1])
		}
		rawOffset := int(uintptr(unsafe.Pointer(&raw[0])) % uintptr(page))
		lazy := m.Q4KLazy(names[1])
		if value == "" {
			if lazy || rawOffset != 0 {
				t.Fatalf("owned control %s: lazy=%v raw page offset=%d, want eager page-aligned bytes", names[1], lazy, rawOffset)
			}
		} else if !lazy || rawOffset != fileOffset {
			// Existing owned fallback bytes are page-aligned (offset zero). This
			// nonzero file-relative pointer proves actual mapped CPU backing;
			// retaining a lazy descriptor alone would not prove that selection.
			t.Fatalf("FAK_GGUF_MMAP=%q %s: lazy=%v raw page offset=%d, want mapped file page offset=%d", value, names[1], lazy, rawOffset, fileOffset)
		}
		hash := sha256.Sum256(raw)
		receipt.Backing = append(receipt.Backing, mappedQ4KServeBackingReceipt{
			Name: names[1], FilePageOffset: fileOffset, RawPageOffset: rawOffset,
			Lazy: lazy, Bytes: len(raw), SHA256: fmt.Sprintf("%x", hash),
		})
	}
	for _, logits := range mappedQ4KServeCPULogits(t, m) {
		bits := make([]uint32, len(logits))
		for i, value := range logits {
			bits[i] = math.Float32bits(value)
		}
		receipt.LogitBits = append(receipt.LogitBits, bits)
	}
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	return receipt
}

func runMappedQ4KServeCase(t *testing.T, value string) mappedQ4KServeReceipt {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeResidentMappedQ4KSelectionAndCPULogitParity$", "-test.count=1", "-test.v", "-test.timeout=30s")
	for _, entry := range os.Environ() {
		if !strings.HasPrefix(entry, "FAK_GGUF_MMAP=") && !strings.HasPrefix(entry, mappedQ4KServeCaseEnv+"=") {
			cmd.Env = append(cmd.Env, entry)
		}
	}
	cmd.Env = append(cmd.Env, "FAK_GGUF_MMAP="+value, mappedQ4KServeCaseEnv+"=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("isolated FAK_GGUF_MMAP=%q receipt: %v\n%s", value, err, out)
	}
	var receipt mappedQ4KServeReceipt
	var found int
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, mappedQ4KServeReceiptMarker) {
			found++
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, mappedQ4KServeReceiptMarker)), &receipt); err != nil {
				t.Fatalf("decode isolated CPU receipt: %v", err)
			}
		}
	}
	if found != 1 || receipt.MMap != value || len(receipt.Backing) != 2 || len(receipt.LogitBits) != 3 {
		t.Fatalf("isolated CPU receipt count=%d receipt=%+v", found, receipt)
	}
	return receipt
}

// fak-test:runtime medium est=5s lane=default
// The default estimate covers the generated fixture and isolated CPU cases.
// The real-artifact subtest requires FAK_TEST_MAPPED_CPU_ARTIFACT explicitly;
// its witnessed A/B cost is 130.244s, with a 2m30s estimate for that opt-in run.
// FAK_TEST_MAPPED_CPU_HEAP_DIAGNOSTIC optionally adds heap diagnostic receipts.
func TestServeResidentMappedQ4KSelectionAndCPULogitParity(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Darwin retained GGUF mapping selection")
	}
	if os.Getenv(mappedQ4KServeCaseEnv) == "artifact" {
		receipt := captureMappedQ4KServeArtifactReceipt(t)
		encoded, err := json.Marshal(receipt)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Printf("%s%s\n", mappedQ4KServeReceiptMarker, encoded)
		return
	}
	if os.Getenv(mappedQ4KServeCaseEnv) == "1" {
		receipt := captureMappedQ4KServeReceipt(t)
		encoded, err := json.Marshal(receipt)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Printf("%s%s\n", mappedQ4KServeReceiptMarker, encoded)
		return
	}
	want := runMappedQ4KServeCase(t, "")
	encoded, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("owned resident CPU receipt: %s", encoded)
	for _, value := range []string{"1", "on", "true"} {
		t.Run(value, func(t *testing.T) {
			got := runMappedQ4KServeCase(t, value)
			for i := range want.Backing {
				if got.Backing[i].Name != want.Backing[i].Name || got.Backing[i].Bytes != want.Backing[i].Bytes || got.Backing[i].SHA256 != want.Backing[i].SHA256 {
					t.Fatalf("mapped CPU byte receipt=%+v, want owned byte receipt=%+v", got.Backing[i], want.Backing[i])
				}
			}
			for step := range want.LogitBits {
				if len(got.LogitBits[step]) != len(want.LogitBits[step]) {
					t.Fatalf("mapped CPU step=%d logit count=%d, want %d", step, len(got.LogitBits[step]), len(want.LogitBits[step]))
				}
				for token := range want.LogitBits[step] {
					if got.LogitBits[step][token] != want.LogitBits[step][token] {
						t.Fatalf("mapped CPU step=%d token=%d logit bits=%x, want owned-resident bits %x", step, token, got.LogitBits[step][token], want.LogitBits[step][token])
					}
				}
			}
			encoded, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("mapped resident CPU receipt: %s", encoded)
		})
	}
	// The normal suite keeps its tiny generated fixture. The audited artifact
	// branch is explicit opt-in, with each load isolated and bounded to 90s.
	if path := os.Getenv(mappedQ4KServeArtifactEnv); path != "" {
		t.Run("real-artifact", func(t *testing.T) {
			testMappedQ4KServeArtifact(t, path)
		})
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestServeResidentMappedQ4KPreservesExplicitDenseStreaming(t *testing.T) {
	mappedQ4KServeCPUEnv(t)
	path := createMappedQ4KServeCompleteGGUF(t)
	for _, tc := range []struct {
		name string
		env  string
		opts []ggufload.Q4KLoadOption
	}{
		{name: "dense environment", env: "FAK_STREAM_Q4K"},
		{name: "Metal dense environment", env: "FAK_METAL_STREAM_Q4K"},
		{name: "dense option", opts: []ggufload.Q4KLoadOption{ggufload.WithStreamedDenseQ4K(true)}},
		{name: "bounded dense option", opts: []ggufload.Q4KLoadOption{ggufload.WithStreamedDenseQ4KWorkingSet(1 << 20)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAK_GGUF_MMAP", "1")
			if tc.env != "" {
				t.Setenv(tc.env, "1")
			}
			m, _, _ := loadResidentQ4KProfiled(path, time.Now(), tc.opts...)
			t.Cleanup(func() { _ = m.CloseWeights() })
			if !m.Q4KLazy("model.layers.0.self_attn.v_proj.weight") {
				t.Fatal("explicit dense streaming lost its checkpoint range descriptor")
			}
			s := m.NewSession()
			defer s.Close()
			s.Q4K = true
			// Explicit streaming must not silently become whole-model CPU residency
			// when mmap is also enabled. Its CPU guard remains the prior behavior.
			defer func() {
				message, ok := recover().(string)
				if !ok || !strings.Contains(message, "Q4_K") || !strings.Contains(message, "CPU") {
					t.Fatalf("explicit dense streaming CPU refusal=%q, want its named Q4_K CPU guard", message)
				}
			}()
			s.Step(2)
		})
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestServeResidentMappedQ4KPreservesExplicitExpertOption(t *testing.T) {
	const helperEnv = "FAK_TEST_MAPPED_CPU_EXPERT_REFUSAL"
	if os.Getenv(helperEnv) != "1" {
		// The production startup helper exits on a loader refusal. Isolate that
		// boundary so the explicit-expert contract cannot terminate this suite.
		ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeResidentMappedQ4KPreservesExplicitExpertOption$", "-test.count=1", "-test.v", "-test.timeout=30s")
		for _, entry := range os.Environ() {
			if !strings.HasPrefix(entry, helperEnv+"=") && !strings.HasPrefix(entry, "FAK_GGUF_MMAP=") {
				cmd.Env = append(cmd.Env, entry)
			}
		}
		cmd.Env = append(cmd.Env, helperEnv+"=1", "FAK_GGUF_MMAP=1")
		out, err := cmd.CombinedOutput()
		exit, exited := err.(*exec.ExitError)
		if !exited || exit.ExitCode() != 1 {
			t.Fatalf("explicit expert refusal exit=%v, want production startup exit 1\n%s", err, out)
		}
		const refusal = "streamed routed experts requested, but this llama checkpoint carries no fused expert slab the tier can serve"
		if !strings.Contains(string(out), refusal) {
			t.Fatalf("explicit expert refusal missing named no-slab reason:\n%s", out)
		}
		t.Logf("explicit expert precedence receipt: mmap=1 exit=1 refusal=%q", refusal)
		return
	}
	mappedQ4KServeCPUEnv(t)
	path := createMappedQ4KServeCompleteGGUF(t)
	// This ordinary Llama fixture has no fused routed-expert slab. The explicit
	// request must select its established refusal even when mmap residency is on.
	if m, _, _ := loadResidentQ4KProfiled(path, time.Now(), ggufload.WithStreamedExperts(0)); m != nil {
		_ = m.CloseWeights()
	}
	t.Fatal("ordinary Llama artifact unexpectedly admitted explicit streamed experts")
}

func createMappedQ4KServeCompleteGGUF(t *testing.T) string {
	t.Helper()
	const dim, vocab = 256, 16
	type fixtureTensor struct {
		name   string
		shape  []uint64
		kind   ggufload.TensorType
		bytes  []byte
		offset uint64
	}
	f32 := func(count int, norm bool) []byte {
		out := make([]byte, count*4)
		for i := range count {
			value := float32(i%29-14) / 32
			if norm {
				value = 1
			}
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(value))
		}
		return out
	}
	q4k := func(rows, seed int) []byte {
		const blockBytes = 144
		out := make([]byte, rows*blockBytes)
		for row := range rows {
			block := out[row*blockBytes : (row+1)*blockBytes]
			binary.LittleEndian.PutUint16(block, 0x2400) // finite d = 1/64
			binary.LittleEndian.PutUint16(block[2:], 0x2000)
			for i := 4; i < 16; i++ {
				block[i] = byte(1 + (row+seed+i)%3)
			}
			for i := 16; i < blockBytes; i++ {
				block[i] = byte(1+(row+seed+i)%15) | byte(1+(2*row+seed+3*i)%15)<<4
			}
		}
		return out
	}
	tensors := []fixtureTensor{
		{name: "token_embd.weight", shape: []uint64{dim, vocab}, kind: ggufload.TensorF32, bytes: f32(dim*vocab, false)},
		{name: "blk.0.attn_norm.weight", shape: []uint64{dim}, kind: ggufload.TensorF32, bytes: f32(dim, true)},
		{name: "blk.0.ffn_norm.weight", shape: []uint64{dim}, kind: ggufload.TensorF32, bytes: f32(dim, true)},
		{name: "output_norm.weight", shape: []uint64{dim}, kind: ggufload.TensorF32, bytes: f32(dim, true)},
	}
	for seed, name := range []string{"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"} {
		tensors = append(tensors, fixtureTensor{name: "blk.0." + name + ".weight", shape: []uint64{dim, dim}, kind: ggufload.TensorQ4_K, bytes: q4k(dim, seed+1)})
	}
	tensors = append(tensors, fixtureTensor{name: "output.weight", shape: []uint64{dim, vocab}, kind: ggufload.TensorQ4_K, bytes: q4k(vocab, 17)})
	var payload bytes.Buffer
	for i := range tensors {
		padToAlignmentForTest(&payload, 32)
		tensors[i].offset = uint64(payload.Len())
		payload.Write(tensors[i].bytes)
	}
	var header bytes.Buffer
	writeMinimalHeaderForTest(&header, uint64(len(tensors)), 12)
	writeKVStringForTest(&header, "general.architecture", "llama")
	writeKVUint32ForTest(&header, "general.alignment", 32)
	writeKVUint32ForTest(&header, "llama.embedding_length", dim)
	writeKVUint32ForTest(&header, "llama.block_count", 1)
	writeKVUint32ForTest(&header, "llama.attention.head_count", 1)
	writeKVUint32ForTest(&header, "llama.attention.head_count_kv", 1)
	writeKVUint32ForTest(&header, "llama.attention.key_length", dim)
	writeKVUint32ForTest(&header, "llama.feed_forward_length", dim)
	writeKVUint32ForTest(&header, "llama.context_length", 16)
	writeKVFloat32ForTest(&header, "llama.attention.layer_norm_rms_epsilon", 1e-6)
	writeKVFloat32ForTest(&header, "llama.rope.freq_base", 10000)
	writeStringForTest(&header, "tokenizer.ggml.tokens")
	_ = binary.Write(&header, binary.LittleEndian, uint32(ggufload.TypeArray))
	_ = binary.Write(&header, binary.LittleEndian, uint32(ggufload.TypeString))
	_ = binary.Write(&header, binary.LittleEndian, uint64(vocab))
	for i := range vocab {
		writeStringForTest(&header, string(rune('a'+i)))
	}
	for _, tensor := range tensors {
		writeTensorInfoForTest(&header, tensor.name, tensor.shape, uint32(tensor.kind), tensor.offset)
	}
	padToAlignmentForTest(&header, 32)
	header.Write(payload.Bytes())
	// The loader maps whole file pages. Keep the final LM-head payload inside
	// that mappable prefix so this fixture witnesses mapping for both tensors.
	padToAlignmentForTest(&header, os.Getpagesize())
	path := filepath.Join(t.TempDir(), "complete-nonzero-q4k.gguf")
	if err := os.WriteFile(path, header.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
