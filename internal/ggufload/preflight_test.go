package ggufload

import (
	"errors"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type dualCapacityBackend struct {
	capBackend
	hostTotal int64
	hostFree  int64
	hostKnown bool
	name      string
	tier      string
}

func (b dualCapacityBackend) Name() string {
	if b.name != "" {
		return b.name
	}
	return b.capBackend.Name()
}

func (b dualCapacityBackend) Tier() string {
	if b.tier != "" {
		return b.tier
	}
	return b.capBackend.Tier()
}

func (b dualCapacityBackend) Caps() compute.Caps {
	c := b.capBackend.Caps()
	c.HostCapacityProbe = b.hostKnown
	return c
}

func (b dualCapacityBackend) HostMemory() (int64, int64, bool) {
	return b.hostTotal, b.hostFree, b.hostKnown
}

var _ compute.HostCapacity = dualCapacityBackend{}

// readyWeightSource builds a header that BOTH parses an architecture (so File.Config succeeds)
// AND carries the same two synth tensors as synthWeightSource, so the lean EstimateLoadBytes is
// the known 1638400 B. It supplies the minimal required GGUF config keys File.Config reads
// (embedding_length divisible by head_count, block_count, feed_forward_length, the rms epsilon).
func readyWeightSource(t *testing.T) *WeightSource {
	t.Helper()
	f := &File{
		Metadata: map[string]Value{
			"general.architecture":                   {Type: TypeString, Value: "llama"},
			"llama.embedding_length":                 {Type: TypeUint32, Value: uint32(256)},
			"llama.block_count":                      {Type: TypeUint32, Value: uint32(2)},
			"llama.attention.head_count":             {Type: TypeUint32, Value: uint32(8)},
			"llama.feed_forward_length":              {Type: TypeUint32, Value: uint32(512)},
			"llama.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
		},
		Tensors: []TensorInfo{
			{Name: "a", Dims: []uint64{256 * 1024}, Type: TensorF32},  // 262144 elems * 4 = 1 MiB
			{Name: "b", Dims: []uint64{256 * 4096}, Type: TensorQ4_K}, // 1048576 elems / 256 * 144 = 589824 B
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// archlessWeightSource is a header with tensors but no general.architecture, so File.Config()
// fails the arch gate — the REFUSE_BAD_ARCH case without a real malformed checkpoint.
func archlessWeightSource(t *testing.T) *WeightSource {
	t.Helper()
	f := &File{
		Tensors: []TensorInfo{
			{Name: "a", Dims: []uint64{256}, Type: TensorF32},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// mixedVulkanWeightSource is the smallest header that exercises every dense Vulkan
// -q4k storage class: Q4_K/Q2_K packed residency, unsupported Q3_K -> Q8,
// an until-#11942 Q2_K embedding -> F32, and a small F32 norm.
func mixedVulkanWeightSource(t *testing.T) *WeightSource {
	t.Helper()
	f := &File{
		Metadata: map[string]Value{
			"general.architecture":                   {Type: TypeString, Value: "llama"},
			"llama.embedding_length":                 {Type: TypeUint32, Value: uint32(256)},
			"llama.block_count":                      {Type: TypeUint32, Value: uint32(1)},
			"llama.attention.head_count":             {Type: TypeUint32, Value: uint32(1)},
			"llama.feed_forward_length":              {Type: TypeUint32, Value: uint32(256)},
			"llama.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
		},
		Tensors: []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{256, 4}, Type: TensorQ2_K},
			{Name: "output.weight", Dims: []uint64{256, 4}, Type: TensorQ2_K},
			{Name: "blk.0.ffn_up.weight", Dims: []uint64{256, 256}, Type: TensorQ4_K},
			{Name: "blk.0.attn_v.weight", Dims: []uint64{256, 256}, Type: TensorQ3_K},
			{Name: "output_norm.weight", Dims: []uint64{256}, Type: TensorF32},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

func mixedVulkanQwen35WeightSource(t *testing.T) *WeightSource {
	t.Helper()
	ws := mixedVulkanWeightSource(t)
	ws.File.Metadata = map[string]Value{
		"general.architecture":                    {Type: TypeString, Value: "qwen35"},
		"qwen35.context_length":                   {Type: TypeUint64, Value: uint64(16)},
		"qwen35.embedding_length":                 {Type: TypeUint64, Value: uint64(256)},
		"qwen35.block_count":                      {Type: TypeUint64, Value: uint64(1)},
		"qwen35.feed_forward_length":              {Type: TypeUint64, Value: uint64(256)},
		"qwen35.attention.head_count":             {Type: TypeUint64, Value: uint64(1)},
		"qwen35.attention.head_count_kv":          {Type: TypeUint64, Value: uint64(1)},
		"qwen35.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
		"qwen35.full_attention_interval":          {Type: TypeUint64, Value: uint64(4)},
	}
	const vocab = uint64(1024)
	ws.File.Tensors[0].Dims = []uint64{256, vocab}
	ws.File.Tensors[1].Dims = []uint64{256, vocab}
	ws.File.Tensors[1].Offset = 1 << 20 // distinct untied output table
	return ws
}

func conservativeTestAllocation(n int64) int64 {
	if n <= 0 {
		return 0
	}
	page := int64(os.Getpagesize())
	return ((n + page - 1) / page * page) + page
}

func TestPreflightVulkanMixedQ4KAccountsPackedQ8F32AndBoundedStaging(t *testing.T) {
	t.Setenv("FAK_GGUF_LOAD_WORKERS", "2")
	ws := mixedVulkanWeightSource(t)
	pf := BuildModelPreflight(PreflightInput{
		Source:         ws,
		Backend:        dualCapacityBackend{capBackend: capBackend{total: 8 << 20, free: 8 << 20, known: true}, hostTotal: 8 << 20, hostFree: 8 << 20, hostKnown: true},
		Headroom:       0.15,
		VulkanMixedQ4K: true,
	})
	if pf.Verdict != PreflightReady || pf.FitState != FitOK {
		t.Fatalf("mixed preflight = %+v, want READY/FIT_OK", pf)
	}

	// Packed: output Q2_K=4 blocks*84, Q4_K=256 blocks*144.
	// Q8 fallback: 256*256 codes + (256*8) f32 scales.
	const q2Payload = int64(4 * 84)
	const q4Payload = int64(256 * 144)
	const q3Payload = int64(256 * 110)
	const q8Codes = int64(256 * 256)
	const q8Scales = int64(256 * 8 * 4)
	const embedF32 = int64(256 * 4 * 4)
	const normF32 = int64(256 * 4)

	wantDevice := q2Payload + q4Payload + q8Codes + q8Scales + embedF32 + normF32
	wantHostResident := conservativeTestAllocation(q2Payload) + conservativeTestAllocation(q4Payload) +
		conservativeTestAllocation(q8Codes) + conservativeTestAllocation(q8Scales) + 2*(embedF32+normF32)
	// Two worker slots: Q3 conversion (raw + old/new f32 normalization buffers)
	// and retained Q4_K are the two largest simultaneous staging demands.
	wantStaging := (q3Payload + 2*256*256*4) + q4Payload
	wantRead := 2*q2Payload + q4Payload + q3Payload + normF32
	if pf.EstDeviceResidentBytes != wantDevice || pf.EstHostResidentBytes != wantHostResident ||
		pf.EstLoadStagingBytes != wantStaging || pf.EstReadBytes != wantRead {
		t.Fatalf("mixed estimate = read/device/host/stage %d/%d/%d/%d, want %d/%d/%d/%d",
			pf.EstReadBytes, pf.EstDeviceResidentBytes, pf.EstHostResidentBytes, pf.EstLoadStagingBytes,
			wantRead, wantDevice, wantHostResident, wantStaging)
	}
	if pf.EstLoadBytes != wantDevice+wantHostResident+wantStaging {
		t.Fatalf("total = %d, want %d", pf.EstLoadBytes, wantDevice+wantHostResident+wantStaging)
	}
	// The embedding is Q2_K on disk but not a matmul weight before #11942. Its
	// host F32 reservation and raw+2*f32 staging are both included above.
	if pf.EstHostResidentBytes <= conservativeTestAllocation(q2Payload)+conservativeTestAllocation(q4Payload) {
		t.Fatal("Q2_K embedding was incorrectly treated as packed resident")
	}
}

func TestPreflightVulkanMixedQ4KUsesHostCapacityAndHeadroom(t *testing.T) {
	t.Setenv("FAK_GGUF_LOAD_WORKERS", "1")
	ws := mixedVulkanWeightSource(t)
	pf := BuildModelPreflight(PreflightInput{
		Source: ws,
		Backend: dualCapacityBackend{
			capBackend: capBackend{total: 8 << 20, free: 8 << 20, known: true},
			hostTotal:  512 << 10, hostFree: 512 << 10, hostKnown: true,
		},
		Headroom:       0.15,
		VulkanMixedQ4K: true,
	})
	if pf.Verdict != PreflightRefuseTooBig || pf.FitState != FitTooBigState || pf.FitScope != string(compute.MemoryScopeHost) {
		t.Fatalf("host-constrained mixed preflight = %+v, want host REFUSE_TOO_BIG", pf)
	}
	wantAvail := compute.BudgetAfterHeadroom(512<<10, 0.15)
	if pf.HostAvailBytes != wantAvail {
		t.Fatalf("headroom-adjusted host availability = %d, want %d", pf.HostAvailBytes, wantAvail)
	}
}

func TestPreflightVulkanMixedQ4KCountsUnifiedAPUMemoryOnce(t *testing.T) {
	t.Setenv("FAK_GGUF_LOAD_WORKERS", "2")
	ws := mixedVulkanWeightSource(t)
	probe := BuildModelPreflight(PreflightInput{
		Source: ws,
		Backend: dualCapacityBackend{
			capBackend: capBackend{total: 8 << 20, free: 8 << 20, known: true},
			hostTotal:  8 << 20, hostFree: 8 << 20, hostKnown: true,
		},
		VulkanMixedQ4K: true,
	})
	if probe.Verdict != PreflightReady {
		t.Fatalf("sizing probe = %+v, want READY", probe)
	}
	hostWant := probe.EstHostResidentBytes + probe.EstLoadStagingBytes
	deviceWant := probe.EstDeviceResidentBytes
	separateMax := max(hostWant, deviceWant)
	sharedFree := separateMax + (probe.EstLoadBytes-separateMax)/2
	if sharedFree <= separateMax || sharedFree >= probe.EstLoadBytes {
		t.Fatalf("invalid fixture: host=%d device=%d total=%d shared=%d", hostWant, deviceWant, probe.EstLoadBytes, sharedFree)
	}

	backend := dualCapacityBackend{
		capBackend: capBackend{total: sharedFree, free: sharedFree, known: true},
		hostTotal:  sharedFree, hostFree: sharedFree, hostKnown: true,
		name: "vulkan", tier: "integrated:test-apu",
	}
	pf := BuildModelPreflight(PreflightInput{Source: ws, Backend: backend, VulkanMixedQ4K: true})
	if pf.Verdict != PreflightRefuseTooBig || pf.FitState != FitTooBigState || pf.FitScope != string(compute.MemoryScopeHost) {
		t.Fatalf("integrated mixed preflight = %+v, want unified host REFUSE_TOO_BIG", pf)
	}
	if pf.EstDeviceResidentBytes > pf.HostAvailBytes || hostWant > pf.HostAvailBytes || pf.EstLoadBytes <= pf.HostAvailBytes {
		t.Fatalf("fixture did not isolate combined-pool refusal: device=%d host=%d total=%d avail=%d",
			pf.EstDeviceResidentBytes, hostWant, pf.EstLoadBytes, pf.HostAvailBytes)
	}
	if !strings.Contains(pf.Reason, "integrated Vulkan unified-memory admission") {
		t.Fatalf("refusal reason %q does not identify the shared physical pool", pf.Reason)
	}

	backend.tier = "discrete:test-gpu"
	pf = BuildModelPreflight(PreflightInput{Source: ws, Backend: backend, VulkanMixedQ4K: true})
	if pf.Verdict != PreflightReady || pf.FitState != FitOK {
		t.Fatalf("discrete mixed preflight = %+v, want independent-pool READY/FIT_OK", pf)
	}
}

func TestPreflightVulkanMixedQ4KAccountsResidentQ2KEmbedding(t *testing.T) {
	t.Setenv("FAK_GGUF_LOAD_WORKERS", "2")
	ws := mixedVulkanQwen35WeightSource(t)
	backend := dualCapacityBackend{
		capBackend: capBackend{total: 8 << 30, free: 8 << 30, known: true},
		hostTotal:  8 << 30, hostFree: 8 << 30, hostKnown: true,
	}
	base := BuildModelPreflight(PreflightInput{Source: ws, Backend: backend, VulkanMixedQ4K: true})
	packed := BuildModelPreflight(PreflightInput{
		Source: ws, Backend: backend, VulkanMixedQ4K: true, ResidentQ2KEmbedding: true,
	})
	if base.Verdict != PreflightReady || packed.Verdict != PreflightReady {
		t.Fatalf("base=%+v packed=%+v, want READY", base, packed)
	}
	const (
		vocab       = int64(1024)
		hidden      = int64(256)
		packedBytes = vocab * (hidden / 256) * 84
		f32Bytes    = vocab * hidden * 4
	)
	if got, want := base.EstDeviceResidentBytes-packed.EstDeviceResidentBytes, f32Bytes; got != want {
		t.Fatalf("device reduction=%d, want removed whole-table F32 %d", got, want)
	}
	if got, want := base.EstHostResidentBytes-packed.EstHostResidentBytes, 2*f32Bytes-conservativeTestAllocation(packedBytes); got != want {
		t.Fatalf("host reduction=%d, want %d", got, want)
	}
	if got, want := base.EstLoadStagingBytes-packed.EstLoadStagingBytes, 2*f32Bytes; got != want {
		t.Fatalf("staging reduction=%d, want removed two F32 work buffers %d", got, want)
	}
}

func TestPreflightVulkanMixedQ4KFailsClosedForUnknownAndSplit(t *testing.T) {
	tests := map[string]func(*WeightSource){
		"unknown_tensor": func(ws *WeightSource) {
			ws.File.Tensors = append(ws.File.Tensors, TensorInfo{Name: "mystery.weight", Dims: []uint64{256, 256}, Type: TensorQ2_K})
		},
		"split_source": func(ws *WeightSource) {
			ws.readerFor = make([]io.ReaderAt, len(ws.File.Tensors))
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			ws := mixedVulkanWeightSource(t)
			mutate(ws)
			pf := BuildModelPreflight(PreflightInput{Source: ws, VulkanMixedQ4K: true})
			if pf.Verdict != PreflightRefuseHeader || !pf.Refused() {
				t.Fatalf("mixed preflight = %+v, want fail-closed REFUSE_BAD_HEADER", pf)
			}
		})
	}
}

func TestPreflightReadyFitOKOnKnownBigDevice(t *testing.T) {
	// synthWeightSource fails File.Config() (no arch) — so to exercise READY we need a header
	// that parses an arch. Build one with the minimum required GGUF config keys plus the two
	// synth tensors so EstimateLoadBytes is the known 1638400 B.
	ws := readyWeightSource(t)
	big := capBackend{total: 2 << 20, free: 2 << 20, known: true}
	pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", Source: ws, Backend: big, Lean: true})
	if pf.Verdict != PreflightReady {
		t.Fatalf("verdict = %s, want READY (reason: %s)", pf.Verdict, pf.Reason)
	}
	if pf.FitState != FitOK {
		t.Fatalf("fit = %s, want FIT_OK", pf.FitState)
	}
	if pf.EstLoadBytes != 1638400 {
		t.Fatalf("est bytes = %d, want 1638400 (the synth lean footprint)", pf.EstLoadBytes)
	}
	if pf.Arch == "" {
		t.Fatalf("arch not populated on READY")
	}
	if pf.ETASecondsEst <= 0 {
		t.Fatalf("ETA must be positive on a non-empty model, got %v", pf.ETASecondsEst)
	}
	if pf.Refused() {
		t.Fatalf("READY must not report Refused()")
	}
}

func TestPreflightRefuseTooBigCarriesDeviceAvail(t *testing.T) {
	ws := readyWeightSource(t)
	small := capBackend{total: 1 << 20, free: 1 << 20, known: true}
	pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", Source: ws, Backend: small, Lean: true})
	if pf.Verdict != PreflightRefuseTooBig {
		t.Fatalf("verdict = %s, want REFUSE_TOO_BIG", pf.Verdict)
	}
	if pf.FitState != FitTooBigState {
		t.Fatalf("fit = %s, want FIT_TOO_BIG", pf.FitState)
	}
	if pf.DeviceAvailBytes != 1<<20 {
		t.Fatalf("device avail = %d, want %d (from the FitError)", pf.DeviceAvailBytes, 1<<20)
	}
	if !pf.Refused() {
		t.Fatalf("REFUSE_TOO_BIG must report Refused()")
	}
	if !strings.Contains(pf.Reason, "FitTooBig") {
		t.Fatalf("reason should carry the typed FitError text, got %q", pf.Reason)
	}
}

func TestPreflightFitUnknownFailsOpen(t *testing.T) {
	ws := readyWeightSource(t)
	// Both a nil backend and a backend that cannot probe must yield READY + FIT_UNKNOWN — the
	// fail-open contract that keeps the portable floor loadable.
	for name, be := range map[string]compute.Backend{
		"nil":         nil,
		"unprobeable": capBackend{known: false},
	} {
		pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", Source: ws, Backend: be, Lean: true})
		if pf.Verdict != PreflightReady {
			t.Fatalf("[%s] verdict = %s, want READY (fail-open)", name, pf.Verdict)
		}
		if pf.FitState != FitUnknown {
			t.Fatalf("[%s] fit = %s, want FIT_UNKNOWN", name, pf.FitState)
		}
		if pf.DeviceAvailBytes != 0 {
			t.Fatalf("[%s] unprobeable device must not report avail bytes, got %d", name, pf.DeviceAvailBytes)
		}
	}
}

func TestPreflightRefuseBadArch(t *testing.T) {
	ws := archlessWeightSource(t)
	pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", Source: ws, Lean: true})
	if pf.Verdict != PreflightRefuseArch {
		t.Fatalf("verdict = %s, want REFUSE_BAD_ARCH", pf.Verdict)
	}
	if !pf.Refused() {
		t.Fatalf("REFUSE_BAD_ARCH must report Refused()")
	}
	if !strings.Contains(pf.Reason, "architecture") {
		t.Fatalf("reason should name the missing architecture, got %q", pf.Reason)
	}
}

func TestPreflightRefuseBadHeader(t *testing.T) {
	pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", OpenErr: errors.New("gguf: bad magic"), Source: nil})
	if pf.Verdict != PreflightRefuseHeader {
		t.Fatalf("verdict = %s, want REFUSE_BAD_HEADER", pf.Verdict)
	}
	if !pf.Refused() {
		t.Fatalf("REFUSE_BAD_HEADER must report Refused()")
	}
	if !strings.Contains(pf.Reason, "bad magic") {
		t.Fatalf("reason should carry the open error, got %q", pf.Reason)
	}
	// A nil source with no error is also a header refusal (defensive).
	pf2 := BuildModelPreflight(PreflightInput{Path: "x.gguf"})
	if pf2.Verdict != PreflightRefuseHeader {
		t.Fatalf("nil source / nil err verdict = %s, want REFUSE_BAD_HEADER", pf2.Verdict)
	}
}

func TestPreflightDeterministic(t *testing.T) {
	ws := readyWeightSource(t)
	in := PreflightInput{Path: "x.gguf", Source: ws, Backend: capBackend{total: 2 << 20, free: 2 << 20, known: true}, Lean: true}
	a := BuildModelPreflight(in)
	b := BuildModelPreflight(in)
	if a != b {
		t.Fatalf("preflight not deterministic:\n a=%+v\n b=%+v", a, b)
	}
}

func TestPreflightETAScalesWithBytes(t *testing.T) {
	// A higher assumed throughput must shorten the ETA for the same byte estimate.
	ws := readyWeightSource(t)
	slow := BuildModelPreflight(PreflightInput{Source: ws, Lean: true, AssumedGiBPerSec: 0.1})
	fast := BuildModelPreflight(PreflightInput{Source: ws, Lean: true, AssumedGiBPerSec: 10})
	if !(slow.ETASecondsEst > fast.ETASecondsEst) {
		t.Fatalf("slower assumed rate must give a longer ETA: slow=%v fast=%v", slow.ETASecondsEst, fast.ETASecondsEst)
	}
	if fast.ETASecondsEst <= 0 {
		t.Fatalf("ETA must stay positive, got %v", fast.ETASecondsEst)
	}
}

func TestPreflightF32PathEstimatesLargerThanLean(t *testing.T) {
	// The default (non-lean) GGUF path dequantizes to f32 resident, so its estimate must exceed
	// the lean raw-payload estimate for the same header — proving the regime selection is wired.
	ws := readyWeightSource(t)
	lean := BuildModelPreflight(PreflightInput{Source: ws, Lean: true})
	f32 := BuildModelPreflight(PreflightInput{Source: ws}) // default path
	if !(f32.EstLoadBytes > lean.EstLoadBytes) {
		t.Fatalf("f32 estimate (%d) must exceed lean estimate (%d)", f32.EstLoadBytes, lean.EstLoadBytes)
	}
}

func TestPreflightRenderMentionsVerdictAndEstimate(t *testing.T) {
	ws := readyWeightSource(t)
	pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", Source: ws, Lean: true})
	out := pf.Render()
	for _, want := range []string{pf.Verdict, "load:", "GiB", "ETA", "estimate", "fit:"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render %q missing %q", out, want)
		}
	}
}
