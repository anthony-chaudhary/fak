package model

import (
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// immutableWeightRecordingBackend makes the device-memory ownership boundary observable
// without requiring a GPU. Its operations remain the cpu-ref oracle; only Caps and the
// upload/free ledger model a device whose immutable uploads have a real lifetime.
type immutableWeightRecordingBackend struct {
	compute.Backend
	mu            sync.Mutex
	weightUploads map[string]int
	weightBuffers map[compute.Buffer]string
	freeCalls     map[compute.Buffer]int
	dtypeUploads  map[compute.Dtype]int
}

func newImmutableWeightRecordingBackend() *immutableWeightRecordingBackend {
	return &immutableWeightRecordingBackend{
		Backend:       compute.Default(),
		weightUploads: make(map[string]int),
		weightBuffers: make(map[compute.Buffer]string),
		freeCalls:     make(map[compute.Buffer]int),
		dtypeUploads:  make(map[compute.Dtype]int),
	}
}

func (b *immutableWeightRecordingBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	out := b.Backend.Upload(t, as)
	b.mu.Lock()
	b.dtypeUploads[as]++
	if as != compute.F32 {
		b.weightBuffers[out.Buf()] = "upload " + as.String()
	}
	b.mu.Unlock()
	return out
}

func (*immutableWeightRecordingBackend) Name() string                    { return "recording-device" }
func (*immutableWeightRecordingBackend) Tier() string                    { return "recording" }
func (*immutableWeightRecordingBackend) Class() compute.CorrectnessClass { return compute.Approx }
func (b *immutableWeightRecordingBackend) Caps() compute.Caps {
	caps := b.Backend.Caps()
	caps.DeviceMemory = true
	return caps
}

func (b *immutableWeightRecordingBackend) UploadClass(t compute.Tensor, as compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	out := b.Backend.Upload(t, as)
	if class == compute.MemoryWeights {
		b.mu.Lock()
		b.weightUploads[site]++
		b.weightBuffers[out.Buf()] = site
		b.mu.Unlock()
	}
	return out
}

func (b *immutableWeightRecordingBackend) Free(t compute.Tensor) {
	b.mu.Lock()
	b.freeCalls[t.Buf()]++
	b.mu.Unlock()
	b.Backend.Free(t)
}

func (b *immutableWeightRecordingBackend) immutableUploadCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	total := 0
	for _, n := range b.weightUploads {
		total += n
	}
	return total
}

func (b *immutableWeightRecordingBackend) assertWeightFrees(t *testing.T, want int) {
	t.Helper()
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.weightBuffers) == 0 {
		t.Fatal("recording backend observed no immutable weight uploads")
	}
	for buf, site := range b.weightBuffers {
		if got := b.freeCalls[buf]; got != want {
			t.Errorf("immutable weight %q free calls=%d, want %d", site, got, want)
		}
	}
}

func immutableResidencyTestConfig() Config {
	return Config{
		HiddenSize:        16,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           4,
		IntermediateSize:  32,
		VocabSize:         64,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
}

func TestDeviceImmutableWeightsSurviveSessionCloseAndServeSnapshotRestore(t *testing.T) {
	m := NewSynthetic(immutableResidencyTestConfig())
	be := newImmutableWeightRecordingBackend()
	first, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{3, 7, 11, 19}
	first.Prefill(prompt)
	snapshot, err := first.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()

	continuation := []int{23, 29}
	want := make([][]float32, len(continuation))
	for i, token := range continuation {
		want[i] = first.Step(token)
	}
	uploads := be.immutableUploadCount()
	if uploads == 0 {
		t.Fatal("prefill staged no immutable weights")
	}
	first.Close()
	be.assertWeightFrees(t, 0)

	restored, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	if err := snapshot.Restore(restored); err != nil {
		restored.Close()
		t.Fatal(err)
	}
	for i, token := range continuation {
		assertSameF32(t, "restored continuation step "+itoa(i), want[i], restored.Step(token))
	}
	if got := be.immutableUploadCount(); got != uploads {
		t.Fatalf("restored session immutable uploads=%d, want original %d (no repeated staging)", got, uploads)
	}
	restored.Close()
	be.assertWeightFrees(t, 0)

	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	be.assertWeightFrees(t, 1)
}

func TestDeviceImmutableWeightResidencyCoalescesConcurrentSessions(t *testing.T) {
	m := NewSynthetic(immutableResidencyTestConfig())
	be := newImmutableWeightRecordingBackend()
	a, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.NewBackendSessionChecked(be)
	if err != nil {
		a.Close()
		t.Fatal(err)
	}

	const name = "model.norm.weight"
	start := make(chan struct{})
	resident := make(chan compute.Tensor, 2)
	for _, s := range []*Session{a, b} {
		go func(s *Session) {
			<-start
			resident <- s.weightHAL(name)
		}(s)
	}
	close(start)
	ta, tb := <-resident, <-resident
	if ta.Buf() != tb.Buf() {
		t.Fatal("concurrent sessions received different immutable weight handles")
	}
	if got := be.immutableUploadCount(); got != 1 {
		t.Fatalf("concurrent immutable uploads=%d, want 1", got)
	}
	a.Close()
	be.assertWeightFrees(t, 0)
	_ = b.Backend.Read(tb)
	b.Close()
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	be.assertWeightFrees(t, 1)
}

func TestDeviceImmutableWeightResidencyIsBackendIdentityScoped(t *testing.T) {
	m := NewSynthetic(immutableResidencyTestConfig())
	aBackend := newImmutableWeightRecordingBackend()
	bBackend := newImmutableWeightRecordingBackend()
	a, err := m.NewBackendSessionChecked(aBackend)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.NewBackendSessionChecked(bBackend)
	if err != nil {
		a.Close()
		t.Fatal(err)
	}

	const name = "model.norm.weight"
	if a.weightHAL(name).Buf() == b.weightHAL(name).Buf() {
		t.Fatal("same-named backend instances shared a device weight handle")
	}
	if aBackend.Name() != bBackend.Name() || aBackend.immutableUploadCount() != 1 || bBackend.immutableUploadCount() != 1 {
		t.Fatalf("backend-scoped uploads=%d/%d names=%q/%q", aBackend.immutableUploadCount(), bBackend.immutableUploadCount(), aBackend.Name(), bBackend.Name())
	}
	a.Close()
	b.Close()
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	aBackend.assertWeightFrees(t, 1)
	bBackend.assertWeightFrees(t, 1)
}

func TestDeviceImmutableQ4KWeightUsesFormatScopedModelResidency(t *testing.T) {
	const out, in = 8, 256
	qt := quantizeQ4KFromRaw(buildRawQ4K(t, out, in, 9420), out, in)
	m := &Model{q4kw: map[string]*q4kTensor{"dense.weight": qt}}
	be := newImmutableWeightRecordingBackend()
	newSession := func() *Session {
		return &Session{
			M: m, Backend: be,
			halW: make(map[string]compute.Tensor), borrowedHALW: make(map[string]struct{}),
		}
	}
	first := newSession()
	firstWeight := first.weightHALQ4K("dense.weight", qt)
	first.Close()
	be.assertWeightFrees(t, 0)

	second := newSession()
	secondWeight := second.weightHALQ4K("dense.weight", qt)
	if firstWeight.Buf() != secondWeight.Buf() {
		t.Fatal("sequential Q4_K sessions received different immutable handles")
	}
	be.mu.Lock()
	uploads := be.dtypeUploads[compute.Q4_K]
	be.mu.Unlock()
	if uploads != 1 {
		t.Fatalf("Q4_K immutable uploads=%d, want 1", uploads)
	}
	second.Close()
	be.assertWeightFrees(t, 0)
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	be.assertWeightFrees(t, 1)
}

func TestDeviceImmutableWeightDeferredModelCloseFreesOnce(t *testing.T) {
	m := NewSynthetic(immutableResidencyTestConfig())
	be := newImmutableWeightRecordingBackend()
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	_ = s.weightHAL("model.norm.weight")
	if err := m.CloseWeights(); err == nil {
		t.Fatal("CloseWeights admitted while a backend session still borrowed model residency")
	}
	be.assertWeightFrees(t, 0)
	s.Close()
	be.assertWeightFrees(t, 1)
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	be.assertWeightFrees(t, 1)
}

func TestSessionCloseUsesBorrowedKeyOwnershipNotTensorIdentity(t *testing.T) {
	m := NewSynthetic(immutableResidencyTestConfig())
	be := newImmutableWeightRecordingBackend()
	weight := compute.NewF32(compute.Default(), []int{1}, []float32{1})
	s := &Session{
		M: m, Backend: be,
		halW:         map[string]compute.Tensor{"borrowed": weight, "private": weight},
		borrowedHALW: map[string]struct{}{"borrowed": {}},
	}
	s.Close()
	be.mu.Lock()
	frees := be.freeCalls[weight.Buf()]
	be.mu.Unlock()
	if frees != 1 {
		t.Fatalf("same tensor under borrowed and private keys freed %d times, want private key exactly once", frees)
	}
}

func TestQwenHybridDerivedWeightsShareModelResidency(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	be := newRecordingQwen35Backend(m)
	be.deviceMemory = true
	countWeightUploads := func() int {
		total := 0
		for i, class := range be.classes {
			if class == compute.MemoryWeights && strings.HasPrefix(be.sites[i], "hal-weight ") {
				total++
			}
		}
		return total
	}
	assertWeightFrees := func(want int) {
		t.Helper()
		seen := 0
		for buf, site := range be.tensorSites {
			if !strings.HasPrefix(site, "hal-weight ") {
				continue
			}
			seen++
			if got := be.freeCalls[buf]; got != want {
				t.Errorf("Qwen immutable weight %q free calls=%d, want %d", site, got, want)
			}
		}
		if seen == 0 {
			t.Fatal("Qwen backend observed no immutable weight uploads")
		}
	}

	first, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	first.Prefill([]int{3, 7, 11})
	uploads := countWeightUploads()
	first.Close()
	assertWeightFrees(0)

	second, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	second.Prefill([]int{3, 7, 11})
	if got := countWeightUploads(); got != uploads {
		second.Close()
		t.Fatalf("second Qwen session immutable uploads=%d, want %d", got, uploads)
	}
	second.Close()
	assertWeightFrees(0)
	be.reference.Close()
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	assertWeightFrees(1)
}

func TestCPUReferenceKeepsSessionLocalWeightMemoizer(t *testing.T) {
	m := NewSynthetic(immutableResidencyTestConfig())
	s, err := m.NewBackendSessionChecked(compute.Default())
	if err != nil {
		t.Fatal(err)
	}
	const name = "model.norm.weight"
	_ = s.weightHAL(name)
	if _, ok := s.halW[name]; !ok {
		t.Fatal("cpu-ref immutable weight did not use the established session-local memoizer")
	}
	s.Close()
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
}

func TestImmutableWeightUploadSitesRemainWeightClassed(t *testing.T) {
	m := NewSynthetic(immutableResidencyTestConfig())
	be := newImmutableWeightRecordingBackend()
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	_ = s.weightHAL("model.norm.weight")
	be.mu.Lock()
	for site := range be.weightUploads {
		if !strings.HasPrefix(site, "hal-weight ") {
			t.Errorf("immutable upload site=%q, want hal-weight classification", site)
		}
	}
	be.mu.Unlock()
	s.Close()
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
}
