package model

import (
	"bytes"
	"errors"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func mappedCheckpointFixture(t *testing.T, quant ExpertCheckpointQuant, budget int64, page, overlay bool) (*ExpertCheckpointTier, *mappedExpertReader, string, int) {
	t.Helper()
	_, stride, _ := quant.blockGeometry()
	r := &mappedExpertReader{data: make([]byte, 2*stride)}
	for i := range r.data {
		r.data[i] = byte(i*7 + 11)
	}
	tier := NewExpertCheckpointTier(budget)
	var data []byte
	if page {
		data = r.data
	}
	const fused = "blk.0.ffn_gate_exps.weight"
	if err := tier.AddShardData(r, int64(len(r.data)), data, []FusedExpertTensor{{Name: fused, Layer: 0, Proj: "gate_proj", Quant: quant, Experts: 2, Rows: 1, Cols: 256}}); err != nil {
		t.Fatal(err)
	}
	if overlay {
		// Even an overlay without this row must retain the historical dense fallback.
		tier.overlays = map[string]*ExpertSparseOverlay{fused: {}}
	}
	return tier, r, expertName(0, 0, "gate_proj.weight"), stride
}

func mappedCheckpointRaw(w expertWeight) []byte {
	if w.kq != nil {
		return w.kq.raw
	}
	if w.q4 != nil {
		return w.q4.raw
	}
	return nil
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointMappedRangeEligibility(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name                  string
		quant                 ExpertCheckpointQuant
		budget                int64
		page, overlay, borrow bool
	}{
		{name: "q2", quant: ExpertCheckpointQ2K, borrow: true},
		{name: "q3", quant: ExpertCheckpointQ3K, borrow: true},
		{name: "q4", quant: ExpertCheckpointQ4K, borrow: true},
		{name: "q5", quant: ExpertCheckpointQ5K, borrow: true},
		{name: "q6", quant: ExpertCheckpointQ6K, borrow: true},
		{name: "retained", quant: ExpertCheckpointQ2K, budget: 4096},
		{name: "overlay_missing_row", quant: ExpertCheckpointQ2K, overlay: true},
		{name: "page_cache", quant: ExpertCheckpointQ3K, page: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tier, r, name, stride := mappedCheckpointFixture(t, tc.quant, tc.budget, tc.page, tc.overlay)
			w, err := tier.fault(name)
			raw := mappedCheckpointRaw(w)
			if err != nil || !bytes.Equal(raw, r.data[:stride]) {
				t.Fatalf("fault bytes differ: %v", err)
			}
			if aliased := &raw[0] == &r.data[0]; aliased != tc.borrow {
				t.Fatalf("alias=%t want=%t", aliased, tc.borrow)
			}
			wantRanges, wantReads := int64(0), int64(1)
			if tc.borrow {
				wantRanges, wantReads = 1, 0
				if cap(raw) != stride {
					t.Fatalf("borrowed capacity=%d want=%d", cap(raw), stride)
				}
			}
			st := tier.Stats()
			if r.ranges.Load() != wantRanges || r.reads.Load() != wantReads || st.Reads != 1 || st.BytesRead != int64(stride) || st.Failures != 0 {
				t.Fatalf("capability/accounting mismatch: ranges=%d reads=%d stats=%+v", r.ranges.Load(), r.reads.Load(), st)
			}
			if tc.budget == 0 && (st.ResidentBytes != 0 || st.ResidentCount != 0 || st.PeakBytes != 0) {
				t.Fatalf("stream-through retained host bytes: %+v", st)
			}
		})
	}
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointMappedRangeConstructorsPreserveSource(t *testing.T) {
	t.Parallel()
	for _, quant := range []ExpertCheckpointQuant{ExpertCheckpointQ2K, ExpertCheckpointQ3K} {
		tier, r, name, stride := mappedCheckpointFixture(t, quant, 0, false, false)
		before := append([]byte(nil), r.data...)
		safe := &mappedBorrowSafeUpload{&mappedUploadWitness{Backend: compute.Default(), label: "opted-in"}}
		stage, ok := tier.stagingForBackend(name, safe)
		if !ok || r.ranges.Load() != 0 {
			t.Fatal("staging descriptor faulted eagerly")
		}
		host := stage.mk()
		hb, ok := host.Buf().(compute.HostBuffer)
		if !ok || len(hb.I8()) != stride || mappedHostPointer(host) != uintptr(unsafe.Pointer(&r.data[0])) || !bytes.Equal(before, r.data) {
			t.Fatal("host constructor changed or expanded mapped source")
		}
		for i, b := range hb.I8() {
			if b != int8(before[i]) {
				t.Fatalf("host constructor changed byte %d", i)
			}
		}
		// The odd Q3_K block remains 110 bytes here. Vulkan owns its separate
		// 112-byte padded upload allocation; it must never pad this mapped source.
		if quant == ExpertCheckpointQ3K && len(hb.I8()) != 110 { //boundarylint:ignore CHANGE_DETECTOR_TEST Q3_K packed ABI is 110 bytes per block; Vulkan upload padding must not alter mapped input
			t.Fatal("Q3_K host source lost exact payload length")
		}
	}
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointMappedRangeSessionLifetime(t *testing.T) {
	t.Parallel()
	for _, quant := range []ExpertCheckpointQuant{ExpertCheckpointQ2K, ExpertCheckpointQ3K} {
		tier, r, name, stride := mappedCheckpointFixture(t, quant, 0, false, false)
		m := NewSynthetic(Config{VocabSize: 8, HiddenSize: 256, NumLayers: 0})
		m.SetExpertCheckpoint(tier)
		closer := &countWeightCloser{}
		m.SetWeightCloser(closer)
		safe := &mappedBorrowSafeUpload{&mappedUploadWitness{Backend: compute.Default(), label: "opted-in"}}
		s := m.NewBackendSession(safe)
		t.Cleanup(s.Close)
		weight, ok := s.resolveExpertWeight(name)
		if !ok || weight.ck == nil {
			t.Fatal("missing mapped expert staging")
		}
		// Retain the borrowed host tensor before its synchronous Upload while
		// the session owns the model/checkpoint lifetime.
		host := weight.ck.mk()
		hb, ok := host.Buf().(compute.HostBuffer)
		if !ok || len(hb.I8()) != stride || mappedHostPointer(host) != uintptr(unsafe.Pointer(&r.data[0])) || r.ranges.Load() != 1 || r.reads.Load() != 0 {
			t.Fatal("safe staging did not retain the borrowed host tensor")
		}
		var active *WeightSessionsActiveError
		if err := m.CloseWeights(); !errors.As(err, &active) || active.Count != 1 || closer.n.Load() != 0 {
			t.Fatalf("borrowed source closed while session remained: %v", err)
		}
		for i, b := range hb.I8() {
			if b != int8(r.data[i]) {
				t.Fatalf("borrowed byte %d changed before session close", i)
			}
		}
		s.Close()
		if closer.n.Load() != 1 {
			t.Fatalf("source closes=%d want=1", closer.n.Load())
		}
		// Borrowed bytes are never accessed after the final session close.
		if err := m.CloseWeights(); err != nil || closer.n.Load() != 1 {
			t.Fatalf("repeated source close: %v count=%d", err, closer.n.Load())
		}
	}
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointMappedRangeOwnerWaiters(t *testing.T) {
	t.Parallel()
	const callers = 8
	boom := errors.New("mapped range failed")
	for _, mode := range []string{"success", "short", "long", "nil", "error", "panic"} {
		t.Run(mode, func(t *testing.T) {
			tier, r, name, stride := mappedCheckpointFixture(t, ExpertCheckpointQ2K, 0, false, false)
			entered := make(chan struct{}, 1)
			release := make(chan struct{})
			unblock := sync.OnceFunc(func() { close(release) })
			t.Cleanup(unblock)
			r.rangeFn = func(off, size int64) ([]byte, error) {
				entered <- struct{}{}
				<-release
				switch mode {
				case "short":
					return r.data[:stride-1], nil
				case "long":
					return r.data[:stride+1], nil
				case "nil":
					return nil, nil
				case "error":
					return r.data[:stride], boom
				case "panic":
					panic(boom)
				}
				return r.data[off : off+size : off+size], nil
			}
			names := make([]string, callers)
			for i := range names {
				names[i] = name
			}
			results := checkpointFlightFaults(tier, names...)
			checkpointFlightWait(t, func() bool { return len(entered) == 1 && checkpointFlightCoalesced(tier) == callers-1 })
			if st := tier.Stats(); st.Reads != 0 || st.ResidentBytes != 0 {
				t.Fatalf("incomplete owner published bytes: %+v", st)
			}
			unblock()
			var shared error
			panics := 0
			for _, result := range checkpointFlightCollect(t, results, callers) {
				if mode == "success" {
					raw := mappedCheckpointRaw(result.weight)
					if result.err != nil || result.panic != nil || len(raw) != stride || &raw[0] != &r.data[0] {
						t.Fatalf("waiter did not receive complete borrowed view: %+v", result)
					}
					continue
				}
				if result.panic != nil {
					if mode != "panic" || result.panic != boom {
						t.Fatalf("unexpected owner panic: %v", result.panic)
					}
					panics++
					continue
				}
				want := ErrGGUFExpertMetadata
				if mode == "error" {
					want = boom
				}
				if mode == "panic" {
					want = ErrExpertCheckpointLoadAborted
				}
				if !errors.Is(result.err, want) || mappedCheckpointRaw(result.weight) != nil {
					t.Fatalf("failed owner published a weight: %+v", result)
				}
				if shared == nil {
					shared = result.err
				} else if shared != result.err {
					t.Fatal("waiters received separate errors")
				}
			}
			st := tier.Stats()
			if r.ranges.Load() != 1 || r.reads.Load() != 0 || st.Coalesced != callers-1 || st.ResidentCount != 0 {
				t.Fatalf("owner/coalescing changed: ranges=%d reads=%d stats=%+v", r.ranges.Load(), r.reads.Load(), st)
			}
			if mode == "success" {
				if st.Reads != 1 || st.BytesRead != int64(stride) || st.Failures != 0 {
					t.Fatalf("success accounting: %+v", st)
				}
			} else {
				if st.Reads != 0 || st.BytesRead != 0 || st.Failures != 1 {
					t.Fatalf("failure accounting: %+v", st)
				}
				if (mode == "panic") != (panics == 1) {
					t.Fatalf("owner panics=%d mode=%s", panics, mode)
				}
			}
			// A new fault after complete publication still borrows at budget zero;
			// a failed or panicking owner leaves no stale loading entry.
			r.rangeFn = nil
			w, err := tier.fault(name)
			if err != nil || len(mappedCheckpointRaw(w)) != stride || r.ranges.Load() != 2 || r.reads.Load() != 0 {
				t.Fatalf("retry failed: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointMappedRangeIndependentKeys(t *testing.T) {
	t.Parallel()
	tier, r, name, stride := mappedCheckpointFixture(t, ExpertCheckpointQ3K, 0, false, false)
	release := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	t.Cleanup(unblock)
	r.rangeFn = func(off, size int64) ([]byte, error) {
		<-release
		return r.data[off : off+size : off+size], nil
	}
	other := expertName(0, 1, "gate_proj.weight")
	results := checkpointFlightFaults(tier, name, other)
	// Both ranges must enter while the first is held, proving mapping access
	// does not hold the tier bookkeeping mutex or serialize different experts.
	checkpointFlightWait(t, func() bool { return r.ranges.Load() == 2 })
	unblock()
	for _, result := range checkpointFlightCollect(t, results, 2) {
		off := 0
		if result.name == other {
			off = stride
		}
		raw := mappedCheckpointRaw(result.weight)
		if result.err != nil || result.panic != nil || len(raw) != stride || &raw[0] != &r.data[off] {
			t.Fatalf("independent mapped expert: %+v", result)
		}
	}
	if st := tier.Stats(); st.Reads != 2 || st.BytesRead != int64(2*stride) || st.Coalesced != 0 || st.Failures != 0 || r.reads.Load() != 0 {
		t.Fatalf("independent-key accounting: %+v", st)
	}
}

// mappedUploadWitness models host-pointer interning without a physical device.
// Free visibly invalidates one returned allocation so peer ownership is tested.
type mappedUploadWitness struct {
	compute.Backend
	label  string
	intern bool
	mu     sync.Mutex
	inputs []uintptr
	keep   []compute.Tensor
	cache  map[uintptr]compute.Tensor
}

func (b *mappedUploadWitness) Name() string { return b.label }

func mappedHostPointer(t compute.Tensor) uintptr {
	raw := t.Buf().(compute.HostBuffer).I8()
	return uintptr(unsafe.Pointer(&raw[0]))
}

func (b *mappedUploadWitness) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	b.mu.Lock()
	defer b.mu.Unlock()
	ptr := mappedHostPointer(t)
	b.inputs = append(b.inputs, ptr)
	if b.intern {
		if cached, ok := b.cache[ptr]; ok {
			return cached
		}
		b.keep = append(b.keep, t) // mirror the interning backend's retained host owner
	}
	i8 := t.Buf().(compute.HostBuffer).I8()
	raw := make([]byte, len(i8))
	for i, v := range i8 {
		raw[i] = byte(v)
	}
	var out compute.Tensor
	switch as {
	case compute.Q2_K:
		out = compute.NewQ2K(compute.Default(), t.Shape, raw)
	case compute.Q3_K:
		out = compute.NewQ3K(compute.Default(), t.Shape, raw)
	case compute.Q4_K:
		out = compute.NewQ4K(compute.Default(), t.Shape, raw)
	case compute.Q5_K:
		out = compute.NewQ5K(compute.Default(), t.Shape, raw)
	case compute.Q6_K:
		out = compute.NewQ6K(compute.Default(), t.Shape, raw)
	default:
		panic("unexpected mapped upload dtype")
	}
	if b.intern {
		if b.cache == nil {
			b.cache = map[uintptr]compute.Tensor{}
		}
		b.cache[ptr] = out
	}
	return out
}

func (b *mappedUploadWitness) Free(t compute.Tensor) {
	if hb, ok := t.Buf().(compute.HostBuffer); ok {
		clear(hb.I8())
	}
}

// The positive test backend copies synchronously and never interns ownership.
type mappedBorrowSafeUpload struct{ *mappedUploadWitness }

func (*mappedBorrowSafeUpload) SupportsBorrowedExpertUploads() bool { return true }

type mappedBorrowDeclinedUpload struct{ *mappedUploadWitness }

func (*mappedBorrowDeclinedUpload) SupportsBorrowedExpertUploads() bool { return false }

func mappedBackendSession(t *testing.T, m *Model, be compute.Backend) *Session {
	t.Helper()
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	return s
}

func mappedSessionStage(s *Session, name string) compute.Tensor {
	w, ok := s.resolveExpertWeight(name)
	if !ok || w.ck == nil {
		panic("missing checkpoint staging")
	}
	return s.weightHALStagedBounded(w.ck.key, name, w.ck.mk, w.ck.dt, w.ck.bytes)
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointMappedRangeUnsafeConsumersOwnCopies(t *testing.T) {
	t.Parallel()
	for _, label := range []string{"cuda", "vulkan", "unknown", "declined"} {
		t.Run(label, func(t *testing.T) {
			tier, r, name, stride := mappedCheckpointFixture(t, ExpertCheckpointQ2K, 0, false, false)
			m := NewSynthetic(Config{VocabSize: 8, HiddenSize: 256, NumLayers: 0})
			m.SetExpertCheckpoint(tier)
			be := &mappedUploadWitness{Backend: compute.Default(), label: label, intern: true}
			var backend compute.Backend = be
			if label == "declined" {
				backend = &mappedBorrowDeclinedUpload{be}
			}
			first, second := mappedBackendSession(t, m, backend), mappedBackendSession(t, m, backend)
			one, two := mappedSessionStage(first, name), mappedSessionStage(second, name)
			mapped := uintptr(unsafe.Pointer(&r.data[0]))
			if len(be.inputs) != 2 || be.inputs[0] == mapped || be.inputs[1] == mapped || be.inputs[0] == be.inputs[1] || one.Buf() == two.Buf() {
				t.Fatal("unknown or name-only backend shared mapped host/device ownership")
			}
			first.Close()
			for i, b := range two.Buf().(compute.HostBuffer).I8() {
				if b != int8(r.data[i]) {
					t.Fatalf("peer close invalidated byte %d", i)
				}
			}
			if r.ranges.Load() != 2 || r.reads.Load() != 0 || tier.Stats().BytesRead != int64(2*stride) {
				t.Fatal("unsafe fallback changed logical source accounting")
			}
		})
	}
}

type mappedOwnedReader struct{ data, last []byte }

func (r *mappedOwnedReader) ReadAt(p []byte, off int64) (int, error) {
	r.last = p
	return bytes.NewReader(r.data).ReadAt(p, off)
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointMappedRangePreservesOwnedReadAtResult(t *testing.T) {
	t.Parallel()
	r := &mappedOwnedReader{data: bytes.Repeat([]byte{31}, q2kBlockBytes)}
	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShard(r, int64(len(r.data)), []FusedExpertTensor{{Name: "gate", Layer: 0, Proj: "gate_proj", Quant: ExpertCheckpointQ2K, Experts: 1, Rows: 1, Cols: 256}}); err != nil {
		t.Fatal(err)
	}
	stage, ok := tier.stagingForBackend(expertName(0, 0, "gate_proj.weight"), &mappedUploadWitness{Backend: compute.Default(), label: "unknown", intern: true})
	if !ok {
		t.Fatal("missing owned staging")
	}
	host := stage.mk()
	if len(r.last) != q2kBlockBytes || mappedHostPointer(host) != uintptr(unsafe.Pointer(&r.last[0])) || !bytes.Equal(r.last, r.data) {
		t.Fatal("ordinary ReadAt result was copied again or changed")
	}
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointMappedRangeMixedBackendWaiters(t *testing.T) {
	t.Parallel()
	tier, r, name, stride := mappedCheckpointFixture(t, ExpertCheckpointQ3K, 0, false, false)
	m := NewSynthetic(Config{VocabSize: 8, HiddenSize: 256, NumLayers: 0})
	m.SetExpertCheckpoint(tier)
	safe := &mappedBorrowSafeUpload{&mappedUploadWitness{Backend: compute.Default(), label: "opted-in"}}
	unsafeBackend := &mappedUploadWitness{Backend: compute.Default(), label: "vulkan", intern: true}
	sessions := []*Session{mappedBackendSession(t, m, safe), mappedBackendSession(t, m, unsafeBackend)}
	release := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	t.Cleanup(unblock)
	r.rangeFn = func(off, size int64) ([]byte, error) { <-release; return r.data[off : off+size : off+size], nil }
	results := make(chan compute.Tensor, 2)
	for _, s := range sessions {
		go func() { results <- mappedSessionStage(s, name) }()
	}
	checkpointFlightWait(t, func() bool { return r.ranges.Load() == 1 && tier.Stats().Coalesced == 1 })
	unblock()
	checkpointFlightWait(t, func() bool { return len(results) == 2 })
	one, two := <-results, <-results
	mapped := uintptr(unsafe.Pointer(&r.data[0]))
	if len(safe.inputs) != 1 || safe.inputs[0] != mapped || len(unsafeBackend.inputs) != 1 || unsafeBackend.inputs[0] == mapped || one.Buf() == two.Buf() {
		t.Fatal("mixed waiters lost per-consumer borrowed/owned policy")
	}
	if st := tier.Stats(); st.Reads != 1 || st.BytesRead != int64(stride) || st.Coalesced != 1 || st.Failures != 0 || st.ResidentCount != 0 || r.reads.Load() != 0 {
		t.Fatalf("mixed-backend source accounting: %+v", st)
	}
	for _, tensor := range []compute.Tensor{one, two} {
		for i, b := range tensor.Buf().(compute.HostBuffer).I8() {
			if b != int8(r.data[i]) {
				t.Fatalf("mixed consumer byte %d changed", i)
			}
		}
	}
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointWarmMappedConsumerOwnership(t *testing.T) {
	t.Parallel()
	for _, quant := range []ExpertCheckpointQuant{ExpertCheckpointQ2K, ExpertCheckpointQ3K, ExpertCheckpointQ4K, ExpertCheckpointQ5K, ExpertCheckpointQ6K} {
		t.Run(quant.String(), func(t *testing.T) {
			_, stride, _ := quant.blockGeometry()
			for _, budget := range []struct {
				name  string
				bytes int64
			}{{"stream-through", 0}, {"retained", int64(stride)}} {
				t.Run(budget.name, func(t *testing.T) {
					for _, policy := range []string{"unknown", "declined", "admitted"} {
						t.Run(policy, func(t *testing.T) {
							tier, r, name, _ := mappedCheckpointFixture(t, quant, budget.bytes, true, false)
							// Prime only source pages: no tier host entry may hide the warm fault.
							_, warm, err := tier.shards[0].readExpertPageCache("blk.0.ffn_gate_exps.weight", 0)
							if err != nil || warm || tier.Stats().Reads != 0 || r.reads.Load() != 1 {
								t.Fatalf("source-only cold prime failed: warm=%t err=%v", warm, err)
							}
							before := append([]byte(nil), r.data...)
							m := NewSynthetic(Config{VocabSize: 8, HiddenSize: 256, NumLayers: 0})
							m.SetExpertCheckpoint(tier)
							closer := &countWeightCloser{}
							m.SetWeightCloser(closer)
							t.Cleanup(func() { _ = m.CloseWeights() })
							rec := &mappedUploadWitness{Backend: compute.Default(), label: policy, intern: policy != "admitted"}
							var backend compute.Backend = rec
							switch policy {
							case "declined":
								backend = &mappedBorrowDeclinedUpload{rec}
							case "admitted":
								backend = &mappedBorrowSafeUpload{rec}
							}
							first, second := mappedBackendSession(t, m, backend), mappedBackendSession(t, m, backend)
							one, two := mappedSessionStage(first, name), mappedSessionStage(second, name)
							mapped := uintptr(unsafe.Pointer(&r.data[0]))
							mayBorrow := policy == "admitted"
							if len(rec.inputs) != 2 || (rec.inputs[0] == mapped) != mayBorrow || (rec.inputs[1] == mapped) != mayBorrow || (policy != "admitted" && rec.inputs[0] == rec.inputs[1]) || one.Buf() == two.Buf() {
								t.Fatalf("warm mapping lost per-consumer ownership: inputs=%v mapped=%d mayBorrow=%t sharedOutput=%t", rec.inputs, mapped, mayBorrow, one.Buf() == two.Buf())
							}
							if one.Dtype != tier.index[name].dtype() || two.Dtype != one.Dtype {
								t.Fatal("upload witness changed the packed dtype")
							}
							wantReads, wantHits, wantResident := 2, 0, 0
							if budget.bytes > 0 {
								wantReads, wantHits, wantResident = 1, 1, 1
							}
							st := tier.Stats()
							pc, _ := tier.shards[0].pageCacheSession()
							if r.reads.Load() != 1 || r.ranges.Load() != 0 || st.Reads != wantReads || st.Hits != wantHits || st.BytesRead != int64(wantReads*stride) || st.ResidentCount != wantResident || st.ResidentBytes != int64(wantResident*stride) || st.Failures != 0 || pc.WarmHits != wantReads {
								t.Fatalf("warm staging changed source/accounting: tier=%+v page=%+v", st, pc)
							}
							var active *WeightSessionsActiveError
							if err := m.CloseWeights(); !errors.As(err, &active) || active.Count != 2 || closer.n.Load() != 0 {
								t.Fatalf("live source pin lost: %v", err)
							}
							first.Close()
							if closer.n.Load() != 0 {
								t.Fatal("peer close retired the source")
							}
							for i, b := range two.Buf().(compute.HostBuffer).I8() {
								if b != int8(before[i]) {
									t.Fatalf("peer close invalidated byte %d", i)
								}
							}
							if !bytes.Equal(before, r.data) {
								t.Fatal("consumer changed the source mapping")
							}
							second.Close()
							if closer.n.Load() != 1 {
								t.Fatal("final session did not retire the source once")
							}
						})
					}
				})
			}
		})
	}
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointColdMappedOwnedStagingIdentity(t *testing.T) {
	t.Parallel()
	r := &mappedOwnedReader{data: bytes.Repeat([]byte{31}, 2*q2kBlockBytes)}
	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShardData(r, int64(len(r.data)), r.data, []FusedExpertTensor{{Name: "gate", Layer: 0, Proj: "gate_proj", Quant: ExpertCheckpointQ2K, Experts: 2, Rows: 1, Cols: 256}}); err != nil {
		t.Fatal(err)
	}
	stage, ok := tier.stagingForBackend(expertName(0, 0, "gate_proj.weight"), &mappedUploadWitness{Backend: compute.Default(), label: "unknown", intern: true})
	if !ok {
		t.Fatal("missing cold mapped staging")
	}
	host := stage.mk()
	if len(r.last) != len(r.data) || mappedHostPointer(host) != uintptr(unsafe.Pointer(&r.last[0])) || mappedHostPointer(host) == uintptr(unsafe.Pointer(&r.data[0])) {
		t.Fatal("cold owned page-cache result was copied again or borrowed")
	}
	pc, _ := tier.shards[0].pageCacheSession()
	if pc.WarmHits != 0 || pc.ExpertsRead != 1 || tier.Stats().Reads != 1 {
		t.Fatal("cold control did not execute one owned read")
	}
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointSharedOwnedConsumerOwnership(t *testing.T) {
	t.Parallel()
	for _, quant := range []ExpertCheckpointQuant{ExpertCheckpointQ2K, ExpertCheckpointQ3K, ExpertCheckpointQ4K, ExpertCheckpointQ5K, ExpertCheckpointQ6K} {
		t.Run(quant.String(), func(t *testing.T) {
			_, stride, _ := quant.blockGeometry()
			for _, mode := range []string{"retained", "coalesced"} {
				t.Run(mode, func(t *testing.T) {
					for _, policy := range []string{"unknown", "declined", "admitted"} {
						t.Run(policy, func(t *testing.T) {
							data := make([]byte, 2*stride)
							for i := range data {
								data[i] = byte(i*7 + 11)
							}
							// Ordinary ReaderAt owns its output; no mapped-range capability.
							r := &checkpointFlightReader{data: data, entered: make(chan int64, 4), release: make(chan struct{})}
							budget := int64(0)
							if mode == "retained" {
								budget = int64(stride)
								r.unblock()
							}
							tier := NewExpertCheckpointTier(budget)
							if err := tier.AddShard(r, int64(len(data)), []FusedExpertTensor{{Name: "gate", Layer: 0, Proj: "gate_proj", Quant: quant, Experts: 2, Rows: 1, Cols: 256}}); err != nil {
								t.Fatal(err)
							}
							name := expertName(0, 0, "gate_proj.weight")
							m := NewSynthetic(Config{VocabSize: 8, HiddenSize: 256, NumLayers: 0})
							m.SetExpertCheckpoint(tier)
							closer := &countWeightCloser{}
							m.SetWeightCloser(closer)
							rec := &mappedUploadWitness{Backend: compute.Default(), label: policy, intern: policy != "admitted"}
							var backend compute.Backend = rec
							switch policy {
							case "declined":
								backend = &mappedBorrowDeclinedUpload{rec}
							case "admitted":
								backend = &mappedBorrowSafeUpload{rec}
							}
							var sessions [2]*Session
							var finished []chan struct{}
							t.Cleanup(func() {
								r.unblock()
								timer := time.NewTimer(5 * time.Second)
								defer timer.Stop()
								for _, done := range finished {
									select {
									case <-done:
									case <-timer.C:
										t.Error("shared owned staging did not retire; session cleanup withheld")
										return
									}
								}
								for _, s := range sessions {
									if s != nil {
										s.Close()
									}
								}
								if err := m.CloseWeights(); err != nil {
									t.Error(err)
								}
							})
							for i := range sessions {
								var err error
								sessions[i], err = m.NewBackendSessionChecked(backend)
								if err != nil {
									t.Fatal(err)
								}
							}
							var got [2]compute.Tensor
							var original expertWeight
							if mode == "retained" {
								got[0] = mappedSessionStage(sessions[0], name)
								got[1] = mappedSessionStage(sessions[1], name)
								tier.mu.Lock()
								for _, w := range tier.host {
									original = w
								}
								tier.mu.Unlock()
							} else {
								type outcome struct {
									slot     int
									tensor   compute.Tensor
									panic    any
									complete bool
								}
								results := make(chan outcome, 2)
								start := func(slot int) {
									done := make(chan struct{})
									finished = append(finished, done)
									go func() {
										result := outcome{slot: slot}
										defer close(done)
										defer func() { result.panic = recover(); results <- result }()
										result.tensor = mappedSessionStage(sessions[slot], name)
										result.complete = true
									}()
								}
								start(0)
								checkpointFlightWait(t, func() bool { return len(r.entered) == 1 })
								var load *expertCheckpointLoad
								tier.mu.Lock()
								for _, pending := range tier.loading {
									load = pending
								}
								tier.mu.Unlock()
								if load == nil {
									t.Fatal("gated owner was not registered")
								}
								start(1)
								checkpointFlightWait(t, func() bool { return checkpointFlightCoalesced(tier) == 1 })
								r.unblock()
								timer := time.NewTimer(5 * time.Second)
								defer timer.Stop()
								for range sessions {
									select {
									case result := <-results:
										if !result.complete || result.panic != nil {
											t.Fatalf("shared owned staging did not return: %+v", result)
										}
										got[result.slot] = result.tensor
									case <-timer.C:
										t.Fatal("shared owned staging result timed out")
									}
								}
								for _, done := range finished {
									select {
									case <-done:
									case <-timer.C:
										t.Fatal("shared owned staging did not finish its deferred result")
									}
								}
								original = load.weight
							}
							raw := mappedCheckpointRaw(original)
							if len(raw) != stride || original.borrowed || !bytes.Equal(raw, data[:stride]) {
								t.Fatal("shared record lost original owned provenance or bytes")
							}
							ptr := uintptr(unsafe.Pointer(&raw[0]))
							originalInputs := 0
							for _, input := range rec.inputs {
								if input == ptr {
									originalInputs++
								}
							}
							wantOriginal := 1 // the exclusive cold owner keeps its existing identity
							if policy == "admitted" {
								wantOriginal = 2
							}
							if len(rec.inputs) != 2 || originalInputs != wantOriginal || got[0].Buf() == got[1].Buf() {
								t.Fatalf("shared owned slab lost per-consumer ownership: originalInputs=%d want=%d sharedOutput=%t", originalInputs, wantOriginal, got[0].Buf() == got[1].Buf())
							}
							if got[0].Dtype != tier.index[name].dtype() || got[1].Dtype != got[0].Dtype {
								t.Fatal("shared owned upload changed dtype")
							}
							wantHits, wantCoalesced, wantResident := 0, 1, 0
							if mode == "retained" {
								wantHits, wantCoalesced, wantResident = 1, 0, 1
							}
							st := tier.Stats()
							if r.calls.Load() != 1 || st.Reads != 1 || st.Hits != wantHits || st.Coalesced != wantCoalesced || st.BytesRead != int64(stride) || st.ResidentCount != wantResident || st.ResidentBytes != int64(wantResident*stride) || st.BudgetBytes != budget || st.PeakBytes != budget || st.Failures != 0 {
								t.Fatalf("shared owned accounting changed: %+v", st)
							}
							// One real eviction while both uploaded session handles remain live.
							if mode == "retained" && quant == ExpertCheckpointQ3K && policy == "unknown" {
								if _, err := tier.fault(expertName(0, 1, "gate_proj.weight")); err != nil {
									t.Fatal(err)
								}
								st = tier.Stats()
								if st.Evictions != 1 || st.ResidentCount != 1 || st.ResidentBytes != int64(stride) || st.Reads != 2 || st.BytesRead != int64(2*stride) || st.BudgetBytes != budget || st.PeakBytes != budget || !bytes.Equal(raw, data[:stride]) {
									t.Fatalf("eviction changed held slab/accounting: %+v", st)
								}
							}
							var active *WeightSessionsActiveError
							if err := m.CloseWeights(); !errors.As(err, &active) || active.Count != 2 || closer.n.Load() != 0 {
								t.Fatalf("shared owned source pin lost: %v", err)
							}
							sessions[0].Close()
							if closer.n.Load() != 0 {
								t.Fatal("peer close retired shared owned source")
							}
							for i, b := range got[1].Buf().(compute.HostBuffer).I8() {
								if b != int8(data[i]) {
									t.Fatalf("peer close invalidated shared owned byte %d", i)
								}
							}
							if !bytes.Equal(raw, data[:stride]) {
								t.Fatal("consumer mutated shared owned slab")
							}
							sessions[1].Close()
							if closer.n.Load() != 1 {
								t.Fatal("shared owned source did not retire exactly once")
							}
						})
					}
				})
			}
		})
	}
}

// fak-test:runtime fast est=100ms
func TestExpertCheckpointReadyOwnedStagingIdentity(t *testing.T) {
	t.Parallel()
	r := &checkpointFlightReader{data: bytes.Repeat([]byte{31}, q2kBlockBytes), entered: make(chan int64, 1), release: make(chan struct{})}
	r.unblock()
	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShard(r, int64(len(r.data)), []FusedExpertTensor{{Name: "gate", Layer: 0, Proj: "gate_proj", Quant: ExpertCheckpointQ2K, Experts: 1, Rows: 1, Cols: 256}}); err != nil {
		t.Fatal(err)
	}
	name := expertName(0, 0, "gate_proj.weight")
	if tier.Prefetch([]string{name}) != 1 {
		t.Fatal("exclusive ready fixture was not admitted")
	}
	checkpointFlightWait(t, func() bool {
		tier.mu.Lock()
		defer tier.mu.Unlock()
		return len(tier.ready) == 1 && len(tier.loading) == 0 && tier.prefetchJobs == 0
	})
	var original expertWeight
	tier.mu.Lock()
	for _, ready := range tier.ready {
		original = ready.weight
	}
	tier.mu.Unlock()
	raw := mappedCheckpointRaw(original)
	if len(raw) != q2kBlockBytes || original.borrowed {
		t.Fatal("ready record was not an exclusive owned read")
	}
	stage, ok := tier.stagingForBackend(name, &mappedUploadWitness{Backend: compute.Default(), label: "unknown", intern: true})
	if !ok {
		t.Fatal("exclusive ready staging missing")
	}
	host := stage.mk()
	if mappedHostPointer(host) != uintptr(unsafe.Pointer(&raw[0])) || !bytes.Equal(raw, r.data) {
		t.Fatal("exclusive ready handoff was copied or changed")
	}
	tier.mu.Lock()
	remaining := len(tier.ready)
	tier.mu.Unlock()
	if st := tier.Stats(); remaining != 0 || r.calls.Load() != 1 || st.Reads != 1 || st.PrefetchHits != 1 || st.Coalesced != 0 || st.ResidentBytes != 0 {
		t.Fatalf("exclusive ready accounting changed: %+v", st)
	}
}
