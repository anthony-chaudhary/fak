package model

import (
	"bytes"
	"errors"
	"io"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41FaultReader struct {
	mu     sync.Mutex
	data   []byte
	reads  [][2]int64
	closed bool
	err    error
	short  bool
}

func (r *v41FaultReader) ReadAt(dst []byte, off int64) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return 0, errors.New("read after close")
	}
	r.reads = append(r.reads, [2]int64{off, int64(len(dst))})
	if r.err != nil {
		return 0, r.err
	}
	if r.short {
		return 0, nil
	}
	return bytes.NewReader(r.data).ReadAt(dst, off)
}

func (r *v41FaultReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closed = true
	return nil
}

// fak-test:runtime fast est=100ms lane=race
func TestV41ExpertFaultIn(t *testing.T) {
	saved := v41ExpertQuantSpecs
	defer func() { v41ExpertQuantSpecs = saved }()
	const rows, cols, scaleCols, base = 2, 16, 1, 7
	stem, weights, scales := v41ExpertDeviceFixture(t, rows, cols, scaleCols)
	name, scaleName := stem+".weight", stem+".scale"
	wb, sb := len(weights), len(scales)
	we := stEntry{Dtype: "I8", Shape: []int{rows, cols}, DataOffsets: []int{3, 3 + wb}}
	se := stEntry{Dtype: "F8_E8M0", Shape: []int{rows, scaleCols}, DataOffsets: []int{5 + wb, 5 + wb + sb}}
	newCase := func() (*Model, *v41FaultReader) {
		r := &v41FaultReader{data: make([]byte, base+we.DataOffsets[1]+sb+9)}
		copy(r.data[base+we.DataOffsets[0]:], weights)
		copy(r.data[base+se.DataOffsets[0]:], scales)
		m := &Model{Cfg: Config{NumLayers: 1, DeepSeekV41: &DeepSeekV41Config{}}}
		m.SetWeightCloser(r)
		t.Cleanup(func() { _ = m.CloseWeights() })
		return m, r
	}
	install := func(m *Model, r *v41FaultReader, budget int) error {
		return installV41ExpertQuantSource(name, scaleName, we, se, r, base, int64(len(r.data)), budget, m)
	}
	x := make([]float32, cols*2)

	t.Run("metadata_only_exact_extents_and_reusable_slab", func(t *testing.T) {
		m, r := newCase()
		if err := install(m, r, wb+sb); err != nil {
			t.Fatal(err)
		}
		store := v41ExpertDeviceStore(m, false)
		if len(r.reads) != 0 || store.faultSlab != nil || len(store.weights[name].weight) != 0 || !m.V41ExpertDeviceInstalled(name) {
			t.Fatal("registration materialized bytes or hid the source")
		}
		be := &v41PackedLifetimeBackend{Backend: compute.Default()}
		if _, err := m.V41ExpertDeviceMatmul(name, be, x); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(be.weight, weights) || !bytes.Equal(be.scales, scales) {
			t.Fatal("source pair changed")
		}
		if len(r.reads) != 2 || r.reads[0] != [2]int64{base + 3, int64(wb)} || r.reads[1] != [2]int64{int64(base + 5 + wb), int64(sb)} {
			t.Fatalf("reads: %v", r.reads)
		}
		p := &store.faultSlab[0]
		if _, err := m.V41ExpertDeviceMatmul(name, be, x); err != nil {
			t.Fatal(err)
		}
		if len(store.faultSlab) != wb+sb || cap(store.faultSlab) != wb+sb || p != &store.faultSlab[0] {
			t.Fatal("staging slab grew or changed")
		}
		if err := install(m, r, wb+sb); err == nil {
			t.Fatal("duplicate replaced immutable binding")
		}
		if err := m.CloseWeights(); err != nil {
			t.Fatal(err)
		}
		if store.faultSlab != nil || m.V41ExpertDeviceInstalled(name) || !r.closed {
			t.Fatal("source/staging not retired")
		}
	})

	t.Run("budget_and_extent_refusals_before_io", func(t *testing.T) {
		m, r := newCase()
		if err := install(m, r, wb+sb-1); err == nil {
			t.Fatal("underbudget accepted")
		}
		bad := se
		bad.DataOffsets = []int{3, 3 + sb}
		if err := installV41ExpertQuantSource(name, scaleName, we, bad, r, base, int64(len(r.data)), wb+sb, m); err == nil {
			t.Fatal("overlapping pair accepted")
		}
		if err := installV41ExpertQuantSource(name, scaleName, we, se, r, base, int64(base+se.DataOffsets[1]-1), wb+sb, m); err == nil {
			t.Fatal("truncated extent accepted")
		}
		if err := installV41ExpertQuantSource(name, "layers.0.ffn.experts.1.w1.scale", we, se, r, base, int64(len(r.data)), wb+sb, m); err == nil {
			t.Fatal("foreign scale accepted")
		}
		if len(r.reads) != 0 || m.V41ExpertDeviceInstalled(name) {
			t.Fatal("refusal read or published source")
		}
	})

	t.Run("selected_io_and_scale_errors_do_not_reach_kernel", func(t *testing.T) {
		m, r := newCase()
		if err := install(m, r, wb+sb); err != nil {
			t.Fatal(err)
		}
		be := &v41PackedLifetimeBackend{Backend: compute.Default()}
		sentinel := errors.New("selected source failure")
		r.err = sentinel
		if _, err := m.V41ExpertDeviceMatmul(name, be, x); !errors.Is(err, sentinel) {
			t.Fatalf("lost IO error: %v", err)
		}
		r.err, r.short = nil, true
		if _, err := m.V41ExpertDeviceMatmul(name, be, x); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("short read: %v", err)
		}
		r.short = false
		r.data[base+se.DataOffsets[0]] = 0xff
		if _, err := m.V41ExpertDeviceMatmul(name, be, x); !errors.Is(err, ErrV4ExpertQuantMetadata) {
			t.Fatalf("NaN scale: %v", err)
		}
		if be.weight != nil {
			t.Fatal("failed source reached kernel")
		}
		r.data[base+se.DataOffsets[0]] = scales[0]
		if _, err := m.V41ExpertDeviceMatmul(name, be, x); err != nil {
			t.Fatalf("retry after failed read: %v", err)
		}
	})

	t.Run("missing_kernel_does_not_fault", func(t *testing.T) {
		m, r := newCase()
		if err := install(m, r, wb+sb); err != nil {
			t.Fatal(err)
		}
		if _, err := m.V41ExpertDeviceMatmul(name, compute.Default(), x); !errors.Is(err, compute.ErrExpertDeviceKernelAbsent) {
			t.Fatalf("missing kernel: %v", err)
		}
		if len(r.reads) != 0 {
			t.Fatal("unsupported backend read checkpoint")
		}
	})

	t.Run("source_pinned_through_synchronous_kernel", func(t *testing.T) {
		m, r := newCase()
		if err := install(m, r, wb+sb); err != nil {
			t.Fatal(err)
		}
		be := &v41PackedLifetimeBackend{Backend: compute.Default()}
		be.during = func() {
			var active *WeightSessionsActiveError
			if err := m.CloseWeights(); !errors.As(err, &active) {
				t.Fatalf("close did not defer: %v", err)
			}
			if r.closed {
				t.Fatal("source closed inside kernel")
			}
		}
		if _, err := m.V41ExpertDeviceMatmul(name, be, x); err != nil {
			t.Fatal(err)
		}
		if !r.closed || m.V41ExpertDeviceInstalled(name) {
			t.Fatal("last source call did not retire owner")
		}
		if _, err := m.V41ExpertDeviceMatmul(name, be, x); err == nil {
			t.Fatal("closed source reused")
		}
	})
}
