package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestV4ExpertStreamingFixtureSpine is the bounded CPU spine for #4809. It drives
// official DeepSeek-V4 expert names through the real safetensors ReaderAt seam and
// the real pagedRing, then compares every streamed GEMM with an always-resident
// cpu-ref result. This is a fixture witness, not a full-checkpoint performance claim.
func TestV4ExpertStreamingFixtureSpine(t *testing.T) {
	const (
		experts     = 8
		topK        = 6
		out         = 2
		in          = 2
		weightBytes = int64(out * in * 4)
	)

	tensors := make(map[string]tinySTTensor, experts)
	weights := make(map[string][]float32, experts)
	for expert := 0; expert < experts; expert++ {
		name := fmt.Sprintf("model.layers.0.ffn.experts.%d.w1.weight", expert)
		w := []float32{
			float32(expert + 1), float32(expert) + 0.25,
			-float32(expert) - 0.5, float32(expert + 2),
		}
		weights[name] = w
		tensors[name] = tinySTTensor{dtype: "F32", shape: []int{out, in}, data: f32TestBytes(w)}
	}
	buf := tinySafetensorsBytes(t, tensors)
	rr := &v4RangeReaderAt{data: buf}
	sf, err := newSafetensorsFile(rr, int64(len(buf)), nil)
	if err != nil {
		t.Fatalf("newSafetensorsFile: %v", err)
	}

	entries := make(map[string]stEntry, experts)
	for name, raw := range sf.hdr {
		var entry stEntry
		if err := json.Unmarshal(raw, &entry); err != nil {
			t.Fatalf("decode %s header: %v", name, err)
		}
		entries[name] = entry
	}
	rr.dataBase = sf.dataBase

	// Deliberately non-monotonic scores make the exact selected set and order visible.
	picks := routeTopKSoftmax([]float32{5, 2, -3, -4, 3, -2, 8, 6}, topK)
	wantExperts := []int{6, 7, 0, 4, 1, 5}
	if len(picks) != topK {
		t.Fatalf("selected experts=%d, want exactly %d", len(picks), topK)
	}
	selected := make([]string, topK)
	for i, pick := range picks {
		if pick.expert != wantExperts[i] {
			t.Fatalf("selected expert[%d]=%d, want %d", i, pick.expert, wantExperts[i])
		}
		name := fmt.Sprintf("model.layers.0.ffn.experts.%d.w1.weight", pick.expert)
		if got, ok := classifyV4Tensor(name); !ok || got != V4ClassRoutedExpert {
			t.Fatalf("classifyV4Tensor(%q)=%v, want routed expert", name, got)
		}
		selected[i] = name
	}

	be := compute.Default()
	x := be.Upload(compute.NewF32(be, []int{in}, []float32{0.75, -1.25}), compute.F32)
	// Two weights fit, fewer than the eight-weight fixture and the six selected experts.
	ring := newPagedRing(be, 2*weightBytes)
	peakResidentBytes := int64(0)

	run := func(name string) []float32 {
		t.Helper()
		entry := entries[name]
		got := ring.matMulStaged(name, func() compute.Tensor {
			raw, err := sf.tensorBytes(entry)
			if err != nil {
				t.Fatalf("stream %s: %v", name, err)
			}
			w := decodeV4FixtureF32(t, raw)
			return compute.NewF32(be, []int{out, in}, w)
		}, compute.F32, x, weightBytes, false)
		want := ringResident(t, be, out, in, weights[name], x)
		if !ringEqual(got, want) {
			t.Fatalf("%s streamed output %v != always-resident %v", name, got, want)
		}
		if ring.used() > peakResidentBytes {
			peakResidentBytes = ring.used()
		}
		if ring.used() > ring.budget() {
			t.Fatalf("resident bytes=%d exceed budget=%d", ring.used(), ring.budget())
		}
		return got
	}

	for _, name := range selected {
		run(name)
	}
	// The newest expert is a true cache hit: matMulStaged must not invoke its reader closure.
	run(selected[len(selected)-1])
	readsBeforeReload := rr.tensorReads
	// The first expert was evicted by the bounded two-slot ring and must stream again.
	run(selected[0])
	if rr.tensorReads != readsBeforeReload+1 {
		t.Fatalf("reload tensor reads=%d, want %d", rr.tensorReads, readsBeforeReload+1)
	}

	if ring.pageIn != 7 || ring.hit != 1 || ring.evict != 5 {
		t.Fatalf("cache trace pageIn/hit/evict=%d/%d/%d, want 7/1/5", ring.pageIn, ring.hit, ring.evict)
	}
	if ring.residentCount() != 2 || peakResidentBytes != 2*weightBytes {
		t.Fatalf("resident count/peak bytes=%d/%d, want 2/%d", ring.residentCount(), peakResidentBytes, 2*weightBytes)
	}
	if rr.tensorBytes != 7*weightBytes {
		t.Fatalf("streamed range bytes=%d, want %d", rr.tensorBytes, 7*weightBytes)
	}
	for expert := 0; expert < experts; expert++ {
		name := fmt.Sprintf("model.layers.0.ffn.experts.%d.w1.weight", expert)
		wantReads := 0
		for _, selectedName := range selected {
			if selectedName == name {
				wantReads = 1
			}
		}
		if name == selected[0] {
			wantReads++ // deterministic post-eviction reload
		}
		if got := rr.readsByRange[v4EntryRangeKey(sf.dataBase, entries[name])]; got != wantReads {
			t.Fatalf("%s range reads=%d, want %d (unselected experts must remain unread)", name, got, wantReads)
		}
	}
}

type v4RangeReaderAt struct {
	data         []byte
	dataBase     int64
	tensorReads  int
	tensorBytes  int64
	readsByRange map[string]int
}

func (r *v4RangeReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if r.dataBase != 0 && off >= r.dataBase {
		if r.readsByRange == nil {
			r.readsByRange = make(map[string]int)
		}
		r.tensorReads++
		r.tensorBytes += int64(len(p))
		r.readsByRange[fmt.Sprintf("%d:%d", off, off+int64(len(p)))]++
	}
	if off < 0 || off > int64(len(r.data)) {
		return 0, io.EOF
	}
	n := copy(p, r.data[off:])
	if n != len(p) {
		return n, io.EOF
	}
	return n, nil
}

func v4EntryRangeKey(dataBase int64, entry stEntry) string {
	return fmt.Sprintf("%d:%d", dataBase+int64(entry.DataOffsets[0]), dataBase+int64(entry.DataOffsets[1]))
}

func decodeV4FixtureF32(t *testing.T, raw []byte) []float32 {
	t.Helper()
	if len(raw)%4 != 0 {
		t.Fatalf("fixture F32 byte length=%d is not divisible by 4", len(raw))
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}
