package model

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func writeV4Shard(t *testing.T, path string, tensors map[string][]float32) {
	t.Helper()
	h := make(map[string]any, len(tensors))
	data := make([]byte, 0)
	for name, vals := range tensors {
		start := len(data)
		for _, v := range vals {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			data = append(data, b[:]...)
		}
		h[name] = map[string]any{"dtype": "F32", "shape": []int{1, len(vals)}, "data_offsets": []int{start, len(data)}}
	}
	hb, err := json.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}
	blob := make([]byte, 8)
	binary.LittleEndian.PutUint64(blob, uint64(len(hb)))
	blob = append(blob, hb...)
	blob = append(blob, data...)
	if err := os.WriteFile(path, blob, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestV4ShardedExpertSourceRoutesSelectedTensors(t *testing.T) {
	dir := t.TempDir()
	n := func(e int, w string) string {
		return "model.layers.0.ffn.experts." + string(rune('0'+e)) + "." + w + ".weight"
	}
	a1, a2, a3 := n(1, "w1"), n(1, "w2"), n(1, "w3")
	b1, b2, b3 := n(2, "w1"), n(2, "w2"), n(2, "w3")
	writeV4Shard(t, filepath.Join(dir, "a.safetensors"), map[string][]float32{a1: {1}, a3: {3}, b2: {20}})
	writeV4Shard(t, filepath.Join(dir, "b.safetensors"), map[string][]float32{a2: {2}, b1: {10}, b3: {30}})
	wm := map[string]string{a1: "a.safetensors", a2: "b.safetensors", a3: "a.safetensors", b1: "b.safetensors", b2: "a.safetensors", b3: "b.safetensors", "model.embed_tokens.weight": "b.safetensors"}
	ib, _ := json.Marshal(map[string]any{"weight_map": wm})
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600); err != nil {
		t.Fatal(err)
	}

	s, err := newV4ShardedExpertSource(dir, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if s.openCount != 0 || s.readCount != 0 {
		t.Fatalf("constructor performed shard IO: opens=%d reads=%d", s.openCount, s.readCount)
	}
	batch, err := s.readV4ExpertBatch(0, []int{2, 1}, 24)
	if err != nil {
		t.Fatal(err)
	}
	if batch.Plan.Bytes != 24 || len(batch.Tensors) != 6 || s.readCount != 6 {
		t.Fatalf("plan=%+v tensors=%d reads=%d", batch.Plan, len(batch.Tensors), s.readCount)
	}
	if len(s.open) > 1 || s.openCount < 2 {
		t.Fatalf("bounded handles violated: resident=%d cumulative=%d", len(s.open), s.openCount)
	}
	want := map[string]float32{a1: 1, a2: 2, a3: 3, b1: 10, b2: 20, b3: 30}
	for _, x := range batch.Tensors {
		if len(x.Bytes) != 4 || math.Float32frombits(binary.LittleEndian.Uint32(x.Bytes)) != want[x.Name] {
			t.Fatalf("bad tensor %s", x.Name)
		}
	}
}

func TestV4ShardedExpertStagerReadsAcrossShardBoundary(t *testing.T) {
	dir := t.TempDir()
	w1 := "model.layers.0.ffn.experts.1.w1.weight"
	w2 := "model.layers.0.ffn.experts.1.w2.weight"
	w3 := "model.layers.0.ffn.experts.1.w3.weight"
	writeV4Shard(t, filepath.Join(dir, "a.safetensors"), map[string][]float32{w1: {1, 0}, w3: {3, 0}})
	writeV4Shard(t, filepath.Join(dir, "b.safetensors"), map[string][]float32{w2: {2, 0}})
	ib, _ := json.Marshal(map[string]any{"weight_map": map[string]string{w1: "a.safetensors", w2: "b.safetensors", w3: "a.safetensors"}})
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600); err != nil {
		t.Fatal(err)
	}
	s, err := newV4ShardedExpertSource(dir, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	plan, err := s.planV4ExpertBatch(0, []int{1}, 24)
	if err != nil {
		t.Fatal(err)
	}
	be := compute.Default()
	stager, err := newV4ShardedExpertStager(s, newPagedRing(be, 16), plan, compute.F32, func(tensor v4ExpertTensor) (compute.Tensor, error) {
		vals := make([]float32, len(tensor.Bytes)/4)
		for i := range vals {
			vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(tensor.Bytes[i*4:]))
		}
		return compute.NewF32(be, tensor.Shape, vals), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	var input [4]byte
	binary.LittleEndian.PutUint32(input[:], math.Float32bits(2))
	got, err := stager.matMul(w1, be.Upload(compute.NewF32(be, []int{2}, []float32{math.Float32frombits(binary.LittleEndian.Uint32(input[:])), 0}), compute.F32))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0] != 2 {
		t.Fatalf("matmul=%v want [2]", got)
	}
	if stager.Stats().SourceReads != 1 || s.readCount != 1 {
		t.Fatalf("stager=%+v source reads=%d", stager.Stats(), s.readCount)
	}
}

func TestV4ShardedExpertSourceRejectsUnsafeIndexBeforeShardOpen(t *testing.T) {
	for _, shard := range []string{"../escape.safetensors", "missing.bin"} {
		t.Run(shard, func(t *testing.T) {
			dir := t.TempDir()
			ib, _ := json.Marshal(map[string]any{"weight_map": map[string]string{"model.layers.0.ffn.experts.1.w1.weight": shard}})
			os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600)
			if _, err := newV4ShardedExpertSource(dir, 1); err == nil {
				t.Fatal("expected refusal")
			}
		})
	}
}

func TestV4ShardedExpertSourceMissingIndexedTensorFailsBeforePayloadRead(t *testing.T) {
	dir := t.TempDir()
	name := "model.layers.0.ffn.experts.1.w1.weight"
	writeV4Shard(t, filepath.Join(dir, "a.safetensors"), map[string][]float32{"model.layers.0.ffn.experts.2.w1.weight": {1}})
	ib, _ := json.Marshal(map[string]any{"weight_map": map[string]string{name: "a.safetensors"}})
	os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600)
	s, err := newV4ShardedExpertSource(dir, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err = s.readV4ExpertBatch(0, []int{1}, 4); err == nil {
		t.Fatal("expected missing tensor refusal")
	}
	if s.readCount != 0 {
		t.Fatalf("payload reads=%d", s.readCount)
	}
}

// v4ReadAtProbe wraps an io.ReaderAt and records the maximum number of ReadAt calls it ever saw
// in flight at once. It is the witness for the batch readahead: the serial batch path issues one
// ReadAt at a time (peak 1), whereas a batch readahead fans the whole batch across the shared
// ReadRanges workers so the peak rises above 1.
//
// The overlap is observed through a synchronization barrier, not a wall-clock sample. A reader that
// arrives while fewer than `want` reads are in flight waits on a condvar until the `want`-th read
// arrives (the fan-out case: every worker is already running, so the barrier releases
// deterministically) or `hold` elapses (the genuinely serial case: the lone reader proceeds alone,
// so a regression still reports peak 1 instead of deadlocking). A fixed spin-loop window was too
// short to overlap the goroutines on the WSL drvfs /mnt/c path even though ReadRanges does spawn
// them, which made the assertion filesystem/scheduler dependent.
type v4ReadAtProbe struct {
	inner io.ReaderAt
	want  int

	mu       sync.Mutex
	cond     *sync.Cond
	inFlight int
	maxSeen  int
	reads    int
}

// v4ReadAtProbeHold bounds how long a lone reader waits for peers. It is only reached on the
// serial-regression path; a real fan-out satisfies the barrier as soon as the peers are scheduled.
const v4ReadAtProbeHold = 250 * time.Millisecond

func newV4ReadAtProbe(inner io.ReaderAt, want int) *v4ReadAtProbe {
	if want < 1 {
		want = 1
	}
	p := &v4ReadAtProbe{inner: inner, want: want}
	p.cond = sync.NewCond(&p.mu)
	return p
}

func (p *v4ReadAtProbe) ReadAt(b []byte, off int64) (int, error) {
	p.mu.Lock()
	p.inFlight++
	if p.inFlight > p.maxSeen {
		p.maxSeen = p.inFlight
	}
	p.reads++
	if p.inFlight >= p.want {
		// The expected fan-out is reached: release every held reader at once.
		p.cond.Broadcast()
	} else {
		p.waitForPeersLocked()
	}
	p.mu.Unlock()

	n, err := p.inner.ReadAt(b, off)

	p.mu.Lock()
	p.inFlight--
	p.cond.Broadcast()
	p.mu.Unlock()
	return n, err
}

// waitForPeersLocked blocks until `want` reads are concurrently in flight, or until the hold
// elapses. Called with p.mu held; returns with p.mu held. The bounded hold is driven by a timer
// goroutine so the wait never depends on how long the underlying ReadAt itself takes.
func (p *v4ReadAtProbe) waitForPeersLocked() {
	timedOut := false
	timer := time.AfterFunc(v4ReadAtProbeHold, func() {
		p.mu.Lock()
		timedOut = true
		p.cond.Broadcast()
		p.mu.Unlock()
	})
	defer timer.Stop()
	for p.inFlight < p.want && !timedOut {
		p.cond.Wait()
	}
	if timedOut {
		// Release any peer still held behind this reader so the serial path cannot wedge.
		p.cond.Broadcast()
	}
}

func (p *v4ReadAtProbe) Peak() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.maxSeen
}

func (p *v4ReadAtProbe) Reads() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.reads
}

func TestV4ShardedExpertBatchIssuesReadAheadAndIsByteIdentical(t *testing.T) {
	dir := t.TempDir()
	n := func(e int, w string) string {
		return "model.layers.0.ffn.experts." + string(rune('0'+e)) + "." + w + ".weight"
	}
	a1, a2, a3 := n(1, "w1"), n(1, "w2"), n(1, "w3")
	b1, b2, b3 := n(2, "w1"), n(2, "w2"), n(2, "w3")
	writeV4Shard(t, filepath.Join(dir, "a.safetensors"), map[string][]float32{a1: {1}, a3: {3}, b2: {20}})
	writeV4Shard(t, filepath.Join(dir, "b.safetensors"), map[string][]float32{a2: {2}, b1: {10}, b3: {30}})
	wm := map[string]string{a1: "a.safetensors", a2: "b.safetensors", a3: "a.safetensors", b1: "b.safetensors", b2: "a.safetensors", b3: "b.safetensors"}
	ib, _ := json.Marshal(map[string]any{"weight_map": wm})
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o600); err != nil {
		t.Fatal(err)
	}

	// Baseline bytes via the unchanged serial path.
	base, err := newV4ShardedExpertSource(dir, 1)
	if err != nil {
		t.Fatal(err)
	}
	wantBatch, err := base.readV4ExpertBatch(0, []int{2, 1}, 24)
	if err != nil {
		t.Fatal(err)
	}
	base.Close()

	s, err := newV4ShardedExpertSource(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	// Resolve both shards first, then wrap each payload reader in a probe.
	if _, err := s.read(a1); err != nil {
		t.Fatal(err)
	}
	if _, err := s.read(a2); err != nil {
		t.Fatal(err)
	}
	var probes []*v4ReadAtProbe
	for _, h := range s.open {
		p := newV4ReadAtProbe(h.src.file.r, 3)
		h.src.file.r = p
		probes = append(probes, p)
	}
	if len(probes) == 0 {
		t.Fatal("no shard handles to probe")
	}

	got, err := s.readV4ExpertBatch(0, []int{2, 1}, 24)
	if err != nil {
		t.Fatal(err)
	}

	// Byte-identity against the serial baseline.
	if len(got.Tensors) != len(wantBatch.Tensors) {
		t.Fatalf("tensor count=%d want %d", len(got.Tensors), len(wantBatch.Tensors))
	}
	want := map[string][]byte{}
	for _, x := range wantBatch.Tensors {
		want[x.Name] = x.Bytes
	}
	for _, x := range got.Tensors {
		if !bytes.Equal(x.Bytes, want[x.Name]) {
			t.Fatalf("tensor %s bytes differ from serial baseline: got %v want %v", x.Name, x.Bytes, want[x.Name])
		}
	}

	// Readahead: at least one probe saw two payload reads in flight at once. The serial path is
	// physically incapable of this.
	peak, reads := 0, 0
	for _, p := range probes {
		if p.Peak() > peak {
			peak = p.Peak()
		}
		reads += p.Reads()
	}
	if reads < 6 {
		t.Fatalf("expected 6 payload reads for the batch, saw %d", reads)
	}
	if peak < 2 {
		t.Fatalf("batch reads were issued serially (peak concurrency %d); expected readahead fan-out", peak)
	}
}
