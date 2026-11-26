package ggufload

import (
	"bytes"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func inkling2Offset(base, b []byte) int {
	return int(uintptr(unsafe.Pointer(&b[0])) - uintptr(unsafe.Pointer(&base[0])))
}

// fak-test:runtime fast est=50ms lane=default
func TestInkling2MappedReadaheadRetargetSkipsReadChunksAndPacesFromCursor(t *testing.T) {
	buf := make([]byte, 96)
	got := make(chan int, 16)
	r := startInkling2MappedReadahead([][]byte{buf[0:64]}, 16, 8, func(b []byte) { got <- inkling2Offset(buf, b) }, nil)
	recv := func(want ...int) {
		t.Helper()
		for _, w := range want {
			select {
			case off := <-got:
				if off != w {
					t.Fatalf("read at %d, want %d", off, w)
				}
			case <-time.After(5 * time.Second):
				t.Fatalf("read at %d never came", w)
			}
		}
	}
	quiet := func() {
		t.Helper()
		select {
		case off := <-got:
			t.Fatalf("read at %d ran past the window", off)
		case <-time.After(20 * time.Millisecond):
		}
	}
	recv(0, 8) // the map-time walk stops at the window with no cursor
	quiet()
	// Upload chunks: [0:8]@0 [8:12]@8 [16:24]@12 [24:32]@20 [32:40]@28 [40:48]@36.
	// The walk already read [0:16), so only the rest is read, paced from offset 0.
	r.retarget([][]byte{buf[0:12], buf[16:48]}, 8)
	recv(16)
	quiet()
	r.advance(12)
	recv(24)
	quiet()
	r.advance(12)
	recv(32, 40)
	// Out of upload chunks, the reader exits on its own; close must not hang.
	r.done.Wait()
	r.close()
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestInkling2ReadaheadCloseWaitsOutTheInFlightRead(t *testing.T) {
	buf := make([]byte, 64)
	entered, gate := make(chan struct{}), make(chan struct{})
	var reads, released atomic.Int32
	r := startInkling2MappedReadahead([][]byte{buf}, 64, 8, func([]byte) {
		if reads.Add(1) == 1 {
			close(entered)
			<-gate
		}
	}, func() { released.Add(1) })
	<-entered
	closed := make(chan struct{})
	go func() { r.close(); close(closed) }()
	select {
	case <-closed:
		t.Fatal("close returned while a read was in flight")
	case <-time.After(20 * time.Millisecond):
	}
	close(gate)
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("close never returned")
	}
	r.close()
	if n := reads.Load(); n != 1 {
		t.Fatalf("%d reads, want none after close", n)
	}
	if n := released.Load(); n != 1 {
		t.Fatalf("release ran %d times, want 1", n)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2ReadaheadReaderFallsBackOnlyWhenPopulateIsUnsupported(t *testing.T) {
	buf := make([]byte, 64)
	var populates, preads int
	var popErr []error
	rd := &inkling2ReadaheadReader{
		populate: func([]byte) error {
			populates++
			if len(popErr) == 0 {
				return nil
			}
			err := popErr[0]
			popErr = popErr[1:]
			return err
		},
		pread: func([]byte) bool { preads++; return true },
		page:  16,
	}
	popErr = []error{syscall.EINTR, syscall.EINTR}
	rd.read(buf)
	if populates != 3 || preads != 0 {
		t.Fatalf("EINTR: %d populates %d preads, want a retry to success", populates, preads)
	}
	popErr = []error{syscall.ENOMEM}
	rd.read(buf)
	if preads != 0 || rd.noAdvice {
		t.Fatal("a transient populate failure switched to pread")
	}
	popErr = []error{syscall.EINVAL}
	rd.read(buf)
	rd.read(buf)
	if populates != 5 || preads != 2 || !rd.noAdvice {
		t.Fatalf("EINVAL: %d populates %d preads, want pread from then on", populates, preads)
	}
	// No pread source: the page-touch walk still reads the span.
	walk := &inkling2ReadaheadReader{populate: func([]byte) error { return syscall.ENOSYS }, pread: func([]byte) bool { return false }}
	walk.read(buf)
	walk.read(nil)
	if !walk.noAdvice {
		t.Fatal("ENOSYS did not select the fallback")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInkling2ReadaheadFilesPreadTheSpanOffset(t *testing.T) {
	data := make([]byte, 4096)
	for i := range data {
		data[i] = byte(i * 7)
	}
	path := filepath.Join(t.TempDir(), "shard.gguf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	mapped := append([]byte(nil), data...)
	f := newInkling2ReadaheadFiles([]inkling2MappedShard{{data: make([]byte, 8)}, {data: mapped, path: path}})
	defer f.close()
	if !f.read(mapped[100:1000]) {
		t.Fatal("span inside the shard was not read")
	}
	if !bytes.Equal(f.scratch[:900], data[100:1000]) {
		t.Fatal("pread read the wrong file offset")
	}
	if f.read(make([]byte, 16)) {
		t.Fatal("span outside every shard reported read")
	}
	f.close()
	if f.files[1] != nil {
		t.Fatal("close left the shard file open")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInkling2UploadPopulateReadsAMappedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.bin")
	if err := os.WriteFile(path, make([]byte, 3<<16), 0o644); err != nil {
		t.Fatal(err)
	}
	data, closer, ok, err := model.MmapOpen(path)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Skip("no mmap on this host")
	}
	defer closer.Close()
	// A mid-page start exercises the round-down to the page boundary.
	err = inkling2UploadPopulate(data[100 : 2<<16])
	if !inkling2UploadAdviseOn {
		if !inkling2PopulateUnsupported(err) {
			t.Fatalf("populate without the Linux path = %v, want unsupported", err)
		}
		return
	}
	if err != nil && !inkling2PopulateUnsupported(err) {
		t.Fatalf("populate: %v", err)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2MappedRegionsSkipHeaderAndTokenEmbedding(t *testing.T) {
	data := make([]byte, 100)
	f := &File{TensorDataOffset: 16, Tensors: []TensorInfo{
		{Name: "blk.0.attn_q.weight", FileOffset: 16},
		{Name: "token_embd.weight", FileOffset: 32},
		{Name: "output.weight", FileOffset: 60},
	}}
	regions := inkling2MappedRegions([]inkling2MappedShard{{data: data}}, f)
	if len(regions) != 2 || inkling2Offset(data, regions[0]) != 16 || len(regions[0]) != 16 ||
		inkling2Offset(data, regions[1]) != 60 || len(regions[1]) != 40 {
		t.Fatalf("regions %d, want [16:32] and [60:100]", len(regions))
	}
	// A trailing embedding leaves only the front.
	f.Tensors[1].FileOffset, f.Tensors[2].FileOffset = 80, 40
	if regions = inkling2MappedRegions([]inkling2MappedShard{{data: data}}, f); len(regions) != 1 || len(regions[0]) != 64 {
		t.Fatalf("trailing embedding: %d regions", len(regions))
	}
	// Split shards are walked whole, in address order.
	both := make([]byte, 64)
	regions = inkling2MappedRegions([]inkling2MappedShard{{data: both[32:]}, {data: both[:32]}, {}}, nil)
	if len(regions) != 2 || inkling2Offset(both, regions[0]) != 0 || inkling2Offset(both, regions[1]) != 32 {
		t.Fatal("split shards not walked whole in address order")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2UploadReadaheadLeadTracksAvailableRAM(t *testing.T) {
	env := func(kv map[string]string) func(string) (string, bool) {
		return func(k string) (string, bool) { v, ok := kv[k]; return v, ok }
	}
	probe := func(avail int64, known bool) func() (int64, int64, bool) {
		return func() (int64, int64, bool) { return 128 << 30, avail, known }
	}
	cuda := inkling2TierBackend{name: "cuda", tier: "sm_120"}
	on := func(v int64) int64 {
		if inkling2UploadAdviseOn {
			return v
		}
		return 0
	}
	cases := []struct {
		be    compute.Backend
		kv    map[string]string
		avail int64
		known bool
		want  int64
	}{
		{cuda, nil, 64 << 30, true, on(inkling2UploadReadaheadDefault)},
		{cuda, nil, 2 << 30, true, on(512 << 20)},
		{cuda, nil, 64 << 20, true, on(inkling2UploadReadaheadChunk)},
		{cuda, nil, 0, false, on(inkling2UploadReadaheadUnknownRAM)},
		{cuda, nil, compute.FreeUnknown, true, on(inkling2UploadReadaheadUnknownRAM)},
		{cuda, map[string]string{"FAK_INKLING2_UPLOAD_READAHEAD_MB": "8192"}, 2 << 30, true, on(8192 << 20)},
		{cuda, map[string]string{"FAK_INKLING2_UPLOAD_READAHEAD_MB": "0"}, 64 << 30, true, 0},
		{inkling2TierBackend{name: "vulkan"}, nil, 64 << 30, true, 0},
	}
	for i, c := range cases {
		if got := inkling2UploadReadaheadLead(c.be, env(c.kv), probe(c.avail, c.known)); got != c.want {
			t.Fatalf("case %d: lead %d, want %d", i, got, c.want)
		}
	}
	// Disabling the reader keeps the mapping-order sort.
	if inkling2UploadReadaheadOn(cuda) != inkling2UploadAdviseOn || inkling2UploadReadaheadOn(nil) {
		t.Fatal("mapping-order sort gate")
	}
}

type inkling2MappingToucher struct {
	data   []byte
	closed int
}

// Close reads the mapping, which faults if the shard was already unmapped.
func (c *inkling2MappingToucher) Close() error {
	c.closed++
	_ = c.data[len(c.data)-1]
	return nil
}

// fak-test:runtime fast est=50ms lane=default
func TestInkling2OpenStartsTheReaderAtMapAndClosesItBeforeUnmap(t *testing.T) {
	dir := t.TempDir()
	open := func(arch string) (*Inkling2LoopModel, *inkling2UploadReadahead, *inkling2MappingToucher, *atomic.Int32, error) {
		path := writeInk2sFile(t, dir, arch+".gguf", buildInk2sGGUF(ink2sOpts{arch: arch, floor: 8192}, ink2sTensors()))
		var touch *inkling2MappingToucher
		var ra *inkling2UploadReadahead
		reads := new(atomic.Int32)
		m, err := openInkling2Loop(path, func(shards []inkling2MappedShard, f *File) io.Closer {
			if len(shards) != 1 || shards[0].path != path || f == nil {
				t.Errorf("hook got %d shards, header %v", len(shards), f != nil)
			}
			ra = startInkling2MappedReadahead(inkling2MappedRegions(shards, f), 1<<20, 64, func([]byte) { reads.Add(1) }, nil)
			touch = &inkling2MappingToucher{data: shards[0].data}
			return inkling2Closers{ra, touch}
		})
		return m, ra, touch, reads, err
	}
	m, ra, touch, reads, err := open("inkling2")
	if err != nil {
		t.Fatal(err)
	}
	for deadline := time.Now().Add(5 * time.Second); reads.Load() == 0; time.Sleep(time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("the map-time reader never read")
		}
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	ra.done.Wait()
	if touch.closed != 1 {
		t.Fatalf("closer before the unmap ran %d times", touch.closed)
	}
	// A bind that fails after the map still stops the reader before the unmap.
	if _, ra, touch, _, err = open("bogus"); err == nil {
		t.Fatal("bogus architecture bound")
	}
	ra.done.Wait()
	if touch.closed != 1 {
		t.Fatalf("failed bind ran the pre-unmap closer %d times", touch.closed)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestInkling2StreamedUploadRetargetsAndClosesTheMapTimeReader(t *testing.T) {
	r := rand.New(rand.NewSource(9))
	quant := func(typ TensorType, rows, cols int) *inkling2QuantMatrix {
		rb, _ := inkling2RowBytes(typ, cols)
		return &inkling2QuantMatrix{name: typ.String(), typ: typ, raw: randomInkling2Raw(t, r, typ, rows, cols), rows: rows, cols: cols, rowBytes: rb}
	}
	out, wq := quant(TensorQ8_0, 5, 64), quant(TensorQ8_0, 3, 64)
	m := &Inkling2LoopModel{Weights: &model.Inkling2LoopWeights{
		Output: out,
		Layers: []model.Inkling2LoopLayerWeights{{Wq: wq}},
	}}
	var reads atomic.Int32
	// A zero window holds the walk until the upload cursor moves.
	ra := startInkling2MappedReadahead([][]byte{out.raw}, 0, 8, func([]byte) { reads.Add(1) }, nil)
	s := &inkling2DeviceSet{be: compute.Default()}
	defer s.Close()
	if err := s.uploadStreamed(m, nil, true, 0, ra); err != nil {
		t.Fatal(err)
	}
	ra.done.Wait()
	if !ra.final || len(ra.chunks.spans) != 2 {
		t.Fatalf("reader not retargeted onto the two uploads (final %v, %d spans)", ra.final, len(ra.chunks.spans))
	}
	if _, ok := m.Weights.Output.(*inkling2DeviceMatrix); !ok {
		t.Fatalf("output stayed %T", m.Weights.Output)
	}
	if len(s.tensors) != 2 {
		t.Fatalf("set owns %d tensors, want 2", len(s.tensors))
	}
}
