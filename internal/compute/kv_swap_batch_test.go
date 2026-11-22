package compute

// CPU witnesses for the batched KV block transport (kv_swap_batch.go, device
// twin kv_swap_batch.cu). The expected memory images come from an oracle that
// never touches the code under test: a sequential byte-by-byte copy over the
// test's own offset table, plus hand-computed route and stride values taken
// from the pinned vLLM swap_blocks_batch constants (NUM_SMS=12,
// THRESHOLD_BYTES=28*1024, MIN_N=16).

import (
	"bytes"
	"errors"
	"reflect"
	"testing"
)

const (
	ksbHostVA   = uint64(0x7f00_0010_0000)
	ksbDeviceVA = uint64(0x0002_0000_0000)
	ksbRegion   = 2 << 20
)

// ksbMove is one block in offset form: the oracle indexes the byte images with
// it directly, the port sees it only as addresses.
type ksbMove struct {
	toDevice bool
	srcOff   int
	dstOff   int
	size     int
}

type ksbFixture struct {
	host, dev             []byte
	wantHost, wantDev     []byte
	slabHostVA, slabDevVA uint64
	space                 *KVSwapSpace
}

func ksbFill(buf []byte, seed uint32) {
	x := seed
	for i := range buf {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		buf[i] = byte(x >> 11)
	}
}

func newKsbFixture(t *testing.T, slabDeviceVA uint64) *ksbFixture {
	t.Helper()
	f := &ksbFixture{
		host:       make([]byte, ksbRegion),
		dev:        make([]byte, ksbRegion),
		slabHostVA: ksbHostVA,
		slabDevVA:  slabDeviceVA,
		space:      &KVSwapSpace{},
	}
	ksbFill(f.host, 0x9e3779b9)
	ksbFill(f.dev, 0x85ebca6b)
	f.wantHost = bytes.Clone(f.host)
	f.wantDev = bytes.Clone(f.dev)
	if err := f.space.AddPinnedSlab(f.slabHostVA, f.slabDevVA, f.host); err != nil {
		t.Fatalf("AddPinnedSlab: %v", err)
	}
	if err := f.space.AddDeviceRegion(ksbDeviceVA, f.dev); err != nil {
		t.Fatalf("AddDeviceRegion: %v", err)
	}
	return f
}

// oracle applies the moves one after another, a byte at a time, to the
// expected images.
func (f *ksbFixture) oracle(moves []ksbMove) {
	for _, m := range moves {
		src, dst := f.wantDev, f.wantHost
		if m.toDevice {
			src, dst = f.wantHost, f.wantDev
		}
		for i := 0; i < m.size; i++ {
			dst[m.dstOff+i] = src[m.srcOff+i]
		}
	}
}

func (f *ksbFixture) blocks(moves []ksbMove) []KVSwapBlock {
	out := make([]KVSwapBlock, len(moves))
	for i, m := range moves {
		if m.toDevice {
			out[i] = KVSwapBlock{Src: f.slabHostVA + uint64(m.srcOff), Dst: ksbDeviceVA + uint64(m.dstOff), Size: int64(m.size)}
		} else {
			out[i] = KVSwapBlock{Src: ksbDeviceVA + uint64(m.srcOff), Dst: f.slabHostVA + uint64(m.dstOff), Size: int64(m.size)}
		}
	}
	return out
}

func (f *ksbFixture) assertImages(t *testing.T) {
	t.Helper()
	if !bytes.Equal(f.host, f.wantHost) {
		t.Fatalf("host slab differs from the sequential-copy oracle at byte %d", ksbFirstDiff(f.host, f.wantHost))
	}
	if !bytes.Equal(f.dev, f.wantDev) {
		t.Fatalf("device region differs from the sequential-copy oracle at byte %d", ksbFirstDiff(f.dev, f.wantDev))
	}
}

func ksbFirstDiff(a, b []byte) int {
	for i := range a {
		if a[i] != b[i] {
			return i
		}
	}
	return -1
}

// ksbLayout places one move per size. Sources fill the lower half of their
// region and destinations the upper half, each on an 8-byte boundary with a
// guard gap, alternating direction so both regions are read and written.
func ksbLayout(sizes []int) []ksbMove {
	moves := make([]ksbMove, len(sizes))
	srcOff, dstOff := 64, ksbRegion/2+64
	for i, size := range sizes {
		moves[i] = ksbMove{toDevice: i%2 == 0, srcOff: srcOff, dstOff: dstOff, size: size}
		step := (size + 7 + 24) &^ 7
		srcOff += step
		dstOff += step
	}
	return moves
}

// 40 blocks: empty, sub-word, exact-word, word-plus-tail, and blocks at and
// just under the 28 KiB threshold.
var ksbMixedSizes = []int{
	0, 1, 7, 8, 9, 15, 16, 17, 23, 24,
	31, 33, 63, 64, 65, 127, 129, 255, 256, 257,
	1000, 1023, 1024, 1025, 2047, 2049, 4095, 4096, 4097, 5003,
	8191, 8192, 8193, 12345, 16384, 16389, 28671, 28672, 28665, 3,
}

func TestSwapBlocksBatchMatchesPerBlockCopy(t *testing.T) {
	t.Run("upstream tuned constants", func(t *testing.T) {
		if KVSwapNumSMs != 12 || KVSwapThresholdBytes != 28672 || KVSwapMinN != 16 {
			t.Fatalf("constants NUM_SMS/THRESHOLD_BYTES/MIN_N = %d/%d/%d, want 12/28672/16",
				KVSwapNumSMs, KVSwapThresholdBytes, KVSwapMinN)
		}
	})

	t.Run("mixed sizes with unaligned tails equal the sequential copy", func(t *testing.T) {
		for _, chunk := range []int{8, 24, 64, 2048, 1 << 20} {
			f := newKsbFixture(t, ksbHostVA)
			moves := ksbLayout(ksbMixedSizes)
			f.oracle(moves)
			got, err := KVSwapBatch(f.space, f.blocks(moves), KVSwapBatchOptions{KernelRoute: true, BytesPerChunk: chunk})
			if err != nil {
				t.Fatalf("chunk %d: KVSwapBatch: %v", chunk, err)
			}
			if want := (KVSwapBatchResult{Route: KVSwapRouteKernel, Programs: 12}); got != want {
				t.Fatalf("chunk %d: result %+v, want %+v", chunk, got, want)
			}
			f.assertImages(t)
		}
	})

	t.Run("programs stride over every job exactly once", func(t *testing.T) {
		f := newKsbFixture(t, ksbHostVA)
		moves := ksbLayout(ksbMixedSizes)
		f.oracle(moves)
		d, err := BuildKVSwapDescriptors(f.blocks(moves))
		if err != nil {
			t.Fatalf("BuildKVSwapDescriptors: %v", err)
		}
		perProg, err := KVSwapExecKernel(f.space, d, 64)
		if err != nil {
			t.Fatalf("KVSwapExecKernel: %v", err)
		}
		// 40 jobs over 12 programs: pid p owns p, p+12, p+24, p+36, so pids 0..3
		// take four jobs and pids 4..11 take three.
		want := []int{4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3}
		if !reflect.DeepEqual(perProg, want) {
			t.Fatalf("jobs per program %v, want %v", perProg, want)
		}
		f.assertImages(t)

		// Fewer jobs than NUM_SMS launch one program per job.
		f = newKsbFixture(t, ksbHostVA)
		moves = ksbLayout([]int{9, 4096, 0, 17, 1023})
		f.oracle(moves)
		if d, err = BuildKVSwapDescriptors(f.blocks(moves)); err != nil {
			t.Fatalf("BuildKVSwapDescriptors: %v", err)
		}
		if perProg, err = KVSwapExecKernel(f.space, d, 2048); err != nil {
			t.Fatalf("KVSwapExecKernel: %v", err)
		}
		if want := []int{1, 1, 1, 1, 1}; !reflect.DeepEqual(perProg, want) {
			t.Fatalf("jobs per program %v, want %v", perProg, want)
		}
		f.assertImages(t)
	})

	t.Run("crossover routes small or oversized batches to DMA with identical bytes", func(t *testing.T) {
		sizes := func(n, size int) []int {
			out := make([]int, n)
			for i := range out {
				out[i] = size + i%5 // tails 0..4 bytes past a word boundary
			}
			return out
		}
		over := sizes(16, 512)
		over[9] = 28*1024 + 1
		atLimit := sizes(16, 512)
		atLimit[9] = 28 * 1024
		cases := []struct {
			name   string
			sizes  []int
			kernel bool
			want   KVSwapBatchResult
		}{
			{"n=15 below MIN_N", sizes(15, 512), true, KVSwapBatchResult{Route: KVSwapRouteDMA, Copies: 15}},
			{"n=16 at MIN_N", sizes(16, 512), true, KVSwapBatchResult{Route: KVSwapRouteKernel, Programs: 12}},
			{"one block above THRESHOLD_BYTES", over, true, KVSwapBatchResult{Route: KVSwapRouteDMA, Copies: 16}},
			{"one block at THRESHOLD_BYTES", atLimit, true, KVSwapBatchResult{Route: KVSwapRouteKernel, Programs: 12}},
			{"gate off keeps a large batch on DMA", ksbMixedSizes, false, KVSwapBatchResult{Route: KVSwapRouteDMA, Copies: 39}},
			{"empty batch", nil, true, KVSwapBatchResult{Route: KVSwapRouteDMA}},
		}
		for _, tc := range cases {
			f := newKsbFixture(t, ksbHostVA)
			moves := ksbLayout(tc.sizes)
			f.oracle(moves)
			got, err := KVSwapBatch(f.space, f.blocks(moves), KVSwapBatchOptions{KernelRoute: tc.kernel})
			if err != nil {
				t.Fatalf("%s: KVSwapBatch: %v", tc.name, err)
			}
			if got != tc.want {
				t.Fatalf("%s: result %+v, want %+v", tc.name, got, tc.want)
			}
			f.assertImages(t)
		}

		if got := KVSwapBatchRoute(make([]int64, 15)); got != KVSwapRouteDMA {
			t.Fatalf("route for 15 descriptors = %v, want dma", got)
		}
		if got := KVSwapBatchRoute(make([]int64, 16)); got != KVSwapRouteKernel {
			t.Fatalf("route for 16 descriptors = %v, want kernel", got)
		}
	})

	t.Run("a misaligned endpoint takes DMA with identical bytes", func(t *testing.T) {
		f := newKsbFixture(t, ksbHostVA)
		moves := ksbLayout(ksbMixedSizes)
		moves[5].srcOff += 3
		f.oracle(moves)
		got, err := KVSwapBatch(f.space, f.blocks(moves), KVSwapBatchOptions{KernelRoute: true})
		if err != nil {
			t.Fatalf("KVSwapBatch: %v", err)
		}
		if want := (KVSwapBatchResult{Route: KVSwapRouteDMA, Copies: 39}); got != want {
			t.Fatalf("result %+v, want %+v", got, want)
		}
		f.assertImages(t)
	})

	t.Run("gate reads the opt-in environment variable", func(t *testing.T) {
		t.Setenv(KVSwapBatchEnv, "")
		if KVSwapBatchEnabled() {
			t.Fatal("kernel route enabled with the variable empty")
		}
		t.Setenv(KVSwapBatchEnv, "1")
		if !KVSwapBatchEnabled() {
			t.Fatal("kernel route not enabled by FAK_KV_SWAP_BATCH=1")
		}
	})

	t.Run("pinned slab host VA and device VA stay equivalent across the transfer", func(t *testing.T) {
		f := newKsbFixture(t, ksbHostVA)
		if !f.space.UVAEquivalent() {
			t.Fatal("slab mapped at its host VA is not reported UVA-equivalent")
		}
		moves := ksbLayout(ksbMixedSizes)
		f.oracle(moves)
		blocks := f.blocks(moves)
		if _, err := KVSwapBatch(f.space, blocks, KVSwapBatchOptions{KernelRoute: true}); err != nil {
			t.Fatalf("KVSwapBatch: %v", err)
		}
		f.assertImages(t)
		// Every host-built slab address must name the same bytes on both sides,
		// before and after the kernel wrote through the device view.
		for i, m := range moves {
			if m.size == 0 {
				continue
			}
			addr, off := blocks[i].Dst, m.dstOff
			if m.toDevice {
				addr, off = blocks[i].Src, m.srcOff
			}
			hostView, err := f.space.resolve(addr, int64(m.size), false)
			if err != nil {
				t.Fatalf("block %d host view: %v", i, err)
			}
			devView, err := f.space.resolve(addr, int64(m.size), true)
			if err != nil {
				t.Fatalf("block %d device view: %v", i, err)
			}
			if &hostView[0] != &devView[0] || &hostView[0] != &f.host[off] {
				t.Fatalf("block %d: host VA %#x resolves to different storage per side", i, addr)
			}
			if !bytes.Equal(devView, f.wantHost[off:off+m.size]) {
				t.Fatalf("block %d: device view of the slab differs from the oracle", i)
			}
		}
	})

	t.Run("a slab mapped at a different device VA never reaches the kernel", func(t *testing.T) {
		f := newKsbFixture(t, ksbHostVA+0x4000_0000)
		if f.space.UVAEquivalent() {
			t.Fatal("slab with a distinct device VA reported UVA-equivalent")
		}
		moves := ksbLayout(ksbMixedSizes)
		blocks := f.blocks(moves)
		d, err := BuildKVSwapDescriptors(blocks)
		if err != nil {
			t.Fatalf("BuildKVSwapDescriptors: %v", err)
		}
		if _, err := KVSwapExecKernel(f.space, d, 64); !errors.Is(err, ErrKVSwapUVAMismatch) {
			t.Fatalf("KVSwapExecKernel error = %v, want ErrKVSwapUVAMismatch", err)
		}
		f.assertImages(t) // refusal moved nothing

		f.oracle(moves)
		got, err := KVSwapBatch(f.space, blocks, KVSwapBatchOptions{KernelRoute: true})
		if err != nil {
			t.Fatalf("KVSwapBatch: %v", err)
		}
		if want := (KVSwapBatchResult{Route: KVSwapRouteDMA, Copies: 39}); got != want {
			t.Fatalf("result %+v, want %+v", got, want)
		}
		f.assertImages(t)
	})

	t.Run("descriptor arrays carry the blocks in order", func(t *testing.T) {
		d, err := BuildKVSwapDescriptors([]KVSwapBlock{
			{Src: 0x1000, Dst: 0x9000, Size: 13},
			{Src: 0x2000, Dst: 0xa000, Size: 0},
			{Src: 0x1000, Dst: 0xb000, Size: 64}, // two jobs may share a source
		})
		if err != nil {
			t.Fatalf("BuildKVSwapDescriptors: %v", err)
		}
		want := &KVSwapDescriptors{
			SrcAddrs: []int64{0x1000, 0x2000, 0x1000},
			DstAddrs: []int64{0x9000, 0xa000, 0xb000},
			Sizes:    []int64{13, 0, 64},
		}
		if !reflect.DeepEqual(d, want) {
			t.Fatalf("descriptors %+v, want %+v", d, want)
		}
	})

	t.Run("refusals leave memory untouched", func(t *testing.T) {
		cases := []struct {
			name   string
			blocks []KVSwapBlock
			want   error
		}{
			{"negative size", []KVSwapBlock{{Src: ksbHostVA, Dst: ksbDeviceVA, Size: -1}}, ErrKVSwapGeometry},
			{"destinations overlap", []KVSwapBlock{
				{Src: ksbHostVA, Dst: ksbDeviceVA, Size: 64},
				{Src: ksbHostVA + 128, Dst: ksbDeviceVA + 63, Size: 64},
			}, ErrKVSwapOverlap},
			{"destination overlaps another job's source", []KVSwapBlock{
				{Src: ksbHostVA, Dst: ksbDeviceVA, Size: 64},
				{Src: ksbDeviceVA + 32, Dst: ksbHostVA + 4096, Size: 64},
			}, ErrKVSwapOverlap},
			{"destination overlaps its own source", []KVSwapBlock{
				{Src: ksbHostVA, Dst: ksbHostVA + 8, Size: 64},
			}, ErrKVSwapOverlap},
			{"source outside every region", []KVSwapBlock{
				{Src: 0x10, Dst: ksbDeviceVA, Size: 64},
			}, ErrKVSwapUnmapped},
			{"destination runs off the region end", []KVSwapBlock{
				{Src: ksbHostVA, Dst: ksbDeviceVA + ksbRegion - 8, Size: 64},
			}, ErrKVSwapUnmapped},
		}
		for _, tc := range cases {
			f := newKsbFixture(t, ksbHostVA)
			if _, err := KVSwapBatch(f.space, tc.blocks, KVSwapBatchOptions{KernelRoute: true}); !errors.Is(err, tc.want) {
				t.Fatalf("%s: error = %v, want %v", tc.name, err, tc.want)
			}
			f.assertImages(t)
		}

		f := newKsbFixture(t, ksbHostVA)
		d, err := BuildKVSwapDescriptors(f.blocks(ksbLayout(ksbMixedSizes)))
		if err != nil {
			t.Fatalf("BuildKVSwapDescriptors: %v", err)
		}
		for _, chunk := range []int{0, 4, 12, -8} {
			if _, err := KVSwapExecKernel(f.space, d, chunk); !errors.Is(err, ErrKVSwapGeometry) {
				t.Fatalf("bytesPerChunk %d: error = %v, want ErrKVSwapGeometry", chunk, err)
			}
		}
		f.assertImages(t)
	})
}
