package compute

// Batched KV block transport between a pinned host slab and device memory: the
// CPU reference of kv_swap_batch.cu.
//
// DIRECT-PORT of vllm-project/vllm vllm/v1/kv_offload/cpu/swap_blocks_triton.py
// at pin ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0).
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// This file is a modified derivative of the Triton _swap_blocks_kernel and its
// swap_blocks_batch dispatcher.
//
// The file is pure Go on purpose: it compiles and is tested in every build,
// while the device kernel ships behind the cuda build tag once a consumer wires
// kv_swap_batch.cu into the single libfakcuda translation unit. Declaring the C
// ABI here would break every -tags cuda link against a libfakcuda.a that does
// not carry the symbols yet.
//
// What is kept from upstream, entry for entry:
//   - the tuned crossover constants NUM_SMS, THRESHOLD_BYTES and MIN_N;
//   - the launch grid min(NUM_SMS, n) and the persistent job stride
//     (job = pid; job += num_progs);
//   - the masked 8-byte word body walked in BYTES_PER_CHUNK windows.
//
// Deliberate divergences:
//   - upstream moves size // 8 words and drops the remainder; the size % 8
//     trailing bytes are copied here, so any block length equals a plain copy;
//   - THRESHOLD_BYTES, which upstream leaves to its caller, is part of the
//     route decision next to MIN_N;
//   - addresses are resolved through an explicit KVSwapSpace instead of raw
//     pointers, so the reference never dereferences an integer and the
//     host-VA/device-VA equivalence the kernel depends on is a checked input.

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"sort"
)

const (
	// KVSwapNumSMs is upstream NUM_SMS: the most programs one launch spreads over.
	KVSwapNumSMs = 12
	// KVSwapThresholdBytes is upstream THRESHOLD_BYTES: the largest block for
	// which the kernel is expected to beat plain async copies.
	KVSwapThresholdBytes = 28 * 1024
	// KVSwapMinN is upstream MIN_N: the smallest batch that amortizes a launch.
	KVSwapMinN = 16
	// KVSwapDefaultBytesPerChunk is the chunk window used when a caller does
	// not choose one. Upstream has no default; this matches the device CTA
	// width of 256 words.
	KVSwapDefaultBytesPerChunk = 256 * 8

	// KVSwapBatchEnv opts a process into the kernel route. Unset or any value
	// other than "1" keeps every batch on the DMA route.
	KVSwapBatchEnv = "FAK_KV_SWAP_BATCH"
)

var (
	// ErrKVSwapGeometry refuses malformed descriptors, chunk sizes or regions.
	ErrKVSwapGeometry = errors.New("compute: kv swap batch geometry refused")
	// ErrKVSwapOverlap refuses a batch whose destinations overlap each other or
	// any source; concurrent programs would make its result order-dependent.
	ErrKVSwapOverlap = errors.New("compute: kv swap batch blocks overlap")
	// ErrKVSwapUnmapped refuses an address range no region of the space covers.
	ErrKVSwapUnmapped = errors.New("compute: kv swap batch address not mapped")
	// ErrKVSwapUVAMismatch refuses the kernel route when a pinned slab's device
	// VA differs from its host VA.
	ErrKVSwapUVAMismatch = errors.New("compute: kv swap batch pinned slab is not UVA-equivalent")
)

// KVSwapRoute names the transport a batch takes.
type KVSwapRoute int

const (
	// KVSwapRouteDMA is one plain copy per block, in descriptor order.
	KVSwapRouteDMA KVSwapRoute = iota
	// KVSwapRouteKernel is one persistent-CTA launch over the whole batch.
	KVSwapRouteKernel
)

func (r KVSwapRoute) String() string {
	if r == KVSwapRouteKernel {
		return "kernel"
	}
	return "dma"
}

// KVSwapBlock is one block move, addressed the way the host sees memory: a
// pinned-slab endpoint carries its host VA, a device endpoint its device VA.
type KVSwapBlock struct {
	Src  uint64
	Dst  uint64
	Size int64
}

// KVSwapDescriptors is the structure-of-arrays form the kernel consumes:
// upstream's src_addrs, dst_addrs and sizes tensors.
type KVSwapDescriptors struct {
	SrcAddrs []int64
	DstAddrs []int64
	Sizes    []int64
}

// Len is the number of jobs.
func (d *KVSwapDescriptors) Len() int { return len(d.Sizes) }

// WordAligned reports whether every non-empty job starts on an 8-byte boundary
// at both ends. The device kernel loads 64-bit words, which must be naturally
// aligned there; a batch that is not goes to DMA.
func (d *KVSwapDescriptors) WordAligned() bool {
	for j, size := range d.Sizes {
		if size > 0 && (d.SrcAddrs[j]|d.DstAddrs[j])&7 != 0 {
			return false
		}
	}
	return true
}

type kvSwapSpan struct {
	lo, hi uint64 // [lo, hi)
	dst    bool
}

// BuildKVSwapDescriptors validates blocks and lays them out as descriptor
// arrays. It refuses negative sizes, ranges that wrap the address space or do
// not fit the signed descriptor ABI, and any destination that overlaps another
// destination or any source.
func BuildKVSwapDescriptors(blocks []KVSwapBlock) (*KVSwapDescriptors, error) {
	const maxInt32 = 1<<31 - 1
	const maxAddr = uint64(1<<63 - 1)
	if len(blocks) > maxInt32 {
		return nil, fmt.Errorf("%w: %d blocks overflow the int32 ABI", ErrKVSwapGeometry, len(blocks))
	}
	d := &KVSwapDescriptors{
		SrcAddrs: make([]int64, len(blocks)),
		DstAddrs: make([]int64, len(blocks)),
		Sizes:    make([]int64, len(blocks)),
	}
	spans := make([]kvSwapSpan, 0, 2*len(blocks))
	for j, b := range blocks {
		if b.Size < 0 {
			return nil, fmt.Errorf("%w: block %d has size %d", ErrKVSwapGeometry, j, b.Size)
		}
		size := uint64(b.Size)
		if b.Src > maxAddr-size || b.Dst > maxAddr-size {
			return nil, fmt.Errorf("%w: block %d range leaves the signed address ABI", ErrKVSwapGeometry, j)
		}
		d.SrcAddrs[j] = int64(b.Src)
		d.DstAddrs[j] = int64(b.Dst)
		d.Sizes[j] = b.Size
		if size == 0 {
			continue
		}
		spans = append(spans,
			kvSwapSpan{lo: b.Src, hi: b.Src + size},
			kvSwapSpan{lo: b.Dst, hi: b.Dst + size, dst: true})
	}
	sort.Slice(spans, func(a, b int) bool { return spans[a].lo < spans[b].lo })
	// Sweep in address order: an overlap is harmless only between two sources.
	var reach uint64
	reachHasDst := false
	for i, s := range spans {
		if i > 0 && s.lo < reach && (s.dst || reachHasDst) {
			return nil, fmt.Errorf("%w: range [%#x,%#x) collides with an earlier block", ErrKVSwapOverlap, s.lo, s.hi)
		}
		if i == 0 || s.lo >= reach {
			reach, reachHasDst = s.hi, s.dst
			continue
		}
		if s.hi > reach {
			reach = s.hi
		}
	}
	return d, nil
}

// KVSwapBatchRoute is the kernel-vs-DMA crossover: fewer than KVSwapMinN
// descriptors, or any block above KVSwapThresholdBytes, routes to DMA.
func KVSwapBatchRoute(sizes []int64) KVSwapRoute {
	if len(sizes) < KVSwapMinN {
		return KVSwapRouteDMA
	}
	for _, s := range sizes {
		if s > KVSwapThresholdBytes {
			return KVSwapRouteDMA
		}
	}
	return KVSwapRouteKernel
}

// KVSwapBatchEnabled reports whether the process opted into the kernel route.
func KVSwapBatchEnabled() bool { return os.Getenv(KVSwapBatchEnv) == "1" }

// KVSwapPrograms is the launch grid for n jobs: upstream min(NUM_SMS, n).
func KVSwapPrograms(n int) int {
	if n < KVSwapNumSMs {
		return n
	}
	return KVSwapNumSMs
}

type kvSwapRegion struct {
	hostVA   uint64
	deviceVA uint64
	mem      []byte
	pinned   bool
}

// KVSwapSpace is the address space a batch runs in: pinned host slabs, each
// visible at a host VA and a device VA, and device regions visible only at a
// device VA. The host builds descriptors and performs DMA with host-side
// addresses; the kernel dereferences the same integers as device addresses.
type KVSwapSpace struct {
	regions []kvSwapRegion
}

// AddPinnedSlab registers a pinned host slab. deviceVA is the address the
// device maps it at; under unified addressing it equals hostVA.
func (s *KVSwapSpace) AddPinnedSlab(hostVA, deviceVA uint64, mem []byte) error {
	return s.add(kvSwapRegion{hostVA: hostVA, deviceVA: deviceVA, mem: mem, pinned: true})
}

// AddDeviceRegion registers device memory at va.
func (s *KVSwapSpace) AddDeviceRegion(va uint64, mem []byte) error {
	return s.add(kvSwapRegion{hostVA: va, deviceVA: va, mem: mem})
}

func (s *KVSwapSpace) add(r kvSwapRegion) error {
	n := uint64(len(r.mem))
	const maxAddr = uint64(1<<63 - 1)
	if n == 0 || r.hostVA == 0 || r.deviceVA == 0 || r.hostVA > maxAddr-n || r.deviceVA > maxAddr-n {
		return fmt.Errorf("%w: region host=%#x device=%#x len=%d", ErrKVSwapGeometry, r.hostVA, r.deviceVA, n)
	}
	for _, o := range s.regions {
		on := uint64(len(o.mem))
		if r.hostVA < o.hostVA+on && o.hostVA < r.hostVA+n ||
			r.deviceVA < o.deviceVA+on && o.deviceVA < r.deviceVA+n {
			return fmt.Errorf("%w: region host=%#x device=%#x overlaps a registered region",
				ErrKVSwapGeometry, r.hostVA, r.deviceVA)
		}
	}
	s.regions = append(s.regions, r)
	return nil
}

// UVAEquivalent reports whether every pinned slab is mapped on the device at
// its host VA, the precondition for the kernel to follow host-built addresses.
func (s *KVSwapSpace) UVAEquivalent() bool {
	for _, r := range s.regions {
		if r.pinned && r.hostVA != r.deviceVA {
			return false
		}
	}
	return true
}

// resolve returns the bytes at [addr, addr+size) as seen from the host side or
// the device side.
func (s *KVSwapSpace) resolve(addr uint64, size int64, deviceSide bool) ([]byte, error) {
	if size == 0 {
		return nil, nil
	}
	for _, r := range s.regions {
		base := r.hostVA
		if deviceSide {
			base = r.deviceVA
		}
		if addr >= base && addr-base <= uint64(len(r.mem)) && uint64(size) <= uint64(len(r.mem))-(addr-base) {
			off := addr - base
			return r.mem[off : off+uint64(size) : off+uint64(size)], nil
		}
	}
	side := "host"
	if deviceSide {
		side = "device"
	}
	return nil, fmt.Errorf("%w: [%#x,+%d) on the %s side", ErrKVSwapUnmapped, addr, size, side)
}

// kvSwapResolveAll resolves every job's endpoints before any byte moves, so a
// refused batch leaves memory untouched.
func kvSwapResolveAll(s *KVSwapSpace, d *KVSwapDescriptors, deviceSide bool) (src, dst [][]byte, err error) {
	n := d.Len()
	if len(d.SrcAddrs) != n || len(d.DstAddrs) != n {
		return nil, nil, fmt.Errorf("%w: descriptor arrays disagree on length", ErrKVSwapGeometry)
	}
	src = make([][]byte, n)
	dst = make([][]byte, n)
	for j := 0; j < n; j++ {
		if d.Sizes[j] < 0 {
			return nil, nil, fmt.Errorf("%w: job %d has size %d", ErrKVSwapGeometry, j, d.Sizes[j])
		}
		if src[j], err = s.resolve(uint64(d.SrcAddrs[j]), d.Sizes[j], deviceSide); err != nil {
			return nil, nil, fmt.Errorf("job %d source: %w", j, err)
		}
		if dst[j], err = s.resolve(uint64(d.DstAddrs[j]), d.Sizes[j], deviceSide); err != nil {
			return nil, nil, fmt.Errorf("job %d destination: %w", j, err)
		}
	}
	return src, dst, nil
}

// KVSwapExecDMA is the reference of the DMA route: one copy per block in
// descriptor order, endpoints resolved with host-side addresses.
func KVSwapExecDMA(s *KVSwapSpace, d *KVSwapDescriptors) error {
	src, dst, err := kvSwapResolveAll(s, d, false)
	if err != nil {
		return err
	}
	for j := range src {
		copy(dst[j], src[j])
	}
	return nil
}

// KVSwapExecKernel is the reference of k_ksb_swap_blocks: min(NUM_SMS, n)
// programs each stride over the jobs, moving every block as 8-byte words in
// bytesPerChunk windows followed by its byte tail. Endpoints are resolved as
// the device sees them. It returns how many jobs each program handled.
func KVSwapExecKernel(s *KVSwapSpace, d *KVSwapDescriptors, bytesPerChunk int) ([]int, error) {
	if bytesPerChunk < 8 || bytesPerChunk%8 != 0 {
		return nil, fmt.Errorf("%w: bytesPerChunk=%d must be a positive multiple of 8", ErrKVSwapGeometry, bytesPerChunk)
	}
	if !s.UVAEquivalent() {
		return nil, ErrKVSwapUVAMismatch
	}
	src, dst, err := kvSwapResolveAll(s, d, true)
	if err != nil {
		return nil, err
	}
	nJobs := d.Len()
	numProgs := KVSwapPrograms(nJobs)
	wordsPerChunk := bytesPerChunk / 8
	jobsPerProg := make([]int, numProgs)
	for pid := 0; pid < numProgs; pid++ {
		for job := pid; job < nJobs; job += numProgs {
			words := int(d.Sizes[job] / 8)
			for start := 0; start < words; start += wordsPerChunk {
				for off := 0; off < wordsPerChunk; off++ {
					idx := start + off
					if idx >= words { // upstream mask = idx < words
						break
					}
					w := binary.LittleEndian.Uint64(src[job][idx*8:])
					binary.LittleEndian.PutUint64(dst[job][idx*8:], w)
				}
			}
			for b := words * 8; b < int(d.Sizes[job]); b++ {
				dst[job][b] = src[job][b]
			}
			jobsPerProg[pid]++
		}
	}
	return jobsPerProg, nil
}

// KVSwapBatchOptions tunes one KVSwapBatch call.
type KVSwapBatchOptions struct {
	// KernelRoute admits the kernel route for batches past the crossover. It is
	// false by default; callers normally pass KVSwapBatchEnabled().
	KernelRoute bool
	// BytesPerChunk is the kernel's chunk window; zero selects
	// KVSwapDefaultBytesPerChunk.
	BytesPerChunk int
}

// KVSwapBatchResult reports how one batch was moved.
type KVSwapBatchResult struct {
	Route KVSwapRoute
	// Programs is the launch grid on the kernel route, zero on DMA.
	Programs int
	// Copies is the number of non-empty per-block copies on the DMA route, zero
	// on kernel.
	Copies int
}

// KVSwapBatch moves blocks through space on the route the crossover picks. The
// kernel route is taken only when opts admits it, the batch clears MIN_N and
// THRESHOLD_BYTES, every endpoint is word-aligned and every pinned slab is
// UVA-equivalent; anything else takes DMA, which produces the same bytes.
func KVSwapBatch(space *KVSwapSpace, blocks []KVSwapBlock, opts KVSwapBatchOptions) (KVSwapBatchResult, error) {
	d, err := BuildKVSwapDescriptors(blocks)
	if err != nil {
		return KVSwapBatchResult{}, err
	}
	if opts.KernelRoute && KVSwapBatchRoute(d.Sizes) == KVSwapRouteKernel && d.WordAligned() && space.UVAEquivalent() {
		chunk := opts.BytesPerChunk
		if chunk == 0 {
			chunk = KVSwapDefaultBytesPerChunk
		}
		perProg, err := KVSwapExecKernel(space, d, chunk)
		if err != nil {
			return KVSwapBatchResult{}, err
		}
		return KVSwapBatchResult{Route: KVSwapRouteKernel, Programs: len(perProg)}, nil
	}
	if err := KVSwapExecDMA(space, d); err != nil {
		return KVSwapBatchResult{}, err
	}
	copies := 0
	for _, size := range d.Sizes {
		if size > 0 {
			copies++
		}
	}
	return KVSwapBatchResult{Route: KVSwapRouteDMA, Copies: copies}, nil
}
