package ggufload

import (
	"errors"
	"os"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	// inkling2UploadReadaheadDefault caps how far the reader runs ahead of the
	// upload cursor: about 1 s of reads at NVMe speed, enough to keep the disk
	// busy across the repack and upload stalls. The chunk list may admit one
	// extra chunk.
	inkling2UploadReadaheadDefault = int64(4) << 30
	// inkling2UploadReadaheadUnknownRAM is the lead when host memory cannot be
	// probed.
	inkling2UploadReadaheadUnknownRAM = int64(1) << 30
	// inkling2UploadReadaheadChunk splits large stacks so the window advances in
	// steps the cursor can track, not whole multi-hundred-MB tensors.
	inkling2UploadReadaheadChunk = 32 << 20
)

// inkling2UploadItem is one weight bindInkling2Device may move to the device:
// exactly one of mat or exps is set.
type inkling2UploadItem struct {
	mat  *model.Inkling2LoopMatrix
	exps *model.Inkling2LoopExperts
}

// raw is the mapped payload the upload reads, nil for a weight that stays on
// the host.
func (it inkling2UploadItem) raw() []byte {
	if it.mat != nil {
		if q, ok := (*it.mat).(*inkling2QuantMatrix); ok && !inkling2MatrixStaysHost(q) {
			return q.raw
		}
		return nil
	}
	if it.exps != nil {
		if q, ok := (*it.exps).(*inkling2QuantExperts); ok {
			return q.raw
		}
	}
	return nil
}

// inkling2MatrixStaysHost keeps small dense float matrices on the host;
// quantized and i-quant matrices always go to the device.
func inkling2MatrixStaysHost(q *inkling2QuantMatrix) bool {
	switch q.typ {
	case TensorF32, TensorF16, TensorBF16:
		return q.rows*q.cols < inkling2HostDenseLimit
	default:
		return false
	}
}

// inkling2UploadItems lists the weights in bind order: output, per-layer
// matrices, resident expert stacks in residency order, then controllers.
func inkling2UploadItems(m *Inkling2LoopModel, resident []bool) []inkling2UploadItem {
	w := m.Weights
	var items []inkling2UploadItem
	mat := func(p *model.Inkling2LoopMatrix) {
		if *p != nil {
			items = append(items, inkling2UploadItem{mat: p})
		}
	}
	exps := func(p *model.Inkling2LoopExperts) {
		if *p != nil {
			items = append(items, inkling2UploadItem{exps: p})
		}
	}
	mat(&w.Output)
	for i := range w.Layers {
		lw := &w.Layers[i]
		for _, p := range []*model.Inkling2LoopMatrix{&lw.Wq, &lw.Wk, &lw.Wv, &lw.Wo, &lw.Gate, &lw.Rel,
			&lw.SinkProj, &lw.Router, &lw.KConvProj, &lw.VConvProj} {
			mat(p)
		}
	}
	for _, i := range inkling2ExpertResidencyOrder(m.HP, len(w.Layers)) {
		if i < len(resident) && resident[i] {
			exps(&w.Layers[i].GateUp)
			exps(&w.Layers[i].Down)
		}
	}
	for i := range w.Controllers {
		cw := &w.Controllers[i]
		for c := range cw.Down {
			mat(&cw.Down[c])
			mat(&cw.Gain[c])
			mat(&cw.Forget[c])
		}
	}
	return items
}

// inkling2SortUploadsByMapping reorders uploads into mapping-address order, so
// a cold load reads each shard front to back instead of seeking between the
// core-first expert stacks. Residency is already decided and every resident
// tensor is uploaded, so the order changes no placement. Host-staying items
// sort last; ties keep bind order.
func inkling2SortUploadsByMapping(items []inkling2UploadItem) {
	key := func(it inkling2UploadItem) (uintptr, bool) {
		r := it.raw()
		if len(r) == 0 {
			return 0, false
		}
		return uintptr(unsafe.Pointer(&r[0])), true
	}
	sort.SliceStable(items, func(a, b int) bool {
		ka, okA := key(items[a])
		kb, okB := key(items[b])
		if okA != okB {
			return okA
		}
		return ka < kb
	})
}

// inkling2ReadaheadChunks hands out upload payload chunks once each, in upload
// order, while they start within window bytes of the upload cursor, so read
// but unconsumed bytes stay under window plus one chunk.
type inkling2ReadaheadChunks struct {
	spans  [][]byte
	starts []int64 // upload-byte offset of each span
	window int64
	next   int
}

func newInkling2ReadaheadChunks(payloads [][]byte, window int64, chunk int) *inkling2ReadaheadChunks {
	p := &inkling2ReadaheadChunks{window: window}
	var at int64
	for _, b := range payloads {
		for len(b) > 0 {
			n := min(len(b), chunk)
			p.spans = append(p.spans, b[:n:n])
			p.starts = append(p.starts, at)
			at += int64(n)
			b = b[n:]
		}
	}
	return p
}

// due returns the spans not yet handed out that start before consumed+window.
func (p *inkling2ReadaheadChunks) due(consumed int64) [][]byte {
	first := p.next
	for p.next < len(p.spans) && p.starts[p.next] < consumed+p.window {
		p.next++
	}
	return p.spans[first:p.next]
}

// take hands out the next due span, nil when none is due yet.
func (p *inkling2ReadaheadChunks) take(consumed int64) []byte {
	if p.next >= len(p.spans) || p.starts[p.next] >= consumed+p.window {
		return nil
	}
	p.next++
	return p.spans[p.next-1]
}

func (p *inkling2ReadaheadChunks) exhausted() bool { return p.next >= len(p.spans) }

// drop removes the spans read reports as already read, keeping every other
// span at its upload offset so the window still measures from the cursor.
func (p *inkling2ReadaheadChunks) drop(read func([]byte) bool) {
	spans, starts := p.spans[:0], p.starts[:0]
	for i, s := range p.spans {
		if !read(s) {
			spans = append(spans, s)
			starts = append(starts, p.starts[i])
		}
	}
	p.spans, p.starts = spans, starts
}

// inkling2AddrRanges is the merged set of address ranges a reader has handed
// out; a reader walks in address order, so it stays a handful of ranges.
type inkling2AddrRanges [][2]uintptr

func inkling2SpanRange(b []byte) [2]uintptr {
	lo := uintptr(unsafe.Pointer(&b[0]))
	return [2]uintptr{lo, lo + uintptr(len(b))}
}

func (r *inkling2AddrRanges) add(b []byte) {
	s := inkling2SpanRange(b)
	if n := len(*r); n > 0 && (*r)[n-1][1] == s[0] {
		(*r)[n-1][1] = s[1]
		return
	}
	*r = append(*r, s)
}

func (r inkling2AddrRanges) covers(b []byte) bool {
	s := inkling2SpanRange(b)
	for _, x := range r {
		if x[0] <= s[0] && s[1] <= x[1] {
			return true
		}
	}
	return false
}

// inkling2UploadReadahead reads the chunk list spans into the page cache from
// a background goroutine as the upload cursor advances; a read blocks on the
// disk, and the upload must not wait on it. A reader started at map time walks
// the mapped file until the bind retargets it onto the upload payloads.
type inkling2UploadReadahead struct {
	read     func([]byte)
	mu       sync.Mutex
	chunks   *inkling2ReadaheadChunks
	final    bool // chunks lists the uploads; running out of them ends the reader
	handed   inkling2AddrRanges
	consumed atomic.Int64
	wake     chan struct{}
	stop     chan struct{}
	stopOnce sync.Once
	done     sync.WaitGroup
	// release runs once the goroutine has exited.
	release func()
}

func startInkling2UploadReadahead(chunks *inkling2ReadaheadChunks, read func([]byte)) *inkling2UploadReadahead {
	return startInkling2Readahead(chunks, true, read, nil)
}

// startInkling2MappedReadahead reads regions in order before the uploads are
// known, up to window bytes, and waits for retarget.
func startInkling2MappedReadahead(regions [][]byte, window int64, chunk int, read func([]byte), release func()) *inkling2UploadReadahead {
	return startInkling2Readahead(newInkling2ReadaheadChunks(regions, window, chunk), false, read, release)
}

func startInkling2Readahead(chunks *inkling2ReadaheadChunks, final bool, read func([]byte), release func()) *inkling2UploadReadahead {
	r := &inkling2UploadReadahead{read: read, chunks: chunks, final: final, release: release,
		wake: make(chan struct{}, 1), stop: make(chan struct{})}
	r.done.Add(1)
	go r.run()
	return r
}

func (r *inkling2UploadReadahead) run() {
	defer r.done.Done()
	if r.release != nil {
		defer r.release()
	}
	for {
		select {
		case <-r.stop:
			return
		default:
		}
		r.mu.Lock()
		s := r.chunks.take(r.consumed.Load())
		if s != nil && !r.final {
			r.handed.add(s)
		}
		end := r.final && r.chunks.exhausted()
		r.mu.Unlock()
		if s != nil {
			r.read(s)
			continue
		}
		if end {
			return
		}
		select {
		case <-r.stop:
			return
		case <-r.wake:
		}
	}
}

// retarget replaces the map-time walk with the upload payloads, in upload
// order, skipping chunks the walk already handed out; the window carries over.
func (r *inkling2UploadReadahead) retarget(payloads [][]byte, chunk int) {
	r.mu.Lock()
	c := newInkling2ReadaheadChunks(payloads, r.chunks.window, chunk)
	c.drop(r.handed.covers)
	r.chunks, r.final, r.handed = c, true, nil
	r.mu.Unlock()
	r.signal()
}

// advance records n more payload bytes uploaded.
func (r *inkling2UploadReadahead) advance(n int) {
	r.consumed.Add(int64(n))
	r.signal()
}

func (r *inkling2UploadReadahead) signal() {
	select {
	case r.wake <- struct{}{}:
	default:
	}
}

// close stops the goroutine and waits out its in-flight read, so no read
// outlives the mapping. It is idempotent.
func (r *inkling2UploadReadahead) close() {
	r.stopOnce.Do(func() { close(r.stop) })
	r.done.Wait()
}

// Close lets the reader sit ahead of the mapping in its closer chain.
func (r *inkling2UploadReadahead) Close() error {
	r.close()
	return nil
}

// inkling2MappedShard is one mapped GGUF shard and the file behind it.
type inkling2MappedShard struct {
	data []byte
	path string
}

// inkling2MappedRegions is what the map-time reader walks, in address order:
// each shard whole, except that a single-file GGUF skips its header and the
// token embedding, which stays on the host and is read a few rows per token.
func inkling2MappedRegions(shards []inkling2MappedShard, f *File) [][]byte {
	var regions [][]byte
	if len(shards) == 1 && f != nil {
		data := shards[0].data
		lo, hi := int64(len(data)), int64(len(data))
		for _, t := range f.Tensors {
			if t.Name == "token_embd.weight" {
				lo = t.FileOffset
			}
		}
		for _, t := range f.Tensors {
			if t.FileOffset > lo && t.FileOffset < hi {
				hi = t.FileOffset
			}
		}
		start := min(max(f.TensorDataOffset, 0), int64(len(data)))
		lo, hi = max(lo, start), max(hi, start)
		if lo > start {
			regions = append(regions, data[start:lo])
		}
		if hi < int64(len(data)) {
			regions = append(regions, data[hi:])
		}
	} else {
		for _, s := range shards {
			if len(s.data) > 0 {
				regions = append(regions, s.data)
			}
		}
	}
	sort.SliceStable(regions, func(a, b int) bool {
		return uintptr(unsafe.Pointer(&regions[a][0])) < uintptr(unsafe.Pointer(&regions[b][0]))
	})
	return regions
}

// inkling2ReadaheadReader reads a span into the page cache: MADV_POPULATE_READ
// where the kernel has it, else pread into a reused scratch buffer, else a
// page-touch walk. Only its reader goroutine calls it.
type inkling2ReadaheadReader struct {
	populate func([]byte) error
	pread    func([]byte) bool
	noAdvice bool
	page     int
	sink     byte
}

// inkling2PopulateUnsupported reports a kernel without MADV_POPULATE_READ
// (Linux < 5.14 rejects the advice value).
func inkling2PopulateUnsupported(err error) bool {
	return errors.Is(err, syscall.EINVAL) || errors.Is(err, syscall.ENOSYS)
}

func (r *inkling2ReadaheadReader) read(b []byte) {
	if len(b) == 0 {
		return
	}
	if !r.noAdvice {
		err := r.populate(b)
		for i := 0; i < 3 && errors.Is(err, syscall.EINTR); i++ {
			err = r.populate(b)
		}
		if !inkling2PopulateUnsupported(err) {
			return
		}
		r.noAdvice = true
	}
	if r.pread != nil && r.pread(b) {
		return
	}
	page := max(r.page, 1)
	for i := 0; i < len(b); i += page {
		r.sink += b[i]
	}
}

// inkling2ReadaheadFiles preads spans of the mapped shards through their own
// file descriptors, opened on first use.
type inkling2ReadaheadFiles struct {
	shards  []inkling2MappedShard
	files   []*os.File
	scratch []byte
}

func newInkling2ReadaheadFiles(shards []inkling2MappedShard) *inkling2ReadaheadFiles {
	return &inkling2ReadaheadFiles{shards: shards, files: make([]*os.File, len(shards))}
}

// read reports whether it read b, false when b lies in no shard or the
// shard's file cannot be read.
func (f *inkling2ReadaheadFiles) read(b []byte) bool {
	s := inkling2SpanRange(b)
	for i, sh := range f.shards {
		if len(sh.data) == 0 || sh.path == "" {
			continue
		}
		base := inkling2SpanRange(sh.data)
		if s[0] < base[0] || s[1] > base[1] {
			continue
		}
		if f.files[i] == nil {
			fh, err := os.Open(sh.path)
			if err != nil {
				return false
			}
			f.files[i] = fh
		}
		if f.scratch == nil {
			f.scratch = make([]byte, inkling2UploadReadaheadChunk)
		}
		off, rest := int64(s[0]-base[0]), len(b)
		for rest > 0 {
			n, err := f.files[i].ReadAt(f.scratch[:min(rest, len(f.scratch))], off)
			if n <= 0 {
				return err == nil
			}
			off, rest = off+int64(n), rest-n
		}
		return true
	}
	return false
}

func (f *inkling2ReadaheadFiles) close() {
	for i, fh := range f.files {
		if fh != nil {
			fh.Close()
			f.files[i] = nil
		}
	}
	f.scratch = nil
}

// inkling2UploadReadaheadOn reports whether be streams its cold load: only the
// CUDA backend on a host with the Linux readahead path.
func inkling2UploadReadaheadOn(be compute.Backend) bool {
	return inkling2UploadAdviseOn && be != nil && be.Name() == "cuda"
}

// inkling2UploadReadaheadWindow is the window for be, 0 when the streamed
// cold-load path is off, and FAK_INKLING2_UPLOAD_READAHEAD_MB overrides the
// size (0 disables).
func inkling2UploadReadaheadWindow(be compute.Backend, lookup func(string) (string, bool)) int64 {
	if !inkling2UploadReadaheadOn(be) {
		return 0
	}
	if v, ok := lookup("FAK_INKLING2_UPLOAD_READAHEAD_MB"); ok {
		mb, err := strconv.ParseInt(v, 10, 64)
		if err != nil || mb <= 0 || mb > 1<<20 {
			return 0
		}
		return mb << 20
	}
	return inkling2UploadReadaheadDefault
}

// inkling2UploadReadaheadLead is the window with the default sized to host RAM:
// read-ahead pages are clean file-backed cache (mapped into this process under
// MADV_POPULATE_READ), which reclaim drops without writeback, so the lead stays
// at a quarter of MemAvailable and reclaim never starts on it. After the
// retarget the map-time walk's unconsumed bytes plus the new window can reach
// about twice the lead, still under half of MemAvailable.
// An explicit FAK_INKLING2_UPLOAD_READAHEAD_MB is taken as given.
func inkling2UploadReadaheadLead(be compute.Backend, lookup func(string) (string, bool), probe func() (int64, int64, bool)) int64 {
	w := inkling2UploadReadaheadWindow(be, lookup)
	if w == 0 {
		return 0
	}
	if _, explicit := lookup("FAK_INKLING2_UPLOAD_READAHEAD_MB"); explicit {
		return w
	}
	_, avail, known := probe()
	if !known || avail <= 0 || avail == compute.FreeUnknown {
		return min(w, inkling2UploadReadaheadUnknownRAM)
	}
	return max(min(w, avail/4), inkling2UploadReadaheadChunk)
}

// startInkling2MappedUploadReadahead starts the map-time reader over shards,
// nil when lead is 0. Its Close must run before the shards are unmapped.
func startInkling2MappedUploadReadahead(shards []inkling2MappedShard, f *File, lead int64) *inkling2UploadReadahead {
	if lead <= 0 {
		return nil
	}
	files := newInkling2ReadaheadFiles(shards)
	rd := &inkling2ReadaheadReader{populate: inkling2UploadPopulate, pread: files.read, page: os.Getpagesize()}
	return startInkling2MappedReadahead(inkling2MappedRegions(shards, f), lead, inkling2UploadReadaheadChunk, rd.read, files.close)
}

// newInkling2UploadReader is the read for a reader started at bind, without
// the shard files the pread fallback needs.
func newInkling2UploadReader() func([]byte) {
	rd := &inkling2ReadaheadReader{populate: inkling2UploadPopulate, page: os.Getpagesize()}
	return rd.read
}
