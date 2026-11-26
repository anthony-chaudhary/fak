package ggufload

import (
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	// inkling2UploadReadaheadDefault is how far the advised range runs ahead of
	// the upload cursor: ~0.3 s of lead at NVMe speed, small against the page
	// cache a cold load already fills. The chunk list may admit one extra chunk.
	inkling2UploadReadaheadDefault = int64(1) << 30
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
// order, while they start within window bytes of the upload cursor, so advised
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

// inkling2UploadReadahead advises the chunk list spans from a background goroutine
// as the upload cursor advances; advice submission can block on the device
// queue, and the upload must not wait on it.
type inkling2UploadReadahead struct {
	chunks   *inkling2ReadaheadChunks
	advise   func([]byte)
	consumed atomic.Int64
	wake     chan struct{}
	stop     chan struct{}
	done     sync.WaitGroup
}

func startInkling2UploadReadahead(chunks *inkling2ReadaheadChunks, advise func([]byte)) *inkling2UploadReadahead {
	r := &inkling2UploadReadahead{chunks: chunks, advise: advise, wake: make(chan struct{}, 1), stop: make(chan struct{})}
	r.done.Add(1)
	go r.run()
	return r
}

func (r *inkling2UploadReadahead) run() {
	defer r.done.Done()
	for {
		for _, s := range r.chunks.due(r.consumed.Load()) {
			select {
			case <-r.stop:
				return
			default:
			}
			r.advise(s)
		}
		if r.chunks.next >= len(r.chunks.spans) {
			return
		}
		select {
		case <-r.stop:
			return
		case <-r.wake:
		}
	}
}

// advance records n more payload bytes uploaded.
func (r *inkling2UploadReadahead) advance(n int) {
	r.consumed.Add(int64(n))
	select {
	case r.wake <- struct{}{}:
	default:
	}
}

// close stops the goroutine and waits for it, so no advice outlives the bind
// (and with it the mapping).
func (r *inkling2UploadReadahead) close() {
	close(r.stop)
	r.done.Wait()
}

// inkling2UploadReadaheadWindow is the window for be, 0 when the streamed
// cold-load path is off: only the CUDA backend on a madvise-capable host takes
// it, and FAK_INKLING2_UPLOAD_READAHEAD_MB overrides the size (0 disables).
func inkling2UploadReadaheadWindow(be compute.Backend, lookup func(string) (string, bool)) int64 {
	if !inkling2UploadAdviseOn || be == nil || be.Name() != "cuda" {
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
