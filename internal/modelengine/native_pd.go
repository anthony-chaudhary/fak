package modelengine

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const nativePDDefaultBlockTokens = 16

// NativePDRole is the native serving role a worker is currently serving.
type NativePDRole string

const (
	NativePDRolePrefill NativePDRole = "prefill"
	NativePDRoleDecode  NativePDRole = "decode"
)

// NativePDResidencyIndex is the subset of the shared per-worker prefix-residency
// index native P/D needs. internal/gateway.PrefixResidencyIndex satisfies this
// interface without modelengine importing gateway (a higher tier).
type NativePDResidencyIndex interface {
	Overlap(worker string, prefix []string) int
	Observe(worker string, prefix []string)
}

// ImportedSequence is a prompt-prefilled sequence handed to a decode scheduler.
// Session must already hold the prompt KV; Logits are the prefill logits for the
// first generated token. AdmitImported queues it into the normal NativeScheduler
// loop, so decode proceeds through continuous batching without re-prefilling.
type ImportedSequence struct {
	Session   *model.Session
	Logits    []float32
	Prompt    []int
	PromptLen int
	Tool      string
	Tokenizer NLTokenizer
	Q4K       bool
}

// AdmitImported admits a sequence whose prompt KV was built elsewhere. It is the
// decode-worker half of native P/D disaggregation: imported lanes enter the same
// WAITING/RUNNING queues as local Admit lanes and are advanced by stepOnce.
func (s *NativeScheduler) AdmitImported(ctx context.Context, c *abi.ToolCall, seq ImportedSequence) (abi.EngineRequest, error) {
	if s == nil {
		return nil, errors.New("modelengine: nil native scheduler")
	}
	if seq.Session == nil || seq.Session.Cache == nil {
		return nil, errors.New("modelengine: imported sequence has no resident KV session")
	}
	if len(seq.Logits) == 0 {
		return nil, errors.New("modelengine: imported sequence has no prefill logits")
	}
	tool := seq.Tool
	if tool == "" && c != nil {
		tool = c.Tool
	}
	promptLen := seq.PromptLen
	if promptLen == 0 {
		promptLen = len(seq.Prompt)
	}
	if promptLen == 0 {
		promptLen = seq.Session.Cache.Len()
	}
	cctx, cancel := context.WithCancel(ctx)
	ln := &schedLane{
		sched:     s,
		ctx:       cctx,
		cancel:    cancel,
		sess:      seq.Session,
		logits:    copyF32(seq.Logits),
		tool:      tool,
		promptLen: promptLen,
		putCtx:    ctx,
		tok:       seq.Tokenizer,
		q4k:       seq.Q4K,
		tokens:    make(chan abi.EngineToken, 1),
		done:      make(chan struct{}),
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		cancel()
		return nil, errSchedClosed
	}
	s.waiting = append(s.waiting, ln)
	s.mu.Unlock()

	s.started.Do(func() { go s.run() })
	s.signal()
	return ln, nil
}

// NativePDRequest is one native P/D admission. Prompt is the already-tokenized
// prefix; if empty, the call is tokenized with the scheduler's byte tokenizer.
type NativePDRequest struct {
	Call             *abi.ToolCall
	Prompt           []int
	PrefixKey        string
	PrefixSegments   []string
	ModelID          string
	TokenizerID      string
	Lease            string
	Taint            abi.TaintLabel
	Scope            abi.ShareScope
	AdmissionVerdict cachemeta.AdmissionVerdict
	AdmittedBy       string
}

// NativePDAdmission records where the request ran and the imported decode state.
type NativePDAdmission struct {
	Request       abi.EngineRequest
	PrefillWorker string
	DecodeWorker  string
	PrefixKey     string
	LocalityHit   bool
	Transfer      model.PagedKVTransferReceipt
	ImportedCache *model.KVCache
}

// NativePDWorkerStats is a point-in-time per-role serving counter set.
type NativePDWorkerStats struct {
	Worker            string
	Role              NativePDRole
	PrefillRequests   int64
	DecodeImports     int64
	SchedulerAdmits   int64
	BytesMoved        int64
	RePrefillOnDecode int64
}

// NativePDWorker is one native prefill or decode worker in the in-process P/D pool.
type NativePDWorker struct {
	name  string
	role  NativePDRole
	sched *NativeScheduler
	pool  *model.PagedKVPool

	mu      sync.Mutex
	stats   NativePDWorkerStats
	entries map[string]cachemeta.Entry
	spans   map[string]*model.PagedKV
}

func newNativePDWorker(name string, role NativePDRole, m *model.Model) *NativePDWorker {
	w := &NativePDWorker{
		name:    name,
		role:    role,
		pool:    model.NewPagedKVPoolWithRaw(m.Cfg, nativePDDefaultBlockTokens),
		entries: make(map[string]cachemeta.Entry),
		spans:   make(map[string]*model.PagedKV),
	}
	w.stats = NativePDWorkerStats{Worker: name, Role: role}
	if role == NativePDRoleDecode {
		w.sched = NewNativeScheduler(m)
	}
	return w
}

func (w *NativePDWorker) Name() string { return w.name }

func (w *NativePDWorker) Role() NativePDRole { return w.role }

func (w *NativePDWorker) Scheduler() *NativeScheduler { return w.sched }

func (w *NativePDWorker) Stats() NativePDWorkerStats {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.stats
}

// TransferEntry returns the trust/residency descriptor the decode worker received
// for spanDigest.
func (w *NativePDWorker) TransferEntry(spanDigest string) (cachemeta.Entry, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	e, ok := w.entries[spanDigest]
	return e, ok
}

func (w *NativePDWorker) close() {
	if w.sched != nil {
		w.sched.Close()
	}
	w.mu.Lock()
	for k, seq := range w.spans {
		if seq != nil {
			seq.Free()
		}
		delete(w.spans, k)
	}
	w.mu.Unlock()
}

func (w *NativePDWorker) addPrefill(bytes int64) {
	w.mu.Lock()
	w.stats.PrefillRequests++
	w.stats.BytesMoved += bytes
	w.mu.Unlock()
}

func (w *NativePDWorker) addImport(receipt model.PagedKVTransferReceipt) {
	digest := receipt.Transfer.SpanDigest
	w.mu.Lock()
	w.stats.DecodeImports++
	w.stats.SchedulerAdmits++
	w.stats.BytesMoved += receipt.Transfer.BytesMoved
	if digest != "" {
		w.entries[digest] = receipt.Entry
		if old := w.spans[digest]; old != nil && old != receipt.KV {
			old.Free()
		}
		w.spans[digest] = receipt.KV
	}
	w.mu.Unlock()
}

// NativePDCluster is the smallest native P/D control surface: one prefill role,
// N decode roles, a prefix-residency index, and per-role metrics. It deliberately
// composes NativeScheduler and model.KVTransport instead of inventing a second
// decode loop.
type NativePDCluster struct {
	m       *model.Model
	prefill *NativePDWorker
	decode  []*NativePDWorker

	mu             sync.Mutex
	residency      NativePDResidencyIndex
	localityHits   int64
	coldRoutes     int64
	lastDecodeName string
}

func NewNativePDCluster(m *model.Model, decodeWorkers int) *NativePDCluster {
	return NewNativePDClusterWithResidency(m, decodeWorkers, nil)
}

// NewNativePDClusterWithResidency builds a native P/D pool over a caller-supplied
// residency view. Passing gateway.PrefixResidencyIndex wires the native role split
// into the same shared routing spine as fleet serving; nil uses a small local
// index for standalone in-process use.
func NewNativePDClusterWithResidency(m *model.Model, decodeWorkers int, residency NativePDResidencyIndex) *NativePDCluster {
	if decodeWorkers < 1 {
		decodeWorkers = 1
	}
	if residency == nil {
		residency = newNativePDLocalResidency()
	}
	c := &NativePDCluster{
		m:         m,
		prefill:   newNativePDWorker("prefill-0", NativePDRolePrefill, m),
		decode:    make([]*NativePDWorker, decodeWorkers),
		residency: residency,
	}
	for i := range c.decode {
		c.decode[i] = newNativePDWorker("decode-"+strconv.Itoa(i), NativePDRoleDecode, m)
	}
	return c
}

func (c *NativePDCluster) Close() {
	if c == nil {
		return
	}
	if c.prefill != nil {
		c.prefill.close()
	}
	for _, w := range c.decode {
		w.close()
	}
}

func (c *NativePDCluster) PrefillWorker() *NativePDWorker { return c.prefill }

func (c *NativePDCluster) DecodeWorkers() []*NativePDWorker {
	out := make([]*NativePDWorker, len(c.decode))
	copy(out, c.decode)
	return out
}

func (c *NativePDCluster) DecodeWorker(name string) (*NativePDWorker, bool) {
	for _, w := range c.decode {
		if w.name == name {
			return w, true
		}
	}
	return nil, false
}

// Admit runs prompt prefill on the prefill worker, moves the resulting paged KV
// span to the routed decode worker, and admits the imported cache into that
// decode worker's continuous-batching scheduler.
func (c *NativePDCluster) Admit(ctx context.Context, req NativePDRequest) (NativePDAdmission, error) {
	if c == nil || c.m == nil || c.prefill == nil || len(c.decode) == 0 {
		return NativePDAdmission{}, errors.New("modelengine: native P/D cluster is not initialized")
	}
	prompt := c.promptFor(ctx, req)
	if len(prompt) == 0 {
		return NativePDAdmission{}, errors.New("modelengine: native P/D request has empty prompt")
	}
	prefixKey := req.PrefixKey
	if prefixKey == "" {
		prefixKey = NativePDPrefixKey(prompt)
	}
	prefixSegments := c.prefixSegments(req, prefixKey)
	decode, _, hit := c.routeDecode(prefixSegments)

	prefillSess := c.m.NewSession()
	logits := prefillSess.Prefill(prompt)
	srcPool := model.NewPagedKVPoolWithRaw(c.m.Cfg, nativePDDefaultBlockTokens)
	src, err := model.KVCacheToPaged(srcPool, prefillSess.Cache)
	prefillSess.Close()
	if err != nil {
		return NativePDAdmission{}, err
	}
	defer src.Free()

	transfer := c.transferDescriptor(req, decode.name, len(prompt))
	receipt, err := (model.LocalKVTransport{Pool: decode.pool}).Send(src, transfer, 0, len(prompt))
	if err != nil {
		return NativePDAdmission{}, err
	}
	c.prefill.addPrefill(receipt.Transfer.BytesMoved)
	cache := receipt.KV.ToKVCache(c.m.Cfg)
	importedSnapshot := cache.Clone()
	decode.addImport(receipt)

	call := req.Call
	if call == nil {
		call = &abi.ToolCall{Tool: "native_pd"}
	}
	r, err := decode.sched.AdmitImported(ctx, call, ImportedSequence{
		Session:   &model.Session{M: c.m, Cache: cache},
		Logits:    logits,
		Prompt:    prompt,
		PromptLen: len(prompt),
		Tool:      call.Tool,
	})
	if err != nil {
		return NativePDAdmission{}, err
	}
	c.mu.Lock()
	c.residency.Observe(decode.name, prefixSegments)
	c.lastDecodeName = decode.name
	c.mu.Unlock()

	return NativePDAdmission{
		Request:       r,
		PrefillWorker: c.prefill.name,
		DecodeWorker:  decode.name,
		PrefixKey:     prefixKey,
		LocalityHit:   hit,
		Transfer:      receipt,
		ImportedCache: importedSnapshot,
	}, nil
}

func (c *NativePDCluster) promptFor(ctx context.Context, req NativePDRequest) []int {
	if len(req.Prompt) > 0 {
		return append([]int(nil), req.Prompt...)
	}
	if req.Call == nil {
		return nil
	}
	return tokenize(req.Call.Tool, refBytes(ctx, req.Call.Args), c.m.Cfg.VocabSize)
}

func (c *NativePDCluster) prefixSegments(req NativePDRequest, fallbackKey string) []string {
	if len(req.PrefixSegments) > 0 {
		return append([]string(nil), req.PrefixSegments...)
	}
	return []string{fallbackKey}
}

func (c *NativePDCluster) transferDescriptor(req NativePDRequest, owner string, n int) cachemeta.KVTransfer {
	modelID := req.ModelID
	if modelID == "" {
		modelID = "native-pd"
	}
	admit := req.AdmissionVerdict
	if admit == "" {
		admit = cachemeta.AdmissionAllow
	}
	admittedBy := req.AdmittedBy
	if admittedBy == "" {
		admittedBy = "native-pd"
	}
	return cachemeta.KVTransfer{
		Direction:        cachemeta.KVMigrate,
		Tokens:           int64(n),
		ModelID:          modelID,
		TokenizerID:      req.TokenizerID,
		SerializerID:     model.PagedKVTransferSerializerID,
		PositionMode:     cachemeta.PositionPrefixAligned,
		FromTier:         cachemeta.TierDRAM,
		ToTier:           cachemeta.TierDRAM,
		Owner:            owner,
		Lease:            req.Lease,
		SecuritySet:      true,
		Taint:            req.Taint,
		Scope:            req.Scope,
		AdmissionVerdict: admit,
		AdmittedBy:       admittedBy,
	}
}

func (c *NativePDCluster) routeDecode(prefix []string) (*NativePDWorker, int, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	bestIdx, bestOverlap := c.bestResidentDecodeLocked(prefix)
	if bestOverlap > 0 {
		if bestOverlap >= len(prefix) {
			c.localityHits++
		}
		return c.decode[bestIdx], bestIdx, bestOverlap >= len(prefix)
	}
	idx := c.coldDecodePlacementLocked()
	c.coldRoutes++
	return c.decode[idx], idx, false
}

func (c *NativePDCluster) bestResidentDecodeLocked(prefix []string) (int, int) {
	if len(prefix) == 0 || c.residency == nil {
		return 0, 0
	}
	bestIdx, bestOverlap := 0, 0
	bestLoad := int(^uint(0) >> 1)
	for i, w := range c.decode {
		overlap := c.residency.Overlap(w.name, prefix)
		load := w.Stats().DecodeImports
		if overlap > bestOverlap ||
			(overlap == bestOverlap && overlap > 0 && int(load) < bestLoad) ||
			(overlap == bestOverlap && overlap > 0 && int(load) == bestLoad && w.name < c.decode[bestIdx].name) {
			bestIdx, bestOverlap, bestLoad = i, overlap, int(load)
		}
	}
	return bestIdx, bestOverlap
}

type nativePDLocalResidency struct {
	mu   sync.Mutex
	held map[string][][]string
}

func newNativePDLocalResidency() *nativePDLocalResidency {
	return &nativePDLocalResidency{held: make(map[string][][]string)}
}

func (r *nativePDLocalResidency) Overlap(worker string, prefix []string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	best := 0
	for _, held := range r.held[worker] {
		if n := nativePDSegmentPrefixLen(held, prefix); n > best {
			best = n
		}
	}
	return best
}

func (r *nativePDLocalResidency) Observe(worker string, prefix []string) {
	if worker == "" || len(prefix) == 0 {
		return
	}
	cp := append([]string(nil), prefix...)
	r.mu.Lock()
	r.held[worker] = append(r.held[worker], cp)
	r.mu.Unlock()
}

func nativePDSegmentPrefixLen(a, b []string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

func (c *NativePDCluster) coldDecodePlacementLocked() int {
	best := 0
	for i := 1; i < len(c.decode); i++ {
		bs, is := c.decode[best].Stats(), c.decode[i].Stats()
		if is.DecodeImports < bs.DecodeImports || (is.DecodeImports == bs.DecodeImports && c.decode[i].name < c.decode[best].name) {
			best = i
		}
	}
	return best
}

// NativePDPrefixKey returns a stable prefix-residency key for a tokenized prompt.
func NativePDPrefixKey(ids []int) string {
	h := sha256.New()
	var buf [8]byte
	for _, id := range ids {
		binary.LittleEndian.PutUint64(buf[:], uint64(int64(id)))
		_, _ = h.Write(buf[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Metrics renders the P/D counters in the same Prometheus text shape as the
// gateway serving surface; callers can splice it into a larger scrape.
func (c *NativePDCluster) Metrics() string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	b.WriteString("# HELP fak_native_pd_worker_requests_total Native prefill/decode worker requests by role.\n")
	b.WriteString("# TYPE fak_native_pd_worker_requests_total counter\n")
	for _, w := range append([]*NativePDWorker{c.prefill}, c.decode...) {
		st := w.Stats()
		var n int64
		switch st.Role {
		case NativePDRolePrefill:
			n = st.PrefillRequests
		case NativePDRoleDecode:
			n = st.DecodeImports
		}
		fmt.Fprintf(&b, "fak_native_pd_worker_requests_total{role=%s,worker=%s} %d\n",
			promQuote(string(st.Role)), promQuote(st.Worker), n)
		fmt.Fprintf(&b, "fak_native_pd_worker_bytes_moved_total{role=%s,worker=%s} %d\n",
			promQuote(string(st.Role)), promQuote(st.Worker), st.BytesMoved)
	}
	c.mu.Lock()
	hits, cold := c.localityHits, c.coldRoutes
	c.mu.Unlock()
	b.WriteString("# HELP fak_native_pd_route_total Native P/D decode routes by locality result.\n")
	b.WriteString("# TYPE fak_native_pd_route_total counter\n")
	fmt.Fprintf(&b, "fak_native_pd_route_total{result=%s} %d\n", promQuote("locality_hit"), hits)
	fmt.Fprintf(&b, "fak_native_pd_route_total{result=%s} %d\n", promQuote("cold_receive"), cold)
	return b.String()
}

func promQuote(s string) string {
	return strconv.Quote(strings.ReplaceAll(s, "\n", `\n`))
}
