package model

// dense_decode_graph.go — portable state of the dense whole-token Q4_K Metal decode
// graph (fak#13599). The darwin route (metal_dense_decode_graph.go) encodes one dense
// PreNorm SwiGLU token — every layer's norms, q/k/v (+bias), RoPE, device-KV
// attention, o_proj, residuals, gate/up/SwiGLU/down and the LM head — into ONE Metal
// command buffer instead of the ~5 synchronous command buffers per layer the blockStep
// path commits. This file owns what must exist on every build: the session-owned device
// KV mirror, its validity rule, the per-token receipt, the kill switch and teardown.
//
// Default ON. The route is fail-open: any decline or pre-mutation error leaves the host
// cache untouched and the token runs through blockStep, so FAK_DENSE_Q4K_DECODE_GRAPH=0
// restores the historical path exactly. The receipt (Session.DenseQ4KDecodeGraphReceipt)
// is the explicit evidence of which path each token took.

import (
	"os"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// Decline reasons recorded by the dense decode graph route. They are a closed set so
// DeclineCounts stays bounded.
const (
	denseDeclineDisabled     = "disabled"
	denseDeclineSession      = "session"
	denseDeclinePosition     = "position"
	denseDeclineToken        = "token"
	denseDeclineObserver     = "observer"
	denseDeclineArchitecture = "architecture"
	denseDeclineGeometry     = "geometry"
	denseDeclineKVPrecision  = "kv_precision"
	denseDeclineTensors      = "tensors"
	denseDeclineProjection   = "projection"
	denseDeclineHead         = "head"
	denseDeclineKVBudget     = "kv_budget"
	denseDeclineDeviceKV     = "device_kv"
	denseDeclineGraph        = "graph_error"
)

// denseQ4KDecodeGraphEnabled is the route's kill switch: FAK_DENSE_Q4K_DECODE_GRAPH=0
// disables it and every token takes the historical blockStep path.
func denseQ4KDecodeGraphEnabled() bool {
	return os.Getenv("FAK_DENSE_Q4K_DECODE_GRAPH") != "0"
}

// denseDecodeKVBudgetBytes bounds one session's device KV mirror (2 sides — KPost and
// V; the host cache keeps KRaw — × layers × capacity × kvWidth × 4 B). A token whose
// context no longer fits declines to blockStep rather than allocating past the budget.
// Qwen2.5-7B (28 layers, kvWidth 512) holds ~37k tokens at the default 4 GiB. The
// process-wide device room (denseDecodeKVLimit) bounds it further.
var denseDecodeKVBudgetBytes atomic.Int64

func init() { denseDecodeKVBudgetBytes.Store(4 << 30) }

// SetDenseQ4KDecodeKVBudgetBytes sets the device KV mirror budget for the dense decode
// graph (config surface; values <= 0 restore the 4 GiB default).
func SetDenseQ4KDecodeKVBudgetBytes(bytes int64) {
	if bytes <= 0 {
		bytes = 4 << 30
	}
	denseDecodeKVBudgetBytes.Store(bytes)
}

// denseDecodeKVSides is the number of f32 device sides a dense mirror holds (KPost, V).
const denseDecodeKVSides = 2

// denseDecodeKVLimit is the byte room a NEW mirror may take: the per-session budget,
// further bounded so the model's resident weights, every other live device KV mirror
// (other sessions; process-wide) and this one stay under metalQ8UploadFraction of the
// device working set — the same headroom the additive Q6_K/Q8 uploads are judged by.
// Unified-memory allocations rarely fail outright (they swap), so the route must
// decline on this estimate rather than wait for an allocation failure. An unknown
// device budget (deviceTotal <= 0) cannot prove anything fits and returns 0.
func denseDecodeKVLimit(sessionBudget, deviceTotal, modelResident, otherKV int64) int64 {
	if deviceTotal <= 0 || sessionBudget <= 0 {
		return 0
	}
	room := int64(metalQ8UploadFraction*float64(deviceTotal)) - modelResident - otherKV
	if room < 0 {
		room = 0
	}
	return min(sessionBudget, room)
}

// denseDecodeKVCapacity picks the mirror capacity for `need` rows: geometric growth
// (2x, at least +256 rows) clamped to the model context while the context still fits
// it (past MaxPositionEmbeddings growth stays geometric rather than +1 per token), then
// to limitBytes. Near the limit the growth step shrinks to whatever room is left; a
// result below need means the context no longer fits.
func denseDecodeKVCapacity(cfg Config, need int, limitBytes int64) int {
	capTokens := 2 * need
	if capTokens < need+256 {
		capTokens = need + 256
	}
	if mp := cfg.MaxPositionEmbeddings; mp >= need && capTokens > mp {
		capTokens = mp
	}
	perToken := int64(denseDecodeKVSides) * int64(cfg.NumLayers) * int64(cfg.NumKVHeads*cfg.HeadDim) * 4
	if perToken <= 0 {
		return 0
	}
	if maxTokens := limitBytes / perToken; int64(capTokens) > maxTokens {
		capTokens = int(maxTokens)
	}
	return capTokens
}

// DenseQ4KDecodeGraphReceipt is the per-token evidence of the dense decode graph route.
// The token fields describe the LAST token the route was consulted for; the counters
// accumulate over the session.
type DenseQ4KDecodeGraphReceipt struct {
	// Accepted is true when the last token ran through the graph and returned its logits.
	Accepted bool
	// DeclineReason names why the last token fell back to blockStep ("" when accepted).
	DeclineReason string
	// CommandBuffers is 1 for an accepted token: the whole forward plus the LM head
	// commit and wait exactly once.
	CommandBuffers                     int
	Encoders                           int
	GPUMilliseconds, WaitMilliseconds  float64
	HostUploadBytes, HostReadbackBytes uint64
	// DeviceKVSeedBytes is the host->device KV traffic this token paid to (re)seed or
	// catch up the mirror before encoding (0 on the steady-state decode path).
	DeviceKVSeedBytes uint64
	DeviceKVCapacity  int

	AcceptedTokens, DeclinedTokens uint64
	// GraphFallbacks counts tokens admitted to the graph that failed after Begin and
	// replayed through blockStep (the host cache had not been mutated).
	GraphFallbacks uint64
	// DeviceKVReseeds counts full mirror reseeds (allocation, growth or invalidation).
	DeviceKVReseeds uint64
	DeclineCounts   map[string]uint64
}

// denseDecodeGraphState is the session-owned dense decode graph state: the device KV
// mirror and the receipt. A Session is single-goroutine, so it needs no lock.
//
// Mirror validity: the device rows [0, rows) equal the host cache's rows exactly when
// the mirror was seeded from / appended alongside THIS *KVCache (pointer identity; the
// mirror holds the pointer, so it cannot be recycled), no non-append mutation happened
// since (KVCache.mutationGeneration), and rows <= Len with every layer's host slices
// still Len rows wide. Appends that bypass the graph (prefill, a declined token) only
// leave rows < Len and are caught up by uploading the missing rows.
type denseDecodeGraphState struct {
	kv       *metalgemm.DeviceKV
	cache    *KVCache
	gen      uint64
	rows     int
	capacity int

	readyModel *Model // model whose tensors + projections were verified device-resident
	// notReadyModel/notReadyReason cache a failed readiness verdict so a model the graph
	// cannot serve (e.g. a Q5_K mix or a Q2_K head) is not re-scanned on every token.
	notReadyModel  *Model
	notReadyReason string
	ones           []float32
	// graphFailures counts consecutive post-Begin failures; the route stops trying for
	// this session after denseDecodeMaxGraphFailures so a persistent native decline
	// does not pay a wasted graph encode on every token.
	graphFailures int
	// injectPostSubmitFailure is a test seam: the next graph reports an accepted
	// post-submit failure (metalgemm.InjectPostSubmitFailureForTest), then clears.
	injectPostSubmitFailure bool

	receipt DenseQ4KDecodeGraphReceipt
}

func (s *Session) denseDecodeState() *denseDecodeGraphState {
	if s.denseDecode == nil {
		s.denseDecode = &denseDecodeGraphState{}
	}
	return s.denseDecode
}

// invalidate frees the device KV mirror; the next accepted token reseeds it.
func (st *denseDecodeGraphState) invalidate() {
	if st == nil {
		return
	}
	if st.kv != nil {
		st.kv.Close()
	}
	st.kv, st.cache, st.gen, st.rows, st.capacity = nil, nil, 0, 0, 0
}

func (st *denseDecodeGraphState) decline(reason string) {
	r := &st.receipt
	r.Accepted, r.DeclineReason = false, reason
	r.CommandBuffers, r.Encoders = 0, 0
	r.GPUMilliseconds, r.WaitMilliseconds = 0, 0
	r.HostUploadBytes, r.HostReadbackBytes, r.DeviceKVSeedBytes = 0, 0, 0
	r.DeclinedTokens++
	if r.DeclineCounts == nil {
		r.DeclineCounts = make(map[string]uint64)
	}
	r.DeclineCounts[reason]++
}

// DenseQ4KDecodeGraphReceipt returns a copy of the dense decode graph receipt. The zero
// value means the route was never consulted (not a MetalQ4K session).
func (s *Session) DenseQ4KDecodeGraphReceipt() DenseQ4KDecodeGraphReceipt {
	if s == nil || s.denseDecode == nil {
		return DenseQ4KDecodeGraphReceipt{}
	}
	out := s.denseDecode.receipt
	if src := s.denseDecode.receipt.DeclineCounts; src != nil {
		out.DeclineCounts = make(map[string]uint64, len(src))
		for k, v := range src {
			out.DeclineCounts[k] = v
		}
	}
	return out
}

// closeDenseDecodeGraph releases the device KV mirror at Session.Close. A Session dropped
// without Close still frees it: metalgemm.DeviceKV carries a runtime cleanup.
func (s *Session) closeDenseDecodeGraph() {
	if s == nil || s.denseDecode == nil {
		return
	}
	s.denseDecode.invalidate()
}
