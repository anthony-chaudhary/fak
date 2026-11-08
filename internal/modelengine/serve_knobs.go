package modelengine

import (
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Serve knobs for landed scheduler ports that had no production caller
// (fak-private#3630). Every knob is read once when the Engine builds its native
// scheduler (the `fak serve` in-kernel path) and is OFF unless its env var holds an
// explicit value, so an unset environment keeps today's scheduler byte-for-byte.
// The one exception is the per-step token budget: when its env var is unset it is
// default-armed from the model's declared context window (nativeTokenBudgetDefault);
// an explicit value still wins verbatim, and an explicit 0 keeps the unlimited
// behavior.
const (
	envNativeMaxTokensPerStep = "FAK_NATIVE_MAX_TOKENS_PER_STEP"
	envNativeDeterministic    = "FAK_NATIVE_DETERMINISTIC"
	envNativeKVAutosize       = "FAK_NATIVE_KV_AUTOSIZE"
	envNativeKVElasticBlocks  = "FAK_NATIVE_KV_ELASTIC_BLOCKS"
	envNativePrefillInterval  = "FAK_NATIVE_PREFILL_INTERVAL"
	envNativeSpecAdmissionPad = "FAK_NATIVE_SPEC_ADMISSION_PAD"
)

func nativeEnvOn(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "on", "true", "yes", "enable", "enabled":
		return true
	default:
		return false
	}
}

// nativeEnvPositiveInt returns a positive integer env value, or 0 for unset,
// malformed, or non-positive input (0 is every knob's disabled value).
func nativeEnvPositiveInt(name string) int {
	n, err := strconv.Atoi(strings.TrimSpace(os.Getenv(name)))
	if err != nil || n <= 0 {
		return 0
	}
	return n
}

// nativeKVFreeBytesProbeFromEnv returns the free-bytes probe only when
// FAK_NATIVE_KV_AUTOSIZE opts in. Without it an assigned probe (the CUDA build
// assigns one) is not consulted, so preemption stays off exactly as before.
func nativeKVFreeBytesProbeFromEnv() func() (uint64, bool) {
	if !nativeEnvOn(envNativeKVAutosize) {
		return nil
	}
	return nativeKVFreeBytesProbe
}

// kvFreeBytesFromBackend adapts compute.DeviceMemoryInfo to the probe shape: it
// reports only a known, positive free reading.
func kvFreeBytesFromBackend(b compute.Backend) (uint64, bool) {
	_, free, known := compute.DeviceMemoryInfo(b)
	if !known || free <= 0 {
		return 0, false
	}
	return uint64(free), true
}

// nativeTokenBudgetContextFraction is the denominator the default per-step token
// budget divides the model's declared context window by. One quarter lets even a
// full-window prompt prefill in four steps while keeping any realistic decode step
// (one token per running lane) far inside the budget; the budget is admission-shaped
// (it defers waiting-lane promotion, never preempts running lanes), so the fraction
// errs large on purpose.
const nativeTokenBudgetContextFraction = 4

// nativeModelContextTokens is the context window the scheduler's model declares
// (MaxPositionEmbeddings). 0 when the model is absent or declares no window —
// e.g. the synthetic checkpoint — which is "nothing to derive from".
func nativeModelContextTokens(sched *NativeScheduler) int {
	if sched == nil || sched.m == nil {
		return 0
	}
	return sched.m.Cfg.MaxPositionEmbeddings
}

// nativeTokenBudgetDefault derives the per-step token budget armed on the serve
// path when FAK_NATIVE_MAX_TOKENS_PER_STEP is unset: one quarter of the model's
// declared context window, minimum 1. A model that declares no window yields 0,
// which keeps the scheduler's unlimited promotion byte-for-byte (the historical
// default-off behavior). The derivation is deliberately conservative: the budget
// only defers waiting-lane promotion that does not fit the remaining step budget
// (skip-not-break) and an empty running set always admits its first candidate, so
// the armed default cannot stall a step.
func nativeTokenBudgetDefault(contextTokens int) int {
	if contextTokens <= 0 {
		return 0
	}
	if b := contextTokens / nativeTokenBudgetContextFraction; b > 0 {
		return b
	}
	return 1
}

// applyNativeServeKnobs arms the env-selected scheduler ports on a freshly built
// scheduler. Each setter runs only for an explicit opt-in value, except the
// per-step token budget, which is default-armed from the model's declared context
// window when its env var is unset (explicit values win verbatim, and an explicit
// 0 keeps the unlimited behavior).
func applyNativeServeKnobs(sched *NativeScheduler, blockTokens int) {
	if sched == nil {
		return
	}
	// Per-step token budget (fak cb8cb8d405, landed default-off). An explicit
	// FAK_NATIVE_MAX_TOKENS_PER_STEP wins verbatim; an explicit 0 — or any value
	// that does not parse to a positive integer — keeps today's unlimited
	// behavior. Unset, the budget is default-armed from the model's declared
	// context window so a single step can never prefill the whole window while
	// other lanes wait; a model that declares no window keeps the unlimited
	// promotion byte-for-byte.
	if raw := strings.TrimSpace(os.Getenv(envNativeMaxTokensPerStep)); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			sched.SetMaxTokensPerStep(n)
		}
	} else if b := nativeTokenBudgetDefault(nativeModelContextTokens(sched)); b > 0 {
		sched.SetMaxTokensPerStep(b)
	}
	if nativeEnvOn(envNativeDeterministic) {
		sched.SetDeterministicMode(true)
	}
	if n := nativeEnvPositiveInt(envNativeKVElasticBlocks); n > 0 {
		if l, err := newNativeElasticLedger(n, blockTokens); err == nil {
			sched.elastic = l
		}
	}
	// FAK_NATIVE_PREFILL_INTERVAL holds prefill to every Nth engine step so decode
	// runs in longer uninterrupted stretches (vLLM local prefill cadence port).
	// Unset or <=1 keeps today's mixed-step order bit-for-bit.
	if n := nativeEnvPositiveInt(envNativePrefillInterval); n > 1 {
		sched.SetPrefillInterval(n)
	}
	// Unset or 0 keeps historical admission byte-for-byte.
	if n := nativeEnvPositiveInt(envNativeSpecAdmissionPad); n > 0 {
		sched.SetSpecAdmissionPad(n)
	}
}

// NativeElasticLedgerStats is the live view of the elastic KV block ledger.
type NativeElasticLedgerStats struct {
	Allocator       ElasticBlockStats
	AdmittedLanes   int64
	AllocatedBlocks int64
	CachedReleases  int64
	Shortfalls      int64
}

// nativeElasticLedger mirrors each running lane's prompt blocks into an
// ElasticBlockAllocator so its FIFO cached-block reuse runs on real serve
// traffic. It is an accounting ledger over a zero-stride pool (no KV bytes):
// it never refuses or delays a lane, it only records what the allocator decided.
type nativeElasticLedger struct {
	mu          sync.Mutex
	alloc       *ElasticBlockAllocator
	blockTokens int
	next        int64
	lanes       map[*schedLane]nativeElasticLane
	stats       NativeElasticLedgerStats
}

type nativeElasticLane struct {
	id     string
	blocks []int
}

func newNativeElasticLedger(blocks, blockTokens int) (*nativeElasticLedger, error) {
	if blockTokens <= 0 {
		blockTokens = nativeKVDefaultBlockTokens
	}
	a, err := NewElasticBlockAllocator(ElasticBlockAllocatorConfig{
		InitialBlocks: blocks,
		ModelCfg:      model.Config{},
		BlockTokens:   blockTokens,
	})
	if err != nil {
		return nil, err
	}
	return &nativeElasticLedger{alloc: a, blockTokens: blockTokens, lanes: make(map[*schedLane]nativeElasticLane)}, nil
}

// laneStarted reserves the lane's prompt blocks. A re-initialized lane keeps its
// existing reservation; capacity exhaustion records a shortfall and stops.
func (l *nativeElasticLedger) laneStarted(ln *schedLane, promptTokens int) {
	if l == nil || ln == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, ok := l.lanes[ln]; ok {
		return
	}
	l.next++
	lane := nativeElasticLane{id: "lane-" + strconv.FormatInt(l.next, 10)}
	if err := l.alloc.RegisterSequence(lane.id, ln.priority); err != nil {
		return
	}
	for i := nativePromptBlocks(promptTokens, l.blockTokens); i > 0; i-- {
		blk, err := l.alloc.AllocBlock(lane.id)
		if err != nil {
			l.stats.Shortfalls++
			break
		}
		lane.blocks = append(lane.blocks, blk)
		l.stats.AllocatedBlocks++
	}
	l.lanes[ln] = lane
	l.stats.AdmittedLanes++
}

// laneFinished releases a finished lane's blocks as cached (its prompt prefix
// stays reusable), which is the ReleaseCachedBlock FIFO path.
func (l *nativeElasticLedger) laneFinished(ln *schedLane) {
	if l == nil || ln == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	lane, ok := l.lanes[ln]
	if !ok {
		return
	}
	delete(l.lanes, ln)
	for _, blk := range lane.blocks {
		if l.alloc.ReleaseCachedBlock(lane.id, blk) == nil {
			l.stats.CachedReleases++
		}
	}
	_, _ = l.alloc.UnregisterSequence(lane.id)
}

func (l *nativeElasticLedger) snapshot() NativeElasticLedgerStats {
	l.mu.Lock()
	defer l.mu.Unlock()
	st := l.stats
	st.Allocator = l.alloc.Stats()
	return st
}

// ElasticLedgerStats reports the elastic KV ledger armed by
// FAK_NATIVE_KV_ELASTIC_BLOCKS; ok is false when the ledger is off.
func (s *NativeScheduler) ElasticLedgerStats() (NativeElasticLedgerStats, bool) {
	if s == nil || s.elastic == nil {
		return NativeElasticLedgerStats{}, false
	}
	return s.elastic.snapshot(), true
}
