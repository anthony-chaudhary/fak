package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

const (
	WarmDiskOutcomeRestored    WarmDiskOutcome = "restored"
	WarmDiskOutcomePersisted                   = "persisted"
	WarmDiskOutcomeStaged                      = "staged"
	WarmDiskOutcomeMiss                        = "miss"
	WarmDiskOutcomeFault                       = "fault"
	WarmDiskOutcomeDisabled                    = "disabled"
	WarmDiskOutcomeUnsupported                 = "unsupported"
	WarmDiskRestored                           = WarmDiskOutcomeRestored
	WarmDiskPersisted                          = WarmDiskOutcomePersisted
	WarmDiskStaged                             = WarmDiskOutcomeStaged
	WarmDiskFault                              = WarmDiskOutcomeFault
	WarmDiskDisabled                           = WarmDiskOutcomeDisabled

	WarmDiskTierLocalSSD             = radixkv.SnapshotTier("local_ssd_l3")
	WarmDiskSnapshotTier             = WarmDiskTierLocalSSD
	warmDiskTier                     = WarmDiskTierLocalSSD
	warmDiskEnvelopeVersion          = 1
	defaultWarmDiskEntries           = 4
	defaultWarmDiskBytes       int64 = 8 << 30
	defaultWarmDiskRecordBytes int64 = 512 << 20
	maxWarmDiskRecordBytes     int64 = 2 << 30
	warmDiskSlotsDir                 = "slots"
	maxWarmDiskSlots                 = 64
	maxWarmDiskSlotBytes             = 1024
)

// WarmDiskOutcome is the closed restart-cache result vocabulary.
type WarmDiskOutcome string

// WarmDiskConfig overrides the bounded per-user restart cache. The zero value
// enables the default store. Disabled is the explicit programmatic opt-out.
type WarmDiskConfig struct {
	Dir            string
	Disabled       bool
	MaxEntries     int
	MaxBytes       int64
	MaxRecordBytes int64
	// ConversationState opts persistence into async conversation-state staging:
	// eligible owned retained states are encoded, copied into a bounded I/O
	// queue, and written + fsync'd by a background writer instead of on the
	// request path. Default off keeps the historical synchronous persist.
	ConversationState        bool
	ConversationStatePermit  int64 // staged bytes held at once; <=0 selects the default
	ConversationStateMaxJobs int   // saves queued plus in flight; <=0 selects the default
}

// WarmDiskReceipt is the bounded, prompt-free result of local restart reuse.
type WarmDiskReceipt struct {
	Outcome        WarmDiskOutcome      `json:"outcome"`
	Tier           radixkv.SnapshotTier `json:"tier,omitempty"`
	ReadBytes      int64                `json:"read_bytes,omitempty"`
	WriteBytes     int64                `json:"write_bytes,omitempty"`
	Reason         string               `json:"reason,omitempty"`
	RestoreOutcome WarmDiskOutcome      `json:"restore_outcome,omitempty"`
	RestoreReason  string               `json:"restore_reason,omitempty"`
	BudgetBytes    int64                `json:"budget_bytes,omitempty"`
}

type warmDiskEnvelope struct {
	Version         int                   `json:"version"`
	Spec            WarmPrefixSpec        `json:"spec"`
	ModelIdentity   string                `json:"model_identity"`
	Tokens          []int                 `json:"tokens"`
	Logits          []float32             `json:"logits"`
	Snapshot        []byte                `json:"snapshot"`
	Codec           string                `json:"codec"`
	Execution       string                `json:"execution"`
	ExecutionPolicy model.ExecutionPolicy `json:"execution_policy"`
	DenseGPULayers  int                   `json:"dense_gpu_layers"`
	GPULayers       int                   `json:"gpu_layers"`
}

// SetWarmDiskConfig overrides local restart-cache placement or disables it.
func (p *InKernelPlanner) SetWarmDiskConfig(cfg WarmDiskConfig) {
	if p == nil {
		return
	}
	p.mu.Lock()
	p.warmDiskConfig = cfg
	p.mu.Unlock()
}

// SetWarmDiskModelDigest binds restart reuse to the load-time artifact identity.
// The legacy name is retained for callers; an empty identity disables disk reuse.
func (p *InKernelPlanner) SetWarmDiskModelDigest(identity string) {
	if p == nil {
		return
	}
	p.mu.Lock()
	p.warmDiskModelIdentity = strings.TrimSpace(identity)
	p.mu.Unlock()
}

func (p *InKernelPlanner) warmDiskState() (WarmDiskConfig, string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.warmDiskConfig, p.warmDiskModelIdentity
}

func resolveWarmDiskConfig(cfg WarmDiskConfig) (WarmDiskConfig, error) {
	cfg = resolveWarmDiskBounds(cfg)
	if strings.TrimSpace(cfg.Dir) == "" {
		if root := strings.TrimSpace(os.Getenv(l3kv.EnvSpec)); strings.EqualFold(root, "off") {
			cfg.Disabled = true
			return cfg, nil
		} else if root != "" {
			cfg.Dir = filepath.Join(root, "native-prefix")
		} else {
			root, err := os.UserCacheDir()
			if err != nil {
				return cfg, err
			}
			cfg.Dir = filepath.Join(root, "fak", "native-prefix")
		}
	}
	return cfg, nil
}

func resolveWarmDiskBounds(cfg WarmDiskConfig) WarmDiskConfig {
	if cfg.MaxEntries <= 0 {
		cfg.MaxEntries = defaultWarmDiskEntries
	}
	if cfg.MaxBytes <= 0 {
		cfg.MaxBytes = defaultWarmDiskBytes
	}
	_, free, known := compute.HostSystemMemoryInfo()
	cfg.MaxRecordBytes = resolveWarmDiskRecordBudget(cfg.MaxRecordBytes, cfg.MaxBytes, free, known)
	return cfg
}

func newWarmDiskReceipt(cfg WarmDiskConfig, outcome WarmDiskOutcome, reason string) *WarmDiskReceipt {
	return &WarmDiskReceipt{Outcome: outcome, Tier: warmDiskTier, Reason: reason, BudgetBytes: cfg.MaxRecordBytes}
}

// Warm-disk write-admission backoff ladder (adapted from the TensorFold
// learned_disk.zig Admission reserve/backoff): each refusal doubles the window
// from a 1 s base, capped at 60 s once the ladder reaches its 7th rung; any
// admitted reservation or landed write clears the ladder.
const (
	warmDiskWriteBackoffBase    = time.Second
	warmDiskWriteBackoffCap     = 60 * time.Second
	warmDiskWriteBackoffMaxFail = 7
	warmDiskWriteDrainPoll      = 5 * time.Millisecond
)

// Closed refusal contract for warm-disk write admission; callers match with errors.Is.
var (
	errWarmDiskSpaceUnknown    = errors.New("warm disk write: free space unknown")
	errWarmDiskBudgetExhausted = errors.New("warm disk write: budget exhausted")
	errWarmDiskBackoffActive   = errors.New("warm disk write: refusal backoff active")
)

// warmDiskWriteAdmission is the byte-permit pool that gates durable warm-disk
// writes: staging reserves bytes up front, an unmeasured limit or an
// unaffordable request refuses fail-closed, refusals arm a bounded exponential
// backoff, finish releases the reservation, and a landed write clears the
// ladder. Incomplete records never publish because the store commits only via
// temp+fsync+rename.
type warmDiskWriteAdmission struct {
	mu        sync.Mutex
	limit     int64 // reservable bytes measured at first use
	known     bool  // false when the volume's free space could not be measured
	reserved  int64 // bytes held by in-flight staging
	fails     int   // consecutive refusals driving the backoff ladder
	backoff   time.Duration
	refusedAt time.Time
	now       func() time.Time
}

func newWarmDiskWriteAdmission(limit int64, known bool, now func() time.Time) *warmDiskWriteAdmission {
	if now == nil {
		now = time.Now
	}
	return &warmDiskWriteAdmission{limit: limit, known: known, now: now}
}

// nextWarmDiskWriteBackoff is the refusal window after the n-th consecutive
// failure (n >= 1): 1 s doubled per failure, capped at 60 s at the 7th rung.
func nextWarmDiskWriteBackoff(fails int) time.Duration {
	if fails <= 0 {
		return 0
	}
	rung := fails - 1
	if rung >= warmDiskWriteBackoffMaxFail-1 {
		return warmDiskWriteBackoffCap
	}
	d := warmDiskWriteBackoffBase << uint(rung)
	if d <= 0 || d > warmDiskWriteBackoffCap {
		return warmDiskWriteBackoffCap
	}
	return d
}

// reserve admits bytes into the pool before staging. It refuses when the limit
// is unknown, when the request overflows or exceeds the remaining pool, or
// while a refusal backoff window is still open; every refusal arms the next
// window.
func (a *warmDiskWriteAdmission) reserve(bytes int64) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if !a.known {
		return errWarmDiskSpaceUnknown
	}
	if a.fails > 0 && a.now().Before(a.refusedAt.Add(a.backoff)) {
		return errWarmDiskBackoffActive
	}
	if bytes < 0 || bytes > a.limit-a.reserved {
		a.armRefusalLocked()
		return errWarmDiskBudgetExhausted
	}
	a.reserved += bytes
	a.clearLadderLocked()
	return nil
}

// release returns a finished staging's bytes to the pool regardless of outcome.
func (a *warmDiskWriteAdmission) release(bytes int64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if bytes > 0 {
		a.reserved -= bytes
		if a.reserved < 0 {
			a.reserved = 0
		}
	}
}

// clear drops the refusal ladder after a write lands.
func (a *warmDiskWriteAdmission) clear() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.clearLadderLocked()
}

// drain reports whether every in-flight reservation finished before the
// shutdown deadline; it polls until the pool empties or the deadline passes.
func (a *warmDiskWriteAdmission) drain(deadline time.Time) bool {
	for {
		a.mu.Lock()
		pending := a.reserved
		a.mu.Unlock()
		if pending == 0 {
			return true
		}
		if !a.now().Before(deadline) {
			return false
		}
		time.Sleep(warmDiskWriteDrainPoll)
	}
}

// pending reports the bytes currently reserved by in-flight staging.
func (a *warmDiskWriteAdmission) pending() int64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.reserved
}

func (a *warmDiskWriteAdmission) armRefusalLocked() {
	if a.fails < warmDiskWriteBackoffMaxFail {
		a.fails++
	}
	a.backoff = nextWarmDiskWriteBackoff(a.fails)
	a.refusedAt = a.now()
}

func (a *warmDiskWriteAdmission) clearLadderLocked() {
	a.fails = 0
	a.backoff = 0
}

// warmDiskWriteGates holds one admission pool per resolved store directory:
// every planner writing that store shares the same permit pool, so concurrent
// staging cannot collectively overrun the measured budget. Entries live for
// the process; each is a few words per directory.
var warmDiskWriteGates struct {
	sync.Mutex
	m map[string]*warmDiskWriteAdmission
}

// warmDiskDiskInfo probes a store volume's free bytes; production measures the
// real filesystem, tests simulate volumes. A nil probe fails closed as unknown.
var warmDiskDiskInfo = compute.DiskInfo

// warmDiskWritePoolLimit sizes the pool from the store cap and measured free
// bytes; unknown or non-positive inputs stay unknown (the fail-closed arm).
func warmDiskWritePoolLimit(maxBytes, free int64, known bool) int64 {
	if !known || maxBytes <= 0 || free <= 0 {
		return 0
	}
	if free < maxBytes {
		return free
	}
	return maxBytes
}

// warmDiskWriteGateFor resolves (creating on first use) the admission pool
// guarding dir's durable writes: the smaller of the store's byte cap and the
// volume's measured free bytes.
func warmDiskWriteGateFor(dir string, maxBytes int64) *warmDiskWriteAdmission {
	warmDiskWriteGates.Lock()
	defer warmDiskWriteGates.Unlock()
	if ad, ok := warmDiskWriteGates.m[dir]; ok {
		return ad
	}
	var free int64
	var known bool
	if warmDiskDiskInfo != nil {
		_, free, known = warmDiskDiskInfo(dir)
	}
	ad := newWarmDiskWriteAdmission(warmDiskWritePoolLimit(maxBytes, free, known), known && maxBytes > 0, time.Now)
	if warmDiskWriteGates.m == nil {
		warmDiskWriteGates.m = map[string]*warmDiskWriteAdmission{}
	}
	warmDiskWriteGates.m[dir] = ad
	return ad
}

// warmDiskWriteGate resolves this planner's durable-write admission pool from
// its configured store. Callers must not hold p.mu; nil means no gating.
func (p *InKernelPlanner) warmDiskWriteGate() *warmDiskWriteAdmission {
	if p == nil {
		return nil
	}
	cfg, _ := p.warmDiskState()
	resolved, err := resolveWarmDiskConfig(cfg)
	if err != nil || resolved.Disabled || strings.TrimSpace(resolved.Dir) == "" {
		return nil
	}
	return warmDiskWriteGateFor(resolved.Dir, resolved.MaxBytes)
}

func resolveWarmDiskRecordBudget(requested, maxBytes, freeBytes int64, known bool) int64 {
	headroom := defaultWarmDiskRecordBytes
	if known {
		headroom = freeBytes / 8
		if headroom < 1 {
			headroom = 1
		}
		if headroom > maxWarmDiskRecordBytes {
			headroom = maxWarmDiskRecordBytes
		}
	}
	budget := requested
	if budget <= 0 || budget > headroom {
		budget = headroom
	}
	if maxBytes > 0 && budget > maxBytes {
		budget = maxBytes
	}
	return budget
}

func (p *InKernelPlanner) warmDiskExecutionIdentity() string {
	backend := "cpu"
	if p.backend != nil {
		backend = p.backend.Name()
	}
	precision := p.kvPrecision
	if precision == "" {
		precision = model.KVPrecisionFP32
	}
	identity := strings.Join([]string{
		"prefix-runtime-v1", runtime.GOOS, runtime.GOARCH, backend,
		strconv.FormatBool(p.metal), strconv.FormatBool(p.q4k), strconv.FormatBool(p.quant),
		strconv.FormatBool(p.requireDeviceExecution), strconv.Itoa(p.denseGPULayers),
		strconv.FormatBool(p.qwen35MetalGDNSequence), strconv.FormatBool(p.q4kGateUpOutputSlab),
		string(precision), strconv.FormatBool(p.cpuOffloadExperts),
	}, ":")
	if p.deepSeekV41ChatProfile != "" {
		identity += ":chat-profile:" + p.deepSeekV41ChatProfile
	}
	return identity
}

func warmDiskKey(spec WarmPrefixSpec, modelIdentity, executionIdentity string) string {
	sum := sha256.Sum256([]byte("fak-native-prefix/v1\x00" + spec.Identity + "\x00" + modelIdentity + "\x00" + executionIdentity))
	return hex.EncodeToString(sum[:])
}

// warmDiskSlotKey names a prefix slot independent of the model artifact, so a
// restart can tell "never persisted" from "persisted for another incarnation".
func warmDiskSlotKey(spec WarmPrefixSpec, executionIdentity string) string {
	sum := sha256.Sum256([]byte("fak-native-prefix-slot/v1\x00" + spec.Identity + "\x00" + executionIdentity))
	return hex.EncodeToString(sum[:])
}

func openWarmDiskSlots(cfg WarmDiskConfig) (*l3kv.DiskStore, error) {
	return l3kv.NewDiskStore(filepath.Join(cfg.Dir, warmDiskSlotsDir))
}

// warmDiskModelIdentityChanged reports whether the slot was last persisted under
// a different model identity. Any read failure is treated as no evidence.
func warmDiskModelIdentityChanged(ctx context.Context, cfg WarmDiskConfig, spec WarmPrefixSpec, identity, execution string) bool {
	slots, err := openWarmDiskSlots(cfg)
	if err != nil {
		return false
	}
	prior, found, err := slots.GetBounded(ctx, warmDiskSlotKey(spec, execution), maxWarmDiskSlotBytes)
	return err == nil && found && string(prior) != identity
}

// recordWarmDiskSlot is best-effort: the slot only sharpens a miss reason.
func recordWarmDiskSlot(ctx context.Context, cfg WarmDiskConfig, spec WarmPrefixSpec, identity, execution string) {
	if len(identity) > maxWarmDiskSlotBytes {
		return
	}
	slots, err := openWarmDiskSlots(cfg)
	if err != nil {
		return
	}
	key := warmDiskSlotKey(spec, execution)
	if slots.Put(ctx, key, []byte(identity)) == nil {
		_ = slots.PruneContext(ctx, maxWarmDiskSlots, 0, key)
	}
}

func (p *InKernelPlanner) openWarmDisk() (*l3kv.DiskStore, WarmDiskConfig, string, *WarmDiskReceipt) {
	cfg, identity := p.warmDiskState()
	if cfg.Disabled {
		cfg = resolveWarmDiskBounds(cfg)
		return nil, cfg, identity, newWarmDiskReceipt(cfg, WarmDiskOutcomeDisabled, "disabled")
	}
	if identity == "" {
		cfg = resolveWarmDiskBounds(cfg)
		return nil, cfg, identity, newWarmDiskReceipt(cfg, WarmDiskOutcomeUnsupported, "model_identity_unavailable")
	}
	resolved, err := resolveWarmDiskConfig(cfg)
	if err != nil {
		return nil, resolved, identity, newWarmDiskReceipt(resolved, WarmDiskOutcomeFault, "cache_dir_unavailable")
	}
	if resolved.Disabled {
		return nil, resolved, identity, newWarmDiskReceipt(resolved, WarmDiskOutcomeDisabled, "disabled")
	}
	store, err := l3kv.NewDiskStore(resolved.Dir)
	if err != nil {
		return nil, resolved, identity, newWarmDiskReceipt(resolved, WarmDiskOutcomeFault, "store_open_failed")
	}
	return store, resolved, identity, nil
}

func (p *InKernelPlanner) restoreWarmPrefixDisk(ctx context.Context, spec WarmPrefixSpec, tokens []int) *WarmDiskReceipt {
	store, cfg, identity, result := p.openWarmDisk()
	if result != nil {
		result.RestoreOutcome, result.RestoreReason = result.Outcome, result.Reason
		return result
	}
	execution := p.warmDiskExecutionIdentity()
	key := warmDiskKey(spec, identity, execution)
	payload, found, err := store.GetBounded(ctx, key, cfg.MaxRecordBytes)
	if err != nil {
		return &WarmDiskReceipt{Outcome: WarmDiskOutcomeFault, Tier: warmDiskTier, Reason: "read_failed", RestoreOutcome: WarmDiskOutcomeFault, RestoreReason: "read_failed", BudgetBytes: cfg.MaxRecordBytes}
	}
	if !found {
		reason := "absent"
		if warmDiskModelIdentityChanged(ctx, cfg, spec, identity, execution) {
			reason = "model_identity_changed"
		}
		return &WarmDiskReceipt{Outcome: WarmDiskOutcomeMiss, Tier: warmDiskTier, Reason: reason, RestoreOutcome: WarmDiskOutcomeMiss, RestoreReason: reason, BudgetBytes: cfg.MaxRecordBytes}
	}
	receipt := &WarmDiskReceipt{Outcome: WarmDiskOutcomeFault, Tier: warmDiskTier, ReadBytes: int64(len(payload)), RestoreOutcome: WarmDiskOutcomeFault, BudgetBytes: cfg.MaxRecordBytes}
	defer func() {
		if receipt.RestoreReason == "" && receipt.RestoreOutcome != WarmDiskOutcomeRestored {
			receipt.RestoreReason = receipt.Reason
		}
	}()
	var envelope warmDiskEnvelope
	if json.Unmarshal(payload, &envelope) != nil {
		receipt.Reason = "envelope_invalid"
		return receipt
	}
	if envelope.Version != warmDiskEnvelopeVersion || envelope.Spec != spec || envelope.ModelIdentity != identity || envelope.Execution != execution || !slices.Equal(envelope.Tokens, tokens) {
		receipt.Reason = "identity_mismatch"
		return receipt
	}
	if err := ctx.Err(); err != nil {
		receipt.Reason = "cancelled"
		return receipt
	}
	if len(envelope.Logits) != p.m.Cfg.VocabSize {
		receipt.Reason = "logits_shape_mismatch"
		return receipt
	}
	var snap *model.PrefixSnapshot
	err = nil
	switch envelope.Codec {
	case "cpu-v1":
		snap, err = model.DecodeCPUPrefixSnapshot(envelope.Snapshot, p.m.Cfg)
	case "host-v1":
		var host *model.HostPrefixSnapshot
		host, err = model.DecodeHostPrefixSnapshot(envelope.Snapshot, p.backend, p.m.Cfg)
		if err == nil {
			snap, err = host.Restore()
			host.Close()
		}
	default:
		err = errors.New("unknown snapshot codec")
	}
	if err != nil {
		receipt.Reason = "snapshot_decode_failed"
		return receipt
	}
	if envelope.ExecutionPolicy > model.ExecutionPolicyDeviceOnly || envelope.DenseGPULayers < 0 || envelope.GPULayers < 0 ||
		envelope.DenseGPULayers > p.m.Cfg.NumLayers || envelope.GPULayers > p.m.Cfg.NumLayers {
		snap.Close()
		receipt.Reason = "snapshot_policy_invalid"
		return receipt
	}
	snap.ExecutionPolicy = envelope.ExecutionPolicy
	snap.DenseGPULayers = envelope.DenseGPULayers
	snap.GPULayers = envelope.GPULayers
	if snap.Tokens != len(tokens) {
		snap.Close()
		receipt.Reason = "snapshot_token_mismatch"
		return receipt
	}
	if err := p.admitWarmDiskSnapshot(spec.Scope, tokens, snap, envelope.Logits); err != nil {
		snap.Close()
		receipt.Reason = "snapshot_admission_failed"
		return receipt
	}
	receipt.Outcome = WarmDiskOutcomeRestored
	receipt.RestoreOutcome = WarmDiskOutcomeRestored
	receipt.Reason = ""
	return receipt
}

func (p *InKernelPlanner) admitWarmDiskSnapshot(scope radixkv.CacheIdentity, tokens []int, snap *model.PrefixSnapshot, logits []float32) error {
	p.mu.Lock()
	scopedTree, tree := p.scopedTree, p.tree
	p.mu.Unlock()
	// Ordinary host demand restores the historical bare-KV tier. Keep complete
	// PrefixSnapshot admission for device and recurrent routes, whose continuation
	// state is larger than KV.
	if p.backend == nil && !p.hostV41CompleteSnapshot() {
		defer snap.Close()
		if snap.Cache == nil || snap.Cache.Len() != len(tokens) {
			return errors.New("disk snapshot has no complete host cache")
		}
		if scope.Tenant != "" && scopedTree != nil {
			return scopedTree.AdmitPrivate(scope, tokens, snap.Cache, logits)
		}
		p.mu.Lock()
		defer p.mu.Unlock()
		boundary, matched := tree.Lookup(tokens)
		leaf := tree.InsertCloneWithLogits(boundary, tokens[matched:], snap.Cache, logits)
		if leaf != nil {
			tree.Done(leaf)
		}
		return nil
	}
	if scope.Tenant != "" && scopedTree != nil {
		return scopedTree.AdmitPrivateSnapshot(scope, tokens, snap, logits)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	boundary, matched := tree.Lookup(tokens)
	leaf, err := tree.InsertSnapshot(boundary, tokens[matched:], snap, logits)
	if leaf != nil {
		tree.Done(leaf)
	}
	return err
}

// Conversation-state staging bounds, adapted from TensorFold's
// --learn-conversations store (zig/src/sessions/store.zig and docs/api.md,
// "Conversations on disk", at revision
// 9e8c368b0b4c2c7892a0c9f0f302d7c164016209): the engine stages an owned copy
// between rounds under a byte permit and a job cap, an I/O thread writes and
// fsyncs it, and a planned stop drains for 20 seconds before cancelling.
const (
	// ConversationStatePermitDefault caps the staged bytes held queued or in
	// flight at once (upstream: 1 GiB).
	ConversationStatePermitDefault int64 = 1 << 30
	// ConversationStateJobsDefault caps saves queued plus in flight (upstream: 8).
	ConversationStateJobsDefault = 8
	// ConversationStateStopDrain is the planned-stop deadline: wait this long
	// for queued saves to publish, then cancel the remainder (upstream: 20 s).
	ConversationStateStopDrain = 20 * time.Second
)

var (
	// errConversationStateCapacityFull reports a save the staging queue cannot
	// take: it exceeds the byte permit even alone, or still does not fit after
	// one queued write finished. Skipped, never written.
	errConversationStateCapacityFull = errors.New("conversation-state staging capacity full")
	// errConversationStateClosed reports a save offered after the planned-stop
	// drain closed the queue.
	errConversationStateClosed = errors.New("conversation-state staging closed")
	// errConversationStateCancelled is delivered on a queued save's done channel
	// when the planned-stop drain cancels it before its write starts.
	errConversationStateCancelled = errors.New("conversation-state save cancelled before publish")
)

// warmDiskRecordWriter is the durability seam the conversation-state stager
// publishes through. *l3kv.DiskStore satisfies it: Put commits via a temp file
// that is fsync'd and atomically renamed, so a record is reusable only after
// all of its files are written and synced.
type warmDiskRecordWriter interface {
	Put(ctx context.Context, key string, payload []byte) error
}

// warmDiskStagedSave is one encoded conversation-state record awaiting durable
// write. The payload is an owned copy: the RAM state may leave while it waits,
// and a cancelled save simply loses its copy.
type warmDiskStagedSave struct {
	key     string
	payload []byte
	done    chan error // one slot: nil published, sentinel when cancelled
}

// warmDiskStager is the bounded I/O queue behind opt-in conversation-state
// persistence. One writer goroutine (the I/O thread) takes saves FIFO and hands
// each to the record writer; a save becomes visible to readers only after that
// writer has committed it. Staging charges the byte permit for queued plus
// in-flight payloads; a save that finds the capacity full but would fit alone
// waits for the next write to finish, and is skipped if it still does not fit.
// A killed process loses unfinished saves (best effort): nothing half-written
// is ever published, and the private temp files are reaped by PruneContext.
type warmDiskStager struct {
	mu       sync.Mutex
	cond     *sync.Cond
	store    warmDiskRecordWriter
	queue    []warmDiskStagedSave
	waiters  []chan struct{}
	charged  int64 // queued + in-flight payload bytes
	inflight int
	permit   int64
	maxJobs  int
	closed   bool
	written  int
	failed   int
	stopped  chan struct{}
}

func newWarmDiskStager(store warmDiskRecordWriter, permit int64, maxJobs int) *warmDiskStager {
	if permit <= 0 {
		permit = ConversationStatePermitDefault
	}
	if maxJobs <= 0 {
		maxJobs = ConversationStateJobsDefault
	}
	s := &warmDiskStager{store: store, permit: permit, maxJobs: maxJobs, stopped: make(chan struct{})}
	s.cond = sync.NewCond(&s.mu)
	go s.run()
	return s
}

// stage enqueues payload under key, charging the permit. It reports
// errConversationStateCapacityFull for a save that cannot be taken (skipped),
// errConversationStateClosed after the drain, or ctx.Err() when the caller's
// context ends while waiting for room. The returned save's done channel
// delivers nil once the record is published or the cancellation sentinel.
func (s *warmDiskStager) stage(ctx context.Context, key string, payload []byte) (*warmDiskStagedSave, error) {
	size := int64(len(payload))
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, errConversationStateClosed
	}
	if s.fitsLocked(size) {
		save := s.enqueueLocked(key, payload, size)
		s.mu.Unlock()
		return save, nil
	}
	if size > s.permit {
		// Cannot fit even alone: skipped, never written.
		s.mu.Unlock()
		return nil, errConversationStateCapacityFull
	}
	waiter := make(chan struct{})
	s.waiters = append(s.waiters, waiter)
	s.mu.Unlock()
	select {
	case <-waiter:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, errConversationStateClosed
	}
	if !s.fitsLocked(size) {
		return nil, errConversationStateCapacityFull
	}
	return s.enqueueLocked(key, payload, size), nil
}

// fitsLocked reports whether one more save of size bytes fits under the job cap
// and the byte permit.
func (s *warmDiskStager) fitsLocked(size int64) bool {
	return len(s.queue)+s.inflight < s.maxJobs && s.charged+size <= s.permit
}

func (s *warmDiskStager) enqueueLocked(key string, payload []byte, size int64) *warmDiskStagedSave {
	save := &warmDiskStagedSave{key: key, payload: payload, done: make(chan error, 1)}
	s.queue = append(s.queue, *save)
	s.charged += size
	s.cond.Signal()
	return save
}

// run is the I/O thread: it takes queued saves FIFO and publishes each through
// the record writer. It exits only after the drain closes the queue; a write
// already in flight may finish after the deadline using only its private temp
// file and staged bytes.
func (s *warmDiskStager) run() {
	defer close(s.stopped)
	for {
		s.mu.Lock()
		for len(s.queue) == 0 && !s.closed {
			s.cond.Wait()
		}
		if len(s.queue) == 0 {
			s.mu.Unlock()
			return
		}
		save := s.queue[0]
		s.queue = s.queue[1:]
		s.inflight++
		s.mu.Unlock()
		err := s.store.Put(context.Background(), save.key, save.payload)
		s.mu.Lock()
		s.inflight--
		s.charged -= int64(len(save.payload))
		if err != nil {
			s.failed++
		} else {
			s.written++
		}
		s.broadcastLocked()
		s.mu.Unlock()
		save.done <- err // nil when published
	}
}

// broadcastLocked wakes every stage caller waiting for room.
func (s *warmDiskStager) broadcastLocked() {
	for _, w := range s.waiters {
		close(w)
	}
	s.waiters = nil
}

// drain runs the planned-stop protocol: keep publishing until the queue is
// empty or the deadline passes, then close the queue and cancel whatever is
// still queued without writing it. It reports saves published during the drain
// and saves cancelled.
func (s *warmDiskStager) drain(deadline time.Duration) (published, cancelled int) {
	s.mu.Lock()
	start := s.written
	alreadyClosed := s.closed
	s.mu.Unlock()
	if !alreadyClosed {
		deadlineAt := time.Now().Add(deadline)
		for {
			s.mu.Lock()
			empty := len(s.queue) == 0 && s.inflight == 0
			s.mu.Unlock()
			if empty || time.Now().After(deadlineAt) {
				break
			}
			time.Sleep(2 * time.Millisecond)
		}
	}
	s.mu.Lock()
	for i := range s.queue {
		s.queue[i].done <- errConversationStateCancelled
	}
	cancelled = len(s.queue)
	s.queue = nil
	s.charged = 0
	s.closed = true
	s.broadcastLocked()
	s.cond.Signal()
	waitWriter := s.inflight == 0
	s.mu.Unlock()
	if waitWriter {
		<-s.stopped // the writer exits promptly once the queue is empty and closed
	}
	// A write still in flight past the deadline may finish later using only its
	// private temp file and staged bytes; it never blocks the stop.
	s.mu.Lock()
	published = s.written - start
	s.mu.Unlock()
	return published, cancelled
}

// warmDiskStagerByDir binds each conversation-state store directory to its
// single I/O queue for the process lifetime. The queue deliberately outlives
// any one planner: a restarted server re-opens the same directory and must
// keep publishing behind the same byte permit, and the planned-stop drain
// closes it explicitly. Bounded by the number of distinct store directories a
// process ever persists into.
var (
	warmDiskStagerMu    sync.Mutex
	warmDiskStagerByDir = map[string]*warmDiskStager{}
)

// conversationStager returns the I/O queue bound to the resolved store
// directory, creating it on first use with the store first opened for
// persistence.
func conversationStager(dir string, store *l3kv.DiskStore, permit int64, maxJobs int) *warmDiskStager {
	warmDiskStagerMu.Lock()
	defer warmDiskStagerMu.Unlock()
	if s, ok := warmDiskStagerByDir[dir]; ok {
		return s
	}
	s := newWarmDiskStager(store, permit, maxJobs)
	warmDiskStagerByDir[dir] = s
	return s
}

// DrainConversationState runs the planned-stop protocol for opt-in
// conversation-state staging: wait up to deadline (default
// ConversationStateStopDrain) for queued saves to publish, then cancel the
// remainder. It reports saves published during the drain and saves cancelled.
// A killed process instead loses unfinished saves outright; the stale temp
// files stay private and are reaped by PruneContext.
func (p *InKernelPlanner) DrainConversationState(deadline time.Duration) (published, cancelled int) {
	if p == nil {
		return 0, 0
	}
	p.mu.Lock()
	cfg := p.warmDiskConfig
	p.mu.Unlock()
	if !cfg.ConversationState || cfg.Disabled {
		return 0, 0
	}
	resolved, err := resolveWarmDiskConfig(cfg)
	if err != nil {
		return 0, 0
	}
	warmDiskStagerMu.Lock()
	stager := warmDiskStagerByDir[resolved.Dir]
	warmDiskStagerMu.Unlock()
	if stager == nil {
		return 0, 0
	}
	if deadline <= 0 {
		deadline = ConversationStateStopDrain
	}
	return stager.drain(deadline)
}

func (p *InKernelPlanner) persistWarmPrefixDisk(ctx context.Context, spec WarmPrefixSpec, tokens []int, prior *WarmDiskReceipt) *WarmDiskReceipt {
	store, cfg, identity, result := p.openWarmDisk()
	if result != nil {
		return result
	}
	snap, logits, matched, err := p.warmDiskSnapshot(spec.Scope, tokens)
	if err != nil || snap == nil || matched != len(tokens) {
		return newWarmDiskReceipt(cfg, WarmDiskOutcomeUnsupported, "complete_snapshot_unavailable")
	}
	defer snap.Close()
	if len(logits) != p.m.Cfg.VocabSize || snap.Tokens != len(tokens) {
		return newWarmDiskReceipt(cfg, WarmDiskOutcomeUnsupported, "snapshot_shape_mismatch")
	}
	resident := snap.ResidentBytes()
	estimated := resident + resident/2 + int64(len(logits))*4 + int64(len(tokens))*12 + (1 << 20)
	if resident <= 0 || estimated > cfg.MaxRecordBytes {
		return newWarmDiskReceipt(cfg, WarmDiskOutcomeUnsupported, "record_exceeds_budget")
	}
	if !snap.HostDiskSerializable() {
		return newWarmDiskReceipt(cfg, WarmDiskOutcomeUnsupported, "snapshot_state_unserializable")
	}
	codec := "cpu-v1"
	wire, err := snap.MarshalCPU()
	if err != nil {
		codec = "host-v1"
		var host *model.HostPrefixSnapshot
		host, err = snap.CloneToHost()
		if err == nil {
			wire, err = host.MarshalBinary()
			host.Close()
		}
	}
	if err != nil {
		return newWarmDiskReceipt(cfg, WarmDiskOutcomeUnsupported, "complete_snapshot_unserializable")
	}
	payload, err := json.Marshal(warmDiskEnvelope{
		Version: warmDiskEnvelopeVersion, Spec: spec, ModelIdentity: identity,
		Tokens: append([]int(nil), tokens...), Logits: append([]float32(nil), logits...), Snapshot: wire, Codec: codec, Execution: p.warmDiskExecutionIdentity(),
		ExecutionPolicy: snap.ExecutionPolicy, DenseGPULayers: snap.DenseGPULayers, GPULayers: snap.GPULayers,
	})
	if err != nil {
		return newWarmDiskReceipt(cfg, WarmDiskOutcomeFault, "envelope_encode_failed")
	}
	if int64(len(payload)) > cfg.MaxRecordBytes || l3kv.RecordBytes(int64(len(payload))) > cfg.MaxBytes {
		return newWarmDiskReceipt(cfg, WarmDiskOutcomeUnsupported, "record_exceeds_budget")
	}
	if err := ctx.Err(); err != nil {
		return newWarmDiskReceipt(cfg, WarmDiskOutcomeFault, "cancelled")
	}
	execution := p.warmDiskExecutionIdentity()
	key := warmDiskKey(spec, identity, execution)
	var staged bool
	var writeGate *warmDiskWriteAdmission
	if cfg.ConversationState {
		// Opt-in conversation-state staging: hand the owned encoded copy to the
		// I/O queue; the record becomes reusable only after the writer has
		// written and synced every file. The stager's own byte permit governs
		// these saves, so no per-write gate reservation is taken here: ownership
		// of the payload outlives this call.
		if _, err := conversationStager(cfg.Dir, store, cfg.ConversationStatePermit, cfg.ConversationStateMaxJobs).stage(ctx, key, payload); err != nil {
			switch {
			case errors.Is(err, errConversationStateCapacityFull):
				return newWarmDiskReceipt(cfg, WarmDiskOutcomeUnsupported, "staging_capacity_full")
			case errors.Is(err, errConversationStateClosed):
				return newWarmDiskReceipt(cfg, WarmDiskOutcomeFault, "staging_closed")
			default:
				return newWarmDiskReceipt(cfg, WarmDiskOutcomeFault, "cancelled")
			}
		}
		staged = true
	} else {
		// Admit the write against the store's measured byte pool before staging: an
		// unaffordable record is skipped, not faulted. When this planner already
		// holds the capture reservation covering the payload (beginWarmDiskCapture
		// reserves MaxRecordBytes, and the checks above bound the payload to it),
		// the durable write completes under that permit and finish
		// (endWarmDiskCapture) releases it; otherwise this write owns its own
		// reservation until it finishes.
		writeGate = p.warmDiskWriteGate()
		if writeGate != nil {
			p.mu.Lock()
			covered := p.warmDiskCapture && p.warmDiskCaptureBudget >= int64(len(payload))
			p.mu.Unlock()
			if !covered {
				if err := writeGate.reserve(int64(len(payload))); err != nil {
					return newWarmDiskReceipt(cfg, WarmDiskOutcomeUnsupported, "write_budget_refused")
				}
				defer writeGate.release(int64(len(payload)))
			}
		}
		if err := store.Put(ctx, key, payload); err != nil {
			return newWarmDiskReceipt(cfg, WarmDiskOutcomeFault, "write_failed")
		}
	}
	recordWarmDiskSlot(ctx, cfg, spec, identity, execution)
	var outcome WarmDiskOutcome = WarmDiskOutcomePersisted
	if staged {
		outcome = WarmDiskOutcomeStaged
	}
	receipt := &WarmDiskReceipt{Outcome: outcome, Tier: warmDiskTier, WriteBytes: int64(len(payload)), BudgetBytes: cfg.MaxRecordBytes}
	if prior != nil {
		receipt.ReadBytes = prior.ReadBytes
		receipt.RestoreOutcome = prior.RestoreOutcome
		receipt.RestoreReason = prior.RestoreReason
	}
	if err := store.PruneContext(ctx, cfg.MaxEntries, cfg.MaxBytes, key); err != nil {
		receipt.Outcome = WarmDiskOutcomeFault
		receipt.Reason = "prune_failed"
	} else if writeGate != nil {
		writeGate.clear() // a landed write clears the refusal ladder
	}
	return receipt
}

func (p *InKernelPlanner) warmDiskSnapshot(scope radixkv.CacheIdentity, tokens []int) (*model.PrefixSnapshot, []float32, int, error) {
	if snap, logits, ok := p.takeWarmDiskCapture(scope, tokens); ok {
		return snap, logits, len(tokens), nil
	}
	p.mu.Lock()
	scopedTree, tree := p.scopedTree, p.tree
	p.mu.Unlock()
	if scope.Tenant != "" && scopedTree != nil {
		snap, logits, matched, _, _, err := scopedTree.LookupSnapshotTiered(scope, tokens)
		return snap, logits, matched, err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	node, snap, matched, _, err := tree.LookupSnapshotTieredContextLocked(context.Background(), tokens, &p.mu)
	var logits []float32
	if node != nil {
		logits = node.Logits()
		tree.Done(node)
	}
	return snap, logits, matched, err
}

func warmDiskCaptureEligible(receipt *WarmDiskReceipt) bool {
	if receipt == nil || receipt.BudgetBytes <= 0 {
		return false
	}
	if receipt.Outcome == WarmDiskOutcomeMiss {
		return true
	}
	if receipt.Outcome != WarmDiskOutcomeFault {
		return false
	}
	switch receipt.Reason {
	case "cache_dir_unavailable", "store_open_failed", "cancelled":
		return false
	default:
		return true
	}
}

func (p *InKernelPlanner) beginWarmDiskCapture(budget int64) {
	gate := p.warmDiskWriteGate()
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.warmDiskCaptured != nil {
		p.warmDiskCaptured.Close()
	}
	p.warmDiskCapture = true
	p.warmDiskCaptureBudget = budget
	p.warmDiskCaptured = nil
	p.warmDiskCapturedTokens = nil
	p.warmDiskCapturedLogits = nil
	p.warmDiskCapturedScope = radixkv.CacheIdentity{}
	if gate != nil && budget > 0 {
		// Reserve the capture's byte budget before staging: an unaffordable or
		// backoff-gated capture is skipped instead of risking an over-budget record.
		if err := gate.reserve(budget); err != nil {
			p.warmDiskCapture = false
			p.warmDiskCaptureBudget = 0
		}
	}
}

func (p *InKernelPlanner) endWarmDiskCapture() {
	p.mu.Lock()
	capturing := p.warmDiskCapture
	budget := p.warmDiskCaptureBudget
	p.warmDiskCapture = false
	p.warmDiskCaptureBudget = 0
	if p.warmDiskCaptured != nil {
		p.warmDiskCaptured.Close()
	}
	p.warmDiskCaptured = nil
	p.warmDiskCapturedTokens = nil
	p.warmDiskCapturedLogits = nil
	p.mu.Unlock()
	if capturing && budget > 0 {
		if gate := p.warmDiskWriteGate(); gate != nil {
			gate.release(budget) // finish releases the reservation
		}
	}
}

func (p *InKernelPlanner) captureWarmDiskCandidate(ctx context.Context, session *model.Session, tokens []int, logits []float32) {
	p.mu.Lock()
	enabled := p.warmDiskCapture
	budget := p.warmDiskCaptureBudget
	p.mu.Unlock()
	if !enabled || session == nil || session.Cache == nil {
		return
	}
	resident := session.Cache.OwnedPayloadBytes()
	estimated := resident + resident/2 + int64(len(logits))*4 + int64(len(tokens))*12 + (1 << 20)
	if resident <= 0 || estimated > budget {
		return
	}
	snap, err := session.PrefixSnapshot()
	if err != nil {
		return
	}
	scope, _ := prefixCacheIdentityFromContext(ctx)
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.warmDiskCapture {
		snap.Close()
		return
	}
	if p.warmDiskCaptured != nil {
		p.warmDiskCaptured.Close()
	}
	p.warmDiskCaptured = snap
	p.warmDiskCapturedTokens = append([]int(nil), tokens...)
	p.warmDiskCapturedLogits = append([]float32(nil), logits...)
	p.warmDiskCapturedScope = scope
}

func (p *InKernelPlanner) takeWarmDiskCapture(scope radixkv.CacheIdentity, tokens []int) (*model.PrefixSnapshot, []float32, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.warmDiskCaptured == nil || p.warmDiskCapturedScope != scope || !slices.Equal(p.warmDiskCapturedTokens, tokens) {
		return nil, nil, false
	}
	snap := p.warmDiskCaptured
	logits := p.warmDiskCapturedLogits
	p.warmDiskCaptured = nil
	p.warmDiskCapturedTokens = nil
	p.warmDiskCapturedLogits = nil
	return snap, logits, true
}
