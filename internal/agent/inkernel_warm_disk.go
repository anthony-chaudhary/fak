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

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

const (
	WarmDiskOutcomeRestored    WarmDiskOutcome = "restored"
	WarmDiskOutcomePersisted                   = "persisted"
	WarmDiskOutcomeMiss                        = "miss"
	WarmDiskOutcomeFault                       = "fault"
	WarmDiskOutcomeDisabled                    = "disabled"
	WarmDiskOutcomeUnsupported                 = "unsupported"
	WarmDiskRestored                           = WarmDiskOutcomeRestored
	WarmDiskPersisted                          = WarmDiskOutcomePersisted
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
	return strings.Join([]string{
		"prefix-runtime-v1", runtime.GOOS, runtime.GOARCH, backend,
		strconv.FormatBool(p.metal), strconv.FormatBool(p.q4k), strconv.FormatBool(p.quant),
		strconv.FormatBool(p.requireDeviceExecution), strconv.Itoa(p.denseGPULayers),
		strconv.FormatBool(p.qwen35MetalGDNSequence), strconv.FormatBool(p.q4kGateUpOutputSlab),
		string(precision), strconv.FormatBool(p.cpuOffloadExperts),
	}, ":")
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
	if err := store.Put(ctx, key, payload); err != nil {
		return newWarmDiskReceipt(cfg, WarmDiskOutcomeFault, "write_failed")
	}
	recordWarmDiskSlot(ctx, cfg, spec, identity, execution)
	receipt := &WarmDiskReceipt{Outcome: WarmDiskOutcomePersisted, Tier: warmDiskTier, WriteBytes: int64(len(payload)), BudgetBytes: cfg.MaxRecordBytes}
	if prior != nil {
		receipt.ReadBytes = prior.ReadBytes
		receipt.RestoreOutcome = prior.RestoreOutcome
		receipt.RestoreReason = prior.RestoreReason
	}
	if err := store.PruneContext(ctx, cfg.MaxEntries, cfg.MaxBytes, key); err != nil {
		receipt.Outcome = WarmDiskOutcomeFault
		receipt.Reason = "prune_failed"
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
	node, snap, matched, _, err := tree.LookupSnapshotTieredContext(context.Background(), tokens)
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
}

func (p *InKernelPlanner) endWarmDiskCapture() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.warmDiskCapture = false
	p.warmDiskCaptureBudget = 0
	if p.warmDiskCaptured != nil {
		p.warmDiskCaptured.Close()
	}
	p.warmDiskCaptured = nil
	p.warmDiskCapturedTokens = nil
	p.warmDiskCapturedLogits = nil
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
