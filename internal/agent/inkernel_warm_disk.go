package agent

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
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

	// Learned shared-prefix disk index layout (TensorFold imprint adaptation):
	// <root>/clock            global open counter word
	// <root>/<id>/            identity dir, id = learnedPrefixIdentity(scope)
	//   <record-key>          l3kv.DiskStore record files
	//   index                 JSON state index (learnedIndexFile)
	//   pending               in-flight write marker (learnedPendingFile)
	//   used                  last-open clock word (learnedUsedFile)
	learnedIndexFile      = "index"
	learnedPendingFile    = "pending"
	learnedUsedFile       = "used"
	learnedClockFile      = "clock"
	learnedIndexVersion   = 1
	learnedPendingMagic   = "FKNP"
	learnedPendingVersion = uint16(1)
	// maxLearnedIndexBytes sanity-caps reads of foreign/corrupt index files.
	maxLearnedIndexBytes = 1 << 20
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

var errLearnedPending = errors.New("learned prefix write already in flight")

// errLearnedSweepExhausted reports that the cross-identity sweep removed every
// evictable record and the byte cap still cannot hold the new record.
var errLearnedSweepExhausted = errors.New("learned prefix sweep: byte cap unsatisfiable")

// learnedIndexEntry is one durable state in an identity's index.
type learnedIndexEntry struct {
	Key   string `json:"key"`
	Bytes int64  `json:"bytes"`
	Used  int64  `json:"used"`
}

// learnedIndex is the per-identity state index kept at <root>/<id>/index.
type learnedIndex struct {
	Version int                 `json:"version"`
	States  []learnedIndexEntry `json:"states"`
}

// learnedPrefixStore is the identity-scoped view over the warm disk root: its own
// record store under <root>/<id>, plus the legacy flat store rooted at <root> for
// pre-upgrade records that remain readable as the oldest pseudo-identity.
type learnedPrefixStore struct {
	cfg      WarmDiskConfig
	root     string
	identity string
	store    *l3kv.DiskStore // <root>/<id>
	legacy   *l3kv.DiskStore // <root>
	index    learnedIndex
	tick     int64
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

// learnedPrefixIdentity maps the authenticated cache scope to a stable 16-lowercase-hex
// directory name: the low 64 bits (last digest bytes) of the NUL-separated scope hash,
// so distinct tenants/agents/epochs never share an identity dir.
func learnedPrefixIdentity(scope radixkv.CacheIdentity) string {
	sum := sha256.Sum256([]byte("fak-native-prefix-identity/v1\x00" + scope.Tenant + "\x00" + scope.Agent + "\x00" + scope.Epoch))
	return hex.EncodeToString(sum[len(sum)-8:])
}

// isLearnedIdentityName reports whether name is a legal identity dir name:
// exactly 16 lowercase hex characters.
func isLearnedIdentityName(name string) bool {
	if len(name) != 16 {
		return false
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return false
		}
	}
	return true
}

// isLearnedRecordKey mirrors l3kv.validKey: a non-empty hex string of bounded length,
// the only names permitted to be record files under the warm disk root.
func isLearnedRecordKey(key string) bool {
	if key == "" || len(key) > 128 {
		return false
	}
	for i := 0; i < len(key); i++ {
		c := key[i]
		isHex := (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
		if !isHex {
			return false
		}
	}
	return true
}

// readWordFile reads an 8-byte little-endian u64 word file. Missing or malformed
// files report ok=false: word files carry best-effort LRU ordering only, so a bad
// read degrades ordering instead of failing the open.
func readWordFile(path string) (uint64, bool) {
	b, err := os.ReadFile(path)
	if err != nil || len(b) != 8 {
		return 0, false
	}
	return binary.LittleEndian.Uint64(b), true
}

// writeFileAtomic commits b to path through a same-directory temp file that is
// fsync'd and atomically renamed into place — the rename is the durability commit,
// so an interrupted write leaves only a temp file, never a torn final. Mirrors
// l3kv's atomicWrite shape; temps are created 0o600 by os.CreateTemp.
func writeFileAtomic(path string, b []byte) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".fakwarm-*")
	if err != nil {
		return fmt.Errorf("warm disk: temp file in %s: %w", filepath.Dir(path), err)
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		return fmt.Errorf("warm disk: write %s: %w", name, err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return fmt.Errorf("warm disk: fsync %s: %w", name, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("warm disk: close %s: %w", name, err)
	}
	if err := os.Rename(name, path); err != nil {
		return fmt.Errorf("warm disk: commit %s: %w", path, err)
	}
	return nil
}

// writeWordFileAtomic writes v as an 8-byte little-endian u64 word file atomically.
func writeWordFileAtomic(path string, v uint64) error {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], v)
	return writeFileAtomic(path, buf[:])
}

// reapStaleWarmTemps removes .l3kv-* and .fakwarm-* temp files older than 24h in
// dir, mirroring DiskStore.PruneContext's stale-temp rule. Best-effort: a crash
// between Put and index commit may leave these behind, and they must not survive
// restarts forever nor block an open.
func reapStaleWarmTemps(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, ".l3kv-") && !strings.HasPrefix(name, ".fakwarm-") {
			continue
		}
		info, err := entry.Info()
		if err != nil || time.Since(info.ModTime()) <= 24*time.Hour {
			continue
		}
		_ = os.Remove(filepath.Join(dir, name))
	}
}

// learnedPendingMarkerPath returns the pending-marker path for an identity dir.
func learnedPendingMarkerPath(root, identity string) string {
	return filepath.Join(root, identity, learnedPendingFile)
}

// writeLearnedPendingMarker atomically writes the in-flight write marker for key:
// magic || version(u16 LE) || keyLen(u16 LE) || key || sha256(everything before it).
// Written BEFORE the record write and cleared after the index commit, so a crash
// leaves a reconcilable marker and a second concurrent writer defers.
func writeLearnedPendingMarker(path, key string) error {
	if !isLearnedRecordKey(key) {
		return fmt.Errorf("warm disk: invalid pending key %q", key)
	}
	buf := make([]byte, 0, 8+len(key)+sha256.Size)
	buf = append(buf, learnedPendingMagic...)
	var head [4]byte
	binary.LittleEndian.PutUint16(head[:2], learnedPendingVersion)
	binary.LittleEndian.PutUint16(head[2:4], uint16(len(key)))
	buf = append(buf, head[:]...)
	buf = append(buf, key...)
	sum := sha256.Sum256(buf)
	buf = append(buf, sum[:]...)
	return writeFileAtomic(path, buf)
}

// readLearnedPendingMarker decodes and verifies a pending marker. ok=false covers
// every rejection: absent file, short/garbage bytes, wrong magic or version, an
// implausible key length or non-hex key, or a digest mismatch. Callers remove
// any present-but-invalid marker as stale garbage.
func readLearnedPendingMarker(path string) (string, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	minLen := len(learnedPendingMagic) + 2 + 2 + sha256.Size
	if len(b) < minLen || string(b[:len(learnedPendingMagic)]) != learnedPendingMagic {
		return "", false
	}
	if binary.LittleEndian.Uint16(b[4:6]) != learnedPendingVersion {
		return "", false
	}
	keyLen := int(binary.LittleEndian.Uint16(b[6:8]))
	if keyLen == 0 || keyLen > 128 || len(b) != 8+keyLen+sha256.Size {
		return "", false
	}
	key := string(b[8 : 8+keyLen])
	if !isLearnedRecordKey(key) {
		return "", false
	}
	sum := sha256.Sum256(b[:8+keyLen])
	if string(b[8+keyLen:]) != string(sum[:]) {
		return "", false
	}
	return key, true
}

// loadLearnedIndex parses the identity's index file. ok=false means missing;
// valid=false means present but unusable (oversized, bad JSON, or a version this
// build does not read) — the caller rebuilds from the record files on disk.
func loadLearnedIndex(path string) (idx learnedIndex, ok, valid bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return idx, false, false
	}
	if len(b) > maxLearnedIndexBytes {
		return idx, true, false
	}
	if err := json.Unmarshal(b, &idx); err != nil {
		return idx, true, false
	}
	if idx.Version != learnedIndexVersion {
		return idx, true, false
	}
	return idx, true, true
}

// openLearnedPrefixStore opens the identity-scoped store for scope under cfg.Dir:
// it creates the root and identity dirs, reaps stale temps, reconciles the state
// index against the record files (adopting a pending-marked record, dropping
// unindexed strays, removing the pending marker), stamps the global clock and the
// identity's used word, and seeds the LRU tick past every indexed state.
func openLearnedPrefixStore(ctx context.Context, cfg WarmDiskConfig, scope radixkv.CacheIdentity) (*learnedPrefixStore, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	root := cfg.Dir
	identity := learnedPrefixIdentity(scope)
	idDir := filepath.Join(root, identity)
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, fmt.Errorf("warm disk: create root %s: %w", root, err)
	}
	if err := os.MkdirAll(idDir, 0o700); err != nil {
		return nil, fmt.Errorf("warm disk: create identity dir %s: %w", idDir, err)
	}
	reapStaleWarmTemps(root)
	reapStaleWarmTemps(idDir)
	store, err := l3kv.NewDiskStore(idDir)
	if err != nil {
		return nil, err
	}
	legacy, err := l3kv.NewDiskStore(root)
	if err != nil {
		return nil, err
	}
	s := &learnedPrefixStore{cfg: cfg, root: root, identity: identity, store: store, legacy: legacy}

	// Stamp the open FIRST so records reconciled below can carry this open's clock
	// word as their used value. Both writes are best-effort — a failure only
	// degrades LRU/sweep ordering, never the open itself.
	clock, _ := readWordFile(filepath.Join(root, learnedClockFile))
	clock++
	_ = writeWordFileAtomic(filepath.Join(root, learnedClockFile), clock)
	_ = writeWordFileAtomic(filepath.Join(idDir, learnedUsedFile), clock)

	idx, present, valid := loadLearnedIndex(filepath.Join(idDir, learnedIndexFile))
	rebuilding := !present || !valid
	changed := rebuilding
	if rebuilding {
		// No usable index: rebuild from the record files, assigning increasing
		// ticks in mtime order so the oldest record stays the LRU victim.
		var found []struct {
			key  string
			size int64
			mod  int64
		}
		if entries, err := os.ReadDir(idDir); err == nil {
			for _, entry := range entries {
				if entry.IsDir() || !isLearnedRecordKey(entry.Name()) {
					continue
				}
				info, infoErr := entry.Info()
				if infoErr != nil || !info.Mode().IsRegular() {
					continue
				}
				found = append(found, struct {
					key  string
					size int64
					mod  int64
				}{entry.Name(), info.Size(), info.ModTime().UnixNano()})
			}
		}
		sort.Slice(found, func(i, j int) bool {
			if found[i].mod == found[j].mod {
				return found[i].key < found[j].key
			}
			return found[i].mod < found[j].mod
		})
		idx = learnedIndex{Version: learnedIndexVersion}
		for i, rec := range found {
			idx.States = append(idx.States, learnedIndexEntry{Key: rec.key, Bytes: rec.size, Used: int64(i + 1)})
		}
	} else {
		idx.Version = learnedIndexVersion
		// Drop index entries whose record file is gone: the index must never claim
		// state the disk does not hold.
		kept := idx.States[:0]
		for _, st := range idx.States {
			if info, statErr := os.Stat(filepath.Join(idDir, st.Key)); statErr == nil && info.Mode().IsRegular() {
				kept = append(kept, st)
			} else {
				changed = true
			}
		}
		idx.States = kept
	}
	// Reconcile the on-disk record files against the now-established index. A
	// valid pending marker names the one in-flight record allowed to exist
	// outside the index; every other unindexed record file is stray garbage.
	pendingKey, pendingValid := readLearnedPendingMarker(learnedPendingMarkerPath(root, identity))
	if entries, err := os.ReadDir(idDir); err == nil {
		for _, entry := range entries {
			if entry.IsDir() || !isLearnedRecordKey(entry.Name()) {
				continue
			}
			inIndex := false
			for _, st := range idx.States {
				if st.Key == entry.Name() {
					inIndex = true
					break
				}
			}
			if inIndex {
				continue
			}
			info, infoErr := entry.Info()
			if infoErr != nil || !info.Mode().IsRegular() {
				continue
			}
			if pendingValid && entry.Name() == pendingKey {
				// The in-flight write landed (or was written out-of-band): adopt it
				// as freshly opened, so it is not the first LRU victim.
				idx.States = append(idx.States, learnedIndexEntry{Key: entry.Name(), Bytes: info.Size(), Used: int64(clock)})
				changed = true
			} else if removeErr := os.Remove(filepath.Join(idDir, entry.Name())); removeErr != nil && !os.IsNotExist(removeErr) {
				return nil, fmt.Errorf("warm disk: remove unindexed record %s: %w", entry.Name(), removeErr)
			}
		}
	}
	if pendingValid {
		// The marker's write either landed (adopted above) or did not; either way
		// the marker is done with and a later writer must not defer on it.
		_ = os.Remove(learnedPendingMarkerPath(root, identity))
	} else if _, statErr := os.Stat(learnedPendingMarkerPath(root, identity)); statErr == nil {
		// Present but undecodable: stale garbage from a torn marker write.
		_ = os.Remove(learnedPendingMarkerPath(root, identity))
	}
	if changed {
		s.index = idx
		if err := s.writeIndex(); err != nil {
			return nil, err
		}
	}
	tick := int64(0)
	for _, st := range idx.States {
		if st.Used > tick {
			tick = st.Used
		}
	}
	s.index = idx
	s.tick = tick + 1
	return s, nil
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

// openWarmDisk resolves the warm disk config and opens the identity-scoped store
// for scope. The gate order and receipt reasons are the closed contract callers
// rely on: disabled -> "disabled"; empty model identity -> Unsupported
// "model_identity_unavailable"; resolve error -> Fault "cache_dir_unavailable";
// env-disabled -> "disabled"; open failure -> Fault "store_open_failed".
func (p *InKernelPlanner) openWarmDisk(ctx context.Context, scope radixkv.CacheIdentity) (*learnedPrefixStore, WarmDiskConfig, string, *WarmDiskReceipt) {
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
	store, err := openLearnedPrefixStore(ctx, resolved, scope)
	if err != nil {
		return nil, resolved, identity, newWarmDiskReceipt(resolved, WarmDiskOutcomeFault, "store_open_failed")
	}
	return store, resolved, identity, nil
}

// lookupRecord reads a record from this identity's store, falling back to the
// legacy flat records directly under the root (pre-upgrade writes). A clean miss
// in both yields found=false; any read fault propagates.
func (s *learnedPrefixStore) lookupRecord(ctx context.Context, key string, maxPayload int64) ([]byte, bool, error) {
	payload, found, err := s.store.GetBounded(ctx, key, maxPayload)
	if err == nil && !found {
		payload, found, err = s.legacy.GetBounded(ctx, key, maxPayload)
	}
	return payload, found, err
}

// Put durably records payload under key in this identity's store.
func (s *learnedPrefixStore) Put(ctx context.Context, key string, payload []byte) error {
	return s.store.Put(ctx, key, payload)
}

// beginWrite claims the in-flight write slot for key. If a VALID pending marker
// already exists (any key — one in-flight write per identity) it returns
// errLearnedPending so the second writer defers; otherwise it writes the marker
// atomically before the record write starts.
func (s *learnedPrefixStore) beginWrite(key string) error {
	if _, ok := readLearnedPendingMarker(learnedPendingMarkerPath(s.root, s.identity)); ok {
		return errLearnedPending
	}
	return writeLearnedPendingMarker(learnedPendingMarkerPath(s.root, s.identity), key)
}

// commitWrite lands key in the index after its record write succeeded: merges
// this instance's view with the CURRENT on-disk index (other instances or the
// conversation-state stager may have committed since this store opened),
// upserts the entry with its full on-disk size and a fresh LRU tick strictly
// past every observed state, enforces the per-identity entry cap by evicting
// lowest-used states (never key itself), rewrites the index atomically, and
// clears the pending marker. An index write failure is returned to the caller;
// the record stays on disk and the next open adopts it via rescan, so no
// committed record is lost.
func (s *learnedPrefixStore) commitWrite(ctx context.Context, key string, payloadBytes int64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	// Merge with the on-disk index so concurrent store instances over the same
	// identity dir never clobber each other's entries: keep every entry the
	// disk already knows, add the ones only this instance saw, and take the max
	// used word so the new tick stays monotonic across instances.
	if diskIdx, present, valid := loadLearnedIndex(filepath.Join(s.root, s.identity, learnedIndexFile)); present && valid {
		maxUsed := s.tick
		for _, st := range s.index.States {
			if st.Used > maxUsed {
				maxUsed = st.Used
			}
		}
		for _, st := range diskIdx.States {
			if st.Used > maxUsed {
				maxUsed = st.Used
			}
		}
		s.tick = maxUsed
		merged := make([]learnedIndexEntry, 0, len(s.index.States)+len(diskIdx.States))
		seen := make(map[string]bool, len(diskIdx.States))
		for _, st := range diskIdx.States {
			merged = append(merged, st)
			seen[st.Key] = true
		}
		for _, st := range s.index.States {
			if !seen[st.Key] {
				merged = append(merged, st)
			}
		}
		s.index.States = merged
	}
	s.tick++
	entry := learnedIndexEntry{Key: key, Bytes: l3kv.RecordBytes(payloadBytes), Used: s.tick}
	replaced := false
	for i := range s.index.States {
		if s.index.States[i].Key == key {
			s.index.States[i] = entry
			replaced = true
			break
		}
	}
	if !replaced {
		s.index.States = append(s.index.States, entry)
	}
	for len(s.index.States) > s.cfg.MaxEntries {
		victim := -1
		for i, st := range s.index.States {
			if st.Key == key {
				continue
			}
			if victim == -1 || st.Used < s.index.States[victim].Used {
				victim = i
			}
		}
		if victim == -1 {
			break
		}
		if removeErr := os.Remove(filepath.Join(s.root, s.identity, s.index.States[victim].Key)); removeErr != nil && !os.IsNotExist(removeErr) {
			return fmt.Errorf("warm disk: evict record %s: %w", s.index.States[victim].Key, removeErr)
		}
		s.index.States = append(s.index.States[:victim], s.index.States[victim+1:]...)
	}
	if err := s.writeIndex(); err != nil {
		return err
	}
	if err := os.Remove(learnedPendingMarkerPath(s.root, s.identity)); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("warm disk: clear pending marker: %w", err)
	}
	return nil
}

// abortWrite releases the pending marker after a failed record write. It only
// removes a marker that names key, so a marker owned by another writer is
// never touched. Best-effort by contract: a leftover marker is reconciled at
// the next open (record absent -> marker removed).
func (s *learnedPrefixStore) abortWrite(key string) {
	markerPath := learnedPendingMarkerPath(s.root, s.identity)
	k, ok := readLearnedPendingMarker(markerPath)
	if !ok || k != key {
		return
	}
	_ = os.Remove(markerPath)
}

// writeIndex rewrites the identity's index atomically (JSON compact, 0o600).
func (s *learnedPrefixStore) writeIndex() error {
	out, err := json.Marshal(&s.index)
	if err != nil {
		return fmt.Errorf("warm disk: encode index: %w", err)
	}
	return writeFileAtomic(filepath.Join(s.root, s.identity, learnedIndexFile), out)
}

// touch refreshes key's LRU position after a successful read hit. Best-effort by
// contract: index bookkeeping must never fault an otherwise-good restore, so
// callers ignore the error. Unknown keys are a silent no-op.
func (s *learnedPrefixStore) touch(key string) error {
	for i := range s.index.States {
		if s.index.States[i].Key == key {
			s.tick++
			s.index.States[i].Used = s.tick
			return s.writeIndex()
		}
	}
	return nil
}

// sweepCandidate is one eviction candidate in a cross-identity sweep: an identity
// dir (or the legacy flat pseudo-identity) with its records ordered LRU-first.
type sweepCandidate struct {
	name    string
	legacy  bool
	used    int64
	bytes   int64
	records []sweepRecord
}

type sweepRecord struct {
	key  string
	size int64
}

// loadSweepCandidate computes one candidate's total bytes and LRU-ordered record
// list. Identities use their index when readable (falling back to a stat scan);
// the legacy pseudo-identity is always a stat scan with used=0 so pre-upgrade
// records are reclaimed first.
func loadSweepCandidate(root, name string, legacy bool) (*sweepCandidate, error) {
	cand := &sweepCandidate{name: name, legacy: legacy}
	if legacy {
		info, err := os.Stat(filepath.Join(root, name))
		if err != nil || !info.Mode().IsRegular() {
			return nil, nil
		}
		cand.bytes = info.Size()
		cand.records = []sweepRecord{{key: name, size: info.Size()}}
		return cand, nil
	}
	idDir := filepath.Join(root, name)
	if used, ok := readWordFile(filepath.Join(idDir, learnedUsedFile)); ok {
		cand.used = int64(used)
	}
	if idx, _, valid := loadLearnedIndex(filepath.Join(idDir, learnedIndexFile)); valid {
		order := make([]int, len(idx.States))
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool {
			ia, ib := idx.States[order[a]], idx.States[order[b]]
			if ia.Used != ib.Used {
				return ia.Used < ib.Used
			}
			return ia.Key < ib.Key
		})
		for _, i := range order {
			st := idx.States[i]
			info, err := os.Stat(filepath.Join(idDir, st.Key))
			if err != nil || !info.Mode().IsRegular() {
				continue
			}
			cand.bytes += info.Size()
			cand.records = append(cand.records, sweepRecord{key: st.Key, size: info.Size()})
		}
		return cand, nil
	}
	entries, err := os.ReadDir(idDir)
	if err != nil {
		return nil, fmt.Errorf("warm disk: sweep scan %s: %w", idDir, err)
	}
	var scanned []struct {
		key  string
		size int64
		mod  int64
	}
	for _, entry := range entries {
		if entry.IsDir() || !isLearnedRecordKey(entry.Name()) {
			continue
		}
		info, infoErr := entry.Info()
		if infoErr != nil || !info.Mode().IsRegular() {
			continue
		}
		scanned = append(scanned, struct {
			key  string
			size int64
			mod  int64
		}{entry.Name(), info.Size(), info.ModTime().UnixNano()})
	}
	sort.Slice(scanned, func(i, j int) bool {
		if scanned[i].mod == scanned[j].mod {
			return scanned[i].key < scanned[j].key
		}
		return scanned[i].mod < scanned[j].mod
	})
	for _, rec := range scanned {
		cand.bytes += rec.size
		cand.records = append(cand.records, sweepRecord{key: rec.key, size: rec.size})
	}
	return cand, nil
}

// learnedRootSweep enforces the byte cap SHARED by every identity under root:
// while total bytes plus need exceed capBytes it evicts the LRU record of the
// least-recently-opened candidate (oldest used word; ties go to the legacy
// pseudo-identity, then lexicographic name). keepKey inside keepIdentity is
// never evicted. slots/, clock, word files and any non-matching name are never
// touched. It returns nil once the cap fits and errLearnedSweepExhausted when
// nothing evictable remains.
func learnedRootSweep(ctx context.Context, root, keepIdentity, keepKey string, need, capBytes int64) error {
	if ctx == nil {
		ctx = context.Background()
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return fmt.Errorf("warm disk: sweep root %s: %w", root, err)
	}
	var candidates []*sweepCandidate
	total := int64(0)
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		var cand *sweepCandidate
		switch {
		case entry.IsDir():
			if !isLearnedIdentityName(entry.Name()) {
				continue
			}
			cand, err = loadSweepCandidate(root, entry.Name(), false)
		default:
			if !isLearnedRecordKey(entry.Name()) {
				continue
			}
			cand, err = loadSweepCandidate(root, entry.Name(), true)
		}
		if err != nil {
			return err
		}
		if cand == nil || cand.bytes == 0 {
			continue
		}
		if cand.name == keepIdentity && keepKey != "" {
			// keepKey's bytes count toward the cap but are never evictable.
			kept := cand.records[:0]
			for _, rec := range cand.records {
				if rec.key != keepKey {
					kept = append(kept, rec)
				}
			}
			cand.records = kept
		}
		candidates = append(candidates, cand)
		total += cand.bytes
	}
	for total+need > capBytes {
		if err := ctx.Err(); err != nil {
			return err
		}
		best := -1
		for i, cand := range candidates {
			if len(cand.records) == 0 {
				continue
			}
			if best == -1 || sweepCandidateLess(*cand, *candidates[best]) {
				best = i
			}
		}
		if best == -1 {
			return errLearnedSweepExhausted
		}
		rec := candidates[best].records[0]
		path := filepath.Join(root, candidates[best].name, rec.key)
		if candidates[best].legacy {
			path = filepath.Join(root, rec.key)
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("warm disk: sweep evict %s: %w", path, err)
		}
		candidates[best].records = candidates[best].records[1:]
		candidates[best].bytes -= rec.size
		total -= rec.size
	}
	return nil
}

// sweepCandidateLess orders eviction: smallest used word first; on a tie the
// legacy pseudo-identity goes first, then lexicographic name.
func sweepCandidateLess(a, b sweepCandidate) bool {
	if a.used != b.used {
		return a.used < b.used
	}
	if a.legacy != b.legacy {
		return a.legacy
	}
	return a.name < b.name
}

func (p *InKernelPlanner) restoreWarmPrefixDisk(ctx context.Context, spec WarmPrefixSpec, tokens []int) *WarmDiskReceipt {
	store, cfg, identity, result := p.openWarmDisk(ctx, spec.Scope)
	if result != nil {
		result.RestoreOutcome, result.RestoreReason = result.Outcome, result.Reason
		return result
	}
	execution := p.warmDiskExecutionIdentity()
	key := warmDiskKey(spec, identity, execution)
	payload, found, err := store.lookupRecord(ctx, key, cfg.MaxRecordBytes)
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
	// A read hit refreshes this state's LRU position; index bookkeeping must
	// never fault an otherwise-successful restore.
	_ = store.touch(key)
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

// learnedGuardedWriter is the durability seam for record writers whose records
// must also land in the learned state index: publishes go through the
// pending-write guard so a crash leaves a reconcilable marker instead of an
// unindexed record that the next open would reap.
type learnedGuardedWriter interface {
	warmDiskRecordWriter
	beginWrite(key string) error
	commitWrite(ctx context.Context, key string, payloadBytes int64) error
	abortWrite(key string)
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
		err := s.publishSave(&save)
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

// publishSave lands one staged save. Writers that also maintain a learned
// state index (the warm-disk store) go through their pending-write guard so
// the record is indexed before it becomes visible and a crash leaves a
// reconcilable marker; plain writers fall back to a direct Put. A deferral by
// an in-flight write on the same identity counts as a failed save: the stager
// is bounded and the caller sees the error, it never blocks the I/O thread.
func (s *warmDiskStager) publishSave(save *warmDiskStagedSave) error {
	g, ok := s.store.(learnedGuardedWriter)
	if !ok {
		return s.store.Put(context.Background(), save.key, save.payload)
	}
	ctx := context.Background()
	if err := g.beginWrite(save.key); err != nil {
		return err
	}
	if err := g.Put(ctx, save.key, save.payload); err != nil {
		g.abortWrite(save.key)
		return err
	}
	return g.commitWrite(ctx, save.key, int64(len(save.payload)))
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
func conversationStager(dir string, store warmDiskRecordWriter, permit int64, maxJobs int) *warmDiskStager {
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
	store, cfg, identity, result := p.openWarmDisk(ctx, spec.Scope)
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
		// of the payload outlives this call. Staged publishes still land through
		// the learned store's guarded write path (see warmDiskStager.run), so
		// every record under the root stays indexed and sweepable.
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
		// The byte cap is shared by every identity under the root: sweep other
		// identities (oldest-opened first) until this record fits. The per-identity
		// entry cap is enforced inside commitWrite instead of a flat prune pass.
		// need is the FULL on-disk record size (payload + l3kv header) so the shared
		// byte cap is never exceeded by the envelope overhead.
		if err := learnedRootSweep(ctx, cfg.Dir, store.identity, key, l3kv.RecordBytes(int64(len(payload))), cfg.MaxBytes); err != nil {
			if errors.Is(err, errLearnedSweepExhausted) {
				return newWarmDiskReceipt(cfg, WarmDiskOutcomeUnsupported, "record_exceeds_budget")
			}
			return newWarmDiskReceipt(cfg, WarmDiskOutcomeFault, "sweep_failed")
		}
		if err := store.beginWrite(key); err != nil {
			if errors.Is(err, errLearnedPending) {
				return newWarmDiskReceipt(cfg, WarmDiskOutcomeFault, "pending_write_in_flight")
			}
			return newWarmDiskReceipt(cfg, WarmDiskOutcomeFault, "pending_write_failed")
		}
		if err := store.Put(ctx, key, payload); err != nil {
			return newWarmDiskReceipt(cfg, WarmDiskOutcomeFault, "write_failed")
		}
		if err := store.commitWrite(ctx, key, int64(len(payload))); err != nil {
			return newWarmDiskReceipt(cfg, WarmDiskOutcomeFault, "index_commit_failed")
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
	if writeGate != nil {
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
