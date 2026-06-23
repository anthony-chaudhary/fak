package cachemeta

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// MediaType names the physical shape of a reusable payload.
type MediaType string

const (
	MediaBytes          MediaType = "bytes"
	MediaTokenIDs       MediaType = "token_ids"
	MediaKVSpan         MediaType = "kv_span"
	MediaPromptPrefix   MediaType = "prompt_prefix"
	MediaRecallPage     MediaType = "recall_page"
	MediaMemoryView     MediaType = "memory_view"
	MediaPlanTemplate   MediaType = "plan_template"
	MediaIntentKey      MediaType = "intent_key"
	MediaAttentionIndex MediaType = "attention_index"
)

// Plane names the cache plane that produced or serves an entry.
type Plane string

const (
	PlaneBlob           Plane = "blob"
	PlaneToolResult     Plane = "tool_result"
	PlaneContextPage    Plane = "context_page"
	PlaneKVPrefix       Plane = "kv_prefix"
	PlaneKVArtifact     Plane = "kv_artifact"
	PlaneKVTransfer     Plane = "kv_transfer"
	PlanePrompt         Plane = "prompt_prefix"
	PlanePolicy         Plane = "policy"
	PlaneProvider       Plane = "provider"
	PlanePlanTemplate   Plane = "plan_template"
	PlaneSemanticIntent Plane = "semantic_intent"
	PlaneMemoryView     Plane = "memory_view"
	PlaneAttentionIndex Plane = "attention_index"
)

// LengthUnit says what EntryID.Length counts.
type LengthUnit string

const (
	UnitBytes     LengthUnit = "bytes"
	UnitTokens    LengthUnit = "tokens"
	UnitPositions LengthUnit = "positions"
)

// EntryID is stable identity for a reusable object, independent of residency.
type EntryID struct {
	Digest    string
	MediaType MediaType
	Length    int64
	Unit      LengthUnit
}

func (id EntryID) Valid() bool {
	return id.Digest != "" && id.MediaType != "" && id.Unit != "" && id.Length >= 0
}

// Entry is the metadata record for a cacheable object. It carries enough context
// to decide whether a hit may be served, but never carries the payload itself.
type Entry struct {
	ID         EntryID
	Plane      Plane
	Derivation Derivation
	Validity   Validity
	Security   Security
	Residency  Residency
	Coherence  Coherence
	Metrics    Metrics
	Labels     map[string]string
}

// Derivation records how an entry was made and which semantic axes must match.
type Derivation struct {
	Producer     string
	Tool         string
	ArgsDigest   string
	ModelID      string
	TokenizerID  string
	SerializerID string
	PositionMode PositionMode
	SourceRefs   []EntryID
}

// PositionMode describes whether cached token/KV material can be relocated.
type PositionMode string

const (
	PositionUnknown           PositionMode = ""
	PositionPrefixAligned     PositionMode = "prefix_aligned"
	PositionRelocatable       PositionMode = "relocatable"
	PositionRecomputeRequired PositionMode = "recompute_required"
)

// Validity names the witness and epochs that bound freshness/integrity.
type Validity struct {
	Witness         string
	AdmittedAtEpoch string
	TrustEpoch      uint64
	TTLMillis       int64
	PolicyVersion   string
}

// Security carries the authority and admission claim for serving an entry.
type Security struct {
	Taint            abi.TaintLabel
	Scope            abi.ShareScope
	AdmissionVerdict AdmissionVerdict
	AdmittedBy       string
	Reason           string
}

// AdmissionVerdict is the cache-facing form of the admission decision that let
// this entry be stored or reused.
type AdmissionVerdict string

const (
	AdmissionUnknown        AdmissionVerdict = ""
	AdmissionAllow          AdmissionVerdict = "allow"
	AdmissionTransform      AdmissionVerdict = "transform"
	AdmissionQuarantine     AdmissionVerdict = "quarantine"
	AdmissionRequireWitness AdmissionVerdict = "require_witness"
	AdmissionDeny           AdmissionVerdict = "deny"
	AdmissionDefer          AdmissionVerdict = "defer"
)

func AdmissionFromVerdict(k abi.VerdictKind) AdmissionVerdict {
	switch k {
	case abi.VerdictAllow:
		return AdmissionAllow
	case abi.VerdictTransform:
		return AdmissionTransform
	case abi.VerdictQuarantine:
		return AdmissionQuarantine
	case abi.VerdictRequireWitness:
		return AdmissionRequireWitness
	case abi.VerdictDeny:
		return AdmissionDeny
	case abi.VerdictDefer:
		return AdmissionDefer
	default:
		return AdmissionUnknown
	}
}

// Residency records where the payload currently lives. It is advisory metadata;
// the package does not fetch or store the payload.
type Residency struct {
	Tier  ResidencyTier
	Owner string
	Lease string
	// Share advertises how the resident payload can be handed to another consumer
	// zero-copy (a coherent CXL region, a shared mmap, an RDMA region, a dma-buf), or
	// the zero value (ShareCopy) when it must be copied. See hardware.go.
	Share ShareDescriptor
}

// ZeroCopy reports whether this residency advertises a zero-copy share capability.
func (r Residency) ZeroCopy() bool { return r.Share.ZeroCopy() }

type ResidencyTier string

const (
	TierUnknown   ResidencyTier = ""
	TierHBM       ResidencyTier = "hbm"
	TierDRAM      ResidencyTier = "dram"
	TierDisk      ResidencyTier = "disk"
	TierRemote    ResidencyTier = "remote"
	TierProvider  ResidencyTier = "provider"
	TierRecompute ResidencyTier = "recompute"
)

// Consumer identifies a session, agent, or derived entry that consumed a shared
// cache entry. Shared entries need this graph for causal invalidation.
type Consumer struct {
	Kind    string
	ID      string
	AgentID string
	TraceID string
}

// Coherence records dependency and invalidation metadata.
type Coherence struct {
	Consumers        []Consumer
	Parents          []EntryID
	InvalidationMode InvalidationMode
}

type InvalidationMode string

const (
	InvalidationNone               InvalidationMode = ""
	InvalidationLRU                InvalidationMode = "lru"
	InvalidationTTL                InvalidationMode = "ttl"
	InvalidationWriteEpoch         InvalidationMode = "write_epoch"
	InvalidationExternalRefutation InvalidationMode = "external_refutation"
	InvalidationPolicy             InvalidationMode = "policy"
)

// Metrics are counters or probes associated with a cache entry. A zero value is
// simply "not yet observed".
type Metrics struct {
	Hits               uint64
	Misses             uint64
	Fills              uint64
	Evictions          uint64
	PrefillTokensSaved int64
	BytesTransferred   int64
	FalseHitFaults     uint64
	QualityDeltaProbe  float64
	Coverage           float64
	FaithfulnessProbe  float64
}

// LookupVerdict is the typed result of asking whether an entry may be reused.
type LookupVerdict struct {
	Kind   LookupKind
	Reason LookupReason
	Entry  Entry
	Handle EntryID
	Meta   map[string]string
}

type LookupKind string

const (
	LookupHit        LookupKind = "hit"
	LookupMiss       LookupKind = "miss"
	LookupRevalidate LookupKind = "revalidate"
	LookupTransform  LookupKind = "transform"
	LookupQuarantine LookupKind = "quarantine"
	LookupFault      LookupKind = "fault"
)

type LookupReason string

const (
	ReasonNone               LookupReason = ""
	ReasonAbsent             LookupReason = "absent"
	ReasonCold               LookupReason = "cold"
	ReasonStale              LookupReason = "stale"
	ReasonExpiredTTL         LookupReason = "expired_ttl"
	ReasonRefutedWitness     LookupReason = "refuted_witness"
	ReasonScopeDenied        LookupReason = "scope_denied"
	ReasonTaintDenied        LookupReason = "taint_denied"
	ReasonModelMismatch      LookupReason = "model_mismatch"
	ReasonTokenizerMismatch  LookupReason = "tokenizer_mismatch"
	ReasonSerializerMismatch LookupReason = "serializer_mismatch"
	ReasonPositionMismatch   LookupReason = "position_mismatch"
	ReasonPolicyMismatch     LookupReason = "policy_mismatch"
	ReasonAdmitterMismatch   LookupReason = "admitter_mismatch"
	ReasonApproxFault        LookupReason = "approximate_fault"
	ReasonResidencyFault     LookupReason = "residency_fault"
	ReasonRestoreMiss        LookupReason = "restore_miss"
	ReasonManifestMismatch   LookupReason = "manifest_mismatch"
	ReasonUnsignedArtifact   LookupReason = "unsigned_artifact"
	ReasonMissingProvenance  LookupReason = "missing_provenance"
	ReasonAccessControlReq   LookupReason = "access_control_required"
	ReasonIncompleteBinding  LookupReason = "incomplete_binding"
	ReasonIndexMismatch      LookupReason = "index_mismatch"
	ReasonNonCausalIndex     LookupReason = "non_causal_index"
)

func Hit(e Entry) LookupVerdict {
	return LookupVerdict{Kind: LookupHit, Entry: e, Handle: e.ID}
}

func Miss(reason LookupReason) LookupVerdict {
	return LookupVerdict{Kind: LookupMiss, Reason: reason}
}

func Revalidate(e Entry, reason LookupReason) LookupVerdict {
	return LookupVerdict{Kind: LookupRevalidate, Reason: reason, Entry: e, Handle: e.ID}
}

func Transform(e Entry, reason LookupReason) LookupVerdict {
	return LookupVerdict{Kind: LookupTransform, Reason: reason, Entry: e, Handle: e.ID}
}

func Quarantine(e Entry, reason LookupReason) LookupVerdict {
	return LookupVerdict{Kind: LookupQuarantine, Reason: reason, Entry: e, Handle: e.ID}
}

func Fault(e Entry, reason LookupReason) LookupVerdict {
	return LookupVerdict{Kind: LookupFault, Reason: reason, Entry: e, Handle: e.ID}
}

func (v LookupVerdict) CanServe() bool { return v.Kind == LookupHit && v.Handle.Valid() }

// Option mutates an Entry during adapter construction.
type Option func(*Entry)

func WithWitness(w string) Option {
	return func(e *Entry) { e.Validity.Witness = w }
}

func WithEpoch(epoch string) Option {
	return func(e *Entry) { e.Validity.AdmittedAtEpoch = epoch }
}

func WithTrustEpoch(epoch uint64) Option {
	return func(e *Entry) { e.Validity.TrustEpoch = epoch }
}

func WithPolicyVersion(v string) Option {
	return func(e *Entry) { e.Validity.PolicyVersion = v }
}

func WithTTLMillis(ms int64) Option {
	return func(e *Entry) { e.Validity.TTLMillis = ms }
}

func WithAdmission(v AdmissionVerdict, by string) Option {
	return func(e *Entry) {
		e.Security.AdmissionVerdict = v
		e.Security.AdmittedBy = by
	}
}

func WithResidency(t ResidencyTier, owner, lease string) Option {
	return func(e *Entry) {
		e.Residency = Residency{Tier: t, Owner: owner, Lease: lease}
	}
}

func WithModel(modelID, tokenizerID string) Option {
	return func(e *Entry) {
		e.Derivation.ModelID = modelID
		e.Derivation.TokenizerID = tokenizerID
	}
}

func WithSerializer(id string) Option {
	return func(e *Entry) { e.Derivation.SerializerID = id }
}

func WithPositionMode(m PositionMode) Option {
	return func(e *Entry) { e.Derivation.PositionMode = m }
}

func WithConsumer(c Consumer) Option {
	return func(e *Entry) { e.Coherence.Consumers = append(e.Coherence.Consumers, c) }
}

func WithParent(id EntryID) Option {
	return func(e *Entry) { e.Coherence.Parents = append(e.Coherence.Parents, id) }
}

func WithLabel(k, v string) Option {
	return func(e *Entry) {
		if e.Labels == nil {
			e.Labels = map[string]string{}
		}
		e.Labels[k] = v
	}
}

func apply(e *Entry, opts []Option) {
	for _, opt := range opts {
		if opt != nil {
			opt(e)
		}
	}
}

// FromRef describes an abi.Ref as a byte/blob entry. Digest is identity only; the
// returned Security record preserves the ref's taint/scope but does not pretend
// that digest alone proves permission.
func FromRef(r abi.Ref, opts ...Option) Entry {
	length := r.Len
	if length == 0 && r.Kind == abi.RefInline {
		length = int64(len(r.Inline))
	}
	digest := r.Digest
	if digest == "" && r.Kind == abi.RefInline {
		digest = DigestBytes(r.Inline)
	}
	e := Entry{
		ID: EntryID{
			Digest:    digest,
			MediaType: MediaBytes,
			Length:    length,
			Unit:      UnitBytes,
		},
		Plane: PlaneBlob,
		Derivation: Derivation{
			Producer: "abi.Ref",
		},
		Security: Security{
			Taint: r.Taint,
			Scope: r.Scope,
		},
		Residency: Residency{Tier: residencyOfRef(r), Owner: "abi.Ref"},
	}
	apply(&e, opts)
	return e
}

func residencyOfRef(r abi.Ref) ResidencyTier {
	switch r.Kind {
	case abi.RefInline:
		return TierDRAM
	case abi.RefBlob, abi.RefRegion:
		return TierDisk
	default:
		return TierUnknown
	}
}

var ErrBadVDSOKey = errors.New("cachemeta: bad vdso key")

// VDSOKey is the parsed form of vDSO's tier-2 key:
// tool:args-digest:epoch-stamp.
type VDSOKey struct {
	Tool       string
	ArgsDigest string
	EpochStamp string
}

func ParseVDSOKey(key string) (VDSOKey, error) {
	parts := strings.Split(key, ":")
	if len(parts) != 3 || parts[0] == "" || parts[1] == "" || parts[2] == "" {
		return VDSOKey{}, ErrBadVDSOKey
	}
	return VDSOKey{Tool: parts[0], ArgsDigest: parts[1], EpochStamp: parts[2]}, nil
}

// FromVDSOKey adapts a tier-2 vDSO key plus its payload Ref into a tool-result
// cache entry. It does not import vdso, keeping the metadata layer below the
// hot-path implementation.
func FromVDSOKey(key string, payload abi.Ref, opts ...Option) (Entry, error) {
	k, err := ParseVDSOKey(key)
	if err != nil {
		return Entry{}, err
	}
	e := FromRef(payload,
		WithEpoch(k.EpochStamp),
		WithResidency(TierDRAM, "vdso", "lru"),
	)
	e.Plane = PlaneToolResult
	e.Derivation.Producer = "vdso"
	e.Derivation.Tool = k.Tool
	e.Derivation.ArgsDigest = k.ArgsDigest
	e.Coherence.InvalidationMode = InvalidationWriteEpoch
	apply(&e, opts)
	if e.Validity.Witness != "" {
		e.Coherence.InvalidationMode = InvalidationExternalRefutation
	}
	return e, nil
}

// ContextPage is the field-only shape a durable recall/context-page adapter
// lowers into. It mirrors the metadata needed from recall.Page without making
// cachemeta import recall.
type ContextPage struct {
	SessionID   string
	Step        int
	Role        string
	Descriptor  string
	Digest      string
	Len         int64
	Taint       abi.TaintLabel
	Quarantined bool
	QID         string
	Reason      string
	Witness     string
	TrustEpoch  uint64
}

func FromContextPage(p ContextPage, opts ...Option) Entry {
	taint := p.Taint
	admit := AdmissionAllow
	if p.Quarantined {
		taint = abi.TaintQuarantined
		admit = AdmissionQuarantine
	}
	e := Entry{
		ID: EntryID{
			Digest:    p.Digest,
			MediaType: MediaRecallPage,
			Length:    p.Len,
			Unit:      UnitBytes,
		},
		Plane: PlaneContextPage,
		Derivation: Derivation{
			Producer: "recall",
			Tool:     p.Role,
		},
		Validity: Validity{
			Witness:    p.Witness,
			TrustEpoch: p.TrustEpoch,
		},
		Security: Security{
			Taint:            taint,
			Scope:            abi.ScopeAgent,
			AdmissionVerdict: admit,
			AdmittedBy:       "recall",
			Reason:           p.Reason,
		},
		Residency: Residency{Tier: TierDisk, Owner: "recall"},
		Labels: map[string]string{
			"session_id": p.SessionID,
			"step":       strconv.Itoa(p.Step),
			"descriptor": p.Descriptor,
		},
	}
	if p.QID != "" {
		e.Labels["qid"] = p.QID
	}
	if p.Witness != "" {
		e.Coherence.InvalidationMode = InvalidationExternalRefutation
	}
	apply(&e, opts)
	return e
}

// KVPrefix is the field-only shape a KV-prefix cache adapter lowers into.
type KVPrefix struct {
	TokenDigest string
	Tokens      []int
	Length      int
	ModelID     string
	TokenizerID string
	Owner       string
}

func FromKVPrefix(p KVPrefix, opts ...Option) Entry {
	length := p.Length
	if length == 0 {
		length = len(p.Tokens)
	}
	digest := p.TokenDigest
	if digest == "" {
		digest = DigestTokenIDs(p.Tokens)
	}
	owner := p.Owner
	if owner == "" {
		owner = "kv"
	}
	e := Entry{
		ID: EntryID{
			Digest:    digest,
			MediaType: MediaKVSpan,
			Length:    int64(length),
			Unit:      UnitPositions,
		},
		Plane: PlaneKVPrefix,
		Derivation: Derivation{
			Producer:     owner,
			ModelID:      p.ModelID,
			TokenizerID:  p.TokenizerID,
			PositionMode: PositionPrefixAligned,
		},
		Security: Security{
			Taint:            abi.TaintTrusted,
			Scope:            abi.ScopeFleet,
			AdmissionVerdict: AdmissionAllow,
			AdmittedBy:       owner,
		},
		Residency: Residency{Tier: TierDRAM, Owner: owner, Lease: "borrow"},
		Coherence: Coherence{InvalidationMode: InvalidationPolicy},
	}
	apply(&e, opts)
	return e
}

// MemoryView is the field-only shape for a derived context view. A view is a
// recomputable artifact over source pages/entries, not a replacement for raw
// memory.
type MemoryView struct {
	ViewID            string
	ViewType          string
	Digest            string
	Length            int64
	SourceRefs        []EntryID
	Producer          string
	PolicyVersion     string
	Scope             abi.ShareScope
	Taint             abi.TaintLabel
	Coverage          float64
	FaithfulnessProbe float64
	Witness           string
	TTLMillis         int64
}

func FromMemoryView(v MemoryView, opts ...Option) Entry {
	producer := v.Producer
	if producer == "" {
		producer = "memory-view"
	}
	qualityDelta := 0.0
	if v.FaithfulnessProbe != 0 {
		qualityDelta = 1 - v.FaithfulnessProbe
	}
	digest := v.Digest
	if digest == "" {
		digest = DigestMemoryView(v.ViewID, v.ViewType, v.SourceRefs)
	}
	e := Entry{
		ID: EntryID{
			Digest:    digest,
			MediaType: MediaMemoryView,
			Length:    v.Length,
			Unit:      UnitBytes,
		},
		Plane: PlaneMemoryView,
		Derivation: Derivation{
			Producer:   producer,
			SourceRefs: append([]EntryID(nil), v.SourceRefs...),
		},
		Validity: Validity{
			Witness:       v.Witness,
			PolicyVersion: v.PolicyVersion,
			TTLMillis:     v.TTLMillis,
		},
		Security: Security{
			Taint:            v.Taint,
			Scope:            v.Scope,
			AdmissionVerdict: AdmissionAllow,
			AdmittedBy:       producer,
		},
		Residency: Residency{Tier: TierRecompute, Owner: producer},
		Coherence: Coherence{
			Parents:          append([]EntryID(nil), v.SourceRefs...),
			InvalidationMode: InvalidationExternalRefutation,
		},
		Metrics: Metrics{
			Coverage:           v.Coverage,
			FaithfulnessProbe:  v.FaithfulnessProbe,
			QualityDeltaProbe:  qualityDelta,
			PrefillTokensSaved: 0,
		},
		Labels: map[string]string{
			"view_id":   v.ViewID,
			"view_type": v.ViewType,
		},
	}
	apply(&e, opts)
	return e
}

func DigestBytes(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func DigestTokenIDs(ids []int) string {
	h := sha256.New()
	var buf [8]byte
	for _, id := range ids {
		binary.BigEndian.PutUint64(buf[:], uint64(int64(id)))
		_, _ = h.Write(buf[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func DigestMemoryView(viewID, viewType string, sources []EntryID) string {
	h := sha256.New()
	_, _ = h.Write([]byte(viewID))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(viewType))
	_, _ = h.Write([]byte{0})
	for _, s := range sources {
		_, _ = h.Write([]byte(s.Digest))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(s.MediaType))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(strconv.FormatInt(s.Length, 10)))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(s.Unit))
		_, _ = h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}
