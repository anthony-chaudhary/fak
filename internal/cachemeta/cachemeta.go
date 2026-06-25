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

// Valid reports whether the identity is fully populated: a non-empty digest,
// media type, and length unit, with a non-negative length.
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

// AdmissionFromVerdict maps an abi.VerdictKind to its cache-facing
// AdmissionVerdict, returning AdmissionUnknown for an unrecognized kind.
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

// InvalidationMode names what causes an entry to be evicted or refuted (LRU,
// TTL, write-epoch, external refutation, or policy).
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

// LookupKind is the category of a LookupVerdict: hit, miss, revalidate,
// transform, quarantine, or fault.
type LookupKind string

const (
	LookupHit        LookupKind = "hit"
	LookupMiss       LookupKind = "miss"
	LookupRevalidate LookupKind = "revalidate"
	LookupTransform  LookupKind = "transform"
	LookupQuarantine LookupKind = "quarantine"
	LookupFault      LookupKind = "fault"
)

// LookupReason is the specific cause carried by a LookupVerdict (why it missed,
// must revalidate, was denied, or faulted).
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

// Hit builds a hit verdict for e, with its handle set to the entry's ID.
func Hit(e Entry) LookupVerdict {
	return LookupVerdict{Kind: LookupHit, Entry: e, Handle: e.ID}
}

// Miss builds a miss verdict carrying the given reason and no entry.
func Miss(reason LookupReason) LookupVerdict {
	return LookupVerdict{Kind: LookupMiss, Reason: reason}
}

// Revalidate builds a verdict that the entry exists but must be re-checked
// before serving, carrying the reason that triggered revalidation.
func Revalidate(e Entry, reason LookupReason) LookupVerdict {
	return LookupVerdict{Kind: LookupRevalidate, Reason: reason, Entry: e, Handle: e.ID}
}

// Transform builds a verdict that the entry may serve only after a transform,
// carrying the reason the raw entry is not directly servable.
func Transform(e Entry, reason LookupReason) LookupVerdict {
	return LookupVerdict{Kind: LookupTransform, Reason: reason, Entry: e, Handle: e.ID}
}

// Quarantine builds a verdict that the entry is held back from serving, carrying
// the reason it was quarantined.
func Quarantine(e Entry, reason LookupReason) LookupVerdict {
	return LookupVerdict{Kind: LookupQuarantine, Reason: reason, Entry: e, Handle: e.ID}
}

// Fault builds a verdict signaling a residency/approximation fault on the entry,
// carrying the reason it could not be served.
func Fault(e Entry, reason LookupReason) LookupVerdict {
	return LookupVerdict{Kind: LookupFault, Reason: reason, Entry: e, Handle: e.ID}
}

// CanServe reports whether the verdict is a hit with a valid handle, i.e. the
// payload may be reused.
func (v LookupVerdict) CanServe() bool { return v.Kind == LookupHit && v.Handle.Valid() }

// Option mutates an Entry during adapter construction.
type Option func(*Entry)

func WithWitness(w string) Option {
	return func(e *Entry) { e.Validity.Witness = w }
}

// WithEpoch sets the entry's admission epoch stamp.
func WithEpoch(epoch string) Option {
	return func(e *Entry) { e.Validity.AdmittedAtEpoch = epoch }
}

// WithTrustEpoch sets the entry's trust epoch.
func WithTrustEpoch(epoch uint64) Option {
	return func(e *Entry) { e.Validity.TrustEpoch = epoch }
}

// WithPolicyVersion sets the policy version that bounds the entry's validity.
func WithPolicyVersion(v string) Option {
	return func(e *Entry) { e.Validity.PolicyVersion = v }
}

// WithTTLMillis sets the entry's time-to-live in milliseconds.
func WithTTLMillis(ms int64) Option {
	return func(e *Entry) { e.Validity.TTLMillis = ms }
}

// WithAdmission sets the admission verdict and the identity that admitted the
// entry.
func WithAdmission(v AdmissionVerdict, by string) Option {
	return func(e *Entry) {
		e.Security.AdmissionVerdict = v
		e.Security.AdmittedBy = by
	}
}

// WithResidency replaces the entry's residency record with the given tier,
// owner, and lease.
func WithResidency(t ResidencyTier, owner, lease string) Option {
	return func(e *Entry) {
		e.Residency = Residency{Tier: t, Owner: owner, Lease: lease}
	}
}

// WithModel sets the model and tokenizer IDs the entry was derived under.
func WithModel(modelID, tokenizerID string) Option {
	return func(e *Entry) {
		e.Derivation.ModelID = modelID
		e.Derivation.TokenizerID = tokenizerID
	}
}

// WithSerializer sets the serializer ID the entry was derived under.
func WithSerializer(id string) Option {
	return func(e *Entry) { e.Derivation.SerializerID = id }
}

// WithPositionMode sets whether the entry's token/KV material may be relocated.
func WithPositionMode(m PositionMode) Option {
	return func(e *Entry) { e.Derivation.PositionMode = m }
}

// WithConsumer appends a consumer to the entry's coherence graph for causal
// invalidation.
func WithConsumer(c Consumer) Option {
	return func(e *Entry) { e.Coherence.Consumers = append(e.Coherence.Consumers, c) }
}

// WithParent appends a parent entry ID to the entry's coherence graph.
func WithParent(id EntryID) Option {
	return func(e *Entry) { e.Coherence.Parents = append(e.Coherence.Parents, id) }
}

// WithLabel sets a label key/value on the entry, allocating the label map if
// needed.
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

// ErrBadVDSOKey is returned by ParseVDSOKey when a key is not the expected
// three-part tool:args-digest:epoch-stamp form.
var ErrBadVDSOKey = errors.New("cachemeta: bad vdso key")

// VDSOKey is the parsed form of vDSO's tier-2 key:
// tool:args-digest:epoch-stamp.
type VDSOKey struct {
	Tool       string
	ArgsDigest string
	EpochStamp string
}

// ParseVDSOKey splits a tier-2 vDSO key into its tool, args-digest, and
// epoch-stamp parts, returning ErrBadVDSOKey if any of the three is missing.
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

// FromStaticTool adapts a vDSO tier-3 (static-table) answer into a tool-result
// cache entry. Unlike a tier-2 entry, a tier-3 answer is args- AND epoch-independent
// — it is a fixed table entry keyed by tool name alone, served unconditionally — so
// it carries no ArgsDigest, no admission epoch, and is NOT write-epoch invalidated
// (InvalidationNone: a write to the world never strands it). A witness option still
// upgrades it to external-refutation invalidation, for a static tool that chooses to
// govern its own freshness.
func FromStaticTool(tool string, payload abi.Ref, opts ...Option) Entry {
	e := FromRef(payload, WithResidency(TierDRAM, "vdso", "static"))
	e.Plane = PlaneToolResult
	e.Derivation.Producer = "vdso"
	e.Derivation.Tool = tool
	e.Coherence.InvalidationMode = InvalidationNone
	apply(&e, opts)
	if e.Validity.Witness != "" {
		e.Coherence.InvalidationMode = InvalidationExternalRefutation
	}
	return e
}

// FromMiss adapts a vDSO fast-path MISS into a tool-result cache entry that names
// the attempted lookup. A miss has no payload, so the entry carries no digest — only
// the tool that was looked up and the lookup reason that produced the miss (recorded
// in Security.Reason). It is the observable, first-class form of the LookupMiss
// verdict (cachemeta.Miss): a consumer of the cache event stream sees WHY the fast
// path declined to serve, in the same stream as fills/hits/evictions. Residency is
// TierRecompute — the answer is not resident and must be recomputed by the engine.
func FromMiss(tool string, reason LookupReason, opts ...Option) Entry {
	e := Entry{
		Plane:      PlaneToolResult,
		Derivation: Derivation{Producer: "vdso", Tool: tool},
		Security:   Security{Reason: string(reason)},
		Residency:  Residency{Tier: TierRecompute, Owner: "vdso"},
		Coherence:  Coherence{InvalidationMode: InvalidationNone},
	}
	apply(&e, opts)
	return e
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

// FromContextPage adapts a durable recall/context page into a context-page
// Entry, mapping a quarantined page to AdmissionQuarantine and tagging it with
// session/step/descriptor labels.
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

// FromKVPrefix adapts a KV-prefix span into a prefix-aligned KV-span Entry,
// digesting the token IDs when no digest is supplied and admitting it as a
// trusted, fleet-scoped borrow.
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

// FromMemoryView adapts a derived context view into a recomputable memory-view
// Entry, recording its source refs as parents, deriving a digest when none is
// given, and folding the faithfulness probe into a quality-delta metric.
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

// DigestBytes returns the lowercase hex SHA-256 of b, the canonical content
// digest for byte payloads.
func DigestBytes(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// DigestTokenIDs returns the lowercase hex SHA-256 over the token IDs encoded as
// big-endian 64-bit words, a stable digest for a token sequence.
func DigestTokenIDs(ids []int) string {
	h := sha256.New()
	var buf [8]byte
	for _, id := range ids {
		binary.BigEndian.PutUint64(buf[:], uint64(int64(id)))
		_, _ = h.Write(buf[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// DigestMemoryView returns a lowercase hex SHA-256 over the view ID, view type,
// and each source ref's identity fields, null-separated so the digest is a
// stable function of the view's identity and inputs.
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
