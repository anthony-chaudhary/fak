package gateway

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

const (
	FeatureProofSchema          = "fak-feature-proof/1"
	DefaultFeatureProofCapacity = 128
	MaxFeatureProofCapacity     = 256
)

// FeatureProofOutcome describes proof collection following execution, not
// catalog enablement. Proof failure does not imply the underlying feature failed.
type FeatureProofOutcome string

const (
	FeatureProofVerified FeatureProofOutcome = "verified"
	FeatureProofFailed   FeatureProofOutcome = "proof_failed"
)

// FeatureProofFailure is closed so neither payloads nor arbitrary error messages
// can enter the receipt ring or metric labels.
type FeatureProofFailure string

const (
	FeatureProofInvalidDelta      FeatureProofFailure = "INVALID_BYTE_DELTA"
	FeatureProofInvalidBounds     FeatureProofFailure = "INVALID_PROTECTED_BOUNDS"
	FeatureProofPrefixMismatch    FeatureProofFailure = "PREFIX_MISMATCH"
	FeatureProofSuffixMismatch    FeatureProofFailure = "SUFFIX_MISMATCH"
	FeatureProofInvalidTokens     FeatureProofFailure = "INVALID_TOKEN_EVIDENCE"
	FeatureProofAnchorUnavailable FeatureProofFailure = "ANCHOR_UNAVAILABLE"
	FeatureProofInvalidBody       FeatureProofFailure = "INVALID_BODY"
	FeatureProofInvalidCache      FeatureProofFailure = "INVALID_CACHE_EVIDENCE"
	FeatureProofSourceUnavailable FeatureProofFailure = "SOURCE_UNAVAILABLE"
)

func (f FeatureProofFailure) Error() string { return string(f) }

// FeatureProofTokenMethod labels estimates explicitly. The empty method is used
// only when both token measurements are unavailable (JSON null).
type FeatureProofTokenMethod string

const FeatureProofAnthropicEstimate FeatureProofTokenMethod = "anthropic_estimate"
const FeatureProofTypedMessageEstimate FeatureProofTokenMethod = "typed_message_estimate"

// FeatureRewriteWitness supplies the regions chosen by the producer's structural
// verifier. VerifyFeatureRewrite proves byte equality of those regions, not that
// an arbitrary caller chose the correct cache anchor. Anthropic producers must
// first use agent.VerifyAnthropicRewrite or agent.VerifyAnthropicElision.
type FeatureRewriteWitness struct {
	PrefixBytes int
	SuffixBytes int
	ShedBytes   int
	PreTokens   *int
	PostTokens  *int
	TokenMethod FeatureProofTokenMethod
}

// FeatureRewriteProof retains only quantitative evidence and digests.
type FeatureRewriteProof struct {
	SerializationMethod  string                  `json:"serialization_method"`
	ProtectedRegionsKind string                  `json:"protected_regions_kind"`
	PreBytes             int                     `json:"pre_bytes"`
	PostBytes            int                     `json:"post_bytes"`
	ShedBytes            int                     `json:"shed_bytes"`
	PreSHA256            string                  `json:"pre_sha256"`
	PostSHA256           string                  `json:"post_sha256"`
	PrefixBytes          int                     `json:"prefix_bytes"`
	SuffixBytes          int                     `json:"suffix_bytes"`
	PrePrefixSHA256      string                  `json:"pre_prefix_sha256"`
	PostPrefixSHA256     string                  `json:"post_prefix_sha256"`
	PreSuffixSHA256      string                  `json:"pre_suffix_sha256"`
	PostSuffixSHA256     string                  `json:"post_suffix_sha256"`
	PreTokens            *int                    `json:"pre_tokens"`
	PostTokens           *int                    `json:"post_tokens"`
	TokenMethod          FeatureProofTokenMethod `json:"token_method"`
}

// FeatureCacheProof binds a gateway call identity to the served payload. Its hash
// is not presented as the vDSO's internal lookup key. Timing is a signed historical
// estimate from a trusted prior engine span and this lookup, never a counterfactual
// measurement of the avoided call. Unmeasured entries encode null quantities.
type FeatureCacheProof struct {
	HitCount            uint64      `json:"hit_count"`
	AdjudicationVerdict WireVerdict `json:"adjudication_verdict"`
	IdentitySHA256      string      `json:"identity_sha256"`
	PayloadSHA256       string      `json:"payload_sha256"`
	CacheWitnessHash    string      `json:"cache_witness_hash"`
	WitnessKind         string      `json:"witness_kind"`
	DurationSavedNS     *int64      `json:"duration_saved_ns"`
	DurationReason      string      `json:"duration_reason"`
	DurationBasis       string      `json:"duration_basis"`
	HistoricalEngineNS  *int64      `json:"historical_engine_ns"`
	CurrentLookupNS     *int64      `json:"current_lookup_ns"`
}

// FeatureProofReceipt is an in-memory observation of one executed producer.
// Failed proof generation retains no partial hashes or savings quantities.
type FeatureProofReceipt struct {
	Schema     string               `json:"schema"`
	Sequence   uint64               `json:"sequence"`
	ObservedAt string               `json:"observed_at"`
	Feature    ServeFeature         `json:"feature"`
	Outcome    FeatureProofOutcome  `json:"outcome"`
	Reason     string               `json:"reason,omitempty"`
	Failure    FeatureProofFailure  `json:"failure,omitempty"`
	Rewrite    *FeatureRewriteProof `json:"rewrite,omitempty"`
	Cache      *FeatureCacheProof   `json:"cache,omitempty"`
}

// FeatureProofCounter is a cumulative count since collector creation. It is
// independent of bounded receipt retention; failed proofs earn no saved bytes.
type FeatureProofCounter struct {
	Feature    ServeFeature        `json:"feature"`
	Outcome    FeatureProofOutcome `json:"outcome"`
	Count      uint64              `json:"count"`
	SavedBytes uint64              `json:"saved_bytes"`
}

// FeatureProofSummary separates supported producers from observed executions.
// Coverage includes applied Anthropic compaction/elision, successful native
// typed compaction, and accepted identified vDSO serves.
type FeatureProofSummary struct {
	Schema            string                `json:"schema"`
	SupportedFeatures []ServeFeature        `json:"supported_features"`
	Receipts          []FeatureProofReceipt `json:"receipts"`
	EvictedReceipts   uint64                `json:"evicted_receipts"`
	Counters          []FeatureProofCounter `json:"counters"`
	CountersScope     string                `json:"counters_scope"`
	RetainedWindow    FeatureProofWindow    `json:"retained_window"`
}

// FeatureProofWindow aggregates only the receipts currently retained in the ring.
// Its sequence bounds identify the observed window; an empty ring has zero bounds.
type FeatureProofWindow struct {
	ReceiptCount  int                   `json:"receipt_count"`
	FirstSequence uint64                `json:"first_sequence"`
	LastSequence  uint64                `json:"last_sequence"`
	Counters      []FeatureProofCounter `json:"counters"`
}

// FeatureProofCollector retains a bounded ring and fixed-cardinality counters.
// Request bytes are hashed synchronously, never retained, logged, or persisted.
type FeatureProofCollector struct {
	mu       sync.Mutex
	ring     []FeatureProofReceipt
	next     int
	count    int
	sequence uint64
	evicted  uint64
	counters [3][2]FeatureProofCounter
}

var (
	errFeatureProofCapacity    = errors.New("INVALID_FEATURE_PROOF_CAPACITY")
	errFeatureProofUnsupported = errors.New("UNSUPPORTED_PROOF_FEATURE")
	errFeatureProofUnavailable = errors.New("FEATURE_PROOF_COLLECTOR_UNAVAILABLE")
	errFeatureProofFailure     = errors.New("INVALID_PROOF_FAILURE")
)

func proofFeatures() [3]ServeFeature {
	return [3]ServeFeature{FeatureCompactHistory, FeatureElideResults, FeatureVDSO}
}

func proofFeatureIndex(feature ServeFeature) int {
	if !knownServeFeature(feature) {
		return -1
	}
	for i, supported := range proofFeatures() {
		if feature == supported {
			return i
		}
	}
	return -1
}

// NewFeatureProofCollector accepts an explicit capacity in [1,256].
func NewFeatureProofCollector(capacity int) (*FeatureProofCollector, error) {
	if capacity < 1 || capacity > MaxFeatureProofCapacity {
		return nil, errFeatureProofCapacity
	}
	c := &FeatureProofCollector{ring: make([]FeatureProofReceipt, capacity)}
	for i, feature := range proofFeatures() {
		c.counters[i][0] = FeatureProofCounter{Feature: feature, Outcome: FeatureProofVerified}
		c.counters[i][1] = FeatureProofCounter{Feature: feature, Outcome: FeatureProofFailed}
	}
	return c, nil
}

// VerifyFeatureRewrite checks exact byte deltas and both protected regions. It
// is pure; failures neither retain inputs nor return unsanitized parsing errors.
func VerifyFeatureRewrite(before, after []byte, witness FeatureRewriteWitness) (FeatureRewriteProof, error) {
	if len(before) <= len(after) || witness.ShedBytes != len(before)-len(after) {
		return FeatureRewriteProof{}, FeatureProofInvalidDelta
	}
	p, s := witness.PrefixBytes, witness.SuffixBytes
	if p <= 0 || s <= 0 || p > len(after) || s > len(after) || p > len(after)-s {
		return FeatureRewriteProof{}, FeatureProofInvalidBounds
	}
	if !bytes.Equal(before[:p], after[:p]) {
		return FeatureRewriteProof{}, FeatureProofPrefixMismatch
	}
	if !bytes.Equal(before[len(before)-s:], after[len(after)-s:]) {
		return FeatureRewriteProof{}, FeatureProofSuffixMismatch
	}
	if witness.PreTokens == nil && witness.PostTokens == nil {
		if witness.TokenMethod != "" {
			return FeatureRewriteProof{}, FeatureProofInvalidTokens
		}
	} else if witness.PreTokens == nil || witness.PostTokens == nil ||
		*witness.PreTokens < 0 || *witness.PostTokens < 0 || witness.TokenMethod != FeatureProofAnthropicEstimate {
		return FeatureRewriteProof{}, FeatureProofInvalidTokens
	}
	return FeatureRewriteProof{
		SerializationMethod: "anthropic_request_json", ProtectedRegionsKind: "cache_anchor_and_trailing_json",
		PreBytes: len(before), PostBytes: len(after), ShedBytes: witness.ShedBytes,
		PreSHA256: featureProofHash(before), PostSHA256: featureProofHash(after),
		PrefixBytes: p, SuffixBytes: s,
		PrePrefixSHA256: featureProofHash(before[:p]), PostPrefixSHA256: featureProofHash(after[:p]),
		PreSuffixSHA256: featureProofHash(before[len(before)-s:]), PostSuffixSHA256: featureProofHash(after[len(after)-s:]),
		PreTokens: copyProofInt(witness.PreTokens), PostTokens: copyProofInt(witness.PostTokens), TokenMethod: witness.TokenMethod,
	}, nil
}

// recordNativeCompaction consumes the value-only observation produced at the
// actual typed compaction seam, after successful native completion. Its region
// hashes establish shared message bytes, not tokenizer or KV cache identity.
func (c *FeatureProofCollector) recordNativeCompaction(v agent.NativeCompactionObservation) {
	if c == nil || v.DroppedMessages <= 0 {
		return
	}
	defer func() {
		if recover() != nil {
			_ = c.RecordFailure(FeatureCompactHistory, FeatureProofInvalidBody)
		}
	}()
	failure := FeatureProofFailure("")
	switch {
	case v.ProofFailure == "NONPOSITIVE_BYTE_DELTA":
		failure = FeatureProofInvalidDelta
	case v.ProofFailure != "" || v.SerializationMethod != "typed_messages_json":
		failure = FeatureProofInvalidBody
	case v.PostBytes < 0 || v.PreBytes <= v.PostBytes || v.ShedBytes != v.PreBytes-v.PostBytes:
		failure = FeatureProofInvalidDelta
	case v.PrefixBytes < 0 || v.SuffixBytes < 0 || v.PrefixBytes > v.PostBytes || v.SuffixBytes > v.PostBytes-v.PrefixBytes:
		failure = FeatureProofInvalidBounds
	case v.PrePrefixSHA256 != v.PostPrefixSHA256 || (v.PrefixBytes == 0 && v.PrePrefixSHA256 != featureProofHash(nil)):
		failure = FeatureProofPrefixMismatch
	case v.PreSuffixSHA256 != v.PostSuffixSHA256 || (v.SuffixBytes == 0 && v.PreSuffixSHA256 != featureProofHash(nil)):
		failure = FeatureProofSuffixMismatch
	case v.PreEstimatedTokens < 0 || v.PostEstimatedTokens < 0 || v.TokenMethod != string(FeatureProofTypedMessageEstimate):
		failure = FeatureProofInvalidTokens
	}
	for _, digest := range []string{v.PreSHA256, v.PostSHA256, v.PrePrefixSHA256, v.PostPrefixSHA256, v.PreSuffixSHA256, v.PostSuffixSHA256} {
		decoded, err := hex.DecodeString(digest)
		if failure == "" && (err != nil || len(decoded) != sha256.Size) {
			failure = FeatureProofInvalidBody
		}
	}
	if failure != "" {
		_ = c.RecordFailure(FeatureCompactHistory, failure)
		return
	}
	proof := &FeatureRewriteProof{
		SerializationMethod: v.SerializationMethod, ProtectedRegionsKind: "shared_whole_messages",
		PreBytes: v.PreBytes, PostBytes: v.PostBytes, ShedBytes: v.ShedBytes,
		PreSHA256: v.PreSHA256, PostSHA256: v.PostSHA256,
		PrefixBytes: v.PrefixBytes, SuffixBytes: v.SuffixBytes,
		PrePrefixSHA256: v.PrePrefixSHA256, PostPrefixSHA256: v.PostPrefixSHA256,
		PreSuffixSHA256: v.PreSuffixSHA256, PostSuffixSHA256: v.PostSuffixSHA256,
		PreTokens: copyProofInt(&v.PreEstimatedTokens), PostTokens: copyProofInt(&v.PostEstimatedTokens),
		TokenMethod: FeatureProofTypedMessageEstimate,
	}
	_ = c.append(FeatureProofReceipt{Feature: FeatureCompactHistory, Outcome: FeatureProofVerified, Rewrite: proof})
}

func (s *Server) withNativeFeatureProof(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if c := s.featureProofCollector(); c != nil {
			r = r.WithContext(agent.WithNativeCompactionObserver(r.Context(), c.recordNativeCompaction))
		}
		next.ServeHTTP(w, r)
	})
}

// RecordRewrite is called after a producer applies a rewrite and verifies its
// structural anchor. Identity with zero claimed savings is a no-op. Callers must
// never turn a returned telemetry error into a request failure.
func (c *FeatureProofCollector) RecordRewrite(feature ServeFeature, before, after []byte, witness FeatureRewriteWitness) error {
	if feature != FeatureCompactHistory && feature != FeatureElideResults {
		return errFeatureProofUnsupported
	}
	if witness.ShedBytes == 0 && bytes.Equal(before, after) {
		return nil
	}
	proof, err := VerifyFeatureRewrite(before, after, witness)
	if err != nil {
		if recordErr := c.RecordFailure(feature, err.(FeatureProofFailure)); recordErr != nil {
			return recordErr
		}
		return err
	}
	return c.append(FeatureProofReceipt{Feature: feature, Outcome: FeatureProofVerified, Rewrite: &proof})
}

// RecordVDSOServe is called only after a confirmed vDSO source passes freshness
// and payload screening and its result is accepted for serving. The identity
// must be a digest of the actual canonical call, not a random ID or a tool name.
func (c *FeatureProofCollector) RecordVDSOServe(identity [32]byte, payload []byte, verdict WireVerdict) error {
	return c.recordVDSOServeMeasured(identity, payload, verdict, 0, 0)
}

func (c *FeatureProofCollector) recordVDSOServeMeasured(identity [32]byte, payload []byte, verdict WireVerdict, historicalEngineNS, lookupNS int64) error {
	if identity == ([32]byte{}) || verdict.Kind != "ALLOW" || verdict.Reason != "SERVED_INLINE" || verdict.By != "vdso" {
		if err := c.RecordFailure(FeatureVDSO, FeatureProofInvalidCache); err != nil {
			return err
		}
		return FeatureProofInvalidCache
	}
	payloadHash := sha256.Sum256(payload)
	h := sha256.New()
	_, _ = h.Write([]byte("fak-gateway-call-result/1\x00"))
	_, _ = h.Write(identity[:])
	_, _ = h.Write(payloadHash[:])
	proof := &FeatureCacheProof{
		HitCount:            1,
		AdjudicationVerdict: WireVerdict{Kind: "ALLOW", Reason: "SERVED_INLINE", By: "vdso"},
		IdentitySHA256:      hex.EncodeToString(identity[:]), PayloadSHA256: hex.EncodeToString(payloadHash[:]),
		CacheWitnessHash: hex.EncodeToString(h.Sum(nil)), WitnessKind: "gateway_call_result_sha256",
		DurationReason: "MATCHED_TIMING_UNAVAILABLE",
	}
	if historicalEngineNS > 0 && lookupNS >= 0 {
		saved := historicalEngineNS - lookupNS
		proof.DurationSavedNS = &saved
		proof.HistoricalEngineNS = &historicalEngineNS
		proof.CurrentLookupNS = &lookupNS
		proof.DurationReason = "HISTORICAL_TIMING_ESTIMATE"
		proof.DurationBasis = "historical_engine_minus_current_lookup_estimate"
	}
	return c.append(FeatureProofReceipt{Feature: FeatureVDSO, Outcome: FeatureProofVerified, Cache: proof})
}

// RecordFailure records an executed producer whose proof could not be generated.
// Unsupported features and unrecognized failure details cannot expand the ring.
func (c *FeatureProofCollector) RecordFailure(feature ServeFeature, detail FeatureProofFailure) error {
	if proofFeatureIndex(feature) < 0 {
		return errFeatureProofUnsupported
	}
	switch detail {
	case FeatureProofInvalidDelta, FeatureProofInvalidBounds, FeatureProofPrefixMismatch,
		FeatureProofSuffixMismatch, FeatureProofInvalidTokens, FeatureProofAnchorUnavailable,
		FeatureProofInvalidBody, FeatureProofInvalidCache, FeatureProofSourceUnavailable:
	default:
		return errFeatureProofFailure
	}
	return c.append(FeatureProofReceipt{Feature: feature, Outcome: FeatureProofFailed,
		Reason: "PROOF_GENERATION_FAILED", Failure: detail})
}

func (c *FeatureProofCollector) append(receipt FeatureProofReceipt) error {
	if c == nil {
		return errFeatureProofUnavailable
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.ring) == 0 {
		return errFeatureProofUnavailable
	}
	c.sequence++
	receipt.Schema = FeatureProofSchema
	receipt.Sequence = c.sequence
	receipt.ObservedAt = time.Now().UTC().Format(time.RFC3339Nano)
	c.ring[c.next] = receipt
	c.next = (c.next + 1) % len(c.ring)
	if c.count == len(c.ring) {
		c.evicted++
	} else {
		c.count++
	}
	outcome := 0
	if receipt.Outcome == FeatureProofFailed {
		outcome = 1
	}
	counter := &c.counters[proofFeatureIndex(receipt.Feature)][outcome]
	counter.Count++
	if receipt.Rewrite != nil && receipt.Outcome == FeatureProofVerified {
		counter.SavedBytes += uint64(receipt.Rewrite.ShedBytes)
	}
	return nil
}

// Snapshot returns detached chronological receipts, retained-window aggregates,
// and separately labeled cumulative counters.
// A nil collector yields an empty snapshot, never synthetic executions.
func (c *FeatureProofCollector) Snapshot() FeatureProofSummary {
	features := proofFeatures()
	result := FeatureProofSummary{Schema: FeatureProofSchema,
		SupportedFeatures: append([]ServeFeature{}, features[:]...),
		Receipts:          []FeatureProofReceipt{}, Counters: []FeatureProofCounter{},
		CountersScope:  "collector_lifetime",
		RetainedWindow: FeatureProofWindow{Counters: []FeatureProofCounter{}},
	}
	if c == nil {
		return result
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	result.EvictedReceipts = c.evicted
	var retained [3][2]FeatureProofCounter
	for i, feature := range features {
		retained[i][0] = FeatureProofCounter{Feature: feature, Outcome: FeatureProofVerified}
		retained[i][1] = FeatureProofCounter{Feature: feature, Outcome: FeatureProofFailed}
	}
	for i := 0; i < c.count; i++ {
		index := (c.next - c.count + i + len(c.ring)) % len(c.ring)
		r := c.ring[index]
		if r.Rewrite != nil {
			proof := *r.Rewrite
			proof.PreTokens = copyProofInt(proof.PreTokens)
			proof.PostTokens = copyProofInt(proof.PostTokens)
			r.Rewrite = &proof
		}
		if r.Cache != nil {
			proof := *r.Cache
			proof.DurationSavedNS = copyProofInt64(proof.DurationSavedNS)
			proof.HistoricalEngineNS = copyProofInt64(proof.HistoricalEngineNS)
			proof.CurrentLookupNS = copyProofInt64(proof.CurrentLookupNS)
			r.Cache = &proof
		}
		result.Receipts = append(result.Receipts, r)
		outcome := 0
		if r.Outcome == FeatureProofFailed {
			outcome = 1
		}
		counter := &retained[proofFeatureIndex(r.Feature)][outcome]
		counter.Count++
		if r.Outcome == FeatureProofVerified && r.Rewrite != nil {
			counter.SavedBytes += uint64(r.Rewrite.ShedBytes)
		}
	}
	result.RetainedWindow.ReceiptCount = len(result.Receipts)
	if len(result.Receipts) > 0 {
		result.RetainedWindow.FirstSequence = result.Receipts[0].Sequence
		result.RetainedWindow.LastSequence = result.Receipts[len(result.Receipts)-1].Sequence
	}
	for _, outcomes := range retained {
		result.RetainedWindow.Counters = append(result.RetainedWindow.Counters, outcomes[:]...)
	}
	for _, outcomes := range c.counters {
		result.Counters = append(result.Counters, outcomes[:]...)
	}
	return result
}

func featureProofHash(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func copyProofInt(value *int) *int {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func copyProofInt64(value *int64) *int64 {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func (s *Server) featureProofCollector() *FeatureProofCollector {
	if s == nil || s.metrics == nil {
		return nil
	}
	return s.metrics.featureProof
}

func (s *Server) handleFeatureProof(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, http.StatusOK, s.featureProofCollector().Snapshot())
}

func (s *Server) recordAnthropicRewriteProof(feature ServeFeature, before, after []byte, anchor agent.CompactAnchor, shedBytes int) {
	c := s.featureProofCollector()
	if c == nil {
		return
	}
	// A proof failure must not replace the producer's already-applied request.
	defer func() {
		if recover() != nil {
			_ = c.RecordFailure(feature, FeatureProofInvalidBody)
		}
	}()
	var structural agent.AnthropicRewriteProof
	var err error
	if feature == FeatureElideResults {
		structural, err = agent.VerifyAnthropicElision(before, after)
	} else {
		structural, err = agent.VerifyAnthropicRewrite(before, after, anchor)
	}
	if err != nil {
		failure := FeatureProofInvalidBody
		switch {
		case errors.Is(err, agent.ErrAnthropicProofAnchor):
			failure = FeatureProofAnchorUnavailable
		case errors.Is(err, agent.ErrAnthropicProofDelta):
			failure = FeatureProofInvalidDelta
		case errors.Is(err, agent.ErrAnthropicProofPrefix):
			failure = FeatureProofPrefixMismatch
		}
		_ = c.RecordFailure(feature, failure)
		return
	}
	_ = c.RecordRewrite(feature, before, after, FeatureRewriteWitness{
		PrefixBytes: structural.PrefixBytes, SuffixBytes: structural.SuffixBytes,
		ShedBytes: shedBytes, PreTokens: &structural.PreTokens, PostTokens: &structural.PostTokens,
		TokenMethod: FeatureProofAnthropicEstimate,
	})
}

func (s *Server) recordVDSOServeProof(ctx context.Context, call *abi.ToolCall, payload []byte, receipt *vdso.LookupReceipt, result *abi.Result) {
	c := s.featureProofCollector()
	if c == nil {
		return
	}
	defer func() {
		if recover() != nil {
			_ = c.RecordFailure(FeatureVDSO, FeatureProofInvalidCache)
		}
	}()
	if call == nil {
		_ = c.RecordFailure(FeatureVDSO, FeatureProofInvalidCache)
		return
	}
	args := resolveBytes(ctx, call.Args)
	if args == nil && call.Args.Kind != abi.RefInline {
		_ = c.RecordFailure(FeatureVDSO, FeatureProofSourceUnavailable)
		return
	}
	// This is a canonical gateway call witness, not the vDSO key (which also
	// includes private epoch/dependency state). Bind the actual looked-up call's
	// principal without retaining its identity or copying open-ended metadata.
	identity, err := json.Marshal(struct {
		Schema     string `json:"schema"`
		Tool       string `json:"tool"`
		Engine     string `json:"engine"`
		ArgsDigest string `json:"args_digest"`
		Principal  string `json:"principal"`
	}{"fak-gateway-call/1", call.Tool, call.Engine, guardrsi.ArgsDigest(string(args)), call.Meta[vdso.MetaPrincipal]})
	if err != nil {
		_ = c.RecordFailure(FeatureVDSO, FeatureProofInvalidCache)
		return
	}
	engineNS, lookupNS, _ := receipt.Timing(result)
	_ = c.recordVDSOServeMeasured(sha256.Sum256(identity), payload, WireVerdict{
		Kind: "ALLOW", Reason: "SERVED_INLINE", By: "vdso",
	}, engineNS, lookupNS)
}

func (c *FeatureProofCollector) writeMetrics(b *strings.Builder) {
	snapshot := c.Snapshot()
	writeHelpType(b, "fak_feature_activations_total", "Executed proof-producing features by verification outcome; excludes enablement, plans and idle state.", "counter")
	for _, counter := range snapshot.Counters {
		fmt.Fprintf(b, "fak_feature_activations_total{feature=%q,outcome=%q} %d\n", counter.Feature, counter.Outcome, counter.Count)
	}
	writeHelpType(b, "fak_feature_saved_bytes_total", "Verified request bytes shed by supported proof-producing features since startup.", "counter")
	for _, counter := range snapshot.Counters {
		if counter.Outcome == FeatureProofVerified {
			fmt.Fprintf(b, "fak_feature_saved_bytes_total{feature=%q} %d\n", counter.Feature, counter.SavedBytes)
		}
	}
}
