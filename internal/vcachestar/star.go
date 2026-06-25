package vcachestar

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"sort"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// Section is the render bucket a prompt part belongs to. Canonical rendering is
// always tools -> system -> messages, regardless of caller input order.
type Section string

const (
	SectionTools    Section = "tools"
	SectionSystem   Section = "system"
	SectionMessages Section = "messages"
)

// Part is one pre-serialized prompt part before M2 canonicalization.
type Part struct {
	Section Section
	Kind    cachemeta.SegmentKind
	Name    string
	Content []byte
	Tokens  int64
	// JSON asks the serializer to canonicalize Content with encoding/json. Tool
	// parts are always treated as JSON when they parse as JSON, because tool
	// schema key order must not depend on map iteration or source formatting.
	JSON bool
}

// Identity scopes a provider-prefix manifest key. Changing any field is a hard
// miss even when the rendered bytes happen to match.
type Identity struct {
	ModelID          string
	TokenizerEpoch   string
	ToolSetHash      string
	BreakpointLayout string
	TTL              string
	ProviderSurface  string
}

// Complete reports whether all Law-B scope axes are populated.
func (i Identity) Complete() bool {
	return i.ModelID != "" &&
		i.TokenizerEpoch != "" &&
		i.ToolSetHash != "" &&
		i.BreakpointLayout != "" &&
		i.TTL != "" &&
		i.ProviderSurface != ""
}

// ManifestKey is the M2 provider-prefix identity. PrefixHash is ONLY the hash of
// exact serialized prefix bytes; Scope carries the separate hard-miss axes.
type ManifestKey struct {
	PrefixHash   string
	PrefixBytes  int
	PrefixTokens int64
	Scope        Identity
}

// Reason is the typed explanation for a preflight, match, plan, or telemetry
// decision.
type Reason string

const (
	ReasonNone                     Reason = ""
	ReasonEmptyRequest             Reason = "empty_request"
	ReasonIncompleteIdentity       Reason = "incomplete_identity"
	ReasonCanonicalWarmMismatch    Reason = "canonical_warm_mismatch"
	ReasonBelowMinimumAnchor       Reason = "below_minimum_anchor"
	ReasonNoExpectedSiblings       Reason = "no_expected_siblings"
	ReasonPrefixHashMismatch       Reason = "prefix_hash_mismatch"
	ReasonModelMismatch            Reason = "model_mismatch"
	ReasonTokenizerMismatch        Reason = "tokenizer_mismatch"
	ReasonToolSetMismatch          Reason = "tool_set_mismatch"
	ReasonBreakpointLayoutMismatch Reason = "breakpoint_layout_mismatch"
	ReasonTTLMismatch              Reason = "ttl_mismatch"
	ReasonProviderSurfaceMismatch  Reason = "provider_surface_mismatch"
	ReasonBelievedWarmZeroRead     Reason = "believed_warm_zero_read"
)

// Match compares two manifest keys and returns the first hard-miss axis.
func (k ManifestKey) Match(want ManifestKey) (bool, Reason) {
	switch {
	case k.PrefixHash != want.PrefixHash:
		return false, ReasonPrefixHashMismatch
	case k.Scope.ModelID != want.Scope.ModelID:
		return false, ReasonModelMismatch
	case k.Scope.TokenizerEpoch != want.Scope.TokenizerEpoch:
		return false, ReasonTokenizerMismatch
	case k.Scope.ToolSetHash != want.Scope.ToolSetHash:
		return false, ReasonToolSetMismatch
	case k.Scope.BreakpointLayout != want.Scope.BreakpointLayout:
		return false, ReasonBreakpointLayoutMismatch
	case k.Scope.TTL != want.Scope.TTL:
		return false, ReasonTTLMismatch
	case k.Scope.ProviderSurface != want.Scope.ProviderSurface:
		return false, ReasonProviderSurfaceMismatch
	default:
		return true, ReasonNone
	}
}

// Digest returns lowercase hex sha256 for deterministic manifests.
func Digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// PreflightRequest is the M2 canonicalizer-as-gate input.
type PreflightRequest struct {
	Parts           []Part
	Scope           Identity
	MinAnchorTokens int64
	// WarmCandidateBytes, when present, is a caller-proposed warm prefix. It must
	// equal the canonical serialized prefix bytes or the candidate is refused.
	WarmCandidateBytes []byte
}

// PreflightAction is what the caller must do with the prompt layout before a
// provider request may be treated as a warm anchor candidate.
type PreflightAction string

const (
	ActionRefuse  PreflightAction = "refuse"
	ActionAccept  PreflightAction = "accept"
	ActionRewrite PreflightAction = "rewrite"
)

// PreflightResult carries the applied layout, serialized prefix bytes, and
// manifest key. Applied contains the whole request layout after volatile
// segments were hoisted to the tail; Anchor contains only the cacheable front
// run that precedes volatile or sealed material.
type PreflightResult struct {
	Action         PreflightAction
	Reason         Reason
	Recommendation cachemeta.LayoutRecommendation
	Applied        []cachemeta.PromptSegment
	Anchor         []cachemeta.PromptSegment
	PrefixBytes    []byte
	Key            ManifestKey
}

// Preflight applies RecommendLayout, serializes the cacheable anchor prefix, and
// refuses a supplied warm candidate whose bytes do not equal that canonical
// prefix. No manifest key is returned until identity is complete.
func Preflight(req PreflightRequest) PreflightResult {
	if len(req.Parts) == 0 {
		return PreflightResult{Action: ActionRefuse, Reason: ReasonEmptyRequest}
	}

	segs, toolBytes := canonicalSegments(req.Parts)
	scope := req.Scope
	if scope.ToolSetHash == "" {
		scope.ToolSetHash = Digest(toolBytes)
	}
	if !scope.Complete() {
		return PreflightResult{Action: ActionRefuse, Reason: ReasonIncompleteIdentity}
	}

	rec := cachemeta.RecommendLayout(segs)
	applied := cloneSegments(rec.Reordered)
	anchor := cacheableFront(applied)
	tokens := tokenSum(anchor)
	if req.MinAnchorTokens > 0 && tokens < req.MinAnchorTokens {
		return PreflightResult{
			Action:         ActionRefuse,
			Reason:         ReasonBelowMinimumAnchor,
			Recommendation: rec,
			Applied:        applied,
			Anchor:         anchor,
		}
	}

	prefix := serializeSegments(anchor)
	if req.WarmCandidateBytes != nil && !bytes.Equal(req.WarmCandidateBytes, prefix) {
		return PreflightResult{
			Action:         ActionRefuse,
			Reason:         ReasonCanonicalWarmMismatch,
			Recommendation: rec,
			Applied:        applied,
			Anchor:         anchor,
			PrefixBytes:    prefix,
		}
	}

	action := ActionAccept
	if rec.Changed {
		action = ActionRewrite
	}
	return PreflightResult{
		Action:         action,
		Recommendation: rec,
		Applied:        applied,
		Anchor:         anchor,
		PrefixBytes:    prefix,
		Key: ManifestKey{
			PrefixHash:   Digest(prefix),
			PrefixBytes:  len(prefix),
			PrefixTokens: tokens,
			Scope:        scope,
		},
	}
}

func canonicalSegments(parts []Part) ([]cachemeta.PromptSegment, []byte) {
	groups := map[Section][]Part{
		SectionTools:    nil,
		SectionSystem:   nil,
		SectionMessages: nil,
	}
	for _, p := range parts {
		section := canonicalSection(p.Section)
		groups[section] = append(groups[section], p)
	}
	sort.SliceStable(groups[SectionTools], func(i, j int) bool {
		a, b := groups[SectionTools][i], groups[SectionTools][j]
		if a.Name != b.Name {
			return a.Name < b.Name
		}
		return bytes.Compare(a.Content, b.Content) < 0
	})

	var out []cachemeta.PromptSegment
	var toolBytes []byte
	for _, section := range []Section{SectionTools, SectionSystem, SectionMessages} {
		for _, p := range groups[section] {
			content := normalizeWireBytes(p.Content)
			if p.JSON || section == SectionTools {
				if canon, err := canonicalJSON(content); err == nil {
					content = canon
				}
			}
			kind := p.Kind
			if kind == "" {
				kind = defaultKind(section)
			}
			if section == SectionTools {
				toolBytes = append(toolBytes, content...)
				toolBytes = append(toolBytes, '\n')
			}
			out = append(out, cachemeta.PromptSegment{
				Kind:    kind,
				Tokens:  p.Tokens,
				Content: content,
			})
		}
	}
	return out, toolBytes
}

func canonicalSection(section Section) Section {
	switch section {
	case SectionTools, SectionSystem, SectionMessages:
		return section
	default:
		return SectionMessages
	}
}

func defaultKind(section Section) cachemeta.SegmentKind {
	switch section {
	case SectionTools:
		return cachemeta.SegToolSchema
	case SectionSystem:
		return cachemeta.SegStable
	default:
		return cachemeta.SegMessage
	}
}

func canonicalJSON(raw []byte) ([]byte, error) {
	var v any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		return nil, errors.New("vcachestar: trailing json value")
	}
	return json.Marshal(v)
}

func normalizeWireBytes(in []byte) []byte {
	out := bytes.TrimPrefix(in, []byte{0xef, 0xbb, 0xbf})
	out = bytes.ReplaceAll(out, []byte("\r\n"), []byte("\n"))
	out = bytes.ReplaceAll(out, []byte("\r"), []byte("\n"))
	return append([]byte(nil), out...)
}

func cacheableFront(segs []cachemeta.PromptSegment) []cachemeta.PromptSegment {
	var out []cachemeta.PromptSegment
	for _, s := range segs {
		if s.Kind == cachemeta.SegVolatile || s.Kind == cachemeta.SegSealed {
			break
		}
		out = append(out, s)
	}
	return cloneSegments(out)
}

func serializeSegments(segs []cachemeta.PromptSegment) []byte {
	var out []byte
	for _, s := range segs {
		out = append(out, s.Content...)
	}
	return out
}

func tokenSum(segs []cachemeta.PromptSegment) int64 {
	var n int64
	for _, s := range segs {
		n += s.Tokens
	}
	return n
}

func cloneSegments(in []cachemeta.PromptSegment) []cachemeta.PromptSegment {
	out := make([]cachemeta.PromptSegment, len(in))
	for i, s := range in {
		out[i] = s
		out[i].Content = append([]byte(nil), s.Content...)
	}
	return out
}

// StarRequest describes one star workload: one shared anchor and many small
// independent suffix units.
type StarRequest struct {
	Key                  ManifestKey
	AnchorTokens         int64
	MinAnchorTokens      int64
	UnitTokens           int64
	ExpectedSiblingReads int
}

// Strategy is the selected M2 anchor strategy.
type Strategy string

const (
	StrategyNone             Strategy = ""
	StrategyFirstNaturalWarm Strategy = "first_natural_request_warms"
)

// CacheUnit names what the manifest models as cacheable.
type CacheUnit string

const (
	CacheUnitNone   CacheUnit = ""
	CacheUnitAnchor CacheUnit = "anchor"
)

// StarDecision is the M2 plan for a star workload.
type StarDecision struct {
	Strategy                 Strategy
	Reason                   Reason
	CacheUnit                CacheUnit
	DedicatedWarm            bool
	FirstNaturalRequestWarms bool
	AnchorTokens             int64
	UnitTokens               int64
	ExpectedSiblingReads     int
	Key                      ManifestKey
}

// Plan models the anchor, not the suffix unit, as the manifest cache unit. M2
// never spends a dedicated warm: the first natural request writes the anchor and
// later siblings read it if provider telemetry confirms the hit.
func Plan(req StarRequest) StarDecision {
	anchorTokens := req.AnchorTokens
	if anchorTokens == 0 {
		anchorTokens = req.Key.PrefixTokens
	}
	base := StarDecision{
		CacheUnit:            CacheUnitAnchor,
		AnchorTokens:         anchorTokens,
		UnitTokens:           req.UnitTokens,
		ExpectedSiblingReads: req.ExpectedSiblingReads,
		Key:                  req.Key,
	}
	if !req.Key.Scope.Complete() || req.Key.PrefixHash == "" {
		base.CacheUnit = CacheUnitNone
		base.Reason = ReasonIncompleteIdentity
		return base
	}
	if req.MinAnchorTokens > 0 && anchorTokens < req.MinAnchorTokens {
		base.Reason = ReasonBelowMinimumAnchor
		return base
	}
	if req.ExpectedSiblingReads <= 0 {
		base.Reason = ReasonNoExpectedSiblings
		return base
	}
	base.Strategy = StrategyFirstNaturalWarm
	base.FirstNaturalRequestWarms = true
	base.DedicatedWarm = false
	return base
}

// Belief is the local warmth belief for one anchor. It is never authority for
// correctness; it only predicts whether the next provider call may rebate cost.
type Belief struct {
	Key                 ManifestKey
	Warm                bool
	LastPrefix          []cachemeta.PromptSegment
	LastPrefixBytes     []byte
	ConfirmedReadTokens int64
}

// Telemetry is the provider read-back for one completed real call.
type Telemetry struct {
	CacheReadInputTokens int64
	UncachedInputTokens  int64
	CurrentPrefix        []cachemeta.PromptSegment
	CurrentPrefixBytes   []byte
}

// CostBooking records Law A accounting: book the full uncached input first, then
// apply a rebate only when cache_read_input_tokens confirms a hit.
type CostBooking struct {
	BookedUncachedTokens int64
	RebateTokens         int64
}

// FoldResult is the reconcile-after-real-call verdict.
type FoldResult struct {
	Belief                  Belief
	ConfirmedHit            bool
	Demoted                 bool
	Alarm                   bool
	Reason                  Reason
	Divergence              cachemeta.TurnDivergence
	FirstDivergeTokenOffset int64
	FirstDivergeByteOffset  int
	Cost                    CostBooking
}

// FoldTelemetry reconciles a warmth belief with real provider telemetry. A
// believed-warm zero-read demotes immediately and reports both token and byte
// divergence when prior/current prefixes were supplied.
func FoldTelemetry(b Belief, t Telemetry) FoldResult {
	next := b
	if len(t.CurrentPrefix) > 0 {
		next.LastPrefix = cloneSegments(t.CurrentPrefix)
	}
	if t.CurrentPrefixBytes != nil {
		next.LastPrefixBytes = append([]byte(nil), t.CurrentPrefixBytes...)
	}
	res := FoldResult{
		Belief: next,
		Cost: CostBooking{
			BookedUncachedTokens: t.UncachedInputTokens,
		},
		FirstDivergeByteOffset: -1,
	}
	if res.Cost.BookedUncachedTokens == 0 {
		res.Cost.BookedUncachedTokens = tokenSum(t.CurrentPrefix)
	}
	if t.CacheReadInputTokens > 0 {
		res.ConfirmedHit = true
		res.Cost.RebateTokens = t.CacheReadInputTokens
		res.Belief.Warm = true
		res.Belief.ConfirmedReadTokens += t.CacheReadInputTokens
		return res
	}
	if b.Warm {
		res.Demoted = true
		res.Alarm = true
		res.Reason = ReasonBelievedWarmZeroRead
		res.Belief.Warm = false
		res.Divergence = cachemeta.Diverge(b.LastPrefix, t.CurrentPrefix)
		res.FirstDivergeTokenOffset = res.Divergence.FirstDivergeTokenOffset()
		res.FirstDivergeByteOffset = firstByteDiff(b.LastPrefixBytes, t.CurrentPrefixBytes)
	}
	return res
}

func firstByteDiff(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

// KeyString renders a stable audit string for logs and tests.
func (k ManifestKey) KeyString() string {
	return "prefix=" + k.PrefixHash +
		";bytes=" + strconv.Itoa(k.PrefixBytes) +
		";tokens=" + strconv.FormatInt(k.PrefixTokens, 10) +
		";model=" + k.Scope.ModelID +
		";tok=" + k.Scope.TokenizerEpoch +
		";tools=" + k.Scope.ToolSetHash +
		";breakpoints=" + k.Scope.BreakpointLayout +
		";ttl=" + k.Scope.TTL +
		";surface=" + k.Scope.ProviderSurface
}
