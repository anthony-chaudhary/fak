package model

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
)

// MTPHead isolates native Multi-Token Prediction (MTP) head weights for Qwen3.8.
//
// Qwen3.8 bundles an MTP head consisting of:
// - mtp.pre_fc_norm_embedding (RMSNorm on next-token embedding)
// - mtp.pre_fc_norm_hidden (RMSNorm on trunk hidden state)
// - mtp.fc (linear projection combining embedding and hidden state: 2H -> H)
// - mtp.layers.0 (single full-attention Qwen3.8 decoder layer)
// - mtp.norm (final RMSNorm before shared lm_head)
type MTPHead struct {
	Weights map[string][]float32
	FC      []float32
	Norm    []float32
	Layers  map[string][]float32
}

// NewMTPHead constructs an MTPHead from isolated mtp.* weights.
func NewMTPHead(weights map[string][]float32) *MTPHead {
	h := &MTPHead{
		Weights: make(map[string][]float32, len(weights)),
		Layers:  make(map[string][]float32),
	}
	for k, v := range weights {
		copied := append([]float32(nil), v...)
		h.Weights[k] = copied
		if strings.HasPrefix(k, "mtp.layers.") {
			h.Layers[k] = copied
		}
	}
	if fc, ok := h.Weights["mtp.fc.weight"]; ok {
		h.FC = fc
	}
	if norm, ok := h.Weights["mtp.norm.weight"]; ok {
		h.Norm = norm
	}
	return h
}

// IsolateMTPHead separates all "mtp.*" keys from weights into a dedicated MTPHead
// and returns the MTPHead along with trunk weights (with mtp.* stripped).
func IsolateMTPHead(weights map[string][]float32) (*MTPHead, map[string][]float32) {
	mtpWeights := make(map[string][]float32)
	trunkWeights := make(map[string][]float32)
	for k, v := range weights {
		if strings.HasPrefix(k, "mtp.") {
			mtpWeights[k] = v
		} else {
			trunkWeights[k] = v
		}
	}
	return NewMTPHead(mtpWeights), trunkWeights
}

// ExtractMTPHead extracts all "mtp.*" keys from weights into a dedicated MTPHead.
func ExtractMTPHead(weights map[string][]float32) *MTPHead {
	head, _ := IsolateMTPHead(weights)
	return head
}

// InjectMTPHead injects an MTPHead into sanitized trunk weights without modifying
// or corrupting trunk normalization scales. It returns a merged weight map.
func InjectMTPHead(trunkWeights map[string][]float32, head *MTPHead) map[string][]float32 {
	headLen := 0
	if head != nil {
		headLen = len(head.Weights)
	}
	out := make(map[string][]float32, len(trunkWeights)+headLen)
	for k, v := range trunkWeights {
		out[k] = append([]float32(nil), v...)
	}
	if head != nil {
		for k, v := range head.Weights {
			out[k] = append([]float32(nil), v...)
		}
	}
	return out
}

// isTrunkNormKey reports whether a weight tensor name belongs to trunk normalization.
func isTrunkNormKey(name string) bool {
	if strings.HasPrefix(name, "mtp.") {
		return false
	}
	lower := strings.ToLower(name)
	if strings.Contains(lower, "norm") {
		return strings.HasSuffix(lower, ".weight") ||
			strings.HasSuffix(lower, "_weight") ||
			!strings.Contains(lower, ".")
	}
	return false
}

// shouldShiftNorm verifies whether a norm tensor is zero-centered and requires
// the +1.0 shift. If it has already been shifted (mean around 1.0), it returns false
// to prevent double-shift corruption.
func shouldShiftNorm(data []float32) bool {
	if len(data) == 0 {
		return false
	}
	var sum float64
	for _, x := range data {
		sum += float64(x)
	}
	mean := sum / float64(len(data))
	// In zero-centered RMSNorm, unshifted weights have mean ≈ 0.0 (typically [-0.4, 0.4]).
	// Once shifted by +1.0, weights have mean ≈ 1.0 (typically [0.6, 1.4]).
	if mean >= 0.6 && mean <= 1.4 {
		return false
	}
	return true
}

// SanitizeTrunkNorms sanitizes trunk norm weights by applying a +1.0 shift exactly once,
// protecting trunk norms from the double-shift corruption bug when MTP keys are present.
//
// In Qwen3.8 checkpoints, trunk RMSNorm weights are stored in zero-centered form (w),
// requiring a +1.0 offset (1+w) for standard RMSNorm computation. However, production
// MTP weights already have norm offsets baked in. When MTP keys are present during
// unisolated sanitization, naive loaders mistakenly apply +1.0 twice to trunk norms.
// SanitizeTrunkNorms isolates MTP keys before sanitization and ensures trunk norms
// are shifted exactly once (+1.0).
func SanitizeTrunkNorms(weights map[string][]float32) map[string][]float32 {
	// 1. Isolate MTP keys to prevent them from interfering with trunk norm sanitization.
	_, trunk := IsolateMTPHead(weights)

	out := make(map[string][]float32, len(trunk))
	for k, v := range trunk {
		copied := append([]float32(nil), v...)
		if isTrunkNormKey(k) {
			if shouldShiftNorm(copied) {
				for i := range copied {
					copied[i] += 1.0
				}
			}
		}
		out[k] = copied
	}
	return out
}

// RejectionSampler implements speculative rejection sampling using the
// Leviathan-Chen formula (Leviathan et al., 2023; Chen et al., 2023).
type RejectionSampler struct {
	mu            sync.Mutex
	totalDrafted  int64
	totalAccepted int64
	totalRejected int64
	randSource    func() float32
	requestKey    uint64 // per-request RNG identity; 0 means no key (legacy path)
	keyed         bool   // true when a keyed stream is armed
	keyedDraws    int    // monotonic per-request draw counter for the legacy keyed seam
}

// NewRejectionSampler creates a RejectionSampler. If randSource is nil,
// math/rand.Float32 is used.
func NewRejectionSampler(randSource func() float32) *RejectionSampler {
	if randSource == nil {
		randSource = rand.Float32
	}
	return &RejectionSampler{
		randSource: randSource,
	}
}

// AcceptanceProbability computes the Leviathan-Chen acceptance probability:
//
//	P(accept) = min(1, P_target(x) / P_draft(x))
func (s *RejectionSampler) AcceptanceProbability(pTarget, pDraft float32) float32 {
	if pDraft <= 0 {
		if pTarget > 0 {
			return 1.0
		}
		return 0.0
	}
	ratio := pTarget / pDraft
	if ratio > 1.0 {
		return 1.0
	}
	if ratio < 0.0 {
		return 0.0
	}
	return ratio
}

// ResidualDistribution computes the normalized residual distribution on rejection:
//
//	P_residual(x) = max(0, P_target(x) - P_draft(x)) / sum_y max(0, P_target(y) - P_draft(y))
//
// If the distributions are identical (denominator <= 0), it falls back to pTarget.
func (s *RejectionSampler) ResidualDistribution(pTarget, pDraft []float32) []float32 {
	n := len(pTarget)
	if len(pDraft) < n {
		n = len(pDraft)
	}
	residual := make([]float32, n)
	var sum float64
	for i := 0; i < n; i++ {
		diff := float64(pTarget[i]) - float64(pDraft[i])
		if diff > 0 {
			residual[i] = float32(diff)
			sum += diff
		} else {
			residual[i] = 0
		}
	}
	if sum <= 1e-12 {
		copyDist := make([]float32, n)
		copy(copyDist, pTarget[:n])
		return copyDist
	}
	invSum := float32(1.0 / sum)
	var checkSum float32
	for i := range residual {
		residual[i] *= invSum
		checkSum += residual[i]
	}
	if len(residual) > 0 && checkSum > 0 && math.Abs(float64(checkSum-1.0)) > 1e-7 {
		residual[len(residual)-1] += (1.0 - checkSum)
	}
	return residual
}

// SampleFromDistribution draws a token index according to dist using r in [0, 1).
func (s *RejectionSampler) SampleFromDistribution(dist []float32, r float32) int {
	if len(dist) == 0 {
		return 0
	}
	if r < 0 {
		r = 0
	}
	if r >= 1.0 {
		r = 0.9999999
	}
	var cum float32
	for i, p := range dist {
		cum += p
		if r < cum {
			return i
		}
	}
	return len(dist) - 1
}

// Sample draws a token index from dist using the sampler's random source.
// When a per-request key is armed it draws from that keyed stream instead, so
// the sampled token is a function of the request identity and a monotonic step
// counter rather than the shared source (see SetRequestKey).
func (s *RejectionSampler) Sample(dist []float32) int {
	key, keyed, step := s.nextKeyedStep()
	if !keyed {
		return s.SampleFromDistribution(dist, s.randSource())
	}
	return s.SampleFromDistribution(dist, KeyedUniform(key, DomainBonusToken, step))
}

// TokenVerificationResult holds the outcome of verifying one draft token.
type TokenVerificationResult struct {
	Accepted         bool
	DraftToken       int
	ReplacementToken int
	AcceptProb       float32
	ResidualDist     []float32
}

// VerifyToken verifies a single draft token under Leviathan-Chen rejection sampling:
//
//	P(accept) = min(1, P_target(x) / P_draft(x))
//
// If randVal >= 0, it is used for the accept decision; otherwise s.randSource() is called.
func (s *RejectionSampler) VerifyToken(token int, pTarget, pDraft []float32, randVal float32) TokenVerificationResult {
	s.mu.Lock()
	s.totalDrafted++
	s.mu.Unlock()

	if randVal < 0 {
		if key, keyed, step := s.nextKeyedStep(); keyed {
			randVal = KeyedUniform(key, DomainAcceptCoin, step)
		} else {
			randVal = s.randSource()
		}
	}

	pT := float32(0)
	if token >= 0 && token < len(pTarget) {
		pT = pTarget[token]
	}
	pD := float32(0)
	if token >= 0 && token < len(pDraft) {
		pD = pDraft[token]
	}

	alpha := s.AcceptanceProbability(pT, pD)
	if randVal <= alpha {
		s.mu.Lock()
		s.totalAccepted++
		s.mu.Unlock()
		return TokenVerificationResult{
			Accepted:   true,
			DraftToken: token,
			AcceptProb: alpha,
		}
	}

	s.mu.Lock()
	s.totalRejected++
	s.mu.Unlock()

	residual := s.ResidualDistribution(pTarget, pDraft)
	replacement := s.Sample(residual)

	return TokenVerificationResult{
		Accepted:         false,
		DraftToken:       token,
		ReplacementToken: replacement,
		AcceptProb:       alpha,
		ResidualDist:     residual,
	}
}

// DraftVerificationResult summarizes the outcome of verifying a sequence of draft tokens.
type DraftVerificationResult struct {
	AcceptedTokens   []int
	BonusToken       int
	AcceptedCount    int
	TotalDrafted     int
	RejectedAt       int // -1 if all accepted
	ReplacementToken int // valid if RejectedAt >= 0
}

// VerifyDraftSequence verifies a sequence of draft tokens against per-position target
// and draft distributions. The first rejected token halts the draft and triggers residual
// sampling.
func (s *RejectionSampler) VerifyDraftSequence(
	draftTokens []int,
	pTargets [][]float32,
	pDrafts [][]float32,
	randVals []float32,
) DraftVerificationResult {
	k := len(draftTokens)
	res := DraftVerificationResult{
		AcceptedTokens: make([]int, 0, k),
		RejectedAt:     -1,
		TotalDrafted:   k,
	}

	for i := 0; i < k; i++ {
		r := float32(-1)
		if i < len(randVals) {
			r = randVals[i]
		}
		var targetDist, draftDist []float32
		if i < len(pTargets) {
			targetDist = pTargets[i]
		}
		if i < len(pDrafts) {
			draftDist = pDrafts[i]
		}

		ver := s.VerifyToken(draftTokens[i], targetDist, draftDist, r)
		if ver.Accepted {
			res.AcceptedTokens = append(res.AcceptedTokens, draftTokens[i])
			res.AcceptedCount++
		} else {
			res.RejectedAt = i
			res.ReplacementToken = ver.ReplacementToken
			break
		}
	}

	// If all accepted and an extra target distribution is supplied, sample the bonus token.
	if res.RejectedAt == -1 {
		if len(pTargets) > k {
			res.BonusToken = s.Sample(pTargets[k])
		} else if len(pTargets) > 0 {
			res.BonusToken = s.Sample(pTargets[len(pTargets)-1])
		}
	}

	return res
}

// SetRequestKey arms a per-request keyed coin stream. An empty requestID clears
// the key and restores the legacy randSource path (no reproducibility claim is
// made without a key -- fail-closed at the CLAIM level).
//
// Arming a key also resets the monotonic draw counter the legacy Sample/VerifyToken
// seam uses as its step, so a request replay starts from a known point.
func (s *RejectionSampler) SetRequestKey(requestID string) {
	key := HashRequestID(requestID)
	s.mu.Lock()
	s.requestKey = key
	s.keyed = key != 0
	s.keyedDraws = 0
	s.mu.Unlock()
}

// KeyedOn reports whether a keyed stream is armed.
func (s *RejectionSampler) KeyedOn() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.keyed
}

// VerifyDraftSequenceKeyedStep is the keyed accept coin for one draft position:
// the value the Keyed-armed sequence path would use at (domain, step) for the
// sampler's armed request key. It is exported for tests and diagnostics that
// need to inspect the keyed draw without re-deriving the key.
func (s *RejectionSampler) VerifyDraftSequenceKeyedStep(domain DomainTag, step int) (float32, bool) {
	key, keyed := s.keyState()
	if !keyed {
		return 0, false
	}
	return KeyedUniform(key, domain, step), true
}

// KeyedStep is the interface-friendly alias of VerifyDraftSequenceKeyedStep: it
// reports the keyed draw at (domain, step) for the armed request, or ok=false
// when no key is armed.
func (s *RejectionSampler) KeyedStep(domain DomainTag, step int) (float32, bool) {
	return s.VerifyDraftSequenceKeyedStep(domain, step)
}

// KeyedStepU is KeyedStep taking the domain as a plain uint64, so callers in
// packages (or tests) that must not name DomainTag can still read the keyed draw.
func (s *RejectionSampler) KeyedStepU(domain uint64, step int) (float32, bool) {
	return s.VerifyDraftSequenceKeyedStep(DomainTag(domain), step)
}

// VerifyTokenKeyedU is VerifyTokenKeyed taking the domain as a plain uint64.
func (s *RejectionSampler) VerifyTokenKeyedU(token int, pTarget, pDraft []float32, domain uint64, step int) TokenVerificationResult {
	return s.VerifyTokenKeyed(token, pTarget, pDraft, DomainTag(domain), step)
}

// keyState returns the armed key and whether a keyed stream is active.
func (s *RejectionSampler) keyState() (uint64, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requestKey, s.keyed
}

// nextKeyedStep returns the armed key, whether it is active, and a monotonic
// draw counter used as the keyed step for the legacy Sample/VerifyToken seam.
// The counter is a function of how many draws THIS request has made, so it is
// slot-free: replaying the same request in a different batch slot repeats the
// counter from 0 and therefore the same coin sequence.
func (s *RejectionSampler) nextKeyedStep() (uint64, bool, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.keyed {
		return 0, false, 0
	}
	step := s.keyedDraws
	s.keyedDraws++
	return s.requestKey, true, step
}

// SampleKeyed draws a token from dist using the keyed stream for (domain, step).
// When no key is armed it falls back to the legacy Sample(dist) path.
func (s *RejectionSampler) SampleKeyed(dist []float32, domain DomainTag, step int) int {
	key, keyed := s.keyState()
	if !keyed {
		return s.Sample(dist)
	}
	return s.SampleFromDistribution(dist, KeyedUniform(key, domain, step))
}

// VerifyTokenKeyed is VerifyToken with the accept coin drawn from the keyed
// stream for (domain, step) when a key is armed; otherwise it defers to
// VerifyToken's legacy randSource/randVal path.
func (s *RejectionSampler) VerifyTokenKeyed(token int, pTarget, pDraft []float32, domain DomainTag, step int) TokenVerificationResult {
	key, keyed := s.keyState()
	if !keyed {
		return s.VerifyToken(token, pTarget, pDraft, -1)
	}
	return s.verifyTokenWithCoin(token, pTarget, pDraft, KeyedUniform(key, domain, step))
}

// verifyTokenWithCoin is VerifyToken with an explicit accept coin, so the keyed
// path can share the exact accept rule (accept iff coin <= alpha) without
// re-deriving it. No randSource call is made.
func (s *RejectionSampler) verifyTokenWithCoin(token int, pTarget, pDraft []float32, coin float32) TokenVerificationResult {
	s.mu.Lock()
	s.totalDrafted++
	s.mu.Unlock()

	pT := float32(0)
	if token >= 0 && token < len(pTarget) {
		pT = pTarget[token]
	}
	pD := float32(0)
	if token >= 0 && token < len(pDraft) {
		pD = pDraft[token]
	}

	alpha := s.AcceptanceProbability(pT, pD)
	if coin <= alpha {
		s.mu.Lock()
		s.totalAccepted++
		s.mu.Unlock()
		return TokenVerificationResult{
			Accepted:   true,
			DraftToken: token,
			AcceptProb: alpha,
		}
	}

	s.mu.Lock()
	s.totalRejected++
	s.mu.Unlock()

	residual := s.ResidualDistribution(pTarget, pDraft)
	replacement := s.SampleFromDistribution(residual, s.nextResidualCoin(token))

	return TokenVerificationResult{
		Accepted:         false,
		DraftToken:       token,
		ReplacementToken: replacement,
		AcceptProb:       alpha,
		ResidualDist:     residual,
	}
}

// nextResidualCoin supplies the residual draw for the keyed single-token path.
// Callers that need a specific (request, step) keyed residual use
// VerifyDraftSequenceKeyed/SampleKeyed; this fallback keeps VerifyTokenKeyed
// self-contained by reusing the armed key with the residual domain and a step
// derived from the rejected draft token. Keying on the token (rather than a
// fixed step 0) keeps two distinct rejections under the same request on distinct
// coins; it is still slot-free -- the physical batch slot is not an input.
func (s *RejectionSampler) nextResidualCoin(token int) float32 {
	key, keyed := s.keyState()
	if keyed {
		return KeyedUniform(key, DomainResidual, token+1)
	}
	return s.randSource()
}

// VerifyDraftSequenceKeyed is VerifyDraftSequence with every accept coin and the
// bonus/replacement draws taken from the per-request keyed stream. Same accept
// rule (Leviathan-Chen, accept iff randVal <= alpha) -- only the coin SOURCE changes.
//
// The physical batch slot never enters the key: the key is
// (requestKey, domain, step) only, so preemption/re-admission into a different
// slot yields a byte-identical accepted-token sequence.
func (s *RejectionSampler) VerifyDraftSequenceKeyed(
	draftTokens []int,
	pTargets [][]float32,
	pDrafts [][]float32,
) DraftVerificationResult {
	key, keyed := s.keyState()
	if !keyed {
		return s.VerifyDraftSequence(draftTokens, pTargets, pDrafts, nil)
	}

	k := len(draftTokens)
	res := DraftVerificationResult{
		AcceptedTokens: make([]int, 0, k),
		RejectedAt:     -1,
		TotalDrafted:   k,
	}

	for i := 0; i < k; i++ {
		var targetDist, draftDist []float32
		if i < len(pTargets) {
			targetDist = pTargets[i]
		}
		if i < len(pDrafts) {
			draftDist = pDrafts[i]
		}

		ver := s.verifyTokenWithCoin(draftTokens[i], targetDist, draftDist, KeyedUniform(key, DomainAcceptCoin, i))
		if ver.Accepted {
			res.AcceptedTokens = append(res.AcceptedTokens, draftTokens[i])
			res.AcceptedCount++
		} else {
			res.RejectedAt = i
			res.ReplacementToken = s.SampleKeyed(ver.ResidualDist, DomainResidual, i)
			break
		}
	}

	if res.RejectedAt == -1 {
		if len(pTargets) > k {
			res.BonusToken = s.SampleKeyed(pTargets[k], DomainBonusToken, k)
		} else if len(pTargets) > 0 {
			res.BonusToken = s.SampleKeyed(pTargets[len(pTargets)-1], DomainBonusToken, len(pTargets)-1)
		}
	}

	return res
}

// AcceptanceRate returns the fraction of drafted tokens that were accepted.
func (s *RejectionSampler) AcceptanceRate() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	total := s.totalAccepted + s.totalRejected
	if total == 0 {
		return 0.0
	}
	return float64(s.totalAccepted) / float64(total)
}

// Stats returns the raw drafted, accepted, and rejected counts alongside acceptance rate.
func (s *RejectionSampler) Stats() (totalDrafted, totalAccepted, totalRejected int64, acceptanceRate float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	total := s.totalAccepted + s.totalRejected
	var rate float64
	if total > 0 {
		rate = float64(s.totalAccepted) / float64(total)
	}
	return s.totalDrafted, s.totalAccepted, s.totalRejected, rate
}

// Reset clears the acceptance statistics.
func (s *RejectionSampler) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.totalDrafted = 0
	s.totalAccepted = 0
	s.totalRejected = 0
}

// EffectiveDistribution calculates the exact theoretical output distribution
// resulting from Leviathan-Chen rejection sampling:
//
//	P_effective(x) = P_draft(x) * min(1, P_target(x) / P_draft(x)) + (1 - alpha_total) * P_residual(x)
//
// where alpha_total = sum_y min(P_target(y), P_draft(y)).
//
// Leviathan & Chen (2023) Theorem 1 proves that P_effective(x) == P_target(x) for all x.
func (s *RejectionSampler) EffectiveDistribution(pTarget, pDraft []float32) []float32 {
	n := len(pTarget)
	if len(pDraft) < n {
		n = len(pDraft)
	}
	effective := make([]float32, n)
	var alphaTotal float64
	for i := 0; i < n; i++ {
		pT := float64(pTarget[i])
		pD := float64(pDraft[i])
		alphaTotal += math.Min(pT, pD)
	}
	residual := s.ResidualDistribution(pTarget, pDraft)
	probReject := 1.0 - alphaTotal
	if probReject < 0 {
		probReject = 0
	}
	for i := 0; i < n; i++ {
		pT := float64(pTarget[i])
		pD := float64(pDraft[i])
		var acceptMass float64
		if pD > 0 {
			acceptProb := math.Min(1.0, pT/pD)
			acceptMass = pD * acceptProb
		}
		residualMass := probReject * float64(residual[i])
		effective[i] = float32(acceptMass + residualMass)
	}
	return effective
}

// VerifyDistributionConsistency checks whether the effective rejection-sampled distribution
// matches pTarget within the specified tolerance.
func (s *RejectionSampler) VerifyDistributionConsistency(pTarget, pDraft []float32, tol float32) error {
	effective := s.EffectiveDistribution(pTarget, pDraft)
	for i := range effective {
		diff := float32(math.Abs(float64(effective[i] - pTarget[i])))
		if diff > tol {
			return fmt.Errorf("model: distribution inconsistency at token %d: target=%g effective=%g diff=%g > tol=%g",
				i, pTarget[i], effective[i], diff, tol)
		}
	}
	return nil
}
