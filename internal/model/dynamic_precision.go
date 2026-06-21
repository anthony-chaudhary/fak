package model

import "math"

// PrecisionTier is the whole-token matmul precision used for a Prefill or Step call.
// The current dynamic policy moves between the proven f32 path and the Q8_0 path; lower
// tiers can be added behind this enum without changing the session API again.
type PrecisionTier uint8

const (
	PrecisionF32 PrecisionTier = iota
	PrecisionQ8
)

func (p PrecisionTier) String() string {
	switch p {
	case PrecisionF32:
		return "f32"
	case PrecisionQ8:
		return "q8_0"
	default:
		return "unknown"
	}
}

// DynamicPrecisionPolicy is a conservative, online precision controller. It tries Q8_0
// first when quantized weights are available, then accepts that lower-precision result only
// if the returned distribution is confident enough. Rejected Q8 work is discarded by
// restoring the prior KV cache and recomputing the same token/span in f32.
//
// MinTop1Margin is in raw-logit units: top1 - top2 must be at least this value. Zero means
// "do not gate on margin". MinTop1Prob gates on softmax(top1); zero means "do not gate on
// probability". With both zero, the policy accepts every Q8 result and behaves like fixed
// Quant=true, except for stats.
type DynamicPrecisionPolicy struct {
	MinTop1Margin float32
	MinTop1Prob   float32
}

// PrecisionStats records the decisions made by a Session's DynamicPrecisionPolicy.
type PrecisionStats struct {
	Q8Attempts   int
	Q8Calls      int
	Q8Tokens     int
	F32Calls     int
	F32Tokens    int
	Fallbacks    int
	LastTier     PrecisionTier
	LastAccepted bool
	LastMargin   float32
	LastTop1Prob float32
}

func (m *Model) q8Ready() bool {
	return m.q8w != nil && m.q8layers != nil && m.q8head != nil
}

func (s *Session) prefillDynamic(ids []int) []float32 {
	if !s.M.q8Ready() {
		logits := s.prefillF32(ids)
		s.recordPrecision(PrecisionF32, len(ids), true, 0, 0)
		return logits
	}
	before := s.Cache.Clone()
	logits := s.headQ(s.prefillBatchedQ(ids))
	s.PrecisionStats.Q8Attempts++
	margin, prob := logitConfidence(logits)
	if s.PrecisionPolicy.accepts(margin, prob) {
		s.recordPrecision(PrecisionQ8, len(ids), true, margin, prob)
		return logits
	}

	s.Cache = before
	logits = s.prefillF32(ids)
	s.recordPrecision(PrecisionF32, len(ids), false, margin, prob)
	return logits
}

func (s *Session) stepDynamic(id int) []float32 {
	if !s.M.q8Ready() {
		logits := s.stepF32(id)
		s.recordPrecision(PrecisionF32, 1, true, 0, 0)
		return logits
	}
	before := s.Cache.Clone()
	logits := s.headQ(s.tokenHiddenQ(id, s.Cache.Len()))
	s.PrecisionStats.Q8Attempts++
	margin, prob := logitConfidence(logits)
	if s.PrecisionPolicy.accepts(margin, prob) {
		s.recordPrecision(PrecisionQ8, 1, true, margin, prob)
		return logits
	}

	s.Cache = before
	logits = s.stepF32(id)
	s.recordPrecision(PrecisionF32, 1, false, margin, prob)
	return logits
}

func (p *DynamicPrecisionPolicy) accepts(margin, prob float32) bool {
	if p == nil {
		return false
	}
	if p.MinTop1Margin > 0 && margin < p.MinTop1Margin {
		return false
	}
	if p.MinTop1Prob > 0 && prob < p.MinTop1Prob {
		return false
	}
	return true
}

func (s *Session) prefillF32(ids []int) []float32 {
	wasQuant := s.Quant
	s.Quant = false
	defer func() { s.Quant = wasQuant }()
	if cfg := s.M.Cfg; cfg.IsMoE() || cfg.DenseMLP || cfg.Alibi || cfg.IsQwen35Hybrid() || cfg.AttnOutputGate || cfg.BlockTopology != PreNorm || cfg.hasLayerSpecificRopeTheta() {
		var last []float32
		for _, id := range ids {
			last = s.tokenHidden(id, s.Cache.Len())
		}
		return s.head(last)
	}
	return s.head(s.prefillBatched(ids))
}

func (s *Session) stepF32(id int) []float32 {
	wasQuant := s.Quant
	s.Quant = false
	defer func() { s.Quant = wasQuant }()
	return s.head(s.tokenHidden(id, s.Cache.Len()))
}

func (s *Session) recordPrecision(tier PrecisionTier, tokens int, accepted bool, margin, prob float32) {
	switch tier {
	case PrecisionQ8:
		s.PrecisionStats.Q8Calls++
		s.PrecisionStats.Q8Tokens += tokens
	case PrecisionF32:
		s.PrecisionStats.F32Calls++
		s.PrecisionStats.F32Tokens += tokens
		if !accepted {
			s.PrecisionStats.Fallbacks++
		}
	}
	s.PrecisionStats.LastTier = tier
	s.PrecisionStats.LastAccepted = accepted
	s.PrecisionStats.LastMargin = margin
	s.PrecisionStats.LastTop1Prob = prob
}

func logitConfidence(logits []float32) (margin, topProb float32) {
	if len(logits) == 0 {
		return 0, 0
	}
	top1, top2 := logits[0], float32(math.Inf(-1))
	for _, v := range logits[1:] {
		if v > top1 {
			top2 = top1
			top1 = v
		} else if v > top2 {
			top2 = v
		}
	}
	if math.IsInf(float64(top2), -1) {
		margin = float32(math.Inf(1))
		topProb = 1
		return margin, topProb
	}
	margin = top1 - top2
	var sum float64
	maxLogit := float64(top1)
	for _, v := range logits {
		sum += math.Exp(float64(v) - maxLogit)
	}
	if sum == 0 || math.IsInf(sum, 0) || math.IsNaN(sum) {
		return margin, 0
	}
	return margin, float32(1.0 / sum)
}
