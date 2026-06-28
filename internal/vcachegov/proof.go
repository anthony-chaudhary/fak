package vcachegov

import "math"

// ProofStatus is the closed result vocabulary for deterministic vCache economics
// proofs. "Proven" means the supplied workload has a cacheable anchor and produces
// positive token-equivalent savings under the stated provider multipliers.
type ProofStatus string

const (
	ProofProven  ProofStatus = "PROVEN"
	ProofRefuted ProofStatus = "REFUTED"
)

// StarSavingsInput describes the low-risk vCache M2 shape: one stable anchor plus
// N independent suffix requests. It is provider-independent arithmetic over token
// equivalents; real billing must still be reconciled from provider telemetry.
type StarSavingsInput struct {
	AnchorTokens    float64
	SuffixTokens    float64
	Requests        int
	MinPrefixTokens float64
	ReadMult        float64
	WriteMult       float64
	Secret          SecretClassification
}

// StarSavingsProof is a deterministic prove/refute result for the star-anchor
// shape. Token counts are "uncached input-token equivalents": the same convention
// used by the design note's read/write multipliers.
type StarSavingsProof struct {
	Status               ProofStatus `json:"status"`
	Reason               string      `json:"reason"`
	Requests             int         `json:"requests"`
	AnchorTokens         float64     `json:"anchor_tokens"`
	SuffixTokens         float64     `json:"suffix_tokens"`
	MinPrefixTokens      float64     `json:"min_prefix_tokens"`
	ReadMult             float64     `json:"read_mult"`
	WriteMult            float64     `json:"write_mult"`
	BaselineTokenEquiv   float64     `json:"baseline_token_equiv"`
	VCacheTokenEquiv     float64     `json:"vcache_token_equiv"`
	SavedTokenEquiv      float64     `json:"saved_token_equiv"`
	SavedPct             float64     `json:"saved_pct"`
	BreakEvenRequests    int         `json:"break_even_requests"`
	CorrectnessDependsOn bool        `json:"correctness_depends_on_hit"`
}

// TelemetryRow is one provider usage row with prompt-cache counters. It matches
// the Claude usage shape but stays provider-neutral: baseline cost is the whole
// prompt cold, while actual cost applies the observed read/write multipliers.
type TelemetryRow struct {
	InputTokens              float64
	CacheCreationInputTokens float64
	CacheReadInputTokens     float64
	Ephemeral1hInputTokens   float64
	Ephemeral5mInputTokens   float64
}

// TelemetrySavingsInput proves or refutes a realized cache run from provider
// telemetry instead of an idealized planned anchor.
type TelemetrySavingsInput struct {
	Rows        []TelemetryRow
	ReadMult    float64
	Write5mMult float64
	Write1hMult float64
}

// TelemetrySavingsProof is the realized-input-token-equivalent proof for a run.
type TelemetrySavingsProof struct {
	Status                 ProofStatus `json:"status"`
	Reason                 string      `json:"reason"`
	Requests               int         `json:"requests"`
	BaselineTokenEquiv     float64     `json:"baseline_token_equiv"`
	ActualTokenEquiv       float64     `json:"actual_token_equiv"`
	SavedTokenEquiv        float64     `json:"saved_token_equiv"`
	SavedPct               float64     `json:"saved_pct"`
	CacheReadTokens        float64     `json:"cache_read_tokens"`
	CacheCreationTokens    float64     `json:"cache_creation_tokens"`
	Ephemeral1hInputTokens float64     `json:"ephemeral_1h_input_tokens"`
	Ephemeral5mInputTokens float64     `json:"ephemeral_5m_input_tokens"`
	FirstPositiveRequest   int         `json:"first_positive_request"`
	CorrectnessDependsOn   bool        `json:"correctness_depends_on_hit"`
}

// ProveStarSavings proves or refutes whether a star-anchor workload saves input
// token equivalents. It deliberately proves only cost: a cache hit is never a
// trust claim, and correctness_depends_on_hit is always false.
func ProveStarSavings(in StarSavingsInput) StarSavingsProof {
	p := StarSavingsProof{
		Requests:             in.Requests,
		AnchorTokens:         in.AnchorTokens,
		SuffixTokens:         in.SuffixTokens,
		MinPrefixTokens:      in.MinPrefixTokens,
		ReadMult:             in.ReadMult,
		WriteMult:            in.WriteMult,
		CorrectnessDependsOn: false,
	}
	if in.MinPrefixTokens <= 0 {
		p.MinPrefixTokens = 1
	}
	if in.ReadMult < 0 {
		p.ReadMult = 0
	}
	if in.WriteMult <= 0 {
		p.WriteMult = 1
	}
	p.BreakEvenRequests = NaturalBreakEvenRequests(p.WriteMult, p.ReadMult)

	switch {
	case !Warmable(in.Secret):
		p.Status = ProofRefuted
		p.Reason = "prefix content is not allowed in an implicit provider cache"
		return p
	case in.Requests <= 0:
		p.Status = ProofRefuted
		p.Reason = "requests must be positive"
		return p
	case in.AnchorTokens < p.MinPrefixTokens:
		p.Status = ProofRefuted
		p.Reason = "anchor is below the provider minimum cacheable prefix"
		return p
	case in.AnchorTokens <= 0:
		p.Status = ProofRefuted
		p.Reason = "anchor tokens must be positive"
		return p
	case in.SuffixTokens < 0:
		p.Status = ProofRefuted
		p.Reason = "suffix tokens cannot be negative"
		return p
	case p.ReadMult >= 1:
		p.Status = ProofRefuted
		p.Reason = "provider read multiplier must be below 1.0 to save tokens"
		return p
	}

	n := float64(in.Requests)
	p.BaselineTokenEquiv = n * (in.AnchorTokens + in.SuffixTokens)
	p.VCacheTokenEquiv = p.WriteMult*in.AnchorTokens + float64(in.Requests-1)*p.ReadMult*in.AnchorTokens + n*in.SuffixTokens
	p.SavedTokenEquiv = p.BaselineTokenEquiv - p.VCacheTokenEquiv
	if p.BaselineTokenEquiv > 0 {
		p.SavedPct = 100 * p.SavedTokenEquiv / p.BaselineTokenEquiv
	}
	p.Status, p.Reason = concludeProof(p.SavedTokenEquiv,
		"workload is below the natural-cache break-even point",
		"star anchor saves token equivalents when the first natural request warms it and siblings read it")
	return p
}

// ProveTelemetrySavings proves or refutes a realized provider-cache run from
// usage telemetry. It is deliberately conservative: cache savings are recognized
// only from observed cache_read/write counters, and correctness never depends on a
// hit.
func ProveTelemetrySavings(in TelemetrySavingsInput) TelemetrySavingsProof {
	p := TelemetrySavingsProof{
		Requests:             len(in.Rows),
		CorrectnessDependsOn: false,
	}
	readMult := in.ReadMult
	write5m := in.Write5mMult
	write1h := in.Write1hMult
	if readMult < 0 {
		readMult = 0
	}
	if write5m <= 0 {
		write5m = WriteMult5Minutes
	}
	if write1h <= 0 {
		write1h = WriteMult1Hour
	}
	if len(in.Rows) == 0 {
		p.Status = ProofRefuted
		p.Reason = "telemetry has no requests"
		return p
	}

	cumulativeBaseline := 0.0
	cumulativeActual := 0.0
	for i, row := range in.Rows {
		creation := maxFloat(row.CacheCreationInputTokens, 0)
		read := maxFloat(row.CacheReadInputTokens, 0)
		e1h := maxFloat(row.Ephemeral1hInputTokens, 0)
		e5m := maxFloat(row.Ephemeral5mInputTokens, 0)
		unspecifiedWrite := creation - e1h - e5m
		if unspecifiedWrite < 0 {
			unspecifiedWrite = 0
		}
		baseline := maxFloat(row.InputTokens, 0) + creation + read
		actual := maxFloat(row.InputTokens, 0) + write1h*e1h + write5m*(e5m+unspecifiedWrite) + readMult*read

		p.BaselineTokenEquiv += baseline
		p.ActualTokenEquiv += actual
		p.CacheCreationTokens += creation
		p.CacheReadTokens += read
		p.Ephemeral1hInputTokens += e1h
		p.Ephemeral5mInputTokens += e5m

		cumulativeBaseline += baseline
		cumulativeActual += actual
		if p.FirstPositiveRequest == 0 && cumulativeBaseline-cumulativeActual > 0 {
			p.FirstPositiveRequest = i + 1
		}
	}
	p.SavedTokenEquiv = p.BaselineTokenEquiv - p.ActualTokenEquiv
	if p.BaselineTokenEquiv > 0 {
		p.SavedPct = 100 * p.SavedTokenEquiv / p.BaselineTokenEquiv
	}
	if p.CacheReadTokens <= 0 {
		p.Status = ProofRefuted
		p.Reason = "telemetry observed no cache reads"
		return p
	}
	p.Status, p.Reason = concludeProof(p.SavedTokenEquiv,
		"observed cache reads did not repay cache write cost",
		"observed provider telemetry shows positive realized cache savings")
	return p
}

// NaturalBreakEvenRequests returns the first request count where first-natural
// warming beats no cache for a star anchor: w + (N-1)r < N.
func NaturalBreakEvenRequests(writeMult, readMult float64) int {
	if readMult >= 1 {
		return int(^uint(0) >> 1)
	}
	n := int(math.Floor((writeMult-readMult)/(1-readMult))) + 1
	if n < 1 {
		return 1
	}
	return n
}

// concludeProof applies the shared prove/refute verdict tail used by both the
// star-anchor and telemetry proofs: a non-positive realized saving refutes with
// refuteReason, otherwise the workload is proven with provenReason.
func concludeProof(saved float64, refuteReason, provenReason string) (ProofStatus, string) {
	if saved <= 0 {
		return ProofRefuted, refuteReason
	}
	return ProofProven, provenReason
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
