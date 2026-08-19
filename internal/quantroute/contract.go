// Package quantroute filters ordered runtime candidates by declared quantization
// compatibility without changing provider preference or hiding conversions.
package quantroute

import "github.com/anthony-chaudhary/fak/internal/quantcompat"

// Code is the stable route outcome vocabulary.
type Code string

const (
	CodeSelected           Code = "selected"
	CodeEmptyInput         Code = "no-candidates"
	CodeConversionOnly     Code = "conversion-only"
	CodeNoCompatibleTarget Code = "no-compatible-target"
)

// Candidate is one caller-ordered provider/runtime option.
type Candidate struct {
	Provider string
	Runtime  quantcompat.Runtime
	Hardware string
}

// Result preserves both the selected candidate and every compatibility decision
// inspected before it, making fallback behavior auditable.
type Result struct {
	Code      Code
	Index     int
	Candidate *Candidate
	Evaluated []quantcompat.Result
}

// Select returns the first directly supported or externally delegated candidate.
// Conversion-required and rejected candidates remain evidence, never silent fallback.
func Select(artifact quantcompat.Request, candidates []Candidate) Result {
	if len(candidates) == 0 {
		return Result{Code: CodeEmptyInput, Index: -1}
	}
	result := Result{Code: CodeNoCompatibleTarget, Index: -1, Evaluated: make([]quantcompat.Result, 0, len(candidates))}
	conversionSeen := false
	for i := range candidates {
		candidate := candidates[i]
		request := artifact
		request.Runtime = candidate.Runtime
		request.Hardware = candidate.Hardware
		compatibility := quantcompat.Adjudicate(request)
		result.Evaluated = append(result.Evaluated, compatibility)
		switch compatibility.Status {
		case quantcompat.StatusDirect, quantcompat.StatusExternalRuntime:
			result.Code = CodeSelected
			result.Index = i
			result.Candidate = &candidate
			return result
		case quantcompat.StatusConversionRequired:
			conversionSeen = true
		}
	}
	if conversionSeen {
		result.Code = CodeConversionOnly
	}
	return result
}
