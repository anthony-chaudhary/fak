package model

import (
	"fmt"
	"strings"
)

const (
	SpeculatorFinalNormRole = "model.norm.weight"
	SpeculatorLMHeadRole    = "lm_head.weight"
)

// SpeculatorVerifierReceipt records the verifier tensor sources a speculative
// head was trained against. Roles beyond the final norm and LM head allow
// native MTP/speculator heads to bind their own canonical tensors as they are
// introduced.
type SpeculatorVerifierReceipt struct {
	VerifierRoles map[string]string
}

// TensorRoleProvenance describes a manifest-only tensor-role resolution. It
// deliberately contains names only: checking a receipt must not materialize
// checkpoint bytes.
type TensorRoleProvenance struct {
	Canonical        string
	ResolvedSource   string
	CandidateAliases []string
}

var speculatorVerifierAliases = map[string][]string{
	SpeculatorFinalNormRole: {
		"model.norm.weight",
		"model.llm.norm.weight",
		"llm.norm.weight",
	},
	SpeculatorLMHeadRole: {
		"lm_head.weight",
		"model.lm_head.weight",
		"model.llm.lm_head.weight",
		"llm.lm_head.weight",
		// Tied-output checkpoints legitimately verify through the embedding row.
		"model.embed_tokens.weight",
		"model.llm.embed_tokens.weight",
		"llm.embed_tokens.weight",
	},
}

// VerifySpeculatorVerifierReceipt refuses a speculative-training receipt whose
// declared verifier tensor sources differ from the live checkpoint manifest.
// The guard is opt-in: ordinary HF/GGUF loading does not call it. Resolution is
// exact and ordered; in particular it never adds a suffix fallback that could
// select an unrelated norm or head tensor.
func VerifySpeculatorVerifierReceipt(manifestNames []string, receipt SpeculatorVerifierReceipt) ([]TensorRoleProvenance, error) {
	present := make(map[string]struct{}, len(manifestNames))
	for _, name := range manifestNames {
		present[name] = struct{}{}
	}
	if receipt.VerifierRoles == nil {
		return nil, fmt.Errorf("model: speculator verifier receipt has no declared roles")
	}
	if _, ok := receipt.VerifierRoles[SpeculatorFinalNormRole]; !ok {
		return nil, fmt.Errorf("model: speculator verifier receipt lacks required canonical role %q", SpeculatorFinalNormRole)
	}

	roles := make([]string, 0, len(receipt.VerifierRoles))
	roles = append(roles, SpeculatorFinalNormRole)
	if _, ok := receipt.VerifierRoles[SpeculatorLMHeadRole]; ok {
		roles = append(roles, SpeculatorLMHeadRole)
	}
	for canonical := range receipt.VerifierRoles {
		if canonical != SpeculatorFinalNormRole && canonical != SpeculatorLMHeadRole {
			roles = append(roles, canonical)
		}
	}

	out := make([]TensorRoleProvenance, 0, len(roles))
	for _, canonical := range roles {
		candidates := speculatorVerifierAliases[canonical]
		if len(candidates) == 0 {
			candidates = []string{canonical}
		}
		p := TensorRoleProvenance{Canonical: canonical, CandidateAliases: append([]string(nil), candidates...)}
		for _, candidate := range candidates {
			if _, ok := present[candidate]; ok {
				p.ResolvedSource = candidate
				break
			}
		}
		if p.ResolvedSource == "" {
			return nil, fmt.Errorf("model: speculator verifier role %q has no live source (candidate aliases: %s)", canonical, strings.Join(candidates, ", "))
		}
		out = append(out, p)
		if declared := receipt.VerifierRoles[canonical]; declared != p.ResolvedSource {
			return out, fmt.Errorf("model: speculator verifier role %q resolved source %q does not match receipt source %q (candidate aliases: %s)", canonical, p.ResolvedSource, declared, strings.Join(candidates, ", "))
		}
	}
	return out, nil
}
