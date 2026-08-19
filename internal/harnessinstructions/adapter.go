// Package harnessinstructions realizes public harness instruction snapshots through fak's system-prompt MMU.
package harnessinstructions

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// Realization is the exact prompt value plus its deterministic audit record.
type Realization struct {
	SchemaVersion      string                            `json:"schema_version"`
	PromptValue        json.RawMessage                   `json:"prompt_value"`
	Snapshot           harnesskit.InstructionSnapshot    `json:"snapshot"`
	Decisions          []harnesskit.InstructionInclusion `json:"decisions"`
	Digest             string                            `json:"digest"`
	StablePrefixDigest string                            `json:"stable_prefix_digest"`
	PrefixAuditStatus  string                            `json:"prefix_audit_status"`
	EstimatedBytes     int                               `json:"estimated_bytes"`
	EstimatedTokens    int                               `json:"estimated_tokens"`
}

// Resolve invokes a public provider and realizes its typed fragments after fak's immutable base prefix.
func Resolve(ctx context.Context, provider harnesskit.InstructionProvider, req harnesskit.InstructionRequest) (Realization, error) {
	snapshot, err := harnesskit.ResolveInstructions(ctx, provider, req)
	if err != nil {
		return Realization{}, err
	}
	return Realize(snapshot)
}

// Realize composes a validated application snapshot with the kernel-owned stable prefix.
func Realize(snapshot harnesskit.InstructionSnapshot) (Realization, error) {
	snapshot, err := harnesskit.ValidateInstructionSnapshot(snapshot)
	if err != nil {
		return Realization{}, err
	}
	for _, fragment := range snapshot.Fragments {
		if fragment.Trust == harnesskit.TrustHost || fragment.Residency == harnesskit.ResidencyStablePrefix {
			return Realization{}, &harnesskit.Error{Code: harnesskit.CodeDenied, Op: "instructions.realize", Err: fmt.Errorf("provider fragment %q crosses the host-owned stable-prefix boundary", fragment.ID)}
		}
	}

	prompt, decisions, err := compose(snapshot)
	if err != nil {
		return Realization{}, err
	}
	body, err := json.Marshal(map[string]json.RawMessage{"system": prompt})
	if err != nil {
		return Realization{}, &harnesskit.Error{Code: harnesskit.CodeInternal, Op: "instructions.audit", Err: err}
	}
	audit := syspromptmmu.AuditBaseContext(body)
	if audit.Status != syspromptmmu.AuditOK {
		return Realization{}, &harnesskit.Error{Code: harnesskit.CodeInternal, Op: "instructions.audit", Err: fmt.Errorf("stable prefix audit: %s", audit.Status)}
	}

	bytes := snapshot.EstimatedBytes
	for _, segment := range syspromptmmu.BaseContext() {
		bytes += len(segment.Content)
	}
	return Realization{
		SchemaVersion:      harnesskit.InstructionContractVersion,
		PromptValue:        prompt,
		Snapshot:           snapshot,
		Decisions:          decisions,
		Digest:             digest(prompt),
		StablePrefixDigest: audit.GotDigest,
		PrefixAuditStatus:  audit.Status,
		EstimatedBytes:     bytes,
		EstimatedTokens:    (bytes + 3) / 4,
	}, nil
}

func compose(snapshot harnesskit.InstructionSnapshot) (json.RawMessage, []harnesskit.InstructionInclusion, error) {
	var blocks []map[string]any
	base := syspromptmmu.BaseContext()
	for index, segment := range base {
		block := map[string]any{"type": "text", "text": string(segment.Content)}
		if index == len(base)-1 {
			block["cache_control"] = map[string]string{"type": "ephemeral"}
		}
		blocks = append(blocks, block)
	}
	decisions := make([]harnesskit.InstructionInclusion, 0, len(snapshot.Fragments))
	for _, fragment := range snapshot.Fragments {
		blocks = append(blocks, map[string]any{"type": "text", "text": fragment.Content})
		decisions = append(decisions, harnesskit.InstructionInclusion{ID: fragment.ID, Included: true, Reason: "included by typed instruction snapshot"})
	}
	prompt, err := json.Marshal(blocks)
	if err != nil {
		return nil, nil, &harnesskit.Error{Code: harnesskit.CodeInternal, Op: "instructions.compose", Err: err}
	}
	return prompt, decisions, nil
}

func digest(content []byte) string {
	sum := sha256.Sum256(content)
	return "sha256:" + hex.EncodeToString(sum[:])
}
