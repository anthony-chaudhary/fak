package harnesskit

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"sort"
	"strings"
)

// InstructionContractVersion identifies the deterministic instruction snapshot format.
const InstructionContractVersion = "harnesskit-instructions/1"

// InstructionTrust identifies who controls a fragment's content.
type InstructionTrust string

const (
	TrustHost        InstructionTrust = "host"
	TrustApplication InstructionTrust = "application"
	TrustUser        InstructionTrust = "user"
	TrustUntrusted   InstructionTrust = "untrusted"
)

// InstructionLifetime declares how long a fragment remains applicable.
type InstructionLifetime string

const (
	LifetimeRun    InstructionLifetime = "run"
	LifetimeThread InstructionLifetime = "thread"
	LifetimeTurn   InstructionLifetime = "turn"
)

// InstructionResidency declares where a fragment may live in the realized prompt.
type InstructionResidency string

const (
	ResidencyStablePrefix  InstructionResidency = "stable-prefix"
	ResidencyOverlay       InstructionResidency = "overlay"
	ResidencyEphemeralTail InstructionResidency = "ephemeral-tail"
)

// InstructionRequest contains trusted runtime facts available at a resolution boundary.
type InstructionRequest struct {
	RunID       string            `json:"run_id,omitempty"`
	ThreadID    string            `json:"thread_id,omitempty"`
	TurnID      string            `json:"turn_id,omitempty"`
	AgentRole   string            `json:"agent_role,omitempty"`
	Model       string            `json:"model,omitempty"`
	Provider    string            `json:"provider,omitempty"`
	ToolProfile string            `json:"tool_profile,omitempty"`
	Facts       map[string]string `json:"facts,omitempty"`
}

// InstructionFragment is one independently attributable instruction block.
type InstructionFragment struct {
	ID         string               `json:"id"`
	Source     string               `json:"source"`
	Trust      InstructionTrust     `json:"trust"`
	Precedence int                  `json:"precedence"`
	Lifetime   InstructionLifetime  `json:"lifetime"`
	Audience   []string             `json:"audience,omitempty"`
	Residency  InstructionResidency `json:"residency"`
	Content    string               `json:"content"`
	Digest     string               `json:"digest"`
}

// InstructionInclusion explains whether and why a fragment was included.
type InstructionInclusion struct {
	ID       string `json:"id"`
	Included bool   `json:"included"`
	Reason   string `json:"reason"`
}

// InstructionSnapshot is the provider's deterministic, validated result.
type InstructionSnapshot struct {
	SchemaVersion      string                 `json:"schema_version"`
	Fragments          []InstructionFragment  `json:"fragments"`
	Decisions          []InstructionInclusion `json:"decisions,omitempty"`
	Digest             string                 `json:"digest"`
	StablePrefixDigest string                 `json:"stable_prefix_digest,omitempty"`
	EstimatedBytes     int                    `json:"estimated_bytes"`
	EstimatedTokens    int                    `json:"estimated_tokens"`
}

// InstructionProvider resolves instructions without mutating an opaque prompt.
type InstructionProvider interface {
	Resolve(context.Context, InstructionRequest) (InstructionSnapshot, error)
}

// InstructionProviderFunc adapts a function to InstructionProvider.
type InstructionProviderFunc func(context.Context, InstructionRequest) (InstructionSnapshot, error)

func (f InstructionProviderFunc) Resolve(ctx context.Context, req InstructionRequest) (InstructionSnapshot, error) {
	return f(ctx, req)
}

// ResolveInstructions invokes a provider, propagates cancellation, and validates the snapshot.
func ResolveInstructions(ctx context.Context, provider InstructionProvider, req InstructionRequest) (InstructionSnapshot, error) {
	if err := context.Cause(ctx); err != nil {
		return InstructionSnapshot{}, instructionError(CodeCanceled, "instructions.resolve", "resolution canceled", err)
	}
	if provider == nil {
		return InstructionSnapshot{}, instructionError(CodeInvalid, "instructions.resolve", "provider is required", nil)
	}
	snapshot, err := provider.Resolve(ctx, cloneInstructionRequest(req))
	if err != nil {
		if cause := context.Cause(ctx); cause != nil {
			return InstructionSnapshot{}, instructionError(CodeCanceled, "instructions.resolve", "resolution canceled", cause)
		}
		return InstructionSnapshot{}, err
	}
	for _, fragment := range snapshot.Fragments {
		if fragment.Trust == TrustHost || fragment.Residency == ResidencyStablePrefix {
			return InstructionSnapshot{}, instructionError(CodeDenied, "instructions.resolve", "providers cannot claim host trust or stable-prefix residency: "+fragment.ID, nil)
		}
	}
	return ValidateInstructionSnapshot(snapshot)
}

// ValidateInstructionSnapshot normalizes, orders, fingerprints, and freezes provider output.
const maxInstructionContentBytes = 1 << 20

func ValidateInstructionSnapshot(snapshot InstructionSnapshot) (InstructionSnapshot, error) {
	if snapshot.SchemaVersion == "" {
		snapshot.SchemaVersion = InstructionContractVersion
	}
	if snapshot.SchemaVersion != InstructionContractVersion {
		return InstructionSnapshot{}, instructionError(CodeUnsupported, "instructions.validate", "unsupported schema version", nil)
	}

	fragments := append([]InstructionFragment(nil), snapshot.Fragments...)
	seen := make(map[string]struct{}, len(fragments))
	for i := range fragments {
		fragment := &fragments[i]
		fragment.ID = strings.TrimSpace(fragment.ID)
		fragment.Source = strings.TrimSpace(fragment.Source)
		fragment.Content = strings.TrimSpace(fragment.Content)
		fragment.Audience = append([]string(nil), fragment.Audience...)
		sort.Strings(fragment.Audience)
		if fragment.ID == "" || fragment.Source == "" || fragment.Content == "" {
			return InstructionSnapshot{}, instructionError(CodeInvalid, "instructions.validate", "fragment id, source, and content are required", nil)
		}
		if len(fragment.Content) > maxInstructionContentBytes {
			return InstructionSnapshot{}, instructionError(CodeInvalid, "instructions.validate", "fragment content exceeds the 1 MiB admission limit: "+fragment.ID, nil)
		}
		if _, ok := seen[fragment.ID]; ok {
			return InstructionSnapshot{}, instructionError(CodeConflict, "instructions.validate", "duplicate fragment id: "+fragment.ID, nil)
		}
		seen[fragment.ID] = struct{}{}
		if !validInstructionTrust(fragment.Trust) || !validInstructionLifetime(fragment.Lifetime) || !validInstructionResidency(fragment.Residency) {
			return InstructionSnapshot{}, instructionError(CodeInvalid, "instructions.validate", "fragment has invalid trust, lifetime, or residency: "+fragment.ID, nil)
		}
		if fragment.Residency == ResidencyStablePrefix && fragment.Trust != TrustHost {
			return InstructionSnapshot{}, instructionError(CodeDenied, "instructions.validate", "only host fragments may claim stable-prefix residency: "+fragment.ID, nil)
		}
		if fragment.Trust == TrustUntrusted && fragment.Precedence > 0 {
			return InstructionSnapshot{}, instructionError(CodeDenied, "instructions.validate", "untrusted fragments may not claim positive precedence: "+fragment.ID, nil)
		}
		fragment.Digest = instructionDigest(fragment.Content)
	}
	sort.SliceStable(fragments, func(i, j int) bool {
		if fragments[i].Residency != fragments[j].Residency {
			return residencyRank(fragments[i].Residency) < residencyRank(fragments[j].Residency)
		}
		if fragments[i].Precedence != fragments[j].Precedence {
			return fragments[i].Precedence > fragments[j].Precedence
		}
		return fragments[i].ID < fragments[j].ID
	})

	snapshot.Fragments = fragments
	snapshot.Decisions = append([]InstructionInclusion(nil), snapshot.Decisions...)
	sort.SliceStable(snapshot.Decisions, func(i, j int) bool { return snapshot.Decisions[i].ID < snapshot.Decisions[j].ID })
	all, stable, bytes := fingerprintInstructionFragments(fragments)
	snapshot.Digest = all
	snapshot.StablePrefixDigest = stable
	snapshot.EstimatedBytes = bytes
	snapshot.EstimatedTokens = (bytes + 3) / 4
	return snapshot, nil
}

func validInstructionTrust(v InstructionTrust) bool {
	return v == TrustHost || v == TrustApplication || v == TrustUser || v == TrustUntrusted
}
func validInstructionLifetime(v InstructionLifetime) bool {
	return v == LifetimeRun || v == LifetimeThread || v == LifetimeTurn
}
func validInstructionResidency(v InstructionResidency) bool {
	return v == ResidencyStablePrefix || v == ResidencyOverlay || v == ResidencyEphemeralTail
}
func residencyRank(v InstructionResidency) int {
	switch v {
	case ResidencyStablePrefix:
		return 0
	case ResidencyOverlay:
		return 1
	default:
		return 2
	}
}
func instructionDigest(content string) string {
	sum := sha256.Sum256([]byte(content))
	return "sha256:" + hex.EncodeToString(sum[:])
}
func fingerprintInstructionFragments(fragments []InstructionFragment) (string, string, int) {
	all := sha256.New()
	stable := sha256.New()
	bytes := 0
	stableCount := 0
	for _, fragment := range fragments {
		line := fragment.ID + "\x00" + fragment.Source + "\x00" + string(fragment.Trust) + "\x00" + string(fragment.Lifetime) + "\x00" + string(fragment.Residency) + "\x00" + fragment.Digest + "\n"
		_, _ = all.Write([]byte(line))
		if fragment.Residency == ResidencyStablePrefix {
			_, _ = stable.Write([]byte(line))
			stableCount++
		}
		bytes += len(fragment.Content)
	}
	stableDigest := ""
	if stableCount > 0 {
		stableDigest = "sha256:" + hex.EncodeToString(stable.Sum(nil))
	}
	return "sha256:" + hex.EncodeToString(all.Sum(nil)), stableDigest, bytes
}
func cloneInstructionRequest(req InstructionRequest) InstructionRequest {
	if req.Facts != nil {
		facts := make(map[string]string, len(req.Facts))
		for key, value := range req.Facts {
			facts[key] = value
		}
		req.Facts = facts
	}
	return req
}
func instructionError(code Code, op, message string, err error) error {
	if err == nil {
		err = errors.New(message)
	}
	return &Error{Code: code, Op: op, Err: err}
}
