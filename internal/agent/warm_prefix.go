package agent

// warm_prefix.go — #13352 (agent-startup-cache-warm CW-02): the canonical, stable
// cache-warming descriptor the agent startup path warms a native prefix from.
//
// WHY this exists. EncodePrompt (prompt_encoding.go) already shares rendering,
// prompt shrink and tokenization with Complete, and syspromptmmu already owns the
// resident base-context plan. What startup lacked was a SINGLE descriptor that binds
// the stable portion of a request — the effective instructions, the resident system
// blocks and the ordered tool schemas — to the stable token boundary it occupies and
// to every compatibility/authorization axis a warm hit depends on.
//
// CONTRACT (the load-bearing properties the witness pins):
//
//  1. Stability. The stable token prefix is derived from the STABLE inputs only:
//     caller-resolved instruction bytes, the resident syspromptmmu plan and the
//     ordered tool schemas. Two different user suffixes MUST derive the same
//     StableTokenDigest/StableTokens — volatile dates, request ids, conversation
//     history and assistant-generation markers never enter the derivation.
//  2. Identity. Any change to a compatibility axis (model, renderer/template,
//     tokenizer, adapter, KV layout, runtime generation) or an authorization axis
//     (tenant/agent cache scope, instruction bytes, tool policy) MUST change Identity.
//  3. Immutability + boundedness. The descriptor is a value with copied slices; the
//     caller cannot mutate its internals, and every digest is a fixed-width hex string.
//
// This file owns the DESCRIPTOR and its derivation only. It does NOT warm, restore or
// hit anything — realization is the successor startup wiring's job.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

// digestPrefix labels every hex digest this file mints, so a reader knows the bytes
// are a content/identity hash rather than a git object id (mirrors syspromptmmu's
// witnessPrefix discipline).
const digestPrefix = "sha256:"

// WarmPrefixSpec is the canonical, immutable descriptor of the stable agent prefix a
// startup cache warm targets. It is derived once through the production render/
// tokenize helpers (EncodePrompt's seam) and carried to the warming path.
type WarmPrefixSpec struct {
	// StableTokens is the token boundary of the stable prefix: the count of leading
	// tokens that come from instructions + resident system blocks + ordered tool
	// schemas ONLY. A warmed request whose suffix starts at this boundary is eligible
	// for reuse; a suffix is never part of the stable prefix.
	StableTokens int `json:"stable_tokens"`
	// StableTokenDigest is the content-derived digest of the stable token PREFIX
	// (token ids [0:StableTokens]). Two suffixes over the same stable inputs share it.
	StableTokenDigest string `json:"stable_token_digest"`
	// InstructionDigest is the digest of the caller-resolved instruction bytes.
	InstructionDigest string `json:"instruction_digest"`
	// ToolPolicyDigest is the digest of the ordered tool schemas (order-sensitive).
	ToolPolicyDigest string `json:"tool_policy_digest"`
	// ResidentPlanDigest is syspromptmmu's plan digest for the resident base context.
	ResidentPlanDigest string `json:"resident_plan_digest"`

	// Identity axes — a change to any of these MUST change Identity.

	// Scope is the authenticated cache owner (tenant/agent) the warm is bounded to.
	Scope radixkv.CacheIdentity `json:"scope"`
	// ModelID is the planner's model identity.
	ModelID string `json:"model_id"`
	// RendererID identifies the template/renderer (family + thinking mode).
	RendererID string `json:"renderer_id"`
	// TokenizerID identifies the tokenizer.
	TokenizerID string `json:"tokenizer_id"`
	// AdapterID identifies the adapter family the request is served through.
	AdapterID string `json:"adapter_id"`
	// KVLayout identifies the realized native KV storage layout.
	KVLayout string `json:"kv_layout"`
	// RuntimeGeneration is the planner's runtime/decode-path generation token.
	RuntimeGeneration string `json:"runtime_generation"`

	// ResidentBlockDigest is the digest of the joined resident system blocks (empty
	// when the caller supplied none).
	ResidentBlockDigest string `json:"resident_block_digest"`

	// Identity is the digest folded over every axis above plus the stable digests —
	// the one key a warming store is permitted to index on.
	Identity string `json:"identity"`
}

// ErrWarmPrefixUnavailable is the closed refusal for a planner that cannot derive a
// warm descriptor (nil/unconfigured planner, missing model or tokenizer identity).
// It is fail-closed: a caller must never warm on a zero-value spec.
var ErrWarmPrefixUnavailable = errors.New("warm prefix: planner cannot derive a stable prefix descriptor")

// WarmPrefixInputs carries the STABLE inputs a warm descriptor is derived from. The
// instruction bytes are caller-resolved (the same slice the caller would hand the
// forward path); this file does NOT run a second AGENTS scanner. History, the
// assistant generation and any volatile tail are deliberately absent — they are the
// suffix, not the prefix.
type WarmPrefixInputs struct {
	// Instructions is the effective instruction snapshot (e.g. resolved AGENTS text).
	Instructions []byte
	// SystemBlocks are the resident system blocks, in wire order.
	SystemBlocks [][]byte
	// Tools are the ordered tool schemas contributing to the stable prefix.
	Tools []ToolDef
	// KVLayout is the realized native KV storage layout (e.g. the planner's KV tier).
	KVLayout string
	// AdapterID identifies the adapter family; empty means the default family.
	AdapterID string
}

// DeriveWarmPrefix builds the canonical warm descriptor for p. It reuses the same
// renderer + tokenizer as EncodePrompt by encoding the stable-only prompt through the
// production path, so the boundary it reports is the boundary the forward will see.
//
// tenant/agent are the authenticated cache scope; tenant is required (a warm with no
// owner cannot be bounded).
func (p *InKernelPlanner) DeriveWarmPrefix(tenant, agent string, in WarmPrefixInputs) (WarmPrefixSpec, error) {
	if p == nil || p.m == nil || p.tok == nil || strings.TrimSpace(p.modelID) == "" {
		return WarmPrefixSpec{}, ErrWarmPrefixUnavailable
	}
	tokenizerID := p.tok.Identity()
	if tokenizerID == "" {
		return WarmPrefixSpec{}, ErrWarmPrefixUnavailable
	}
	tenant = strings.TrimSpace(tenant)
	if tenant == "" {
		return WarmPrefixSpec{}, ErrWarmPrefixUnavailable
	}

	// The stable inputs render through the SAME renderer/tokenizer as the forward:
	// instructions ride as ordered system messages, then the resident blocks, with the
	// ordered tools attached. Encoding the stable-only prompt yields the exact stable
	// token boundary; the suffix (history, user text) is never part of it.
	stableMsgs := make([]Message, 0, len(in.SystemBlocks)+1)
	if len(in.Instructions) > 0 {
		stableMsgs = append(stableMsgs, Message{Role: RoleSystem, Content: string(in.Instructions)})
	}
	for _, block := range in.SystemBlocks {
		if len(block) == 0 {
			continue
		}
		stableMsgs = append(stableMsgs, Message{Role: RoleSystem, Content: string(block)})
	}
	enc, err := p.EncodePrompt(context.Background(), stableMsgs, in.Tools)
	if err != nil {
		return WarmPrefixSpec{}, err
	}

	adapterID := strings.TrimSpace(in.AdapterID)
	if adapterID == "" {
		adapterID = defaultWarmAdapterID
	}
	scope := radixkv.CacheIdentity{Tenant: tenant, Agent: strings.TrimSpace(agent)}

	spec := WarmPrefixSpec{
		StableTokens:        len(enc.TokenIDs),
		StableTokenDigest:   digestInts(enc.TokenIDs),
		InstructionDigest:   digestBytes(in.Instructions),
		ToolPolicyDigest:    digestToolDefs(in.Tools),
		ResidentBlockDigest: digestBlocks(in.SystemBlocks),
		ResidentPlanDigest:  syspromptmmu.PlanDigest(),
		Scope:               scope,
		ModelID:             enc.ModelID,
		RendererID:          enc.RendererID,
		TokenizerID:         enc.TokenizerID,
		AdapterID:           adapterID,
		KVLayout:            strings.TrimSpace(in.KVLayout),
		RuntimeGeneration:   p.warmRuntimeGeneration(),
	}
	spec.Identity = spec.computeIdentity()
	return spec, nil
}

// Stable returns the spec's stable-prefix fields as a comparable value so a caller can
// test prefix stability (two specs with equal Stable share a cacheable prefix) without
// threading the whole descriptor.
func (s WarmPrefixSpec) Stable() (tokens int, digest string) {
	return s.StableTokens, s.StableTokenDigest
}

// Bounded reports whether the spec is a usable, non-empty descriptor. A false result
// means the spec must NOT be used to warm or index a cache. A stable prefix with no
// stable SOURCE (no instructions and no resident block) is not warmable — the ChatML
// wrapper alone is a rendering artifact, not a cacheable agent prefix — so at least one
// of the two content digests must be present.
func (s WarmPrefixSpec) Bounded() bool {
	hasSource := s.InstructionDigest != "" || s.ResidentBlockDigest != ""
	return hasSource &&
		s.StableTokens > 0 &&
		s.StableTokenDigest != "" &&
		s.AdapterID != "" &&
		s.Identity != "" &&
		strings.TrimSpace(s.Scope.Tenant) != ""
}

// computeIdentity folds every compatibility and authorization axis plus the stable
// digests into one digest. It is a value method over the already-set fields so the
// identity is a pure function of the descriptor's own content.
func (s WarmPrefixSpec) computeIdentity() string {
	h := sha256.New()
	writeField := func(label, value string) {
		h.Write([]byte(label))
		h.Write([]byte{0})
		h.Write([]byte(value))
		h.Write([]byte{0})
	}
	writeField("tenant", s.Scope.Tenant)
	writeField("agent", s.Scope.Agent)
	writeField("model", s.ModelID)
	writeField("renderer", s.RendererID)
	writeField("tokenizer", s.TokenizerID)
	writeField("adapter", s.AdapterID)
	writeField("kv_layout", s.KVLayout)
	writeField("runtime_generation", s.RuntimeGeneration)
	writeField("stable_token_digest", s.StableTokenDigest)
	writeField("instruction_digest", s.InstructionDigest)
	writeField("tool_policy_digest", s.ToolPolicyDigest)
	writeField("resident_block_digest", s.ResidentBlockDigest)
	writeField("resident_plan_digest", s.ResidentPlanDigest)
	writeField("stable_tokens", strconv.Itoa(s.StableTokens))
	return digestPrefix + hex.EncodeToString(h.Sum(nil))
}

// warmRuntimeGeneration names the planner's decode-path/runtime generation. It is the
// axis that invalidates a warm when the runtime changes underneath a stable prompt: a
// quantized (q4k) resident path and the dense f32 path are different generations, so a
// warm captured on one is never restored on the other.
func (p *InKernelPlanner) warmRuntimeGeneration() string {
	gen := "dense"
	if p.q4k {
		gen = "resident-q4k"
	}
	if p.quant {
		gen += "+quant"
	} else {
		gen += "+f32"
	}
	return gen
}

// digestBytes returns the labeled content digest of a byte slice (empty ⇒ empty).
func digestBytes(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	sum := sha256.Sum256(b)
	return digestPrefix + hex.EncodeToString(sum[:])
}

// digestBlocks digests the resident system blocks in wire order, NUL-separated per
// block so no concatenation aliases another block set.
func digestBlocks(blocks [][]byte) string {
	if len(blocks) == 0 {
		return ""
	}
	h := sha256.New()
	for _, b := range blocks {
		h.Write(b)
		h.Write([]byte{0})
	}
	return digestPrefix + hex.EncodeToString(h.Sum(nil))
}

// digestInts digests a token-id sequence NUL-separated per id, so no concatenation of
// ids aliases another sequence.
func digestInts(ids []int) string {
	if len(ids) == 0 {
		return ""
	}
	h := sha256.New()
	for _, id := range ids {
		h.Write([]byte(strconv.Itoa(id)))
		h.Write([]byte{0})
	}
	return digestPrefix + hex.EncodeToString(h.Sum(nil))
}

// digestToolDefs digests the ORDERED tool schemas over their name/description/parameters,
// so tool reordering or a schema edit changes the tool-policy axis.
func digestToolDefs(tools []ToolDef) string {
	if len(tools) == 0 {
		return ""
	}
	h := sha256.New()
	for _, t := range tools {
		h.Write([]byte(t.Type))
		h.Write([]byte{0})
		h.Write([]byte(t.Function.Name))
		h.Write([]byte{0})
		h.Write([]byte(t.Function.Description))
		h.Write([]byte{0})
		h.Write(t.Function.Parameters)
		h.Write([]byte{0})
	}
	return digestPrefix + hex.EncodeToString(h.Sum(nil))
}

// defaultWarmAdapterID names the default adapter family when the caller does not
// select one, so the adapter axis is always populated (never silently empty).
const defaultWarmAdapterID = "native-inkernel"
