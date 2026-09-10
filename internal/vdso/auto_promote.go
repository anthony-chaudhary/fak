package vdso

import (
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// PromotedRegistry is a thread-safe registry of dynamically promoted read-only tool signatures.
type PromotedRegistry struct {
	mu       sync.RWMutex
	promoted map[string]bool
}

// NewPromotedRegistry creates a fresh dynamic promotion table.
func NewPromotedRegistry() *PromotedRegistry {
	return &PromotedRegistry{
		promoted: make(map[string]bool),
	}
}

// Promote marks a tool signature as dynamically promoted to read-only status.
func (p *PromotedRegistry) Promote(tool string) {
	tool = strings.TrimSpace(tool)
	if tool == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.promoted[tool] = true
}

// IsPromoted reports whether the tool signature has been dynamically promoted.
func (p *PromotedRegistry) IsPromoted(tool string) bool {
	tool = strings.TrimSpace(tool)
	if tool == "" {
		return false
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.promoted[tool]
}

// Invalidate removes a tool signature from the dynamic promotion table.
func (p *PromotedRegistry) Invalidate(tool string) {
	tool = strings.TrimSpace(tool)
	if tool == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.promoted, tool)
}

// InvalidateAll clears all promoted tool signatures.
func (p *PromotedRegistry) InvalidateAll() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.promoted = make(map[string]bool)
}

// Len returns the count of currently promoted tools.
func (p *PromotedRegistry) Len() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.promoted)
}

// Tools returns a sorted slice of all promoted tool signatures.
func (p *PromotedRegistry) Tools() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]string, 0, len(p.promoted))
	for t := range p.promoted {
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

// HasIdempotencyReceipt reports whether result metadata carries verified
// read-only or idempotency markers from execution.
func HasIdempotencyReceipt(meta map[string]string) bool {
	if meta == nil {
		return false
	}
	if v := meta["read_only"]; strings.EqualFold(v, "true") || v == "1" {
		return true
	}
	if v := meta["readOnly"]; strings.EqualFold(v, "true") || v == "1" {
		return true
	}
	if v := meta["idempotent"]; strings.EqualFold(v, "true") || v == "1" {
		return true
	}
	if v := meta["idempotency"]; strings.EqualFold(v, "true") || v == "1" {
		return true
	}
	if v := meta["outcome"]; v == "verified_fresh_reuse" || v == "executed_cold_read" {
		return true
	}
	return false
}

// isStateDriftOrMutation checks whether a tool call or result indicates state drift
// or mutation that invalidates dynamic read-only promotion.
func isStateDriftOrMutation(c *abi.ToolCall, r *abi.Result) bool {
	if c != nil && destructive(c) {
		return true
	}
	if r == nil || r.Meta == nil {
		return false
	}
	meta := r.Meta
	if strings.EqualFold(meta["state_drift"], "true") || meta["state_drift"] == "1" {
		return true
	}
	if strings.EqualFold(meta["drift"], "true") || meta["drift"] == "1" {
		return true
	}
	if strings.EqualFold(meta["mutated"], "true") || meta["mutated"] == "1" {
		return true
	}
	if strings.EqualFold(meta["mutation"], "true") || meta["mutation"] == "1" {
		return true
	}
	if strings.EqualFold(meta["destructive"], "true") || meta["destructive"] == "1" {
		return true
	}
	if strings.EqualFold(meta["read_only"], "false") || meta["read_only"] == "0" {
		return true
	}
	if strings.EqualFold(meta["readOnly"], "false") || meta["readOnly"] == "0" {
		return true
	}
	if strings.EqualFold(meta["idempotent"], "false") || meta["idempotent"] == "0" {
		return true
	}
	return false
}

// PromoteReadOnly auto-promotes a tool signature into the dynamic read-only registry.
func (v *VDSO) PromoteReadOnly(tool string) {
	if v == nil {
		return
	}
	v.promotedMu.Lock()
	if v.promoted == nil {
		v.promoted = NewPromotedRegistry()
	}
	v.promotedMu.Unlock()
	v.promoted.Promote(tool)
}

// IsPromotedReadOnly reports whether a tool signature has been dynamically promoted.
func (v *VDSO) IsPromotedReadOnly(tool string) bool {
	if v == nil {
		return false
	}
	v.promotedMu.RLock()
	reg := v.promoted
	v.promotedMu.RUnlock()
	if reg == nil {
		return false
	}
	return reg.IsPromoted(tool)
}

// InvalidatePromoted removes a tool signature from the dynamic read-only registry.
func (v *VDSO) InvalidatePromoted(tool string) {
	if v == nil {
		return
	}
	v.promotedMu.RLock()
	reg := v.promoted
	v.promotedMu.RUnlock()
	if reg != nil {
		reg.Invalidate(tool)
	}
}

// InvalidateAllPromoted removes all tool signatures from the dynamic read-only registry.
func (v *VDSO) InvalidateAllPromoted() {
	if v == nil {
		return
	}
	v.promotedMu.RLock()
	reg := v.promoted
	v.promotedMu.RUnlock()
	if reg != nil {
		reg.InvalidateAll()
	}
}

// PromotedTools returns a sorted slice of currently promoted tool names.
func (v *VDSO) PromotedTools() []string {
	if v == nil {
		return nil
	}
	v.promotedMu.RLock()
	reg := v.promoted
	v.promotedMu.RUnlock()
	if reg == nil {
		return nil
	}
	return reg.Tools()
}

// isReadOnly reports cache eligibility. Explicit replay hints are authoritative;
// tool-name inference and runtime promotion apply only when neither hint is supplied.
func (v *VDSO) isReadOnly(c *abi.ToolCall) bool {
	if c == nil || destructive(c) {
		return false
	}
	_, hasReadHint := c.Meta["readOnlyHint"]
	_, hasIdempotentHint := c.Meta["idempotentHint"]
	if hasReadHint || hasIdempotentHint {
		return metaTrue(c, "readOnlyHint") && metaTrue(c, "idempotentHint")
	}
	if isReadOnlyCall(c) {
		return true
	}
	return v.IsPromotedReadOnly(c.Tool)
}

// PromoteReadOnly promotes tool on the Default vDSO instance.
func PromoteReadOnly(tool string) {
	Default.PromoteReadOnly(tool)
}

// IsPromotedReadOnly reports if tool is promoted on the Default vDSO instance.
func IsPromotedReadOnly(tool string) bool {
	return Default.IsPromotedReadOnly(tool)
}

// InvalidatePromoted invalidates tool on the Default vDSO instance.
func InvalidatePromoted(tool string) {
	Default.InvalidatePromoted(tool)
}
