package headroom

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// HeadroomStatus represents the observed context headroom health state.
type HeadroomStatus string

const (
	HeadroomStatusHealthy  HeadroomStatus = "healthy"
	HeadroomStatusLow      HeadroomStatus = "low"
	HeadroomStatusCritical HeadroomStatus = "critical"
	HeadroomStatusUnknown  HeadroomStatus = "unknown"
)

// ParseHeadroomStatus normalizes a raw status string to a known HeadroomStatus.
func ParseHeadroomStatus(s string) HeadroomStatus {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "healthy", "fresh", "ok", "green":
		return HeadroomStatusHealthy
	case "low", "constrained", "warn", "warning", "yellow":
		return HeadroomStatusLow
	case "critical", "danger", "exhausted", "red", "reset_imminent":
		return HeadroomStatusCritical
	default:
		return HeadroomStatusUnknown
	}
}

// String returns the canonical lower-case string representation.
func (s HeadroomStatus) String() string {
	switch s {
	case HeadroomStatusHealthy:
		return "healthy"
	case HeadroomStatusLow:
		return "low"
	case HeadroomStatusCritical:
		return "critical"
	default:
		return "unknown"
	}
}

// IsValid reports whether s is a recognized headroom status.
func (s HeadroomStatus) IsValid() bool {
	switch s {
	case HeadroomStatusHealthy, HeadroomStatusLow, HeadroomStatusCritical, HeadroomStatusUnknown:
		return true
	default:
		return false
	}
}

// CompressionRisk represents the evaluated risk of compressing tool results or overflowing context.
type CompressionRisk string

const (
	RiskLow      CompressionRisk = "low"
	RiskMedium   CompressionRisk = "medium"
	RiskHigh     CompressionRisk = "high"
	RiskCritical CompressionRisk = "critical"
	RiskUnknown  CompressionRisk = "unknown"
)

// ParseCompressionRisk normalizes a raw risk string to a known CompressionRisk.
func ParseCompressionRisk(s string) CompressionRisk {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "low", "minimal", "none":
		return RiskLow
	case "medium", "moderate", "med":
		return RiskMedium
	case "high", "elevated":
		return RiskHigh
	case "critical", "severe":
		return RiskCritical
	default:
		return RiskUnknown
	}
}

// String returns the canonical lower-case string representation.
func (r CompressionRisk) String() string {
	switch r {
	case RiskLow:
		return "low"
	case RiskMedium:
		return "medium"
	case RiskHigh:
		return "high"
	case RiskCritical:
		return "critical"
	default:
		return "unknown"
	}
}

// ReserveState captures live context reserve tokens and risk signals.
type ReserveState struct {
	Status          HeadroomStatus  `json:"status"`                     // observed context headroom status
	RemainingTokens int             `json:"remaining_tokens,omitempty"` // unallocated tokens in context window
	TotalTokens     int             `json:"total_tokens,omitempty"`     // total context window capacity (0 if unknown)
	ReserveTokens   int             `json:"reserve_tokens,omitempty"`   // floor reserved for reasoning/completion
	CompressionRisk CompressionRisk `json:"compression_risk,omitempty"` // evaluated risk of compression or overrun
}

// EffectiveStatus returns the normalized headroom status, deriving it from token metrics
// if Status was unobserved or Unknown.
func (r ReserveState) EffectiveStatus() HeadroomStatus {
	st := ParseHeadroomStatus(string(r.Status))
	if st != HeadroomStatusUnknown {
		return st
	}
	// Derive status from token metrics when status was not explicitly observed.
	if r.TotalTokens > 0 && r.RemainingTokens >= 0 {
		avail := r.RemainingTokens - r.ReserveTokens
		ratio := float64(r.RemainingTokens) / float64(r.TotalTokens)
		if avail <= 0 || ratio <= 0.10 {
			return HeadroomStatusCritical
		}
		if ratio <= 0.25 || (r.ReserveTokens > 0 && avail <= r.ReserveTokens) {
			return HeadroomStatusLow
		}
		return HeadroomStatusHealthy
	}
	if r.RemainingTokens > 0 && r.ReserveTokens > 0 {
		avail := r.RemainingTokens - r.ReserveTokens
		if avail <= 0 {
			return HeadroomStatusCritical
		}
		if avail <= r.ReserveTokens {
			return HeadroomStatusLow
		}
		return HeadroomStatusHealthy
	}
	return HeadroomStatusUnknown
}

// Tier names for tool result rendering budgets.
const (
	TierStandard     = "standard"
	TierConservative = "conservative"
	TierCautious     = "cautious"
	TierMinimal      = "minimal"
)

// ToolResultBudget defines discrete limits for rendered tool results.
type ToolResultBudget struct {
	MaxItems   int    `json:"max_items"`   // maximum items/lines to render (1, 3, 5, 10)
	ByteLimit  int    `json:"byte_limit"`  // maximum raw bytes for rendered result
	TokenLimit int    `json:"token_limit"` // estimated token ceiling for rendered result
	Tier       string `json:"tier"`        // "standard", "conservative", "cautious", "minimal"
}

// Discrete safe default tiers based on observed context headroom status.
var (
	// BudgetHealthy is the standard budget for healthy context headroom (10 items, 16KB, 4096 tokens).
	BudgetHealthy = ToolResultBudget{
		MaxItems:   10,
		ByteLimit:  16384,
		TokenLimit: 4096,
		Tier:       TierStandard,
	}
	// BudgetLow is the conservative budget for low context headroom (5 items, 8KB, 2048 tokens).
	BudgetLow = ToolResultBudget{
		MaxItems:   5,
		ByteLimit:  8192,
		TokenLimit: 2048,
		Tier:       TierConservative,
	}
	// BudgetUnknown is the cautious safe default when headroom is unobserved (3 items, 4KB, 1024 tokens).
	BudgetUnknown = ToolResultBudget{
		MaxItems:   3,
		ByteLimit:  4096,
		TokenLimit: 1024,
		Tier:       TierCautious,
	}
	// BudgetCritical is the strict minimal/compact budget for critical context headroom (1 item, 2KB, 512 tokens).
	BudgetCritical = ToolResultBudget{
		MaxItems:   1,
		ByteLimit:  2048,
		TokenLimit: 512,
		Tier:       TierMinimal,
	}
)

// DefaultBudgetForStatus returns the discrete safe default tier for a given HeadroomStatus.
func DefaultBudgetForStatus(status HeadroomStatus) ToolResultBudget {
	switch status {
	case HeadroomStatusHealthy:
		return BudgetHealthy
	case HeadroomStatusLow:
		return BudgetLow
	case HeadroomStatusCritical:
		return BudgetCritical
	default:
		return BudgetUnknown
	}
}

// BudgetReceipt explains the computed budget and active reserve state.
type BudgetReceipt struct {
	Schema          string           `json:"schema"`                 // "fak-headroom-budget-receipt/1"
	Status          HeadroomStatus   `json:"status"`                 // evaluated headroom status
	CompressionRisk CompressionRisk  `json:"compression_risk"`       // evaluated compression risk
	Budget          ToolResultBudget `json:"budget"`                 // computed effective budget
	ActiveReserves  ReserveState     `json:"active_reserves"`        // snapshot of the evaluated reserve state
	Clamped         bool             `json:"clamped"`                // whether budget was clamped from standard
	ClampReason     string           `json:"clamp_reason,omitempty"` // explanation of why clamping occurred
	Explanation     string           `json:"explanation"`            // human/agent-readable rationale
}

// String returns a compact single-line rendering of the budget receipt.
func (r BudgetReceipt) String() string {
	if r.Clamped {
		return fmt.Sprintf("budget: %s (tier=%s, items=%d, bytes=%d, tokens=%d, clamped: %s)",
			r.Status, r.Budget.Tier, r.Budget.MaxItems, r.Budget.ByteLimit, r.Budget.TokenLimit, r.ClampReason)
	}
	return fmt.Sprintf("budget: %s (tier=%s, items=%d, bytes=%d, tokens=%d)",
		r.Status, r.Budget.Tier, r.Budget.MaxItems, r.Budget.ByteLimit, r.Budget.TokenLimit)
}

// ComputeBudget determines the deterministic rendering budget for tool results
// by coupling observed context headroom status, live context reserves, and compression risk.
func ComputeBudget(reserves ReserveState) BudgetReceipt {
	status := reserves.EffectiveStatus()
	risk := ParseCompressionRisk(string(reserves.CompressionRisk))

	baseBudget := DefaultBudgetForStatus(status)
	budget := baseBudget
	clamped := false
	var clampReasons []string

	// Rule 1: Critical context headroom ALWAYS clamps to strict minimal/compact budget.
	if status == HeadroomStatusCritical {
		budget = BudgetCritical
		clamped = true
		clampReasons = append(clampReasons, "critical headroom clamped to minimal budget")
	} else if status == HeadroomStatusHealthy {
		// Rule 2: Healthy headroom allows standard budget by default, but elevated risk clamps.
		if risk == RiskCritical {
			budget = BudgetCritical
			clamped = true
			clampReasons = append(clampReasons, "critical compression risk clamped healthy headroom to minimal")
		} else if risk == RiskHigh {
			budget = BudgetLow
			clamped = true
			clampReasons = append(clampReasons, "high compression risk reduced healthy headroom to conservative")
		}
	} else if status == HeadroomStatusLow {
		// Low headroom defaults to conservative; elevated risk clamps to minimal.
		if risk == RiskHigh || risk == RiskCritical {
			budget = BudgetCritical
			clamped = true
			clampReasons = append(clampReasons, fmt.Sprintf("%s compression risk clamped low headroom to minimal", risk))
		}
	} else { // Unknown
		if risk == RiskHigh || risk == RiskCritical {
			budget = BudgetCritical
			clamped = true
			clampReasons = append(clampReasons, fmt.Sprintf("%s compression risk clamped unknown headroom to minimal", risk))
		}
	}

	// Rule 3: Live context reserve limits. If remaining tokens are known, clamp TokenLimit to available tokens.
	if reserves.RemainingTokens > 0 {
		avail := reserves.RemainingTokens - reserves.ReserveTokens
		if avail <= 0 {
			budget.MaxItems = 1
			budget.TokenLimit = 128
			budget.ByteLimit = 512
			budget.Tier = TierMinimal
			clamped = true
			clampReasons = append(clampReasons, "reserve floor exhausted; clamped to minimal safety floor")
		} else if avail < budget.TokenLimit {
			budget.TokenLimit = avail
			if avail*4 < budget.ByteLimit {
				budget.ByteLimit = avail * 4
			}
			if avail < 512 {
				budget.MaxItems = 1
				budget.Tier = TierMinimal
			} else if avail < 2048 && budget.MaxItems > 5 {
				budget.MaxItems = 5
				budget.Tier = TierConservative
			}
			clamped = true
			clampReasons = append(clampReasons, fmt.Sprintf("available reserve tokens (%d) below tier limit; clamped to reserve", avail))
		}
	}

	clampReason := ""
	if len(clampReasons) > 0 {
		clampReason = strings.Join(clampReasons, "; ")
	}

	var explanation string
	if clamped {
		explanation = fmt.Sprintf("context headroom %s (risk=%s) clamped: %s; effective budget: %d items, %d bytes, %d tokens (tier: %s)",
			status, risk, clampReason, budget.MaxItems, budget.ByteLimit, budget.TokenLimit, budget.Tier)
	} else {
		explanation = fmt.Sprintf("context headroom %s (risk=%s); standard %s budget allowed: %d items, %d bytes, %d tokens",
			status, risk, budget.Tier, budget.MaxItems, budget.ByteLimit, budget.TokenLimit)
	}

	active := reserves
	active.Status = status
	active.CompressionRisk = risk

	return BudgetReceipt{
		Schema:          "fak-headroom-budget-receipt/1",
		Status:          status,
		CompressionRisk: risk,
		Budget:          budget,
		ActiveReserves:  active,
		Clamped:         clamped,
		ClampReason:     clampReason,
		Explanation:     explanation,
	}
}

// ComputeBudgetForStatus is a convenience constructor computing the budget solely from HeadroomStatus.
func ComputeBudgetForStatus(status HeadroomStatus) BudgetReceipt {
	return ComputeBudget(ReserveState{Status: status})
}

// ClampResultLines limits a slice of item lines to budget.MaxItems.
// When truncated, it appends an elision notice and reports truncated=true.
func ClampResultLines(lines []string, budget ToolResultBudget) ([]string, bool) {
	if budget.MaxItems <= 0 || len(lines) <= budget.MaxItems {
		return lines, false
	}
	keep := make([]string, budget.MaxItems, budget.MaxItems+1)
	copy(keep, lines[:budget.MaxItems])
	elided := len(lines) - budget.MaxItems
	keep = append(keep, fmt.Sprintf("[... clamped %d item(s) to adhere to %s budget (%d max items) ...]", elided, budget.Tier, budget.MaxItems))
	return keep, true
}

// ClampResultBytes limits raw bytes to budget.ByteLimit.
// When truncated, it trims at a clean boundary and appends a truncation notice.
func ClampResultBytes(raw []byte, budget ToolResultBudget) ([]byte, bool) {
	if budget.ByteLimit <= 0 || len(raw) <= budget.ByteLimit {
		return raw, false
	}
	limit := budget.ByteLimit
	noticeTemplate := "\n[... clamped %d byte(s) to adhere to %s budget (%d max bytes) ...]"
	dummyNotice := fmt.Sprintf(noticeTemplate, len(raw)-limit, budget.Tier, limit)
	target := limit - len(dummyNotice)
	if target < 16 {
		target = limit / 2
	}
	cut := target
	// Try to find a newline within the last 256 bytes of target to avoid tearing words/lines
	if idx := bytes.LastIndexByte(raw[:cut], '\n'); idx > target-256 && idx > 0 {
		cut = idx
	}
	elided := len(raw) - cut
	notice := fmt.Sprintf(noticeTemplate, elided, budget.Tier, limit)
	out := make([]byte, 0, cut+len(notice))
	out = append(out, raw[:cut]...)
	out = append(out, notice...)
	if len(out) >= len(raw) {
		return raw[:limit], true
	}
	if len(out) > limit {
		out = out[:limit]
	}
	return out, true
}

// ClampToolResult applies both item line limits (if multi-line) and byte limits
// to rendered tool result bytes according to budget.
func ClampToolResult(raw []byte, budget ToolResultBudget) ([]byte, bool) {
	if len(raw) == 0 {
		return raw, false
	}
	truncated := false
	work := raw

	// If input has multiple lines, clamp lines if count exceeds MaxItems
	lines := strings.Split(string(work), "\n")
	if len(lines) > budget.MaxItems && budget.MaxItems > 0 {
		clampedLines, didLineClamp := ClampResultLines(lines, budget)
		if didLineClamp {
			candidate := []byte(strings.Join(clampedLines, "\n"))
			if len(candidate) < len(work) || budget.Tier == TierMinimal {
				work = candidate
				truncated = true
			}
		}
	}

	// Check byte limit
	if len(work) > budget.ByteLimit && budget.ByteLimit > 0 {
		clampedBytes, didByteClamp := ClampResultBytes(work, budget)
		if didByteClamp {
			work = clampedBytes
			truncated = true
		}
	}
	return work, truncated
}

// ParseReceipt decodes a JSON-encoded BudgetReceipt and validates its schema.
func ParseReceipt(data []byte) (BudgetReceipt, error) {
	var r BudgetReceipt
	if err := json.Unmarshal(data, &r); err != nil {
		return r, fmt.Errorf("headroom budget: decode receipt: %w", err)
	}
	if r.Schema != "fak-headroom-budget-receipt/1" {
		return r, fmt.Errorf("headroom budget: unsupported schema %q", r.Schema)
	}
	return r, nil
}
