package fitplan

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// FitConfirmSchema stamps the post-load two-phase fit confirmation record. It is the
// llama.cpp breakdown form: total = free_after + self + unaccounted, where self is the
// sum of the plan rows the scope was charged and unaccounted is everything else the load
// consumed (compute buffers, staging, transients, fragmentation, live-reading drift).
const FitConfirmSchema = "fak.fit_confirm.v1"

// ErrFitConfirmInvalid marks a confirm record that fails Validate.
var ErrFitConfirmInvalid = errors.New("fitplan: fit_confirm_invalid")

// ConfirmClass is one per-class planned-vs-allocated row of a fit_confirm record.
type ConfirmClass string

const (
	ConfirmWeights        ConfirmClass = "weights"
	ConfirmKV             ConfirmClass = "kv"
	ConfirmComputeStaging ConfirmClass = "compute/staging"
	ConfirmRing           ConfirmClass = "ring"
	ConfirmTransient      ConfirmClass = "transient"
)

// confirmClassOrder is the canonical row order every record renders in, so a golden
// JSON or log line is stable across builds.
var confirmClassOrder = []ConfirmClass{
	ConfirmWeights, ConfirmKV, ConfirmComputeStaging, ConfirmRing, ConfirmTransient,
}

// ConfirmScopeRow is one scope's (device or host) breakdown: what the plan charged,
// what the capacity probe shows after load, and the residual neither accounts for.
type ConfirmScopeRow struct {
	Scope            string            `json:"scope"`
	TotalBytes       int64             `json:"total_bytes"`
	FreeAfterLoad    int64             `json:"free_after_load_bytes"`
	FreeKnown        bool              `json:"free_known"`
	Classes          []ConfirmClassRow `json:"classes"`
	UnaccountedBytes int64             `json:"unaccounted_bytes"`
}

// ConfirmClassRow is one class's planned vs allocated bytes. Allocated is the
// placement the plan committed for that class; where the writer holds a post-load
// measurement it supplies it, otherwise it equals Planned and Unaccounted carries
// the difference between the plan and the live reading.
type ConfirmClassRow struct {
	Class          ConfirmClass `json:"class"`
	PlannedBytes   int64        `json:"planned_bytes"`
	AllocatedBytes int64        `json:"allocated_bytes"`
}

// FitConfirmRecord is the post-load two-phase confirmation: per scope, the planned
// charge per class against the post-load capacity reading, plus the unaccounted
// residual. Emitted on the startup log line and as JSON.
type FitConfirmRecord struct {
	Schema  string            `json:"schema"`
	Model   string            `json:"model,omitempty"`
	Backend string            `json:"backend,omitempty"`
	Scopes  []ConfirmScopeRow `json:"scopes"`
}

// ConfirmInput is one scope's inputs to BuildFitConfirm.
type ConfirmInput struct {
	// Scope is "device" or "host".
	Scope     string
	Total     int64
	FreeAfter int64
	Known     bool
	// FreeKnown is false when the total is known but the post-load free reading is not
	// probeable; the row then reports no unaccounted number instead of inventing one.
	FreeKnown bool
	// Planned is the plan's charge per class for this scope; zero classes are omitted.
	Planned map[ConfirmClass]int64
}

// BuildFitConfirm assembles the record from per-scope readings and planned charges.
// Scopes keep caller order (device before host by convention); classes render in
// confirmClassOrder. A scope whose capacity is unknown keeps its row with
// TotalBytes 0 so the operator can see which probe failed.
func BuildFitConfirm(model, backend string, scopes ...ConfirmInput) FitConfirmRecord {
	rec := FitConfirmRecord{Schema: FitConfirmSchema, Model: model, Backend: backend, Scopes: make([]ConfirmScopeRow, 0, len(scopes))}
	for _, in := range scopes {
		row := ConfirmScopeRow{
			Scope:     strings.TrimSpace(in.Scope),
			FreeKnown: in.FreeKnown && in.Known,
		}
		if in.Known {
			row.TotalBytes = in.Total
			if row.FreeKnown {
				row.FreeAfterLoad = in.FreeAfter
			}
		}
		self := int64(0)
		for _, c := range confirmClassOrder {
			planned, ok := in.Planned[c]
			if !ok || planned <= 0 {
				continue
			}
			self += planned
			row.Classes = append(row.Classes, ConfirmClassRow{Class: c, PlannedBytes: planned, AllocatedBytes: planned})
		}
		if row.FreeKnown {
			row.UnaccountedBytes = row.TotalBytes - row.FreeAfterLoad - self
		}
		rec.Scopes = append(rec.Scopes, row)
	}
	return rec
}

// Validate reports a record another consumer cannot use; errors.Is matches
// ErrFitConfirmInvalid.
func (r FitConfirmRecord) Validate() error {
	bad := func(why string) error { return fmt.Errorf("%w: %s", ErrFitConfirmInvalid, why) }
	if r.Schema != FitConfirmSchema {
		return bad("schema " + strconv.Quote(r.Schema) + ", want " + strconv.Quote(FitConfirmSchema))
	}
	if len(r.Scopes) == 0 {
		return bad("no scope rows")
	}
	for i, s := range r.Scopes {
		if s.Scope != "device" && s.Scope != "host" {
			return bad(fmt.Sprintf("scope %d names %q, want device or host", i, s.Scope))
		}
		if s.FreeKnown && s.TotalBytes < 0 {
			return bad(fmt.Sprintf("scope %s has negative total", s.Scope))
		}
		for _, c := range s.Classes {
			if c.PlannedBytes < 0 || c.AllocatedBytes < 0 {
				return bad(fmt.Sprintf("scope %s class %s has negative bytes", s.Scope, c.Class))
			}
		}
	}
	return nil
}

// Text renders the record as the one startup log line: per scope, the total, the
// post-load free, each non-zero class, and the unaccounted residual. An unknown
// scope reads "unprobed" so the line never prints a fake zero.
func (r FitConfirmRecord) Text() string {
	var b strings.Builder
	fmt.Fprintf(&b, "fit_confirm")
	if r.Model != "" {
		fmt.Fprintf(&b, " model=%s", r.Model)
	}
	if r.Backend != "" {
		fmt.Fprintf(&b, " backend=%s", r.Backend)
	}
	for _, s := range r.Scopes {
		fmt.Fprintf(&b, " | %s:", s.Scope)
		if !s.FreeKnown {
			fmt.Fprint(&b, " unprobed")
			continue
		}
		fmt.Fprintf(&b, " total=%s free_after=%s", formatConfirmBytes(s.TotalBytes), formatConfirmBytes(s.FreeAfterLoad))
		for _, c := range s.Classes {
			fmt.Fprintf(&b, " %s=%s", c.Class, formatConfirmBytes(c.AllocatedBytes))
		}
		fmt.Fprintf(&b, " unaccounted=%s", formatConfirmSignedBytes(s.UnaccountedBytes))
	}
	return b.String()
}

// formatConfirmBytes renders a byte count in the smallest honest GiB/MiB/KiB form.
func formatConfirmBytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.2fGiB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.1fMiB", float64(n)/(1<<20))
	default:
		return fmt.Sprintf("%dB", n)
	}
}

// formatConfirmSignedBytes keeps a sign on a non-zero value: a negative unaccounted
// means the plan over-charged relative to the observed consumption (or free grew).
func formatConfirmSignedBytes(n int64) string {
	if n > 0 {
		return "+" + formatConfirmBytes(n)
	}
	return formatConfirmBytes(n)
}
