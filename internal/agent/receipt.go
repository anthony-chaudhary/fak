package agent

// receipt.go — the SIGNED TERMINAL TURN RECEIPT (#2415). A harness's final result
// message is a self-accounting turn receipt (cost, turns, denials) a parent must
// take on trust. fak's ArmMetrics are kernel-measured; this extends them into a
// terminal receipt bound to the guard journal's hash chain (internal/journal), with
// every field labeled WITNESSED (kernel-authored, re-derivable from the journal) or
// OBSERVED (relayed provider usage the worker self-reports). A peer, a dispatcher, or
// dos-witness-claim can then VERIFY the receipt against the journal WITHOUT trusting
// the worker — the receipt becomes dos_verify input instead of narrative.
//
// THE NON-FORGEABILITY. VerifyReceipt does not trust the receipt's WITNESSED numbers:
// it re-folds them straight from the journal rows and requires a match. So a worker
// that deflates its denial count and RE-SIGNS the receipt still fails verification —
// the journal, not the receipt, is the authority. The Sig additionally binds EVERY
// field (WITNESSED and OBSERVED) to the journal chain head, so any blob tamper is
// caught even without a journal round-trip. It is a COMMITMENT over the journal's own
// tamper-evident chain, not a PKI signature: forging a receipt means forging the
// journal, which journal.Verify makes detectable.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachewitness"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// Receipt field names. WITNESSED fields are folded from the journal; OBSERVED fields
// are relayed provider usage. Exported so a consumer (the `fak session receipt` shell)
// can lift a specific number by name.
const (
	FieldDenialsByReason  = "denials_by_reason" // WITNESSED
	FieldDenialsTotal     = "denials_total"     // WITNESSED
	FieldAdmitted         = "admitted_results"  // WITNESSED
	FieldTaintHighWater   = "taint_high_water"  // WITNESSED
	FieldWitnessGates     = "witness_gates"     // WITNESSED
	FieldTurns            = "turns"             // OBSERVED
	FieldPromptTokens     = "prompt_tokens"     // OBSERVED
	FieldCompletionTokens = "completion_tokens" // OBSERVED
)

// ReceiptField is one labeled line of a turn receipt: a name, its canonical string
// value, and the provenance class saying whether the number is kernel-authored
// (WITNESSED, recomputable from the journal) or relayed (OBSERVED, a self-reported
// figure a verifier must not treat as kernel proof).
type ReceiptField struct {
	Name  string                  `json:"name"`
	Value string                  `json:"value"`
	Prov  cachewitness.Provenance `json:"prov"`
}

// Receipt is a terminal turn receipt bound to the guard journal's hash chain. Sig
// binds ChainHead plus every field, so a mutated field fails VerifyReceipt.
type Receipt struct {
	TraceID   string         `json:"trace_id"`
	Fields    []ReceiptField `json:"fields"`
	ChainHead string         `json:"chain_head"` // hash of the journal's last row the receipt is bound to
	Sig       string         `json:"sig"`        // commitment over ChainHead + canonical(Fields)
}

// ObservedUsage is the relayed, self-reported side of a receipt: the harness turn
// count and provider token usage. None of it is journal-derivable, so every field it
// contributes is labeled OBSERVED. ArmMetrics.ObservedUsage bridges the existing
// kernel-adjacent metrics into this shape.
type ObservedUsage struct {
	Turns            int
	PromptTokens     int
	CompletionTokens int
}

// ObservedUsage lifts the relayed usage numbers out of an ArmMetrics so a completed
// arm can be turned into a terminal receipt (the "extend ArmMetrics into a terminal
// receipt" seam). Turns/tokens are harness/provider figures — OBSERVED, never
// WITNESSED — so they ride the receipt only as self-reported context.
func (m ArmMetrics) ObservedUsage() ObservedUsage {
	return ObservedUsage{Turns: m.Turns, PromptTokens: m.PromptTokens, CompletionTokens: m.CompletionTokens}
}

// witnessedFacts is the kernel-authored side of a receipt, folded from the journal.
type witnessedFacts struct {
	denialsByReason map[string]int
	denialsTotal    int
	admitted        int
	witnessGates    int
	taintHighWater  string
}

// foldWitnessed folds the durable decision journal for one trace into the WITNESSED
// receipt numbers. traceID == "" folds every row (a whole-journal receipt). Only DENY
// rows count as tool-call denials: a RESULT_DENY / QUARANTINE is a result-floor
// page-out, a distinct phenomenon — the same DENY-not-quarantine discipline
// internal/sessionobs already draws over transcripts.
func foldWitnessed(traceID string, rows []journal.Row) witnessedFacts {
	f := witnessedFacts{denialsByReason: map[string]int{}}
	taintRank := 0
	for _, r := range rows {
		if traceID != "" && r.TraceID != traceID {
			continue
		}
		switch r.Kind {
		case "DENY":
			f.denialsTotal++
			reason := r.Reason
			if reason == "" {
				reason = "unspecified"
			}
			f.denialsByReason[reason]++
		case "DECIDE":
			switch r.Verdict {
			case "ALLOW":
				f.admitted++
			case "WITNESS":
				f.witnessGates++
			}
		}
		if rank := taintOrder(r.Taint); rank > taintRank {
			taintRank = rank
			f.taintHighWater = r.Taint
		}
	}
	return f
}

// taintOrder ranks the result-taint labels so the high-water mark is a max.
func taintOrder(t string) int {
	switch t {
	case "trusted":
		return 1
	case "tainted":
		return 2
	case "quarantined":
		return 3
	}
	return 0
}

// BuildReceipt folds the journal rows for traceID into the WITNESSED fields, appends
// the OBSERVED usage, binds the receipt to the journal's chain head, and signs it.
// Pass the FULL journal (chain head = last row's Hash) so the binding covers the whole
// ledger the WITNESSED numbers were read from.
func BuildReceipt(traceID string, rows []journal.Row, obs ObservedUsage) Receipt {
	f := foldWitnessed(traceID, rows)
	fields := []ReceiptField{
		{FieldDenialsByReason, renderDenials(f.denialsByReason), cachewitness.Witnessed},
		{FieldDenialsTotal, strconv.Itoa(f.denialsTotal), cachewitness.Witnessed},
		{FieldAdmitted, strconv.Itoa(f.admitted), cachewitness.Witnessed},
		{FieldTaintHighWater, orNone(f.taintHighWater), cachewitness.Witnessed},
		{FieldWitnessGates, strconv.Itoa(f.witnessGates), cachewitness.Witnessed},
		{FieldTurns, strconv.Itoa(obs.Turns), cachewitness.Observed},
		{FieldPromptTokens, strconv.Itoa(obs.PromptTokens), cachewitness.Observed},
		{FieldCompletionTokens, strconv.Itoa(obs.CompletionTokens), cachewitness.Observed},
	}
	head := chainHead(rows)
	return Receipt{TraceID: traceID, Fields: fields, ChainHead: head, Sig: signReceipt(head, fields)}
}

// VerifyReceipt checks a receipt against the durable journal WITHOUT trusting the
// worker that emitted it: (1) the journal's own hash chain must be intact, (2) the
// WITNESSED fields must match a fresh re-fold from the journal (so a re-signed denial
// count is still caught), (3) the chain-head binding must match, and (4) the signature
// must recompute over every field. rows must be the FULL journal the receipt was built
// over. A nil return means the receipt is authentic.
func VerifyReceipt(r Receipt, rows []journal.Row) error {
	if _, err := journal.VerifyRows(rows); err != nil {
		return fmt.Errorf("receipt: journal chain not intact: %w", err)
	}
	if head := chainHead(rows); head != r.ChainHead {
		return fmt.Errorf("receipt: chain-head mismatch: receipt=%s journal=%s", short(r.ChainHead), short(head))
	}
	// Re-derive the WITNESSED ground truth from the journal. OBSERVED fields cannot be
	// recomputed (they are relayed), so they are covered by the signature only — which
	// is exactly why they carry the OBSERVED label.
	ref := BuildReceipt(r.TraceID, rows, ObservedUsage{})
	refByName := make(map[string]ReceiptField, len(ref.Fields))
	for _, f := range ref.Fields {
		refByName[f.Name] = f
	}
	for _, f := range r.Fields {
		if f.Prov != cachewitness.Witnessed {
			continue
		}
		want, ok := refByName[f.Name]
		if !ok {
			return fmt.Errorf("receipt: unknown witnessed field %q", f.Name)
		}
		if want.Value != f.Value {
			return fmt.Errorf("receipt: witnessed field %q disagrees with journal: receipt=%q journal=%q", f.Name, f.Value, want.Value)
		}
	}
	if sig := signReceipt(r.ChainHead, r.Fields); sig != r.Sig {
		return fmt.Errorf("receipt: signature mismatch (a field was altered after signing)")
	}
	return nil
}

// FieldValue returns the canonical value of a named receipt field (and whether it was
// present), so a consumer can lift a specific number without re-folding.
func (r Receipt) FieldValue(name string) (string, bool) {
	for _, f := range r.Fields {
		if f.Name == name {
			return f.Value, true
		}
	}
	return "", false
}

// RenderReceipt writes the human, per-field-labeled receipt view plus the
// verification verdict (verifyErr == nil means it verified against the journal).
func RenderReceipt(w io.Writer, r Receipt, verifyErr error) {
	fmt.Fprintf(w, "receipt trace=%s chain_head=%s sig=%s\n", orNone(r.TraceID), short(r.ChainHead), short(r.Sig))
	for _, f := range r.Fields {
		fmt.Fprintf(w, "  [%-9s] %-18s %s\n", f.Prov, f.Name, f.Value)
	}
	if verifyErr != nil {
		fmt.Fprintf(w, "verify: FAIL — %v\n", verifyErr)
		return
	}
	fmt.Fprintln(w, "verify: OK — WITNESSED fields recomputed from the journal hash chain")
}

// signReceipt is the receipt's binding to the journal: a sha256 over the chain head
// and every field (name|value|prov), unit-separated so no concatenation collision is
// possible. Any changed field flips this digest.
func signReceipt(head string, fields []ReceiptField) string {
	h := sha256.New()
	io.WriteString(h, head)
	for _, f := range fields {
		fmt.Fprintf(h, "\x1f%s\x1f%s\x1f%s", f.Name, f.Value, f.Prov)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func chainHead(rows []journal.Row) string {
	if len(rows) == 0 {
		return ""
	}
	return rows[len(rows)-1].Hash
}

// renderDenials canonicalizes the denials-by-reason map to a sorted "reason:count"
// list so the WITNESSED value is deterministic (the signature depends on it).
func renderDenials(m map[string]int) string {
	if len(m) == 0 {
		return "none"
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+":"+strconv.Itoa(m[k]))
	}
	return strings.Join(parts, ",")
}

func orNone(s string) string {
	if s == "" {
		return "none"
	}
	return s
}

func short(s string) string {
	if len(s) <= 12 {
		return s
	}
	return s[:12]
}
