package agent

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachewitness"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// buildTraceJournal writes one allow + one deny for trace onto a real in-memory
// hash-chained journal and returns its rows. The rows carry genuine PrevHash/Hash
// links (the kernel's own chain), so VerifyReceipt exercises the true journal
// round-trip, not a hand-forged one.
func buildTraceJournal(t *testing.T, trace string) []journal.Row {
	t.Helper()
	j := journal.OpenMemory()
	allow := &abi.ToolCall{Tool: "search_kb", TraceID: trace, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}}
	j.Emit(abi.Event{Kind: abi.EvDecide, Call: allow, Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "test"}})
	deny := &abi.ToolCall{Tool: "send_email", TraceID: trace, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"to":"x@y.com"}`)}}
	j.Emit(abi.Event{Kind: abi.EvDeny, Call: deny, Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "floor"}})
	return j.Recent(0)
}

// TestReceiptRoundTripVerifies is the positive control: an untouched receipt built
// over the journal it was folded from verifies, and its WITNESSED fields carry the
// journal-derived numbers (1 admit, 1 deny) while the OBSERVED fields carry the
// relayed usage — the per-field provenance labels the receipt exists to draw.
func TestReceiptRoundTripVerifies(t *testing.T) {
	rows := buildTraceJournal(t, "trace-a")
	r := BuildReceipt("trace-a", rows, ObservedUsage{Turns: 3, PromptTokens: 100, CompletionTokens: 50})

	if err := VerifyReceipt(r, rows); err != nil {
		t.Fatalf("an untouched receipt failed verification: %v", err)
	}
	if got, _ := r.FieldValue(FieldDenialsTotal); got != "1" {
		t.Fatalf("%s = %q, want \"1\" (one DENY row)", FieldDenialsTotal, got)
	}
	if got, _ := r.FieldValue(FieldAdmitted); got != "1" {
		t.Fatalf("%s = %q, want \"1\" (one ALLOW decision)", FieldAdmitted, got)
	}
	if got, _ := r.FieldValue(FieldTurns); got != "3" {
		t.Fatalf("%s = %q, want \"3\" (the relayed turn count)", FieldTurns, got)
	}
	// Provenance labels must partition the fields: the kernel-authored counts are
	// WITNESSED, the relayed usage is OBSERVED. Mislabeling a self-reported number
	// WITNESSED is the exact conflation this receipt exists to remove.
	prov := map[string]cachewitness.Provenance{}
	for _, f := range r.Fields {
		prov[f.Name] = f.Prov
	}
	if prov[FieldDenialsTotal] != cachewitness.Witnessed {
		t.Fatalf("%s labeled %v, want WITNESSED", FieldDenialsTotal, prov[FieldDenialsTotal])
	}
	if prov[FieldTurns] != cachewitness.Observed {
		t.Fatalf("%s labeled %v, want OBSERVED", FieldTurns, prov[FieldTurns])
	}
}

// TestReceiptTamperFailsVerify is the load-bearing non-forgeability witness (#2415):
// a worker that DEFLATES its WITNESSED denial count cannot make the receipt verify,
// even when it re-signs the doctored receipt so the signature is internally
// consistent. VerifyReceipt re-folds the WITNESSED numbers straight from the journal
// hash chain, so the journal — not the receipt's self-report — is the authority.
func TestReceiptTamperFailsVerify(t *testing.T) {
	rows := buildTraceJournal(t, "trace-a")
	r := BuildReceipt("trace-a", rows, ObservedUsage{Turns: 3, PromptTokens: 100, CompletionTokens: 50})
	if err := VerifyReceipt(r, rows); err != nil {
		t.Fatalf("precondition: the honest receipt must verify: %v", err)
	}

	// (1) Naive tamper: deflate the denial count in place, leave the signature. The
	// signature no longer recomputes over the mutated field -> caught.
	naive := deepCopyReceipt(r)
	setField(t, &naive, FieldDenialsTotal, "0")
	if err := VerifyReceipt(naive, rows); err == nil {
		t.Fatal("a mutated denial count with a stale signature verified — tamper undetected")
	}

	// (2) Re-signed tamper (the real forgery attempt): deflate the denial count AND
	// re-sign so the receipt is self-consistent. It must STILL fail, because the
	// WITNESSED numbers are re-derived from the journal, not trusted from the receipt.
	forged := deepCopyReceipt(r)
	setField(t, &forged, FieldDenialsTotal, "0")
	forged.Sig = signReceipt(forged.ChainHead, forged.Fields)
	err := VerifyReceipt(forged, rows)
	if err == nil {
		t.Fatal("a re-signed deflated denial count verified — the receipt was trusted over the journal")
	}
	if !strings.Contains(err.Error(), FieldDenialsTotal) {
		t.Fatalf("verify error did not name the tampered field: %v", err)
	}

	// The honest receipt still verifies (the tamper mutated a copy, not the ledger).
	if err := VerifyReceipt(r, rows); err != nil {
		t.Fatalf("the untampered receipt stopped verifying: %v", err)
	}
}

// TestReceiptRenderLabelsEveryField proves the human view carries a provenance label
// on every line and a verdict line, so an operator reading a receipt sees which
// numbers are kernel-proven and which are self-reported.
func TestReceiptRenderLabelsEveryField(t *testing.T) {
	rows := buildTraceJournal(t, "trace-a")
	r := BuildReceipt("trace-a", rows, ObservedUsage{Turns: 3})
	var buf bytes.Buffer
	RenderReceipt(&buf, r, VerifyReceipt(r, rows))
	out := buf.String()
	if !strings.Contains(out, string(cachewitness.Witnessed)) || !strings.Contains(out, string(cachewitness.Observed)) {
		t.Fatalf("render missing a provenance label:\n%s", out)
	}
	if !strings.Contains(out, "verify: OK") {
		t.Fatalf("render missing the verify verdict:\n%s", out)
	}
}

func deepCopyReceipt(r Receipt) Receipt {
	fields := make([]ReceiptField, len(r.Fields))
	copy(fields, r.Fields)
	r.Fields = fields
	return r
}

func setField(t *testing.T, r *Receipt, name, value string) {
	t.Helper()
	for i := range r.Fields {
		if r.Fields[i].Name == name {
			r.Fields[i].Value = value
			return
		}
	}
	t.Fatalf("field %q not present in receipt", name)
}
