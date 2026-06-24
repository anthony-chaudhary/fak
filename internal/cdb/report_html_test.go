package cdb

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// TestHTMLReportLoadsACoreImageAndListsPages — the #574 smoke test. An
// attached image renders to a self-contained HTML document that:
//   - lists the page-table rows (one per tool result), so the report is a real
//     "inspection UI", not a stub;
//   - surfaces the sealed pages with their reason code in a dedicated
//     quarantine panel — the headline panel of the product;
//   - carries the honest SealPanelNote next to the seals (detector evadable;
//     durable, not improved);
//   - reports the working-set residency for the follow-up query (the demand-
//     paging proof);
//   - refuses sealed-page bytes by construction (no Examine paged them in to
//     render the report — so the report cannot echo poison).
//
// It exercises the full attached-image path (ingest → persist → reload →
// HTMLReport) on the committed real-shaped fixture, so the assertions hold on
// bytes that came off disk.
func TestHTMLReportLoadsACoreImageAndListsPages(t *testing.T) {
	ctx := context.Background()
	im := attachFixture(t)
	query := "what refund fee did the user's account show?"

	var sb strings.Builder
	if err := im.HTMLReport(ctx, query, "cdb-image", "testdata/cdb/session.jsonl", &sb); err != nil {
		t.Fatalf("HTMLReport: %v", err)
	}
	html := sb.String()
	info := im.Info()

	if !strings.Contains(html, "<!DOCTYPE html>") {
		t.Error("report is not a standalone HTML document (missing DOCTYPE)")
	}
	if !strings.Contains(html, "</html>") {
		t.Error("report is not a complete HTML document (missing </html>)")
	}
	// session id present in the header
	if !strings.Contains(html, info.SessionID) {
		t.Errorf("report does not surface the session id %q", info.SessionID)
	}
	// the page table lists every page: assert each benign/ sealed row's role
	// appears in the backtrace section.
	frames := im.Backtrace()
	for _, f := range frames {
		// the role is rendered inside the page table; the descriptor may be
		// html-escaped by the template, but the role strings here are plain.
		if !strings.Contains(html, f.Role) {
			t.Errorf("page table does not list step %d role %q", f.Step, f.Role)
		}
	}
	// the quarantine panel surfaces every sealed page with its reason code
	for _, f := range frames {
		if !f.Sealed {
			continue
		}
		if f.Reason == "" {
			t.Errorf("sealed step %d has empty reason code in the page table", f.Step)
		}
		if !strings.Contains(html, f.Reason) {
			t.Errorf("quarantine panel does not surface sealed step %d reason %q",
				f.Step, f.Reason)
		}
		if !strings.Contains(html, f.QID) {
			t.Errorf("quarantine panel does not surface sealed step %d qid %q",
				f.Step, f.QID)
		}
	}
	// the honest seal note is carried verbatim next to the seals
	if !strings.Contains(html, SealPanelNote) {
		t.Error("the honest SealPanelNote (detector evadable; durable, not improved) is missing from the report")
	}
	// the working-set residency panel reports the demand-paged number
	ws := im.WorkingSet(ctx, query, 0)
	if ws.PagesTouched == 0 {
		t.Fatal("working set is empty; residency panel cannot be asserted")
	}
	residency := fmt.Sprintf("%.2f", ws.ResidencyPct)
	if !strings.Contains(html, residency) {
		t.Errorf("residency panel does not report %.2f%% (got substring miss for %q)",
			ws.ResidencyPct, residency)
	}
}

// TestHTMLReportRefusesSealedBytes — the load-bearing trust property: a sealed
// page's bytes are NEVER paged in to render the report. We assert it by
// constructing a transcript with a known poison payload, attaching it, and
// confirming the report contains the safe sealed descriptor but NOT the poison
// bytes — the same way Backtrace cannot echo them.
func TestHTMLReportRefusesSealedBytes(t *testing.T) {
	ctx := context.Background()
	const poison = "EXFILTRATE_THIS_SECRET_PAYLOAD_AKIAIOSFODNN7EXAMPLE"
	transcript := `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"f":"x"}}]}}` + "\n" +
		`{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"` + poison + `"}]}}`

	rec, st, err := ingest(ctx, strings.NewReader(transcript), "poison-html-test")
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if st.Sealed == 0 {
		t.Fatalf("poison payload was not sealed at ingest (pages=%d sealed=%d)", st.Pages, st.Sealed)
	}
	dir := t.TempDir()
	if err := rec.Persist(dir); err != nil {
		t.Fatalf("persist: %v", err)
	}
	im, err := Attach(dir)
	if err != nil {
		t.Fatalf("attach: %v", err)
	}

	var sb strings.Builder
	if err := im.HTMLReport(ctx, "anything", dir, "poison-test", &sb); err != nil {
		t.Fatalf("HTMLReport: %v", err)
	}
	html := sb.String()
	if strings.Contains(html, poison) {
		t.Fatalf("HTML report leaked sealed-page bytes — the gate must refuse page-in on render")
	}
	// the report DOES surface the sealed-page ROW (with its reason code), so the
	// operator can see WHAT got sealed without seeing the poison itself.
	if !strings.Contains(html, "sealed") {
		t.Errorf("report does not surface the sealed-page row in its quarantine panel")
	}
}
