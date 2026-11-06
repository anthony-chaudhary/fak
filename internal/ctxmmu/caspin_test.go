package ctxmmu_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// Keep churn distinct across repeated invocations in the same test binary.
var casPinChurnSequence atomic.Uint64

// TestQuarantinePageInSurvivesCASEviction proves the held-quarantine CAS pin
// end-to-end: a sealed result paged out to the BOUNDED registered CAS still pages back
// in (after a witness Clear) even after CAS churn that evicts unpinned blobs. The
// gated page-in is not broken by the byte bound — the soundness property the pin
// exists to protect.
func TestQuarantinePageInSurvivesCASEviction(t *testing.T) {
	ctx := context.Background()
	if abi.ActiveResolver() != testCAS {
		t.Fatal("CAS pin witness must use the suite's registered resolver")
	}
	old := testCAS.MaxBytes()
	budget := testCAS.Bytes() + 8192 // allow prior held pins plus this test's payload
	testCAS.SetMaxBytes(budget)
	defer testCAS.SetMaxBytes(old)

	m := ctxmmu.New()

	// A >256B secret-shaped body: quarantines AND pages out to the CAS (not inline).
	secret := []byte("sk-" + strings.Repeat("a", 400))
	c := call("read_secret")
	r := result(c, secret)
	if v := m.Admit(ctx, c, r); v.Kind != abi.VerdictQuarantine {
		t.Fatalf("want Quarantine, got %v", v.Kind)
	}
	id := r.Meta["quarantine_id"]
	if id == "" {
		t.Fatalf("no quarantine_id stamped")
	}
	handle := m.Held()[id]
	if handle.Digest == "" {
		t.Fatal("quarantine handle carries no CAS digest")
	}

	// Control: an UNPINNED CAS blob put now should be evicted by the churn below.
	ctrl, err := testCAS.Put(ctx, bytes.Repeat([]byte{0xAB}, 1000))
	if err != nil {
		t.Fatalf("put control: %v", err)
	}
	churnID := casPinChurnSequence.Add(1)
	for i, n := 0, int(budget/1024)+8; i < n; i++ {
		body := bytes.Repeat([]byte{0xAB}, 1024)
		copy(body, fmt.Sprintf("caspin-quarantine-%d-%d", churnID, i))
		if _, err := testCAS.Put(ctx, body); err != nil {
			t.Fatalf("churn put: %v", err)
		}
	}
	if _, err := testCAS.Resolve(ctx, ctrl); err == nil {
		t.Fatalf("control (unpinned) blob should have been evicted by the bound")
	}
	// Verify the actual CAS bytes, so staging fallback cannot mask a lost pin.
	if got, err := testCAS.Resolve(ctx, handle); err != nil || !bytes.Equal(got, secret) {
		t.Fatalf("held quarantine CAS bytes = (%q, %v), want original secret", got, err)
	}

	// The quarantine bytes are PINNED, so the gated page-in still resolves them.
	m.Clear(id)
	got, err := m.PageIn(ctx, id)
	if err != nil {
		t.Fatalf("pinned quarantine page-in failed under CAS eviction (soundness break): %v", err)
	}
	if !bytes.Equal(got, secret) {
		t.Fatalf("page-in returned wrong bytes")
	}
}

func TestPagedResultSurvivesCASEviction(t *testing.T) {
	ctx := context.Background()
	if abi.ActiveResolver() != testCAS {
		t.Fatal("CAS pin witness must use the suite's registered resolver")
	}
	old := testCAS.MaxBytes()
	budget := testCAS.Bytes() + 8192 // allow prior held pins plus this test's payload
	testCAS.SetMaxBytes(budget)
	defer testCAS.SetMaxBytes(old)

	m := ctxmmu.New()

	// Create an oversize benign result (> 4096 B) and pass to m.Admit(ctx, c, r).
	originalBytes := bytes.Repeat([]byte("benign text content for caspin test. "), 125) // ~4625 bytes (> 4096)
	c := call("fetch_data")
	r := result(c, originalBytes)

	v := m.Admit(ctx, c, r)
	if v.Kind != abi.VerdictTransform {
		t.Fatalf("want VerdictTransform, got %v", v.Kind)
	}

	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("payload not TransformPayload: %T", v.Payload)
	}
	stubBytes := resolveBody(t, ctx, tp.NewArgs)
	var stub map[string]any
	if err := json.Unmarshal(stubBytes, &stub); err != nil {
		t.Fatalf("unmarshal stub: %v, raw=%s", err, stubBytes)
	}
	if paged, _ := stub["_paged"].(bool); !paged {
		t.Fatalf("expected _paged=true in %+v", stub)
	}
	ref, _ := stub["ref"].(string)
	if ref == "" {
		t.Fatalf("expected non-empty ref in %+v", stub)
	}
	if rTool, _ := stub["retrieval_tool"].(string); rTool != "fak_context_restore" {
		t.Fatalf("retrieval_tool = %q, want fak_context_restore", rTool)
	}
	if sizeVal, ok := stub["size"].(float64); !ok || int(sizeVal) != len(originalBytes) {
		t.Fatalf("stub size = %v, want %d", stub["size"], len(originalBytes))
	}

	// Churn the registered CAS past its whole budget with distinct bodies.
	ctrl, err := testCAS.Put(ctx, bytes.Repeat([]byte{0xCD}, 1000))
	if err != nil {
		t.Fatalf("put control: %v", err)
	}
	churnID := casPinChurnSequence.Add(1)
	for i, n := 0, int(budget/1024)+8; i < n; i++ {
		body := bytes.Repeat([]byte{0xCD}, 1024)
		copy(body, fmt.Sprintf("caspin-paged-%d-%d", churnID, i))
		if _, err := testCAS.Put(ctx, body); err != nil {
			t.Fatalf("churn put: %v", err)
		}
	}
	// Assert unpinned control blob is evicted.
	if _, err := testCAS.Resolve(ctx, ctrl); err == nil {
		t.Fatalf("control (unpinned) blob should have been evicted by the bound")
	}
	handle := m.Held()[ref]
	if handle.Digest == "" {
		t.Fatal("paged result carries no held CAS digest")
	}
	if got, err := testCAS.Resolve(ctx, handle); err != nil || !bytes.Equal(got, originalBytes) {
		t.Fatalf("held paged CAS bytes = (%q, %v), want original payload", got, err)
	}

	// Assert ctxmmu.ResolvePaged(ctx, ref) and m.ResolvePagedRef(ctx, ref) successfully resolve the exact original bytes.
	resolved1, ok := ctxmmu.ResolvePaged(ctx, ref)
	if !ok {
		t.Fatalf("ctxmmu.ResolvePaged(%q) failed", ref)
	}
	if !bytes.Equal(resolved1, originalBytes) {
		t.Fatalf("ctxmmu.ResolvePaged returned wrong bytes: got %d bytes, want %d bytes", len(resolved1), len(originalBytes))
	}

	resolved2, ok := m.ResolvePagedRef(ctx, ref)
	if !ok {
		t.Fatalf("m.ResolvePagedRef(%q) failed", ref)
	}
	if !bytes.Equal(resolved2, originalBytes) {
		t.Fatalf("m.ResolvePagedRef returned wrong bytes: got %d bytes, want %d bytes", len(resolved2), len(originalBytes))
	}
}
