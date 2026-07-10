package ctxmmu_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// memwrite_test.go is the contract suite for the memory-write adjudicator
// (issue #2874): every refusal must cite a reason from the abi closed
// vocabulary, carry a witness naming the measured structure, and never echo
// secret bytes. Secret and oversize are asserted as INDEPENDENT checks (each
// trips alone), with the secret verdict taking precedence when both hold.

// wordsFact returns a distilled-fact-shaped body of n distinct words, so the
// dedup assertions control their shingle arithmetic exactly.
func wordsFact(n int) string {
	words := []string{
		"the", "trunk", "guard", "refuses", "any", "commit", "made", "off",
		"main", "branch", "so", "every", "worker", "must", "land", "its",
		"diff", "back", "through", "serialized", "worktree", "verb", "under",
		"a", "lane", "lease", "before", "another", "loop", "starts",
	}
	for len(words) < n {
		words = append(words, words[len(words)%30]+"x")
	}
	return strings.Join(words[:n], " ")
}

func TestMemWriteAdmitsCleanFact(t *testing.T) {
	a := ctxmmu.NewMemoryWriteAdjudicator()
	v := a.AdmitWrite([]byte("the release pipeline signs artifacts with the org key before publishing"))
	if !v.Admit || v.Reason != abi.ReasonNone {
		t.Fatalf("clean fact refused: %+v", v)
	}
	if v.Witness == "" {
		t.Fatalf("admit verdict must still carry a witness claim")
	}
}

func TestMemWriteRefusesOversizeVerbatim(t *testing.T) {
	a := ctxmmu.NewMemoryWriteAdjudicator()
	// Oversize but secret-free: only the size check may fire (independence).
	body := bytes.Repeat([]byte("distinct words flow onward here "), (ctxmmu.MemoryWriteMaxBytes/32)+2)
	if len(body) <= ctxmmu.MemoryWriteMaxBytes {
		t.Fatalf("fixture must exceed MemoryWriteMaxBytes")
	}
	v := a.AdmitWrite(body)
	if v.Admit || v.Reason != abi.ReasonOversize {
		t.Fatalf("oversize verbatim blob not refused OVERSIZE: %+v", v)
	}
	if !strings.Contains(v.Witness, "OVERSIZE") || !strings.Contains(v.Witness, "byte") {
		t.Fatalf("witness must name the closed-vocabulary reason and the measured bound: %q", v.Witness)
	}
}

func TestMemWriteRefusesSecretShapedAndRedactsWitness(t *testing.T) {
	a := ctxmmu.NewMemoryWriteAdjudicator()
	const leaked = "sk-abcdef0123456789abcdef0123"
	// Small body: the size check cannot be what fires (independence).
	v := a.AdmitWrite([]byte("prod api key is " + leaked))
	if v.Admit || v.Reason != abi.ReasonSecretExfil {
		t.Fatalf("secret-shaped write not refused SECRET_EXFIL: %+v", v)
	}
	if !strings.Contains(v.Witness, "SECRET_EXFIL") {
		t.Fatalf("witness must cite the closed-vocabulary reason: %q", v.Witness)
	}
	if strings.Contains(v.Witness, leaked) {
		t.Fatalf("witness echoed the secret bytes: %q", v.Witness)
	}
}

// TestMemWriteSecretPrecedesOversize pins the documented rung order: a body
// that is BOTH oversize and secret-bearing cites the security-relevant reason,
// mirroring MMU.Admit where ScreenBytes runs before the oversize branch.
func TestMemWriteSecretPrecedesOversize(t *testing.T) {
	a := ctxmmu.NewMemoryWriteAdjudicator()
	body := append(bytes.Repeat([]byte("distinct words flow onward here "), (ctxmmu.MemoryWriteMaxBytes/32)+2),
		[]byte("api_key=sk-abcdef0123456789abcdef0123")...)
	v := a.AdmitWrite(body)
	if v.Admit || v.Reason != abi.ReasonSecretExfil {
		t.Fatalf("oversize+secret body must cite SECRET_EXFIL, got %+v", v)
	}
}

func TestMemWriteRefusesNearDuplicate(t *testing.T) {
	a := ctxmmu.NewMemoryWriteAdjudicator()
	orig := wordsFact(30)
	a.Note("mem-001", []byte(orig))

	// Exact re-save: similarity 1.0.
	v := a.AdmitWrite([]byte(orig))
	if v.Admit || v.Reason != ctxmmu.ReasonMemoryNearDuplicate {
		t.Fatalf("exact re-save not refused: %+v", v)
	}
	if v.DuplicateOf != "mem-001" || !strings.Contains(v.Witness, "mem-001") {
		t.Fatalf("duplicate refusal must witness the matched entry id: %+v", v)
	}
	if !strings.Contains(v.Witness, ctxmmu.ReasonMemoryNearDuplicateName) {
		t.Fatalf("witness must cite the registered reason name: %q", v.Witness)
	}

	// One-word substitution mid-fact: 2 of ~29 bigrams change, similarity
	// ~0.87 — still at or above the 0.85 threshold, still a near-duplicate.
	edited := strings.Replace(orig, "serialized", "locked", 1)
	if edited == orig {
		t.Fatalf("fixture must actually differ")
	}
	if v := a.AdmitWrite([]byte(edited)); v.Admit || v.Reason != ctxmmu.ReasonMemoryNearDuplicate {
		t.Fatalf("one-word-edit re-save not refused as near-duplicate: %+v", v)
	}

	// A same-domain but genuinely different fact shares only isolated bigrams
	// and must admit — the strict-threshold under-refuse direction.
	fresh := "memory writes are adjudicated by structure before landing so an oversize or secret shaped entry is refused with a closed vocabulary reason"
	if v := a.AdmitWrite([]byte(fresh)); !v.Admit {
		t.Fatalf("distinct fact wrongly refused as duplicate: %+v", v)
	}
}

// TestMemWriteAdmitDoesNotNoteItself pins the Note contract: AdmitWrite never
// records its own candidate, so an admitted-but-unwritten body may be
// resubmitted without blocking itself.
func TestMemWriteAdmitDoesNotNoteItself(t *testing.T) {
	a := ctxmmu.NewMemoryWriteAdjudicator()
	body := []byte(wordsFact(30))
	if v := a.AdmitWrite(body); !v.Admit {
		t.Fatalf("first admit: %+v", v)
	}
	if v := a.AdmitWrite(body); !v.Admit {
		t.Fatalf("resubmit after unlanded admit must still admit: %+v", v)
	}
	a.Note("mem-002", body)
	if v := a.AdmitWrite(body); v.Admit {
		t.Fatalf("resubmit after Note must refuse: %+v", v)
	}
}

// TestMemWriteNotedLedgerIsBounded pins the FIFO bound: past the construction
// limit the oldest entry stops participating in dedup (cheap degradation).
func TestMemWriteNotedLedgerIsBounded(t *testing.T) {
	a := ctxmmu.NewMemoryWriteAdjudicatorWithLimit(1)
	first := []byte(wordsFact(30))
	second := []byte("memory writes are adjudicated by structure before landing so junk is refused")
	a.Note("old", first)
	a.Note("new", second) // evicts "old"
	if got := a.NotedLen(); got != 1 {
		t.Fatalf("noted ledger not bounded: len=%d", got)
	}
	if v := a.AdmitWrite(first); !v.Admit {
		t.Fatalf("evicted entry must stop blocking its duplicate: %+v", v)
	}
	if v := a.AdmitWrite(second); v.Admit {
		t.Fatalf("surviving entry must still block its duplicate: %+v", v)
	}
}

// TestMemWriteReasonRegistered pins the out-of-tree label-space entry: the
// near-duplicate code resolves to its stable name (never REASON_1060) and back.
func TestMemWriteReasonRegistered(t *testing.T) {
	if got := abi.ReasonName(ctxmmu.ReasonMemoryNearDuplicate); got != ctxmmu.ReasonMemoryNearDuplicateName {
		t.Fatalf("ReasonName = %q, want %q", got, ctxmmu.ReasonMemoryNearDuplicateName)
	}
	c, ok := abi.ReasonByName(ctxmmu.ReasonMemoryNearDuplicateName)
	if !ok || c != ctxmmu.ReasonMemoryNearDuplicate {
		t.Fatalf("ReasonByName = %v,%v", c, ok)
	}
}
