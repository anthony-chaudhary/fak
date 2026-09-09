package abi

import (
	"strings"
	"testing"
)

func assertPanicContains(t *testing.T, desc string, substr string, f func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("%s: expected panic containing %q, but did not panic", desc, substr)
			return
		}
		msg, ok := r.(string)
		if !ok {
			t.Errorf("%s: panic payload is not string: %v", desc, r)
			return
		}
		if !strings.Contains(msg, substr) {
			t.Errorf("%s: panic message %q does not contain %q", desc, msg, substr)
		}
	}()
	f()
}

func TestRegisterReasonClashDiscipline(t *testing.T) {
	ResetForTest()
	defer ResetForTest()

	// 1. Core-range claims must panic.
	assertPanicContains(t, "claim ReasonNone (0)", "closed core range", func() {
		RegisterReason(ReasonNone, "ZERO_CODE")
	})
	assertPanicContains(t, "claim ReasonDefaultDeny (1)", "closed core range", func() {
		RegisterReason(ReasonDefaultDeny, "DEFAULT_DENY_CUSTOM")
	})
	assertPanicContains(t, "claim ReasonCoreMax (1023)", "closed core range", func() {
		RegisterReason(ReasonCoreMax, "BOUNDARY_CORE")
	})

	// 2. Valid first registration beyond ReasonCoreMax.
	const codeA ReasonCode = 1050
	const nameA = "TEST_REASON_ALPHA"
	RegisterReason(codeA, nameA)

	if got := ReasonName(codeA); got != nameA {
		t.Fatalf("ReasonName(%d) = %q, want %q", codeA, got, nameA)
	}
	if gotCode, ok := ReasonByName(nameA); !ok || gotCode != codeA {
		t.Fatalf("ReasonByName(%q) = (%d, %v), want (%d, true)", nameA, gotCode, ok, codeA)
	}

	// 3. Exact (code, name) re-registration is an idempotent no-op (does not panic).
	RegisterReason(codeA, nameA)

	// 4. Code clash: same code under a different name must panic.
	assertPanicContains(t, "same code different name", "duplicate ReasonCode 1050", func() {
		RegisterReason(codeA, "TEST_REASON_BRAVO")
	})

	// 5. Name clash: core reason name under an open-range code must panic.
	assertPanicContains(t, "core name collision", `duplicate ReasonName "POLICY_BLOCK"`, func() {
		RegisterReason(1051, "POLICY_BLOCK")
	})

	// 6. Name clash: registered reason name under a different code must panic.
	const codeB ReasonCode = 1052
	assertPanicContains(t, "registered name collision", `duplicate ReasonName "TEST_REASON_ALPHA"`, func() {
		RegisterReason(codeB, nameA)
	})

	// 7. Distinct valid registration succeeds.
	const codeC ReasonCode = 1053
	const nameC = "TEST_REASON_CHARLIE"
	RegisterReason(codeC, nameC)

	if got := ReasonName(codeC); got != nameC {
		t.Fatalf("ReasonName(%d) = %q, want %q", codeC, got, nameC)
	}
	if gotCode, ok := ReasonByName(nameC); !ok || gotCode != codeC {
		t.Fatalf("ReasonByName(%q) = (%d, %v), want (%d, true)", nameC, gotCode, ok, codeC)
	}
}
