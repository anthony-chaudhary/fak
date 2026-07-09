package linkstate

import (
	"strings"
	"testing"
	"time"
)

func TestPhaseContractAndAdmission(t *testing.T) {
	checkedAt := time.Date(2026, 7, 4, 14, 0, 0, 0, time.UTC)

	clear := New("gpu-server", Clear, "", "", "", checkedAt)
	if clear.Schema != Schema {
		t.Fatalf("schema = %q, want %q", clear.Schema, Schema)
	}
	if clear.CheckedAt != "2026-07-04T14:00:00Z" {
		t.Fatalf("checked_at = %q", clear.CheckedAt)
	}
	if !clear.AdmitDispatch {
		t.Fatal("CLEAR must admit dispatch")
	}
	if clear.Detail != DetailReady || clear.NextAction != "admit-dispatch" {
		t.Fatalf("CLEAR defaults wrong: detail=%q next=%q", clear.Detail, clear.NextAction)
	}
	if probs := clear.Validate(); len(probs) != 0 {
		t.Fatalf("clear record should validate: %v", probs)
	}

	// Every non-CLEAR phase fails closed and still names a next step.
	for _, tc := range []struct {
		phase    Phase
		detail   string
		wantNext string
	}{
		{Working, DetailJobInFlight, "await-completion"},
		{Waiting, DetailGatewayDown, "recover-gateway"},
		{Waiting, DetailAuthBlocked, "fix-auth-or-channel"},
		{Waiting, DetailPrivateRecovery, "confirm-control-session"},
		{Waiting, DetailIndeterminate, "publish-link-state"},
	} {
		s := New("gpu-server", tc.phase, tc.detail, "", "", checkedAt)
		if s.AdmitDispatch {
			t.Fatalf("phase %s must fail closed", tc.phase)
		}
		if s.NextAction != tc.wantNext {
			t.Fatalf("phase %s detail %s: next_action = %q, want %q", tc.phase, tc.detail, s.NextAction, tc.wantNext)
		}
		if s.NextAction == "" {
			t.Fatalf("phase %s must always carry a next_action (never wedged)", tc.phase)
		}
		if probs := s.Validate(); len(probs) != 0 {
			t.Fatalf("record %s/%s should validate: %v", tc.phase, tc.detail, probs)
		}
	}
}

func TestValidateRejectsBadRecords(t *testing.T) {
	bad := State{
		Schema:     Schema,
		Subject:    "gpu.server", // a dotted host — not tokenish
		CheckedAt:  "2026-07-04T14:00:00Z",
		Phase:      "PRIVATE_OK", // not in the closed phase vocabulary
		Detail:     DetailReady,
		NextAction: "see:C123", // colon — not tokenish
		Evidence:   "scrubbed-readback",
	}
	probs := strings.Join(bad.Validate(), " | ")
	for _, want := range []string{"closed link-state vocabulary", "subject", "next_action"} {
		if !strings.Contains(probs, want) {
			t.Fatalf("expected %q in validation problems, got %q", want, probs)
		}
	}

	// Phase/detail inconsistency is refused: CLEAR may only carry the ready detail.
	incon := New("gpu-server", Clear, DetailReady, "admit-dispatch", "scrubbed-readback", time.Now())
	incon.Detail = DetailGatewayDown
	probs = strings.Join(incon.Validate(), " | ")
	if !strings.Contains(probs, "inconsistent with phase") {
		t.Fatalf("expected phase/detail inconsistency to be flagged, got %q", probs)
	}
}

func TestLoadForcesAdmissionBitAndScrubs(t *testing.T) {
	doc := `{
		"schema":"fak.link_state/v1",
		"subject":"gpu-server",
		"checked_at":"2026-07-04T14:00:00Z",
		"phase":"CLEAR",
		"detail":"ready",
		"next_action":"admit-dispatch",
		"evidence":"scrubbed-readback",
		"admit_dispatch":false
	}`
	got, err := Load(strings.NewReader(doc))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !got.AdmitDispatch {
		t.Fatal("loader must derive admit_dispatch from phase, not trust the file's false")
	}

	// A private/raw field is refused (DisallowUnknownFields).
	if _, err := Load(strings.NewReader(`{"schema":"fak.link_state/v1","subject":"gpu-server","phase":"CLEAR","detail":"ready","next_action":"admit-dispatch","evidence":"scrubbed-readback","raw_thread":"secret"}`)); err == nil {
		t.Fatal("unknown raw/private fields must be refused")
	}
}

func TestCoarsenFoldsLegacyStatuses(t *testing.T) {
	for _, tc := range []struct {
		status     string
		wantPhase  Phase
		wantDetail string
	}{
		{"READY_FOR_DEV_WORK", Clear, DetailReady},
		{"WAIT_PRIVATE_RECOVERY", Waiting, DetailPrivateRecovery},
		{"GATEWAY_UNREACHABLE", Waiting, DetailGatewayDown},
		{"AUTH_OR_CHANNEL_BLOCKED", Waiting, DetailAuthBlocked},
		{"INDETERMINATE", Waiting, DetailIndeterminate},
		{"SOMETHING_NEW_WE_NEVER_SAW", Waiting, DetailIndeterminate}, // unknown fails safe
	} {
		phase, detail := Coarsen(tc.status)
		if phase != tc.wantPhase || detail != tc.wantDetail {
			t.Fatalf("Coarsen(%q) = (%s,%s), want (%s,%s)", tc.status, phase, detail, tc.wantPhase, tc.wantDetail)
		}
		if phase == Clear && tc.status != "READY_FOR_DEV_WORK" {
			t.Fatalf("only the ready status may coarsen to CLEAR, got %q", tc.status)
		}
	}
}
