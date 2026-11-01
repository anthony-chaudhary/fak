package dogfoodissues

import (
	"encoding/json"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
)

// These tests exercise the production sync loop with an injected Runner.
// They never invoke gh or use the network; exported-wrapper and CLI integration
// require separate coverage.
const hostileDiagnostic = "SYNTHETIC_UNKNOWN_SECRET_72 ghp_FAKE_NOT_A_CREDENTIAL\n::warning::forged\r\x1b]8;;https://example.invalid/secret\aLINK\x1b]8;;\a\x00\b\u202e\xff"

func typedPlan(action string) PlanRow {
	n := 42
	r := PlanRow{Action: action, Key: "fixture/key", Title: "fixture", Body: "<!-- fak-dogfood-action-key: fixture/key -->\n\nProcess cause: none\n", Milestone: DefaultMilestone}
	if action == "update" {
		r.Number = &n
	}
	return r
}

func typedView(number int, key, milestone string) string {
	b, err := json.Marshal(Issue{Number: number, Body: "<!-- fak-dogfood-action-key: " + key + " -->", URL: "https://example.invalid/fixture/issues/42", Milestone: &IssueMilestone{Title: milestone}})
	if err != nil {
		panic(err)
	}
	return string(b)
}

// fak-test:runtime fast est=1s lane=default
func TestTypedProducerBranches(t *testing.T) {
	cases := []struct {
		name   string
		action string
		stage  SyncFailureStage
		reason SyncFailureReason
		calls  []string
	}{
		{"missing_number", "update", SyncStagePreflight, SyncReasonMissingIssueNumber, nil},
		{"policy_rejected", "create", SyncStagePrepare, SyncReasonPolicyRejected, nil},
		{"write_failed", "create", SyncStageWrite, SyncReasonCommandFailed, []string{"create"}},
		{"create_url_missing", "create", SyncStageWriteResponse, SyncReasonCreatedIssueURLMissing, []string{"create"}},
		{"readback_failed", "update", SyncStageReadback, SyncReasonCommandFailed, []string{"edit", "view"}},
		{"readback_json", "update", SyncStageReadback, SyncReasonInvalidJSON, []string{"edit", "view"}},
		{"readback_number", "update", SyncStageReadback, SyncReasonNumberMismatch, []string{"edit", "view"}},
		{"readback_marker", "update", SyncStageReadback, SyncReasonMarkerMismatch, []string{"edit", "view"}},
		{"readback_milestone", "update", SyncStageReadback, SyncReasonMilestoneMismatch, []string{"edit", "view"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan := typedPlan(tc.action)
			if tc.name == "missing_number" {
				plan.Number = nil
			}
			if tc.name == "policy_rejected" {
				plan.Body = "Process cause: invalid-fixture-cause\n"
			}
			var calls []string
			runner := func(args []string) (string, string, bool) {
				if len(args) < 2 || args[0] != "issue" {
					t.Errorf("unexpected fixture operation: %q", args)
					return "", "unexpected", false
				}
				op := args[1]
				calls = append(calls, op)
				if op != "view" {
					if op != "create" && op != "edit" {
						t.Errorf("unexpected write: %q", args)
						return "", "unexpected", false
					}
					if tc.name == "write_failed" {
						return "", hostileDiagnostic, false
					}
					if tc.name == "create_url_missing" {
						return "created but no URL", hostileDiagnostic, true
					}
					return "https://example.invalid/fixture/issues/42", "", true
				}
				switch tc.name {
				case "readback_failed":
					return "", hostileDiagnostic, false
				case "readback_json":
					return "{invalid", hostileDiagnostic, true
				case "readback_number":
					return typedView(43, plan.Key, plan.Milestone), "", true
				case "readback_marker":
					return typedView(42, "different-marker", plan.Milestone), "", true
				case "readback_milestone":
					return typedView(42, plan.Key, "wrong-milestone"), "", true
				default:
					t.Errorf("unexpected readback in %s", tc.name)
					return "", "unexpected", false
				}
			}
			rows := syncWithRunner([]PlanRow{plan}, "fixture/repo", nil, runner, SyncOptions{Timeout: time.Second})
			if len(rows) != 1 {
				t.Fatalf("rows=%d", len(rows))
			}
			row := rows[0]
			if row.OK || row.Verified || row.FailureStage != tc.stage || row.FailureReason != tc.reason {
				t.Fatalf("producer category: got OK=%t verified=%t %q/%q, want %q/%q", row.OK, row.Verified, row.FailureStage, row.FailureReason, tc.stage, tc.reason)
			}
			if !reflect.DeepEqual(calls, tc.calls) {
				t.Fatalf("operation trace=%v, want %v", calls, tc.calls)
			}
			if tc.stage == SyncStageReadback && (row.Number == nil || *row.Number != 42) {
				t.Fatal("readback failure lost known issue identity")
			}
			result := Result{Schema: Schema, Mode: "live", Report: "fixture/report.json", Planned: []PlanRow{plan}, Synced: rows}
			text := Render(result)
			want := "stage=" + string(tc.stage) + " reason=" + string(tc.reason)
			if !strings.Contains(text, want) {
				t.Fatalf("typed reason absent: want %q in %q", want, text)
			}
			if strings.Contains(text, "SYNTHETIC_UNKNOWN_SECRET_72") || strings.Contains(text, "ghp_FAKE") || strings.Contains(text, "\x1b") {
				t.Fatal("untrusted diagnostic leaked into default text")
			}
			if (tc.name == "write_failed" || tc.name == "readback_failed") && !strings.Contains(row.Stderr, "SYNTHETIC_UNKNOWN_SECRET_72") {
				t.Fatal("legacy raw JSON input was silently removed")
			}
			raw, err := json.Marshal(row)
			if err != nil || !strings.Contains(string(raw), `"failure_stage":"`+string(tc.stage)+`"`) || !strings.Contains(string(raw), `"failure_reason":"`+string(tc.reason)+`"`) {
				t.Fatalf("additive failure JSON missing: %v %s", err, raw)
			}
			now := time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC)
			receipt := BuildReceipt(result, ReceiptModeLive, now)
			if receipt.SyncedOK != 0 || receipt.SyncedFailed != 1 {
				t.Fatalf("failed receipt counters changed: %+v", receipt)
			}
			result.Synced[0].FailureStage, result.Synced[0].FailureReason = "", ""
			if !reflect.DeepEqual(receipt, BuildReceipt(result, ReceiptModeLive, now)) {
				t.Fatal("new diagnosis changed consultation receipt")
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestTypedProducerSuccessAndPartialProgress(t *testing.T) {
	for _, action := range []string{"create", "update"} {
		t.Run(action, func(t *testing.T) {
			plan := typedPlan(action)
			var calls []string
			runner := func(args []string) (string, string, bool) {
				calls = append(calls, args[1])
				if args[1] == "view" {
					return typedView(42, plan.Key, plan.Milestone), "", true
				}
				return "https://example.invalid/fixture/issues/42", "", true
			}
			// Zero timeout exercises the unchanged conservative default branch;
			// the injected runner returns immediately and never starts gh.
			rows := syncWithRunner([]PlanRow{plan}, "", nil, runner, SyncOptions{})
			if len(rows) != 1 || !rows[0].OK || !rows[0].Verified || rows[0].Number == nil || *rows[0].Number != 42 || rows[0].FailureStage != "" || rows[0].FailureReason != "" {
				t.Fatalf("success semantics: %+v", rows)
			}
			wantOp := "create"
			if action == "update" {
				wantOp = "edit"
			}
			if !reflect.DeepEqual(calls, []string{wantOp, "view"}) {
				t.Fatalf("unexpected extra calls: %v", calls)
			}
			b, _ := json.Marshal(rows[0])
			if strings.Contains(string(b), "failure_") {
				t.Fatal("success JSON gained nonempty diagnostic fields")
			}
			// Golden follows the unchanged current-3319 SyncRow field order and
			// the exact fixture outputs; this rejects any added success keys.
			wantJSON := `{"key":"fixture/key","action":"` + action + `","ok":true,"number":42,"url":"https://example.invalid/fixture/issues/42","verified":true,"stdout":"https://example.invalid/fixture/issues/42","stderr":""}`
			if string(b) != wantJSON {
				t.Fatalf("success JSON bytes changed: got %s want %s", b, wantJSON)
			}
		})
	}
	first, second := typedPlan("update"), typedPlan("update")
	second.Key = "fixture/second"
	second.Body = "<!-- fak-dogfood-action-key: fixture/second -->"
	writes := 0
	rows := syncWithRunner([]PlanRow{first, second}, "", nil, func(args []string) (string, string, bool) {
		if args[1] == "edit" {
			writes++
			if writes == 1 {
				return "", hostileDiagnostic, false
			}
			return "", "", true
		}
		return typedView(42, second.Key, second.Milestone), "", true
	}, SyncOptions{Timeout: time.Second})
	if len(rows) != 2 || rows[0].OK || rows[0].FailureStage != SyncStageWrite || !rows[1].OK || !rows[1].Verified || writes != 2 {
		t.Fatalf("partial progress/continuation changed: %+v writes=%d", rows, writes)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestTypedRenderLegacyUnknownAndHostileFields(t *testing.T) {
	cases := []struct {
		name   string
		stage  SyncFailureStage
		reason SyncFailureReason
	}{
		{"legacy", "", ""},
		{"unknown_stage", SyncFailureStage(hostileDiagnostic), SyncReasonMarkerMismatch},
		{"unknown_reason", SyncStageReadback, SyncFailureReason(hostileDiagnostic)},
		{"invalid_pair", SyncStageWrite, SyncReasonMarkerMismatch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, stderr := range []string{"", " \t\r\n", hostileDiagnostic, strings.Repeat("private-fixture-", 8192)} {
				negative := -1
				row := SyncRow{Action: hostileDiagnostic, Key: hostileDiagnostic, URL: hostileDiagnostic, Stdout: hostileDiagnostic, Stderr: stderr, Number: &negative, FailureStage: tc.stage, FailureReason: tc.reason}
				text := strings.Join(syncedFailureLines([]SyncRow{row}), "\n")
				if !strings.Contains(text, "action=unknown issue=unknown stage=unknown reason=unspecified") {
					t.Fatalf("legacy fallback missing: %q", text)
				}
				if strings.Contains(text, "SYNTHETIC_UNKNOWN_SECRET_72") || strings.Contains(text, "private-fixture-") {
					t.Fatal("raw field leaked")
				}
				for _, b := range []byte(text) {
					if (b < 32 && b != '\n') || b > 126 {
						t.Fatalf("non-printable added byte %d", b)
					}
				}
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestTypedRenderOutputBoundsAndOrdinal(t *testing.T) {
	rows := make([]SyncRow, 1001)
	maxInt := int(^uint(0) >> 1)
	for i := range rows {
		rows[i] = SyncRow{Number: &maxInt, Action: "create", Stderr: hostileDiagnostic, FailureStage: SyncStageWriteResponse, FailureReason: SyncReasonCreatedIssueURLMissing}
	}
	rows[0].OK = true
	lines := syncedFailureLines(rows)
	text := strings.Join(lines, "\n")
	if len(text) > 4096 {
		t.Fatalf("added section=%d bytes >4096", len(text))
	}
	if strings.Count(text, "    row=") != 20 || !strings.Contains(text, "sync failed: 1000 row(s)") || !strings.Contains(text, "980 more failed row(s) omitted") {
		t.Fatalf("caps/counts wrong: %q", text)
	}
	if !strings.Contains(text, "row=2 action=create issue=#"+strconv.Itoa(maxInt)) {
		t.Fatal("ordinal or bounded numeric target lost")
	}
	if strings.Contains(text, "SYNTHETIC_UNKNOWN_SECRET_72") {
		t.Fatal("diagnostic leaked")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestTypedRenderSuccessRefusalAndEmptyPlanCompatibility(t *testing.T) {
	base := "dogfood-action-issues: live  1 item(s)\n  report: fixture/report.json\n  [update] #42: fixture (grade= debt=0)"
	plan := typedPlan("update")
	res := Result{Mode: "live", Report: "fixture/report.json", Planned: []PlanRow{plan}, Synced: []SyncRow{{OK: true, Verified: true, FailureStage: SyncStageWrite, FailureReason: SyncReasonCommandFailed}}}
	if got := Render(res); got != base {
		t.Fatalf("success output bytes changed: %q", got)
	}
	res.Synced[0].Verified = false
	if Render(res) != base {
		t.Fatal("OK=true legacy semantics changed or stale codes rendered")
	}
	res.Refused = true
	res.Error = "strict_scope"
	res.Synced[0].OK = false
	want := "dogfood-action-issues: live  1 item(s)\n  report: fixture/report.json\n  refused: strict_scope"
	if got := Render(res); got != want {
		t.Fatalf("strict refusal output changed: %q", got)
	}
	res.Refused = false
	res.Planned = nil
	res.Receipt = "fixture/issues-sync.json"
	text := Render(res)
	if !strings.Contains(text, "stage=write reason=command_failed") || !strings.Contains(text, "no dispatchable scorecard ACTION items found") || !strings.Contains(text, "receipt: fixture/issues-sync.json") {
		t.Fatal("zero-plan early return lost evidence or existing receipt path")
	}
	res.Mode = "dry-run"
	res.Planned = []PlanRow{plan}
	res.Synced = nil
	res.Receipt = ""
	want = "dogfood-action-issues: dry-run  1 item(s)\n  report: fixture/report.json\n  [update] #42: fixture (grade= debt=0)\n  dry-run: pass --live to create/update issues with gh"
	if got := Render(res); got != want {
		t.Fatalf("dry-run bytes changed: %q", got)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestTypedRenderContradictoryFlagsStayFailed(t *testing.T) {
	text := strings.Join(syncedFailureLines([]SyncRow{{OK: false, Verified: true, FailureStage: SyncStageReadback, FailureReason: SyncReasonMarkerMismatch}}), "\n")
	if !strings.Contains(text, "sync failed: 1 row(s)") || !strings.Contains(text, "verified=true") {
		t.Fatal("contradictory supplied flags were hidden/coerced")
	}
}
