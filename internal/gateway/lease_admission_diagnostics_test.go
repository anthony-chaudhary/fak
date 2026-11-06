package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=1s
func TestLeaseAdmissionDiagnosticsDistinctFailures(t *testing.T) {
	type authoritySetup func(t *testing.T, root string)
	emptyPresence := func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	}
	validWorkspace := func(root string, leases ...WorkspaceAdmissionLease) {
		SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
			return WorkspaceLeaseAdmissionView{WorkspaceRoot: root, Leases: leases}, nil
		})
	}
	cases := []struct {
		name, tool    string
		args          map[string]any
		setup         authoritySetup
		wantReason    string
		wantFailure   string
		wantRecovery  string
		wantWorkspace bool
	}{
		{
			name: "unknown shell footprint", tool: "bash",
			args: map[string]any{"command": "python -c 'print(1)'"},
			setup: func(t *testing.T, root string) {
				SetLeasePlaneProviders(nil, emptyPresence)
				validWorkspace(root)
			},
			wantReason: "DEFAULT_DENY", wantFailure: "unknown_write_footprint", wantRecovery: "USE_FILE_TOOLS",
		},
		{
			name: "outside workspace", tool: "allow_write",
			setup: func(t *testing.T, root string) {
				SetLeasePlaneProviders(nil, emptyPresence)
				validWorkspace(root)
			},
			wantReason: "DEFAULT_DENY", wantFailure: "outside_trusted_workspace", wantRecovery: "CHECK_WORKSPACE_BINDING",
		},
		{
			name: "write target parent is unresolved", tool: "allow_write",
			setup: func(t *testing.T, root string) {
				SetLeasePlaneProviders(nil, emptyPresence)
				validWorkspace(root)
			},
			wantReason: "DEFAULT_DENY", wantFailure: "write_target_unresolvable", wantRecovery: "CHECK_WRITE_TARGET", wantWorkspace: true,
		},
		{
			name: "workspace authority unavailable", tool: "allow_write",
			setup: func(t *testing.T, root string) {
				SetLeasePlaneProviders(nil, emptyPresence)
				SetWorkspaceLeaseAdmissionProvider(nil)
			},
			wantReason: "DEFAULT_DENY", wantFailure: "workspace_authority_unavailable", wantRecovery: "RESTORE_LEASE_AUTHORITY",
		},
		{
			name: "lease authority read failed", tool: "allow_write",
			setup: func(t *testing.T, root string) {
				SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
					return LeasePresenceView{}, errors.New("unavailable")
				})
				validWorkspace(root)
			},
			wantReason: "DEFAULT_DENY", wantFailure: "lease_authority_read_failed", wantRecovery: "RESTORE_LEASE_AUTHORITY",
		},
		{
			name: "peer lease held", tool: "allow_write",
			setup: func(t *testing.T, root string) {
				SetLeasePlaneProviders(nil, emptyPresence)
				validWorkspace(root, WorkspaceAdmissionLease{TreeGlobs: []string{"write.txt"}, SessionID: "peer"})
			},
			wantReason: "LEASE_HELD", wantFailure: "peer_lease_held", wantRecovery: "RESPECT_PEER_LEASE",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			t.Cleanup(func() {
				SetLeasePlaneProviders(nil, nil)
				SetWorkspaceLeaseAdmissionProvider(nil)
			})
			tc.setup(t, root)
			args := tc.args
			if args == nil {
				args = map[string]any{"file_path": filepath.Join(root, "write.txt")}
			}
			if tc.name == "outside workspace" {
				args = map[string]any{"file_path": filepath.Join(t.TempDir(), "write.txt")}
			}
			if tc.name == "write target parent is unresolved" {
				args = map[string]any{"file_path": filepath.Join(root, "missing-parent", "probe.txt")}
			}
			encodedArgs, err := json.Marshal(args)
			if err != nil {
				t.Fatal(err)
			}
			request, err := json.Marshal(map[string]any{
				"tool": tc.tool, "arguments": json.RawMessage(encodedArgs), "trace_id": "diagnostic-boundary",
			})
			if err != nil {
				t.Fatal(err)
			}

			response, rpcErr := callFakAdjudicate(t, newPermissiveAdmissionTestServer(t, false), string(request))
			if rpcErr != nil {
				t.Fatalf("fak_adjudicate: %+v", rpcErr)
			}
			if response.Verdict.Kind != "DENY" || response.Verdict.Reason != tc.wantReason || response.Verdict.By != leaseAdmissionBy {
				t.Fatalf("verdict classification = %+v", response.Verdict)
			}
			if got := response.Verdict.Detail["admission_failure"]; got != tc.wantFailure {
				t.Errorf("admission_failure = %q, want %q", got, tc.wantFailure)
			}
			if got := response.Verdict.Detail["recovery"]; got != tc.wantRecovery {
				t.Errorf("recovery = %q, want %q", got, tc.wantRecovery)
			}
			if response.Verdict.Detail["remedy"] == "" {
				t.Error("remedy machine detail is empty")
			}
			if tc.wantWorkspace {
				if got := response.Verdict.Detail["trusted_workspace"]; got != filepath.Clean(root) {
					t.Errorf("trusted_workspace = %q, want canonical root %q", got, filepath.Clean(root))
				}
			}
			assertAdjudicateReceipt(t, response.Receipt, "denied")
		})
	}

	t.Run("atomic execution guard", func(t *testing.T) {
		srv := newPermissiveAdmissionTestServer(t, false)
		verdict, result, err := srv.syscall(context.Background(), "allow_write", `{"file_path":"write.txt"}`, false, "", "diagnostic-atomic")
		if err != nil || result != nil || verdict.Kind != "DENY" || verdict.Reason != "DEFAULT_DENY" {
			t.Fatalf("atomic execution verdict=%+v result=%+v err=%v", verdict, result, err)
		}
		for key, want := range map[string]string{
			"admission_failure": "atomic_guard_unavailable",
			"recovery":          "USE_GUARDED_EXECUTOR",
		} {
			if got := verdict.Detail[key]; got != want {
				t.Errorf("%s = %q, want %q", key, got, want)
			}
		}
		if verdict.Detail["remedy"] == "" {
			t.Error("remedy machine detail is empty")
		}
	})
}

// fak-test:runtime fast est=1s
func TestLeaseAdmissionDiagnosticsRenderFailureSpecificRecovery(t *testing.T) {
	cases := []struct {
		name         string
		failure      string
		reason       string
		wantRecovery string
	}{
		{"unknown footprint", "unknown_write_footprint", "DEFAULT_DENY", "USE_FILE_TOOLS"},
		{"outside workspace", "outside_trusted_workspace", "DEFAULT_DENY", "CHECK_WORKSPACE_BINDING"},
		{"write target unresolved", "write_target_unresolvable", "DEFAULT_DENY", "CHECK_WRITE_TARGET"},
		{"workspace authority", "workspace_authority_unavailable", "DEFAULT_DENY", "RESTORE_LEASE_AUTHORITY"},
		{"lease authority read", "lease_authority_read_failed", "DEFAULT_DENY", "RESTORE_LEASE_AUTHORITY"},
		{"invalid view", "invalid_lease_view", "DEFAULT_DENY", "RESTORE_LEASE_AUTHORITY"},
		{"peer held", "peer_lease_held", "LEASE_HELD", "RESPECT_PEER_LEASE"},
		{"atomic guard", "atomic_guard_unavailable", "DEFAULT_DENY", "USE_GUARDED_EXECUTOR"},
	}
	allRecoveries := []string{
		"USE_FILE_TOOLS", "CHECK_WORKSPACE_BINDING", "CHECK_WRITE_TARGET", "RESTORE_LEASE_AUTHORITY",
		"RESPECT_PEER_LEASE", "USE_GUARDED_EXECUTOR", "INSPECT_LEASE_ADMISSION",
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			verdict := WireVerdict{
				Kind: "DENY", Reason: tc.reason, By: leaseAdmissionBy, Disposition: "TERMINAL",
				Detail: map[string]string{
					"admission_failure": tc.failure,
					"recovery":          tc.wantRecovery,
					"remedy":            "bounded remedy",
				},
			}
			adj := ToolAdjudication{ToolCallID: "call-1", Tool: "exec_command", Verdict: verdict}
			outputs := map[string]string{
				"denySummary":      denySummary([]ToolAdjudication{adj}),
				"adjudicationNote": adjudicationNote([]ToolAdjudication{adj}),
				"deniedToolResult": deniedToolResult(adj),
			}
			for surface, output := range outputs {
				if !strings.Contains(output, tc.wantRecovery) {
					t.Errorf("%s missing recovery %q: %s", surface, tc.wantRecovery, output)
				}
				for _, other := range allRecoveries {
					if other != tc.wantRecovery && strings.Contains(output, other) {
						t.Errorf("%s carries unrelated recovery %q: %s", surface, other, output)
					}
				}
				if tc.failure != "unknown_write_footprint" && strings.Contains(strings.ToLower(output), "write footprint") {
					t.Errorf("%s gives unknown-footprint advice for %q: %s", surface, tc.failure, output)
				}
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestLeaseAdmissionDiagnosticsLegacyAndFutureFailureFallbacks(t *testing.T) {
	cases := []struct {
		name, failure, reason, wantRecovery string
		wantLegacy                          bool
		forbidRecovery                      string
	}{
		{name: "absent legacy code", reason: "DEFAULT_DENY", wantRecovery: "USE_FILE_TOOLS", wantLegacy: true},
		{name: "whitespace legacy code", failure: " \t ", reason: "DEFAULT_DENY", wantRecovery: "USE_FILE_TOOLS", wantLegacy: true},
		{name: "unknown future code", failure: "future_admission_failure", reason: "DEFAULT_DENY", wantRecovery: "INSPECT_LEASE_ADMISSION"},
		{name: "untyped peer lease", reason: "LEASE_HELD", wantRecovery: "INSPECT_LEASE_ADMISSION", forbidRecovery: "USE_FILE_TOOLS"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			detail := map[string]string{}
			if tc.failure != "" {
				detail["admission_failure"] = tc.failure
			}
			adj := ToolAdjudication{
				ToolCallID: "call-fallback",
				Tool:       "exec_command",
				Verdict: WireVerdict{
					Kind: "DENY", Reason: tc.reason, By: leaseAdmissionBy, Disposition: "TERMINAL", Detail: detail,
				},
			}
			for surface, output := range map[string]string{
				"denySummary":      denySummary([]ToolAdjudication{adj}),
				"adjudicationNote": adjudicationNote([]ToolAdjudication{adj}),
				"deniedToolResult": deniedToolResult(adj),
			} {
				if !strings.Contains(output, tc.wantRecovery) {
					t.Errorf("%s missing fallback recovery %q: %s", surface, tc.wantRecovery, output)
				}
				if tc.forbidRecovery != "" && strings.Contains(output, tc.forbidRecovery) {
					t.Errorf("%s carries forbidden recovery %q: %s", surface, tc.forbidRecovery, output)
				}
				lower := strings.ToLower(output)
				if strings.Contains(lower, "write footprint") {
					t.Errorf("%s guesses unknown-footprint cause for %q: %s", surface, tc.failure, output)
				}
				if tc.wantLegacy {
					for _, evidence := range []string{"legacy/untyped", "cause unavailable", "file mutation", "file tools"} {
						if !strings.Contains(lower, evidence) {
							t.Errorf("%s legacy fallback missing %q: %s", surface, evidence, output)
						}
					}
				}
			}
		})
	}
}

// fak-test:runtime fast est=1s
// The composed lease-admission DEFAULT_DENY note must carry the remedy sentence
// EXACTLY ONCE: remedyNote already renders it as the sanctioned alternative, so
// defaultDenyOperatorRemedy may only contribute the typed recovery token. The
// verdict is built through leaseAdmissionRefusal, the same seam the wire uses.
func TestLeaseAdmissionComposedNoteCarriesRemedyOnce(t *testing.T) {
	verdict := leaseAdmissionRefusal(abi.ReasonDefaultDeny, leaseFailureUnknownFootprint, "write footprint is unknown")
	adj := ToolAdjudication{ToolCallID: "call-once", Tool: "bash", Verdict: verdict}
	remedy := verdict.Detail["remedy"]
	surfaces := map[string]string{
		"denySummary":      denySummary([]ToolAdjudication{adj}),
		"adjudicationNote": adjudicationNote([]ToolAdjudication{adj}),
		"deniedToolResult": deniedToolResult(adj),
	}
	if notes, _ := renderRefusalNotes(adj); notes != "" {
		surfaces["renderRefusalNotes"] = notes
	}
	for name, got := range surfaces {
		if n := strings.Count(got, remedy); n != 1 {
			t.Errorf("%s carries the remedy sentence %d times, want exactly 1:\n%s", name, n, got)
		}
		if !strings.Contains(got, "recovery=USE_FILE_TOOLS") {
			t.Errorf("%s missing typed token recovery=USE_FILE_TOOLS:\n%s", name, got)
		}
		alt := strings.Index(got, "sanctioned alternative:")
		tok := strings.Index(got, "recovery=USE_FILE_TOOLS")
		if alt < 0 || tok < 0 || alt > tok {
			t.Errorf("%s order broken (sanctioned alternative:%d, token:%d):\n%s", name, alt, tok, got)
		}
	}
}

// fak-test:runtime fast est=1s
// The unknown_write_footprint remedy keeps its two pinned sentences and states
// the structural truth: re-proposing the identical mutating shell call cannot
// prove its write footprint, so it will be refused again.
func TestLeaseAdmissionUnknownFootprintRemedyStatesStructuralTruth(t *testing.T) {
	diagnostic, ok := diagnosticForLeaseAdmissionFailure(leaseFailureUnknownFootprint)
	if !ok {
		t.Fatal("unknown_write_footprint has no diagnostic")
	}
	for _, want := range []string{
		"Use structured file tools with explicit paths.",
		"Native shell commands have no blanket fak-run exemption: use an actually registered typed operation; generic opaque effects remain denied.",
		"cannot prove its write footprint and will be refused again",
		"move the step to structured file tools or a registered typed operation instead of re-proposing it",
	} {
		if !strings.Contains(diagnostic.remedy, want) {
			t.Errorf("remedy missing %q:\n%s", want, diagnostic.remedy)
		}
	}
}

// fak-test:runtime fast est=1s
// Deduplication must not eat the sentence on wires whose Detail["remedy"] is
// empty or differs from the diagnostic: they keep the full typed composition.
func TestLeaseAdmissionDedupKeepsSentenceOnUntypedOrForeignRemedy(t *testing.T) {
	diag, _ := diagnosticForLeaseAdmissionFailure(leaseFailureUnknownFootprint)
	cases := []struct {
		name   string
		detail map[string]string
	}{
		{"empty remedy detail", map[string]string{"admission_failure": "unknown_write_footprint", "recovery": "USE_FILE_TOOLS"}},
		{"foreign remedy detail", map[string]string{"admission_failure": "unknown_write_footprint", "recovery": "USE_FILE_TOOLS", "remedy": "bounded remedy"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			adj := ToolAdjudication{
				ToolCallID: "call-dedup",
				Tool:       "bash",
				Verdict: WireVerdict{
					Kind: "DENY", Reason: "DEFAULT_DENY", By: leaseAdmissionBy, Disposition: "TERMINAL", Detail: tc.detail,
				},
			}
			want := "recovery=USE_FILE_TOOLS: " + diag.remedy
			notes, _ := renderRefusalNotes(adj)
			if !strings.Contains(notes, want) {
				t.Fatalf("renderRefusalNotes lost the full typed remedy composition:\n%s", notes)
			}
		})
	}
}
