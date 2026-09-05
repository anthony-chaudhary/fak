package gateway

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// The whole point of the refusal-note seam (#2750): a BRAND-NEW refusal kind
// that carries actionable Meta must surface in-band on the content channel by
// registering ONE renderer - with no edit to the denySummary / adjudicationNote
// call sites. Before the seam, every new kind (reversibility, remedy, livelock)
// had to hand-stitch another note call at each site; this test pins that a fresh
// kind now rides the generic fold instead.

func TestRefusalNotesLeadWithAllowedPathAndTrailReason(t *testing.T) {
	adj := ToolAdjudication{
		Tool: "git push", Admitted: false,
		Verdict: WireVerdict{
			Kind: "DENY", Reason: "REVERSIBILITY_CONFIRM", Disposition: "RETRYABLE",
			Detail: map[string]string{
				"confirm_token": "tok", "confirm_key": "_fak_confirm",
				"remedy": "use fak sync push",
			},
		},
	}
	for name, got := range map[string]string{
		"denySummary":      denySummary([]ToolAdjudication{adj}),
		"adjudicationNote": adjudicationNote([]ToolAdjudication{adj}),
	} {
		allowed := strings.Index(got, "sanctioned alternative: use fak sync push")
		reason := strings.Index(got, "REVERSIBILITY_CONFIRM")
		if allowed < 0 || reason < 0 {
			t.Fatalf("%s missing contract fields: %q", name, got)
		}
		if allowed >= reason {
			t.Fatalf("%s order = allowed:%d reason:%d; want allowed < reason: %q", name, allowed, reason, got)
		}
	}
}

func TestDefaultDenySurfacesBoundedLiveOperatorChoice(t *testing.T) {
	adj := ToolAdjudication{
		Tool:     "exec_command",
		Admitted: false,
		Verdict:  WireVerdict{Kind: "DENY", Reason: "DEFAULT_DENY", Disposition: "TERMINAL"},
	}
	for name, got := range map[string]string{
		"denySummary":      denySummary([]ToolAdjudication{adj}),
		"adjudicationNote": adjudicationNote([]ToolAdjudication{adj}),
		"deniedToolResult": deniedToolResult(adj),
	} {
		for _, want := range []string{
			"operator choice (outside this wrapped agent)",
			"tool not permitted by policy",
			"consult operator to widen capability floor",
			"standard harness tool",
			"DEFAULT_DENY",
		} {
			if !strings.Contains(got, want) {
				t.Fatalf("%s missing %q:\n%s", name, want, got)
			}
		}
		// Agent-visible text must never contain runnable fak guard allow commands (#11504)
		for _, forbid := range []string{
			"fak guard allow",
			"`fak guard",
		} {
			if strings.Contains(got, forbid) {
				t.Fatalf("%s contains forbidden executable command %q:\n%s", name, forbid, got)
			}
		}
	}

	// Operator remedy is preserved out-of-band via helper
	cmd := OperatorRemedyCommand(adj)
	if cmd != "fak guard allow --ttl 15m exec_command" {
		t.Fatalf("OperatorRemedyCommand mismatch: got %q, want %q", cmd, "fak guard allow --ttl 15m exec_command")
	}

	unsafe := adj
	unsafe.Tool = "exec_command; injected"
	got := denySummary([]ToolAdjudication{unsafe})
	if strings.Contains(got, "--ttl 15m exec_command;") || strings.Contains(got, "fak guard allow") {
		t.Fatalf("unsafe or executable command entered agent-visible text:\n%s", got)
	}
	unsafeCmd := OperatorRemedyCommand(unsafe)
	if unsafeCmd != "fak guard allow --ttl 15m <tool>" {
		t.Fatalf("OperatorRemedyCommand for unsafe tool mismatch: got %q, want %q", unsafeCmd, "fak guard allow --ttl 15m <tool>")
	}
}

// TestRefusalNotesIsolateOperatorRemedyFromAgentVisibleContext asserts that renderRefusalNotes
// and deniedToolResult for DEFAULT_DENY or POLICY_BLOCK contain zero runnable "fak guard allow"
// shell strings in agent-visible text, while operator remediation is preserved in the header or metadata (#11504).
func TestRefusalNotesIsolateOperatorRemedyFromAgentVisibleContext(t *testing.T) {
	for _, tc := range []struct {
		name   string
		reason string
		tool   string
	}{
		{name: "DefaultDeny", reason: "DEFAULT_DENY", tool: "exec_command"},
		{name: "PolicyBlock", reason: "POLICY_BLOCK", tool: "rm_rf"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			adj := ToolAdjudication{
				Tool:     tc.tool,
				Admitted: false,
				Verdict:  WireVerdict{Kind: "DENY", Reason: tc.reason, Disposition: "TERMINAL"},
			}
			notes, _ := renderRefusalNotes(adj)
			deniedResult := deniedToolResult(adj)

			for label, text := range map[string]string{
				"renderRefusalNotes": notes,
				"deniedToolResult":   deniedResult,
			} {
				if strings.Contains(text, "fak guard allow") {
					t.Fatalf("%s contains runnable 'fak guard allow': %q", label, text)
				}
				if strings.Contains(text, "`fak guard") {
					t.Fatalf("%s contains backtick-wrapped fak guard command: %q", label, text)
				}
			}

			// Verify operator remediation is available via helper/metadata
			cmd := OperatorRemedyCommand(adj)
			expectedCmd := "fak guard allow --ttl 15m " + tc.tool
			if cmd != expectedCmd {
				t.Fatalf("OperatorRemedyCommand got %q, want %q", cmd, expectedCmd)
			}

			AttachOperatorRemedyMetadata(&adj)
			if got := adj.Verdict.Detail["operator_remedy"]; got != expectedCmd {
				t.Fatalf("metadata operator_remedy got %q, want %q", got, expectedCmd)
			}

			rec := httptest.NewRecorder()
			SetOperatorRemedyHeaders(rec, []ToolAdjudication{adj})
			if h := rec.Header().Get("X-Fak-Operator-Remedy"); h != expectedCmd {
				t.Fatalf("X-Fak-Operator-Remedy header got %q, want %q", h, expectedCmd)
			}
			if h := rec.Header().Get("X-Fak-Operator-Action"); h != expectedCmd {
				t.Fatalf("X-Fak-Operator-Action header got %q, want %q", h, expectedCmd)
			}
		})
	}
}

func TestReframeUserSpanUntouched(t *testing.T) {
	userTool := "Do not forget `USER_LITERAL`; never rewrite my words."
	adj := ToolAdjudication{
		Tool:     userTool,
		Admitted: false,
		Verdict: WireVerdict{
			Kind:        "DENY",
			Reason:      "POLICY_BLOCK",
			Disposition: "TERMINAL",
		},
	}

	for name, got := range map[string]string{
		"denySummary":      denySummary([]ToolAdjudication{adj}),
		"adjudicationNote": adjudicationNote([]ToolAdjudication{adj}),
	} {
		if !strings.Contains(got, userTool) {
			t.Fatalf("%s changed opaque user bytes: %q", name, got)
		}
		if strings.Contains(got, "remember `USER_LITERAL`") || strings.Contains(got, "always rewrite my words") {
			t.Fatalf("%s reframed opaque user span: %q", name, got)
		}
	}
}

func TestRefusalNoteSeamSurfacesNewKindWithoutCallSiteEdit(t *testing.T) {
	saved := refusalNotes
	defer func() { refusalNotes = saved }()
	// Register a new refusal kind whose actionable Meta lives in a Detail key that
	// no existing renderer reads - proving the surfacing is generic, not bespoke.
	refusalNotes = append(append([]refusalNote(nil), saved...), refusalNote{
		render: func(a ToolAdjudication) string {
			if h := a.Verdict.Detail["witness_hint"]; h != "" {
				return "witness required: " + h
			}
			return ""
		},
	})

	adj := ToolAdjudication{
		Tool:     "Bash",
		Admitted: false,
		Verdict: WireVerdict{
			Kind:        "DENY",
			Reason:      "NEEDS_WITNESS",
			Disposition: "ESCALATE",
			Detail:      map[string]string{"witness_hint": "attach a failing test"},
		},
	}
	for name, got := range map[string]string{
		"denySummary":      denySummary([]ToolAdjudication{adj}),
		"adjudicationNote": adjudicationNote([]ToolAdjudication{adj}),
	} {
		if !strings.Contains(got, "witness required: attach a failing test") {
			t.Fatalf("%s did not surface the new refusal kind through the seam:\n%s", name, got)
		}
	}
}

// Every refusal must surface the agent's APPEAL channel (`fak complain`) in-band:
// a false-positive DENY is byte-identical to a correct one in the decision journal,
// so the kernel cannot self-detect it — the only signal is the agent saying so, and
// it can only say so if it is told the channel exists. A single-denial turn names the
// concrete reason/tool so the appeal is copy-pasteable, and the hint leads with
// "adapt first" so it never reads as an invitation to appeal in lieu of adapting.
func TestAdjudicationNoteSurfacesComplaintChannel(t *testing.T) {
	note := adjudicationNote([]ToolAdjudication{{
		Tool:     "Write",
		Admitted: false,
		Verdict:  WireVerdict{Kind: "DENY", Reason: "FILE_ADMISSION", Disposition: "TERMINAL"},
	}})
	for _, want := range []string{
		"fak complain",
		"--reason FILE_ADMISSION",
		"--tool Write",
		"--from-journal",
		"Adapt first; appeal only when you are confident",
	} {
		if !strings.Contains(note, want) {
			t.Fatalf("adjudicationNote missing complaint hint %q:\n%s", want, note)
		}
	}
}

// External remedy text is opaque (#4430), while its allowed path still leads every
// structured refusal token (#4439). This pins both properties on the real content path.
func TestRefusalNotesTokensStable(t *testing.T) {
	adj := ToolAdjudication{
		Tool:     "Bash",
		Admitted: false,
		Verdict: WireVerdict{
			Kind:        "DENY",
			Reason:      "POLICY_BLOCK",
			Disposition: "TERMINAL",
			Detail:      map[string]string{"remedy": "do not forget to clear the OFF_TRUNK blocker"},
		},
	}
	note := adjudicationNote([]ToolAdjudication{adj})
	for name, got := range map[string]string{
		"adjudicationNote": note,
		"denySummary":      denySummary([]ToolAdjudication{adj}),
	} {
		// contract tokens survive the reframe on both content-channel choke points
		for _, tok := range []string{"POLICY_BLOCK", "OFF_TRUNK"} {
			if !strings.Contains(got, tok) {
				t.Fatalf("%s dropped contract token %q:\n%s", name, tok, got)
			}
		}
		remedy := "sanctioned alternative: do not forget to clear the OFF_TRUNK blocker"
		if !strings.Contains(got, remedy) {
			t.Fatalf("%s changed opaque remedy bytes:\n%s", name, got)
		}
		if strings.Index(got, remedy) >= strings.Index(got, "POLICY_BLOCK") {
			t.Fatalf("%s did not lead with the allowed path:\n%s", name, got)
		}
	}
	// The load-bearing judgement prohibition (adjudicationNote's re-propose trailer) is preserved
	// verbatim — the reframe flips mechanical idioms only, never a bare prohibition.
	if !strings.Contains(strings.ToLower(note), "do not re-propose") {
		t.Fatalf("adjudicationNote dropped the load-bearing prohibition:\n%s", note)
	}
}

// A single-denial appeal must bind its witness: the copy-pasted `fak complain
// --from-journal` command carries the refused call's args_digest as an exact
// `--args-digest` selector, so SelectDenial attaches the witnessed verdict instead
// of refusing an ambiguous reason/tool match and filing witness-less on a busy
// journal. When the denial carries no digest, the command falls back to the bare
// reason/tool form (no dangling selector flag).
func TestComplaintHintBindsWitnessSelectorOnSingleDenial(t *testing.T) {
	withDigest := adjudicationNote([]ToolAdjudication{{
		Tool:       "PowerShell",
		ArgsDigest: "sha256:deadbeefcafe",
		Admitted:   false,
		Verdict:    WireVerdict{Kind: "DENY", Reason: "POLICY_BLOCK", Disposition: "TERMINAL"},
	}})
	for _, want := range []string{
		"--reason POLICY_BLOCK",
		"--tool PowerShell",
		"--from-journal --args-digest sha256:deadbeefcafe",
	} {
		if !strings.Contains(withDigest, want) {
			t.Fatalf("single-denial complaint hint missing %q:\n%s", want, withDigest)
		}
	}

	// No digest on the adjudication → no selector flag (and no empty `--args-digest`).
	noDigest := adjudicationNote([]ToolAdjudication{{
		Tool:     "Write",
		Admitted: false,
		Verdict:  WireVerdict{Kind: "DENY", Reason: "FILE_ADMISSION", Disposition: "TERMINAL"},
	}})
	if strings.Contains(noDigest, "--args-digest") {
		t.Fatalf("digest-less denial should not emit an --args-digest selector:\n%s", noDigest)
	}
	if !strings.Contains(noDigest, "--tool Write --from-journal`") {
		t.Fatalf("digest-less denial should fall back to the bare copy-paste form:\n%s", noDigest)
	}
}

// A turn with more than one denial keeps the <REASON>/<TOOL> placeholders rather
// than misattributing one refused call's scope to another — the appeal command
// stays generic when it cannot be unambiguously specialized, and emits no digest
// selector (a mixed turn's digests are per-call and cannot be attributed).
func TestComplaintHintKeepsPlaceholdersOnMixedTurn(t *testing.T) {
	note := adjudicationNote([]ToolAdjudication{
		{Tool: "Write", ArgsDigest: "sha256:aaa", Admitted: false, Verdict: WireVerdict{Kind: "DENY", Reason: "FILE_ADMISSION", Disposition: "TERMINAL"}},
		{Tool: "Bash", ArgsDigest: "sha256:bbb", Admitted: false, Verdict: WireVerdict{Kind: "DENY", Reason: "POLICY_BLOCK", Disposition: "TERMINAL"}},
	})
	if !strings.Contains(note, "--reason <REASON> --tool <TOOL>") {
		t.Fatalf("mixed-turn complaint hint should keep placeholders:\n%s", note)
	}
	// Even though each denial carries a digest, a mixed turn cannot attribute one to
	// the appeal, so no --args-digest selector is emitted.
	if strings.Contains(note, "--args-digest") {
		t.Fatalf("mixed-turn complaint hint must not emit a digest selector:\n%s", note)
	}
}

// A clean turn (no denials) says nothing about the complaint channel — the hint
// rides the denial trailer, not every note.
func TestComplaintHintAbsentWithoutDenial(t *testing.T) {
	if h := complaintHint(nil); h != "" {
		t.Fatalf("complaintHint should be empty with no denials, got %q", h)
	}
}

// The seam pins the allowed-first rendering order: sanctioned alternative,
// then the preview-confirm constraint.
func TestRefusalNoteSeamPreservesOrderAndConfirmRecipe(t *testing.T) {
	adj := reversibilityRefusal()
	adj.Verdict.Detail["remedy"] = "run git push --dry-run"
	notes, recipe := renderRefusalNotes(adj)
	if !recipe {
		t.Fatalf("reversibility refusal did not flag a confirm recipe: %q", notes)
	}
	recipeIdx := strings.Index(notes, "preview-confirm gate")
	fixIdx := strings.Index(notes, "sanctioned alternative: run git push --dry-run")
	if recipeIdx < 0 || fixIdx < 0 {
		t.Fatalf("seam dropped a note: recipe=%d fix=%d in %q", recipeIdx, fixIdx, notes)
	}
	if fixIdx > recipeIdx {
		t.Fatalf("seam reordered notes (sanctioned alternative must precede recipe):\n%s", notes)
	}
}

func TestCompactRefusal(t *testing.T) {
	t.Run("FormatExplicitArgs", func(t *testing.T) {
		got := FormatCompactRefusalNote("POLICY_BLOCK", "rm -rf / is blocked by policy", "Use git clean instead")
		lines := strings.Split(got, "\n")
		if len(lines) != 2 {
			t.Fatalf("expected 2 lines, got %d: %q", len(lines), got)
		}
		if lines[0] != "[FAK GATE: POLICY_BLOCK] rm -rf / is blocked by policy" {
			t.Fatalf("line 1 mismatch: %q", lines[0])
		}
		if lines[1] != "Next Action: Use git clean instead" {
			t.Fatalf("line 2 mismatch: %q", lines[1])
		}
	})

	t.Run("FormatEmbeddedNextActionInMessage", func(t *testing.T) {
		msg := "rm -rf / is blocked by policy. Next Action: Use git clean to remove untracked files."
		got := FormatCompactRefusalNote("POLICY_BLOCK", msg, "")
		lines := strings.Split(got, "\n")
		if len(lines) != 2 {
			t.Fatalf("expected 2 lines, got %d: %q", len(lines), got)
		}
		if lines[0] != "[FAK GATE: POLICY_BLOCK] rm -rf / is blocked by policy." {
			t.Fatalf("line 1 mismatch: %q", lines[0])
		}
		if lines[1] != "Next Action: Use git clean to remove untracked files." {
			t.Fatalf("line 2 mismatch: %q", lines[1])
		}
		if strings.Contains(lines[0], "Next Action") {
			t.Fatalf("line 1 should not contain embedded next action: %q", lines[0])
		}
	})

	t.Run("FormatPrefixDeduplication", func(t *testing.T) {
		got := FormatCompactRefusalNote("[FAK GATE: POLICY_BLOCK]", "[FAK GATE: POLICY_BLOCK] rm is blocked", "Next Action: use git clean")
		lines := strings.Split(got, "\n")
		if len(lines) != 2 {
			t.Fatalf("expected 2 lines, got %d: %q", len(lines), got)
		}
		if lines[0] != "[FAK GATE: POLICY_BLOCK] rm is blocked" {
			t.Fatalf("line 1 duplicate prefix: %q", lines[0])
		}
		if lines[1] != "Next Action: use git clean" {
			t.Fatalf("line 2 duplicate prefix: %q", lines[1])
		}
	})

	t.Run("FormatEmptyNextActionDerivesAffordance", func(t *testing.T) {
		got := FormatCompactRefusalNote("OFF_TRUNK", "cannot commit on side branch", "")
		lines := strings.Split(got, "\n")
		if len(lines) != 2 {
			t.Fatalf("expected 2 lines, got %d: %q", len(lines), got)
		}
		if !strings.HasPrefix(lines[0], "[FAK GATE: OFF_TRUNK]") {
			t.Fatalf("line 1 missing reason tag: %q", lines[0])
		}
		if !strings.Contains(lines[1], "commit on main with fak commit") {
			t.Fatalf("line 2 missing known affordance: %q", lines[1])
		}
	})

	t.Run("CompressMultilineVerboseNote", func(t *testing.T) {
		verbose := `Detailed policy refusal explanations can span 10-20 lines of text.
Tool 'rm -rf /' was refused by the active policy gate.
Reason: POLICY_BLOCK
Violation: Recursive root path deletion is forbidden by containment.
Constraint: rm: DENY (POLICY_BLOCK/TERMINAL)
Next Action: Choose an allowed alternative tool or request operator clearance.
Session ID: 991283
Timestamp: 2026-09-03T12:00:00Z
Turn: 3
Policy File: /etc/fak/policy.json
Audit Record: audit-10294-x
Stack Trace: none
Caller: agent-worker-4`

		got := CompressRefusalNote(verbose)
		lines := strings.Split(got, "\n")
		if len(lines) != 2 {
			t.Fatalf("expected at most 2 lines, got %d:\n%s", len(lines), got)
		}
		if !strings.HasPrefix(lines[0], "[FAK GATE: POLICY_BLOCK]") {
			t.Fatalf("line 1 missing extracted reason: %q", lines[0])
		}
		if !strings.Contains(lines[0], "Recursive root path deletion is forbidden by containment.") {
			t.Fatalf("line 1 missing primary informative reason: %q", lines[0])
		}
		if lines[1] != "Next Action: Choose an allowed alternative tool or request operator clearance." {
			t.Fatalf("line 2 missing extracted action: %q", lines[1])
		}
	})

	t.Run("CompressExistingRefusalOutputs", func(t *testing.T) {
		adj := ToolAdjudication{
			Tool:     "git push",
			Admitted: false,
			Verdict: WireVerdict{
				Kind:        "DENY",
				Reason:      "REVERSIBILITY_CONFIRM",
				Disposition: "RETRYABLE",
				Detail: map[string]string{
					"remedy": "use fak sync push",
				},
			},
		}
		raw := denySummary([]ToolAdjudication{adj})
		compressed := CompressRefusalNote(raw)
		lines := strings.Split(compressed, "\n")
		if len(lines) != 2 {
			t.Fatalf("expected 2 lines, got %d:\n%s", len(lines), compressed)
		}
		if !strings.HasPrefix(lines[0], "[FAK GATE: REVERSIBILITY_CONFIRM]") {
			t.Fatalf("line 1 missing reason tag: %q", lines[0])
		}
		if lines[1] != "Next Action: use fak sync push" {
			t.Fatalf("line 2 missing remedy action: %q", lines[1])
		}
	})

	t.Run("CompressIdempotency", func(t *testing.T) {
		initial := "[FAK GATE: POLICY_BLOCK] tool rm is blocked\nNext Action: use git clean"
		once := CompressRefusalNote(initial)
		twice := CompressRefusalNote(once)
		if once != initial {
			t.Fatalf("once != initial:\ngot:\n%s\nwant:\n%s", once, initial)
		}
		if twice != once {
			t.Fatalf("twice != once:\ntwice:\n%s\nonce:\n%s", twice, once)
		}
	})

	t.Run("AtMostTwoLinesGuarantee", func(t *testing.T) {
		testCases := []struct {
			reason     string
			message    string
			nextAction string
		}{
			{"", "", ""},
			{"   ", "   ", "   "},
			{"POLICY_BLOCK", "Line 1\r\nLine 2\r\nLine 3\r\nLine 4\r\nLine 5", "Action 1\r\nAction 2"},
			{"CUSTOM_GATE", strings.Repeat("Very long explanation text without newline. ", 10), "Action"},
			{"DEFAULT_DENY", "Tool not found\n\n\n\n", "Update policy\n"},
		}

		for i, tc := range testCases {
			got := FormatCompactRefusalNote(tc.reason, tc.message, tc.nextAction)
			lines := strings.Split(got, "\n")
			if len(lines) > 2 || len(lines) == 0 {
				t.Fatalf("case %d: expected 1 or 2 lines, got %d: %q", i, len(lines), got)
			}
			if strings.TrimSpace(got) == "" {
				t.Fatalf("case %d: got empty string", i)
			}
			for lineIdx, line := range lines {
				if strings.Contains(line, "\n") || strings.Contains(line, "\r") {
					t.Fatalf("case %d line %d contains newline characters: %q", i, lineIdx, line)
				}
			}
			if !strings.HasPrefix(lines[0], "[FAK GATE: ") {
				t.Fatalf("case %d line 1 must start with [FAK GATE: , got: %q", i, lines[0])
			}
			if len(lines) == 2 && !strings.HasPrefix(lines[1], "Next Action: ") {
				t.Fatalf("case %d line 2 must start with Next Action: , got: %q", i, lines[1])
			}
		}
	})

	t.Run("AffordanceFirstReframing", func(t *testing.T) {
		got := FormatCompactRefusalNote("POLICY_BLOCK", "Do not forget to stamp the commit.", "use git commit")
		lines := strings.Split(got, "\n")
		if !strings.Contains(lines[0], "remember to stamp the commit.") {
			t.Fatalf("failed to reframe negative idiom affordance-first: %q", lines[0])
		}
	})
}

func TestIFCSinkRefusalNoteSurfacesRemedy(t *testing.T) {
	// A refusal on By == "ifc-sink" with Meta["fix"] renders the clear remedy note instead of bare "TRUST_VIOLATION"
	fixText := "IFC egress block: parameter 'to' contains external destination; strip off-box destination keys from send_email or authorize tool in policy"
	v := abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonTrustViolation,
		By:     "ifc-sink",
		Meta: map[string]string{
			"subsystem":         "ifc-sink",
			"deny_rule":         "ifc_taint_egress",
			"taint_source_tool": "read_webpage",
			"offending_arg":     "to",
			"fix":               fixText,
		},
	}
	wv := renderVerdict(v, nil)
	adj := ToolAdjudication{
		Tool:     "send_email",
		Admitted: false,
		Verdict:  wv,
	}

	for name, got := range map[string]string{
		"denySummary":      denySummary([]ToolAdjudication{adj}),
		"adjudicationNote": adjudicationNote([]ToolAdjudication{adj}),
	} {
		wantSanctioned := "sanctioned alternative: " + fixText
		if !strings.Contains(got, wantSanctioned) {
			t.Fatalf("%s missing clear remedy note %q:\n%s", name, wantSanctioned, got)
		}
		if !strings.Contains(got, "TRUST_VIOLATION") {
			t.Fatalf("%s missing reason TRUST_VIOLATION:\n%s", name, got)
		}
		affordance := errorAffordance("TRUST_VIOLATION")
		if strings.Contains(got, affordance) {
			t.Fatalf("%s should render remedy from Meta[fix], but found fallback affordance %q:\n%s", name, affordance, got)
		}
	}

	// Without Meta["fix"], it falls back to the TRUST_VIOLATION errorAffordance rather than bare token
	bareV := abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonTrustViolation,
		By:     "ifc-sink",
	}
	bareWV := renderVerdict(bareV, nil)
	bareAdj := ToolAdjudication{
		Tool:     "send_email",
		Admitted: false,
		Verdict:  bareWV,
	}
	for name, got := range map[string]string{
		"denySummary":      denySummary([]ToolAdjudication{bareAdj}),
		"adjudicationNote": adjudicationNote([]ToolAdjudication{bareAdj}),
	} {
		affordance := errorAffordance("TRUST_VIOLATION")
		if !strings.Contains(got, affordance) {
			t.Fatalf("%s bare refusal missing errorAffordance %q:\n%s", name, affordance, got)
		}
	}
}
