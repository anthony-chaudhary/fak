package adjudicator

// The new-file authorship exemption (fak-private guard complaints #3707/#3708/#3709):
// a typed write of a Go test file that does not yet exist cannot weaken an existing
// gating suite, so it is admitted under ANY lane. Every tamper shape stays gated:
// overwriting an existing test file, TestMain in a new file (the skip-m.Run()
// vacuous-pass vector), testdata/fixtures/configs, shell/exec writes, and any
// target that escapes the workspace (the guard's CWD).

import (
	"context"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const authorshipProbeFile = "zz_authorship_probe_new_test.go"

func authorshipProbeAbsent(t *testing.T) {
	t.Helper()
	_ = os.Remove(authorshipProbeFile)
	if _, err := os.Stat(authorshipProbeFile); err == nil {
		t.Fatalf("probe file %s must not exist for the authorship cases", authorshipProbeFile)
	}
}

func TestImmunityAdmitsNewTestFileAuthorship(t *testing.T) {
	a := New(testImmunityBasePolicy())
	ctx := context.Background()
	authorshipProbeAbsent(t)

	cases := []struct {
		name string
		tool string
		args string
		meta map[string]string
	}{
		{
			name: "Write of a NEW *_test.go under an implementation lane",
			tool: "Write",
			args: `{"filePath":"zz_authorship_probe_new_test.go","content":"package adjudicator\n\nimport \"testing\"\n\nfunc TestAuthorshipProbe(t *testing.T) {}\n"}`,
			meta: map[string]string{"lane": "implementation"},
		},
		{
			name: "Write of a NEW *_test.go under no lane at all",
			tool: "Write",
			args: `{"filePath":"zz_authorship_probe_new_test.go","content":"package adjudicator"}`,
			meta: nil,
		},
		{
			name: "write_file of a NEW *_test.go",
			tool: "write_file",
			args: `{"path":"zz_authorship_probe_new_test.go","content":"package adjudicator"}`,
			meta: map[string]string{"lane": "cmd"},
		},
		{
			name: "Write of a NEW *_test.go by ABSOLUTE path outside the CWD (serving router hosts many workspaces; existence is the gate)",
			tool: "Write",
			args: `{"filePath":"C:/zz_abs_probe_0917_new_test.go","content":"package elsewhere"}`,
			meta: map[string]string{"lane": "implementation"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			call := inlineCall(tc.tool, tc.args)
			call.Meta = tc.meta
			v := a.Adjudicate(ctx, call)
			if v.Kind == abi.VerdictDeny && abi.ReasonName(v.Reason) == "TEST_TAMPER_REFUSED" {
				t.Fatalf("%s: pure authorship of a new test file must not be refused (got %v/%s)",
					tc.name, v.Kind, abi.ReasonName(v.Reason))
			}
			if v.Kind == abi.VerdictDeny {
				t.Fatalf("%s: expected the authorship write to pass the floor, got %v (%s)",
					tc.name, v.Kind, abi.ReasonName(v.Reason))
			}
		})
	}
}

func TestImmunityAdmitsNewTestFileAuthorshipByAbsolutePath(t *testing.T) {
	a := New(testImmunityBasePolicy())
	authorshipProbeAbsent(t)
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	// Absolute path (the shape Claude Code's Write tool emits in practice)
	// exercises the IsAbs branch of workspaceResolvedPath.
	call := inlineCallJSON(t, "Write", map[string]any{
		"filePath": wd + "/zz_authorship_probe_new_test.go",
		"content":  "package adjudicator",
	})
	call.Meta = map[string]string{"lane": "implementation"}
	v := a.Adjudicate(context.Background(), call)
	if v.Kind == abi.VerdictDeny {
		t.Fatalf("absolute-path new-file authorship must be admitted, got %v (%s)", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestLaneMetaPrecedence pins the load-bearing ordering of resolveLane: the
// host-designated Meta["lane"] (launch-time, trusted) outranks wire-supplied
// args["lane"] (model-controlled, never trusted). With a designated test lane an
// edit of an EXISTING test file is admitted (the designated-test-author route);
// with an implementation lane the args-side "test" claim must NOT win.
func TestLaneMetaPrecedence(t *testing.T) {
	a := New(testImmunityBasePolicy())
	ctx := context.Background()

	editArgs := func(argsLane string) map[string]any {
		return map[string]any{
			"filePath":   "decide_test.go",
			"old_string": "package adjudicator",
			"new_string": "package adjudicator\n",
			"lane":       argsLane,
		}
	}

	// Host-designated test lane wins over the wire's implementation claim:
	// the existing-test edit is admitted (designated-test-author route).
	call := inlineCallJSON(t, "Edit", editArgs("implementation"))
	call.Meta = map[string]string{"lane": "test"}
	if v := a.Adjudicate(ctx, call); v.Kind == abi.VerdictDeny && abi.ReasonName(v.Reason) == "TEST_TAMPER_REFUSED" {
		t.Fatalf("designated test lane must admit the existing-test edit (Meta outranks args): got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}

	// The wire's test-lane claim must not win over the host's implementation lane.
	call2 := inlineCallJSON(t, "Edit", editArgs("test"))
	call2.Meta = map[string]string{"lane": "implementation"}
	if v := a.Adjudicate(ctx, call2); v.Kind != abi.VerdictDeny || abi.ReasonName(v.Reason) != "TEST_TAMPER_REFUSED" {
		t.Fatalf("wire args lane must not override the host implementation lane: got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestImmunityStillRefusesTamperShapes(t *testing.T) {
	a := New(testImmunityBasePolicy())
	ctx := context.Background()
	authorshipProbeAbsent(t)

	cases := []struct {
		name string
		tool string
		args string
		meta map[string]string
	}{
		{
			name: "Write of a NEW *_test.go whose body defines TestMain with whitespace evasion (skip-m.Run vector)",
			tool: "Write",
			args: `{"filePath":"zz_authorship_probe_new_test.go","content":"package adjudicator\n\nimport (\n\t\"os\"\n\t\"testing\"\n)\n\nfunc TestMain (m *testing.M) { os.Exit(0) }\n"}`,
			meta: nil,
		},
		{
			name: "Write of a NEW *_test.go whose body calls os.Exit in init (vacuous-pass vector)",
			tool: "Write",
			args: `{"filePath":"zz_authorship_probe_new_test.go","content":"package adjudicator\n\nimport \"os\"\n\nfunc init() { os.Exit(0) }\n"}`,
			meta: nil,
		},
		{
			name: "Write of a NEW *_test.go whose body sets a test flag in init (zero-tests-ok vector)",
			tool: "Write",
			args: `{"filePath":"zz_authorship_probe_new_test.go","content":"package adjudicator\n\nimport \"flag\"\n\nfunc init() { flag.Set(\"test.run\", \"zzz_no_match\") }\n"}`,
			meta: nil,
		},
		{
			name: "Multi-target write mixing a NEW test file with an EXISTING one stays gated",
			tool: "Write",
			args: `{"paths":["zz_authorship_probe_new_test.go","decide_test.go"],"content":"package adjudicator"}`,
			meta: nil,
		},
		{
			name: "Write overwriting an EXISTING *_test.go",
			tool: "Write",
			args: `{"filePath":"decide_test.go","content":"package adjudicator"}`,
			meta: nil,
		},
		{
			name: "Write with no content key fails closed",
			tool: "Write",
			args: `{"filePath":"zz_authorship_probe_new_test.go"}`,
			meta: nil,
		},
		{
			name: "Write targeting a RELATIVE path that escapes the workspace fails closed",
			tool: "Write",
			args: `{"filePath":"../zz_escape_probe_test.go","content":"package other"}`,
			meta: nil,
		},
		{
			name: "Bash heredoc creating a NEW *_test.go stays gated (shell is a wildcard)",
			tool: "Bash",
			args: `{"command":"cat > zz_authorship_probe_new_test.go <<'EOF'\npackage adjudicator\nEOF"}`,
			meta: nil,
		},
		{
			name: "Write of a NEW testdata file stays gated (fixtures change how tests run)",
			tool: "Write",
			args: `{"filePath":"testdata/zz_authorship_probe.json","content":"{}"}`,
			meta: nil,
		},
		{
			name: "Write of a NEW conftest.py stays gated (collection hooks can skip tests)",
			tool: "Write",
			args: `{"filePath":"conftest.py","content":"def pytest_collection_modifyitems(items):\n    items.clear()\n"}`,
			meta: nil,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			call := inlineCall(tc.tool, tc.args)
			call.Meta = tc.meta
			v := a.Adjudicate(ctx, call)
			if v.Kind != abi.VerdictDeny || v.Reason != ReasonTestTamperRefused {
				t.Fatalf("%s: expected Deny/TEST_TAMPER_REFUSED, got %v/%s",
					tc.name, v.Kind, abi.ReasonName(v.Reason))
			}
		})
	}
}
