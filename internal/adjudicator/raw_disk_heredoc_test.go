package adjudicator

import "testing"

// raw_disk_heredoc_test.go — the ordinary gotcha caller must not reuse a
// pre-matched-rule device decision without first ESTABLISHING a device trigger
// (#13867).
//
// The arg-predicate path (evalDeviceOpArgRule) consults
// commandPerformsDeviceOperation only AFTER the rule's own raw regex has matched,
// so the structural walk is handed a string that already names a device verb. The
// gotcha path (matchRawDiskDestruction, driven by EvalDangerousGotchas) had no such
// gate: it handed the walk EVERY command string. The walk decides under BOTH
// dialects and is deliberately FAIL-CLOSED, so a payload that merely fails the
// OTHER dialect's lexer — a Python here-doc body whose apostrophes leave an
// unterminated PowerShell quoted span — was read as a device operation and refused
// as `dangerous_gotcha:raw_disk`, even though it names no device verb at all.
//
// The exact read-only batch witnessed in the report, reproduced WITHOUT executing
// it. The here-doc body is input DATA to the comparison, not a command line.
const readOnlyPythonHeredocBatch = `gofmt -d internal/model/kv_darwin_test.go
git diff --check
git diff -- internal/model/kv_darwin_test.go
git status --short
python3 - <<'PY'
from pathlib import Path
import hashlib,json,subprocess
p=Path('internal/model/kv_darwin_test.go')
b=p.read_bytes()
base=subprocess.check_output(['git','show','HEAD:internal/model/kv_darwin_test.go'])
start=b'\tcfg := Config{'
label=b'func TestAsyncKVBlockITLJitterReduction(t *testing.T) {'
end=b'// TestAsyncKVBlockConcurrencyAndRace'
def body(x):
    f=x.index(label)
    return x[x.index(start,f):x.index(end,f)]
expected=base.replace(b'import (\n\t"sync"',b'import (\n\t"os"\n\t"sync"').replace(b'// TestAsyncKVBlockITLJitterReduction verifies Scoped Acceptance Criteria 3:\n// P99 inter-token latency variance drops by >= 40% in multi-token generation benchmarks.\nfunc TestAsyncKVBlockITLJitterReduction(t *testing.T) {\n',b'// TestAsyncKVBlockITLJitterReduction optionally qualifies Scoped Acceptance Criteria 3:\n// P99 inter-token latency variance drops by >= 40% in multi-token generation benchmarks.\n// Wall-clock qualification follows the package\'s FAK_PERF opt-in convention.\n// fak-test:runtime medium est=2s lane=default\nfunc TestAsyncKVBlockITLJitterReduction(t *testing.T) {\n\tif os.Getenv("FAK_PERF") == "" {\n\t\tt.Skip("set FAK_PERF=1 to run the wall-clock KV jitter qualification")\n\t}\n')
assert b==expected,'unexpected source delta'
assert body(b)==body(base),'qualification body changed'
print(json.dumps({'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'assertion_body_bytes':len(body(b)),'assertion_body_sha256':hashlib.sha256(body(b)).hexdigest(),'original_qualification_body_byte_identical':True,'candidate_matches_exact_admitted_shape':True,'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},indent=2))
PY`

// TestRawDiskGotchaAdmitsReadOnlyPythonHeredoc is the ordinary-caller regression
// named by the issue's definition of done: the exact safe batch must not be refused
// as a raw-disk operation. It goes through EvalDangerousGotchas — the same entry the
// kernel uses — not a helper.
func TestRawDiskGotchaAdmitsReadOnlyPythonHeredoc(t *testing.T) {
	if _, denied := EvalDangerousGotchas("Bash", map[string]any{"command": readOnlyPythonHeredocBatch}); denied {
		t.Fatalf("the read-only Python here-doc batch was refused as dangerous_gotcha:raw_disk")
	}
}

// TestRawDiskGotchaAdmitsPayloadFailingTheOtherDialectLexer pins the actual branch:
// each payload names NO device verb, so no device trigger exists, and the structural
// walk's fail-closed PowerShell branch (an unterminated quoted span) must not stand
// in for one.
//
// The exact reported batch trips THREE gotchas at once — raw_disk via the ASYNC
// walk, and (found while pinning this) privilege_escalation and infra_teardown, whose
// own fail-closed PowerShell/mixed-dialect walks fired on the same Python here-doc
// for the same reason: a caller with no rule of its own handed them a command they
// were built never to see without a trigger. The benign payloads below carry each
// family's accidental mention shape (an apostrophe-bearing Python line, a bare
// `python3 - <<` opener) so all three gates are exercised together.
func TestRawDiskGotchaAdmitsPayloadFailingTheOtherDialectLexer(t *testing.T) {
	safe := []struct {
		name string
		cmd  string
	}{
		{"python single-quoted string swallows the PS quote", "python3 - <<'PY'\nprint(\"it's a plain label\")\nPY"},
		{"python bytes literal with an escape", "python3 - <<'PY'\nb=b'\\\\tX'\nprint(b)\nPY"},
		{"read-only diagnostics only", "gofmt -d internal/model/kv_darwin_test.go\ngit diff --check\ngit status --short"},
	}
	for _, tc := range safe {
		t.Run(tc.name, func(t *testing.T) {
			if _, denied := EvalDangerousGotchas("Bash", map[string]any{"command": tc.cmd}); denied {
				t.Errorf("a payload naming no device verb was refused as a device operation: %q", tc.cmd)
			}
		})
	}
}

// TestElevationAndTeardownGotchasAdmitMentionlessPayloads pins the two sibling
// families the report's batch also tripped. Each payload names neither
// `Start-Process` nor `terraform`, so their fail-closed walks must not stand in for
// a trigger, while a real invocation of each stays denied through the ordinary
// caller.
func TestElevationAndTeardownGotchasAdmitMentionlessPayloads(t *testing.T) {
	safe := []string{
		"python3 - <<'PY'\nprint(\"a runas note: it's harmless\")\nPY",
		"python3 - <<'PY'\nprint('a destroy note')\nPY",
		`grep -rn 'runas' docs/`,
	}
	for _, cmd := range safe {
		t.Run(cmd, func(t *testing.T) {
			if _, denied := EvalDangerousGotchas("Bash", map[string]any{"command": cmd}); denied {
				t.Errorf("a payload mentioning neither elevation verb nor terraform was refused: %q", cmd)
			}
		})
	}
	unsafe := []string{
		`Start-Process -Verb RunAs cmd`,
		`terraform destroy`,
		`terraform destroy -auto-approve`,
	}
	for _, cmd := range unsafe {
		t.Run(cmd, func(t *testing.T) {
			if _, denied := EvalDangerousGotchas("Bash", map[string]any{"command": cmd}); !denied {
				t.Errorf("a real elevation/teardown was admitted: %q", cmd)
			}
		})
	}
}

// TestRawDiskGotchaStillDeniesRealDeviceOperations guards the other direction: the
// gate must establish a trigger, not weaken the deny. Each case names a device verb
// and stays refused through the ordinary caller.
func TestRawDiskGotchaStillDeniesRealDeviceOperations(t *testing.T) {
	unsafe := []struct {
		name string
		cmd  string
	}{
		{"filesystem creation", `mkfs.ext4 /dev/sdb1`},
		{"byte copy onto a device", `dd if=/dev/zero of=/dev/sda`},
		{"redirect into a raw device", `echo x > /dev/sda`},
		{"powershell volume cmdlet", `Format-Volume -DriveLetter D`},
		{"device verb laundered through a shell", `sh -c 'mkfs /dev/sda'`},
		{"device verb inside an evaluator stays refused", `eval "mkfs /dev/sda"`},
		{"device verb behind variable indirection stays refused", `M=mkfs; $M /dev/sda`},
	}
	for _, tc := range unsafe {
		t.Run(tc.name, func(t *testing.T) {
			if _, denied := EvalDangerousGotchas("Bash", map[string]any{"command": tc.cmd}); !denied {
				t.Errorf("a real device operation was admitted: %q", tc.cmd)
			}
		})
	}
}
