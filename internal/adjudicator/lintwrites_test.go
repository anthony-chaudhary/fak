package adjudicator

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // resolver for the args read path
)

// lintPolicy is the opt-in floor under test: write_file is allowed, and the
// in-process write-lint rung is ON (issue #536).
func lintPolicy() Policy {
	return Policy{
		Allow:      map[string]bool{"write_file": true, "edit": true},
		LintWrites: true,
	}
}

// TestLintWritesDeniesBrokenGoWriteWhenEnabled mirrors internal/swebench's
// TestExecToolLintsWriteWhenEnabled at the verdict layer: under the opt-in
// policy a broken Go write is refused with Deny(MALFORMED) and a bounded
// file:line:col witness — the write does NOT land, unlike the fleet's advisory
// append. The witness names only the first finding and never the file content.
func TestLintWritesDeniesBrokenGoWriteWhenEnabled(t *testing.T) {
	a := New(lintPolicy())
	ctx := context.Background()
	// A sentinel comment the parse error cannot echo proves bounded disclosure:
	// the witness must carry the finding, not the file body.
	broken := `{"path":"main.go","content":"// SECRETLINE-marker\npackage x\nfunc ("}`
	v := a.Adjudicate(ctx, inlineCall("write_file", broken))

	if v.Kind != abi.VerdictDeny {
		t.Fatalf("broken Go write: got Kind=%v, want VerdictDeny", v.Kind)
	}
	if v.Reason != abi.ReasonMalformed {
		t.Fatalf("broken Go write: got Reason=%s, want MALFORMED", abi.ReasonName(v.Reason))
	}
	wp, ok := v.Payload.(abi.WitnessPayload)
	if !ok {
		t.Fatalf("Payload type = %T, want abi.WitnessPayload", v.Payload)
	}
	if !strings.HasPrefix(wp.Claim, "main.go:") {
		t.Fatalf("witness must be addressed to the write path (\"main.go:...\"), got %q", wp.Claim)
	}
	// file:line (and ideally :col) — a citable location, not a bare filename.
	if !strings.Contains(wp.Claim, "main.go:3") {
		t.Fatalf("witness should locate the parse error on line 3, got %q", wp.Claim)
	}
	if strings.Contains(wp.Claim, "SECRETLINE-marker") {
		t.Fatalf("witness leaked file content (unbounded disclosure): %q", wp.Claim)
	}
}

// A clean write, an unlinted language, and a language whose only checker shells
// out (Python) all pass through unchanged — the rung fails OPEN. lint is a
// quality signal, never a security gate, so an absent in-process checker DEFERs.
func TestLintWritesAllowsCleanUnlintedAndAbsentChecker(t *testing.T) {
	a := New(lintPolicy())
	ctx := context.Background()

	clean := inlineCall("write_file", `{"path":"ok.go","content":"package x\n\nfunc F() int { return 1 }\n"}`)
	if v := a.Adjudicate(ctx, clean); v.Kind != abi.VerdictAllow {
		t.Fatalf("clean Go write: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}

	unlinted := inlineCall("write_file", `{"path":"notes.txt","content":"anything at all"}`)
	if v := a.Adjudicate(ctx, unlinted); v.Kind != abi.VerdictAllow {
		t.Fatalf("unlinted .txt write: got %v/%s, want Allow (defer)", v.Kind, abi.ReasonName(v.Reason))
	}

	// Python's only checker shells out and lives off the decide path, so there is
	// no in-process opinion here — the write DEFERs and falls through to Allow.
	pyWrite := inlineCall("write_file", `{"path":"tensor_model.py","content":"def f(:\n    return 1\n"}`)
	if v := a.Adjudicate(ctx, pyWrite); v.Kind != abi.VerdictAllow {
		t.Fatalf("absent-checker .py write: got %v/%s, want Allow (fail open)", v.Kind, abi.ReasonName(v.Reason))
	}
}

// A broken JSON write is refused with MALFORMED + a file:line:col witness — the
// second in-process grammar.
func TestLintWritesDeniesBrokenJSONWrite(t *testing.T) {
	a := New(lintPolicy())
	v := a.Adjudicate(context.Background(),
		inlineCall("write_file", `{"path":"config.json","content":"{\"a\": }"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonMalformed {
		t.Fatalf("broken JSON write: got %v/%s, want Deny/MALFORMED", v.Kind, abi.ReasonName(v.Reason))
	}
	wp, _ := v.Payload.(abi.WitnessPayload)
	if !strings.HasPrefix(wp.Claim, "config.json:") {
		t.Fatalf("witness should address config.json, got %q", wp.Claim)
	}
}

// Off by default: with LintWrites false a broken write passes the floor
// unchanged (existing floors are byte-for-byte unaffected unless opted in).
func TestLintWritesOffByDefaultDoesNotDeny(t *testing.T) {
	a := New(Policy{Allow: map[string]bool{"write_file": true}}) // LintWrites == false
	v := a.Adjudicate(context.Background(),
		inlineCall("write_file", `{"path":"main.go","content":"package x\nfunc ("}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("lint off: broken write must pass unchanged, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

// The rung is scoped to WHOLE-FILE writes: a partial edit (a fragment that would
// never parse standalone) is never false-denied as MALFORMED — it DEFERs. This
// is what keeps the rung from false-blocking every edit tool call.
func TestLintWritesScopedToWholeFileWritesNotEdits(t *testing.T) {
	a := New(lintPolicy())
	v := a.Adjudicate(context.Background(),
		inlineCall("edit", `{"path":"main.go","content":"func ("}`))
	if v.Reason == abi.ReasonMalformed {
		t.Fatalf("a partial edit must not be MALFORMED-denied (would false-block edits), got %q",
			abi.ReasonName(v.Reason))
	}
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("edit of allowed tool: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
}
