package adjudicator

import (
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	// Blank-import the blob backend so abi.ActiveResolver() is non-nil on the
	// TRANSFORM path (putJSON re-stores the redacted args as a non-inline Ref,
	// and the test resolves that Ref back). init() in blob registers the
	// region backend; we do NOT call abi.ResetForTest().
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// inlineCall builds a tool call with inline JSON args (no resolver needed for
// the args read path).
func inlineCall(tool, jsonArgs string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(jsonArgs)},
		Meta: map[string]string{"readOnlyHint": "true"},
	}
}

// Unit 13: Default.Adjudicate returns VerdictAllow for an allowed tool.
func TestDefaultAllowsAllowedTool(t *testing.T) {
	v := Default.Adjudicate(context.Background(), inlineCall("get_user_details", `{}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("get_user_details: got Kind=%v, want VerdictAllow", v.Kind)
	}
	if v.By != "monitor" {
		t.Fatalf("By: got %q, want %q", v.By, "monitor")
	}
}

// Unit 15: New(Policy{}) with an EMPTY policy => VerdictDeny / ReasonDefaultDeny.
func TestEmptyPolicyDefaultDeny(t *testing.T) {
	a := New(Policy{})
	v := a.Adjudicate(context.Background(), inlineCall("anything_at_all", `{}`))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("empty policy: got Kind=%v, want VerdictDeny", v.Kind)
	}
	if v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("empty policy: got Reason=%v (%s), want ReasonDefaultDeny",
			v.Reason, abi.ReasonName(v.Reason))
	}
}

// Unit 15 (cont): an UNKNOWN tool under DefaultPolicy => Deny / DefaultDeny.
func TestDefaultPolicyUnknownToolDefaultDeny(t *testing.T) {
	a := New(DefaultPolicy())
	v := a.Adjudicate(context.Background(), inlineCall("totally_unknown_tool", `{}`))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("unknown tool: got Kind=%v, want VerdictDeny", v.Kind)
	}
	if v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("unknown tool: got Reason=%v (%s), want ReasonDefaultDeny",
			v.Reason, abi.ReasonName(v.Reason))
	}
}

func TestAdmitAndLogPostureAllowsOnlyReadShapedDefaultDeny(t *testing.T) {
	a := New(Policy{
		Posture: PostureAdmitAndLog,
		Deny: map[string]abi.ReasonCode{
			"exfiltrate": abi.ReasonSecretExfil,
		},
		SelfModifyGlobs: []string{"internal/abi/"},
	})
	ctx := context.Background()

	v := a.Adjudicate(ctx, inlineCall("read_report", `{}`))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("read-shaped default deny: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v.Meta["posture"] != "admit_and_log" || v.Meta["would_deny"] != "DEFAULT_DENY" {
		t.Fatalf("admit-and-log metadata = %v, want posture + would_deny", v.Meta)
	}
	if v.Reason != abi.ReasonNone {
		t.Fatalf("admitted read should not carry a refusal reason, got %s", abi.ReasonName(v.Reason))
	}

	if v := a.Adjudicate(ctx, inlineCall("delete_account", `{}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("write-shaped default deny: got %v/%s, want Deny/DEFAULT_DENY", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := a.Adjudicate(ctx, inlineCall("exfiltrate", `{}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSecretExfil {
		t.Fatalf("explicit deny: got %v/%s, want Deny/SECRET_EXFIL", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := a.Adjudicate(ctx, inlineCall("write_file", `{"path":"internal/abi/types.go"}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
		t.Fatalf("self-modify deny: got %v/%s, want Deny/SELF_MODIFY", v.Kind, abi.ReasonName(v.Reason))
	}
}

// Unit 20: a write-shaped call whose target matches a SelfModifyGlob =>
// VerdictDeny, Reason ReasonSelfModify, with a bounded-disclosure WitnessPayload
// whose Claim is EXACTLY the offending glob (nothing else).
func TestSelfModifyDeniedWithBoundedWitness(t *testing.T) {
	a := New(DefaultPolicy())
	// write_file is write-shaped ("write"); path contains the first glob
	// "internal/abi/" in SelfModifyGlobs, so matchGlob returns that exact glob.
	v := a.Adjudicate(context.Background(),
		inlineCall("write_file", `{"path":"internal/abi/types.go"}`))

	if v.Kind != abi.VerdictDeny {
		t.Fatalf("self-modify: got Kind=%v, want VerdictDeny", v.Kind)
	}
	if v.Reason != abi.ReasonSelfModify {
		t.Fatalf("self-modify: got Reason=%v (%s), want ReasonSelfModify",
			v.Reason, abi.ReasonName(v.Reason))
	}
	wp, ok := v.Payload.(abi.WitnessPayload)
	if !ok {
		t.Fatalf("self-modify: Payload type = %T, want abi.WitnessPayload", v.Payload)
	}
	// Bounded disclosure: the witness carries ONLY the offending glob string.
	const wantGlob = "internal/abi/"
	if wp.Claim != wantGlob {
		t.Fatalf("witness Claim: got %q, want exactly %q (bounded disclosure)",
			wp.Claim, wantGlob)
	}
	// The glob must be one of the policy's declared SelfModifyGlobs.
	found := false
	for _, g := range DefaultPolicy().SelfModifyGlobs {
		if g == wp.Claim {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("witness Claim %q is not one of SelfModifyGlobs %v",
			wp.Claim, DefaultPolicy().SelfModifyGlobs)
	}
}

// Unit 18: RedactFields triggers a VerdictTransform; the new args (re-stored as a
// non-inline Ref via putJSON) resolve to args with "password" => "[REDACTED]".
func TestRedactTransform(t *testing.T) {
	a := New(Policy{
		Allow:        map[string]bool{"do_thing": true},
		RedactFields: []string{"password"},
	})
	ctx := context.Background()
	v := a.Adjudicate(ctx, inlineCall("do_thing", `{"password":"hunter2","keep":"me"}`))

	if v.Kind != abi.VerdictTransform {
		t.Fatalf("redact: got Kind=%v, want VerdictTransform", v.Kind)
	}
	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		t.Fatalf("redact: Payload type = %T, want abi.TransformPayload", v.Payload)
	}

	res := abi.ActiveResolver()
	if res == nil {
		t.Fatal("ActiveResolver() is nil: blob backend must be registered (blank import)")
	}
	b, err := res.Resolve(ctx, tp.NewArgs)
	if err != nil {
		t.Fatalf("resolve transformed args: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal transformed args: %v", err)
	}
	if got["password"] != "[REDACTED]" {
		t.Fatalf("password: got %v, want %q", got["password"], "[REDACTED]")
	}
	// Untouched field is preserved.
	if got["keep"] != "me" {
		t.Fatalf("keep: got %v, want %q", got["keep"], "me")
	}
}

func TestArgAllowGlobDeniesAllowedToolOutsideBound(t *testing.T) {
	a := New(Policy{
		Allow: map[string]bool{"write_file": true},
		ArgPredicates: []ArgPredicate{{
			Tool: "write_file", Arg: "path", Kind: ArgAllowGlob,
			Glob: "./out/**", Reason: abi.ReasonPolicyBlock,
		}},
	})
	ctx := context.Background()

	if v := a.Adjudicate(ctx, inlineCall("write_file", `{"path":"./out/report.txt"}`)); v.Kind != abi.VerdictAllow {
		t.Fatalf("write under ./out: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	v := a.Adjudicate(ctx, inlineCall("write_file", `{"path":"./out/../secrets.txt"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("path escape: got %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}
	wp, ok := v.Payload.(abi.WitnessPayload)
	if !ok || !strings.Contains(wp.Claim, "write_file.path allow_glob ./out/**") {
		t.Fatalf("bounded witness = %+v, want predicate identity", v.Payload)
	}
	if strings.Contains(wp.Claim, "secrets") {
		t.Fatalf("bounded witness leaked arg value: %q", wp.Claim)
	}
	if v := a.Adjudicate(ctx, inlineCall("write_file", `{}`)); v.Kind != abi.VerdictDeny {
		t.Fatalf("missing constrained arg: got %v, want Deny", v.Kind)
	}
}

func TestArgDenyRegexDeniesAllowedShellCommand(t *testing.T) {
	a := New(Policy{
		Allow: map[string]bool{"run_shell": true},
		ArgPredicates: []ArgPredicate{{
			Tool: "run_shell", Arg: "cmd", Kind: ArgDenyRegex,
			Re: regexp.MustCompile(`rm|push --force`), Reason: abi.ReasonPolicyBlock,
		}},
	})
	ctx := context.Background()

	if v := a.Adjudicate(ctx, inlineCall("run_shell", `{"cmd":"git status --short"}`)); v.Kind != abi.VerdictAllow {
		t.Fatalf("benign shell cmd: got %v/%s, want Allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if v := a.Adjudicate(ctx, inlineCall("run_shell", `{"cmd":"git push --force origin main"}`)); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("dangerous shell cmd: got %v/%s, want Deny/POLICY_BLOCK", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestArgPredicatesAreRestrictOnly(t *testing.T) {
	a := New(Policy{
		ArgPredicates: []ArgPredicate{{
			Tool: "write_file", Arg: "path", Kind: ArgAllowGlob,
			Glob: "./out/**", Reason: abi.ReasonPolicyBlock,
		}},
	})
	v := a.Adjudicate(context.Background(), inlineCall("write_file", `{"path":"./out/report.txt"}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("satisfied predicate must not grant allow: got %v/%s, want Deny/DEFAULT_DENY", v.Kind, abi.ReasonName(v.Reason))
	}
}

// Unit 19: every reason this package emits is in the CLOSED core vocabulary —
// ReasonName never falls through to the REASON_<n> forward-compat rendering.
func TestReasonsAreInClosedVocab(t *testing.T) {
	emitted := []abi.Verdict{
		New(Policy{}).Adjudicate(context.Background(), inlineCall("x", `{}`)),
		New(DefaultPolicy()).Adjudicate(context.Background(), inlineCall("unknown_tool", `{}`)),
		New(DefaultPolicy()).Adjudicate(context.Background(),
			inlineCall("write_file", `{"path":"internal/abi/types.go"}`)),
		// Explicit deny rules from DefaultPolicy.
		New(DefaultPolicy()).Adjudicate(context.Background(), inlineCall("shell_rm_rf", `{}`)),
		New(DefaultPolicy()).Adjudicate(context.Background(), inlineCall("exfiltrate", `{}`)),
	}
	for i, v := range emitted {
		if v.Kind != abi.VerdictDeny {
			continue // only deny verdicts carry a refusal reason
		}
		name := abi.ReasonName(v.Reason)
		if name == "" {
			t.Fatalf("verdict[%d]: empty reason name for code %d", i, v.Reason)
		}
		if len(name) >= len("REASON_") && name[:len("REASON_")] == "REASON_" {
			t.Fatalf("verdict[%d]: reason %d rendered as %q — NOT in the closed vocab",
				i, v.Reason, name)
		}
	}
}

// Units 21,22: a p50<1ms witness — time 1000 Adjudicate calls on an allowed call
// and assert mean and median per-call latency are well under 1ms.
func TestAdjudicateP50UnderOneMillisecond(t *testing.T) {
	const n = 1000
	ctx := context.Background()
	call := inlineCall("get_user_details", `{}`)

	samples := make([]time.Duration, 0, n)
	var total time.Duration
	for i := 0; i < n; i++ {
		start := time.Now()
		v := Default.Adjudicate(ctx, call)
		d := time.Since(start)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("iteration %d: got Kind=%v, want VerdictAllow", i, v.Kind)
		}
		samples = append(samples, d)
		total += d
	}

	mean := total / n

	// median (p50)
	for i := 1; i < len(samples); i++ {
		for j := i; j > 0 && samples[j-1] > samples[j]; j-- {
			samples[j-1], samples[j] = samples[j], samples[j-1]
		}
	}
	median := samples[len(samples)/2]

	if mean >= time.Millisecond {
		t.Fatalf("mean per-call latency %v >= 1ms", mean)
	}
	if median >= time.Millisecond {
		t.Fatalf("median (p50) per-call latency %v >= 1ms", median)
	}
	t.Logf("Adjudicate over %d calls: mean=%v p50=%v", n, mean, median)
}

// #172 Hole 2: the self-modify guard must cover the WITNESS machinery — the
// trees that, if self-edited, would let an RSI loop grade its own homework.
// This is the lane-local floor for the glob-coverage slice (the cross-package
// architest gate stays separate); it asserts both deployable policies guard the
// witness trees, and that a write into a witness tree is denied SELF_MODIFY.
func TestSelfModifyGuardsWitnessMachinery(t *testing.T) {
	// Each witness tree paired with a sample write target inside it. The targets
	// match by substring (matchGlob), the same way Adjudicate matches at runtime.
	witness := map[string]string{
		"internal/adjudicator/": "internal/adjudicator/decide.go",
		"internal/architest/":   "internal/architest/floor_test.go",
		"internal/shipgate/":    "internal/shipgate/gate.go",
		"dos.toml":              "dos.toml",
	}
	for _, p := range []struct {
		name   string
		policy Policy
	}{
		{"DefaultPolicy", DefaultPolicy()},
		{"DevAgentPolicy", DevAgentPolicy()},
	} {
		a := New(p.policy)
		for tree, target := range witness {
			// Glob-set coverage: some witness tree glob must match the target.
			if matchGlob(target, p.policy.SelfModifyGlobs) == "" {
				t.Errorf("%s: SelfModifyGlobs does not cover witness tree %q (no glob matches %q)",
					p.name, tree, target)
			}
			// End-to-end: a write into the witness tree is denied SELF_MODIFY.
			v := a.Adjudicate(context.Background(),
				inlineCall("write_file", `{"path":"`+target+`"}`))
			if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonSelfModify {
				t.Errorf("%s: write to %q got %v/%s, want Deny/SELF_MODIFY",
					p.name, target, v.Kind, abi.ReasonName(v.Reason))
			}
		}
	}
}

// #172 Hole 1: the self-modify guard must also cover the SHELL path. A Bash/exec
// tool carries its write target inside the `command` string, not a `path` arg, and
// its tool NAME contains no write-word — so the file-write guard never fires for it.
// A shell WRITE into a guarded tree must be denied SELF_MODIFY; a shell READ of a
// guarded file, and a write OUTSIDE the guarded trees, must stay allowed (only the
// write-into-guarded-tree case is the self-edit the floor exists to stop).
func TestSelfModifyGuardsShellWritePath(t *testing.T) {
	a := New(DefaultPolicy())
	cases := []struct {
		name    string
		command string
		wantDup bool // want Deny/SELF_MODIFY
	}{
		// Shell WRITES into a guarded tree → SELF_MODIFY (the acceptance criterion:
		// "a Bash command writing into internal/abi/ is denied").
		{"sed -i into abi", "sed -i 's/x/y/' fak/internal/abi/kernel.go", true},
		{"tee into kernel", "echo hi | tee fak/internal/kernel/x.go", true},
		{"redirect into dos.toml", "echo '[x]' > dos.toml", true},
		{"git apply into adjudicator", "git apply < /tmp/p.patch fak/internal/adjudicator/decide.go", true},
		{"git checkout guarded", "git checkout -- internal/shipgate/gate.go", true},
		{"cp over architest", "cp /tmp/evil.go internal/architest/floor_test.go", true},
		{"append into .dos", "echo x >> .dos/state", true},
		{"rm a guarded file", "rm fak/internal/abi/kernel.go", true},
		{"shred a guarded file", "shred -u fak/internal/abi/kernel.go", true},
		// `find … -delete` / `find … -exec` into a guarded tree → SELF_MODIFY: a
		// delete/exec-write idiom that carries none of the shellWriteVerbs tokens,
		// the porous-denylist `find -delete` gap (#172 Hole 1). Without the find
		// rung these would launder a deletion/rewrite of a guarded tree past the floor.
		{"find -delete in abi", "find fak/internal/abi -name '*.go' -delete", true},
		{"find -exec rm in kernel", "find fak/internal/kernel -name '*.go' -exec rm {} +", true},
		// `perl -i`/`ruby -i` in-place edits into a guarded tree → SELF_MODIFY: the
		// sed -i family across other interpreters (#172 Hole 1 residual). They rewrite
		// the file exactly as `sed -i` does but carry none of its leading token.
		{"perl -i into abi", "perl -i -pe 's/x/y/' fak/internal/abi/kernel.go", true},
		{"ruby -i into kernel", "ruby -i -pe 'gsub(/x/,\"y\")' fak/internal/kernel/x.go", true},
		// `git -C <guarded-tree> <mutating-subcommand>` → SELF_MODIFY: the -C dir
		// operand makes the guarded tree the write target even though `reset`/`commit`
		// are not in shellWriteVerbs. `git -C fak/internal/abi reset --hard` rewrites
		// the guarded tree from history past the file-write tokens caught above.
		{"git -C reset into abi", "git -C fak/internal/abi reset --hard HEAD", true},
		{"git -C clean into kernel", "git -C fak/internal/kernel clean -fd", true},
		// `python -c`/`node -e` inline programs that open a guarded file for writing →
		// SELF_MODIFY: a general-purpose interpreter rewrites the file from inside the
		// program string, carrying none of the shellWriteVerbs tokens and no `>`
		// redirect (#172 Hole 1 residual, the perl/ruby -i gap one interpreter out —
		// python/node/ruby are the runtimes most likely on a coding agent's PATH; the
		// `ruby -e` rows close the rulesynth-loop-surfaced asymmetry (`ruby -i` caught,
		// `ruby -e` eval-write slipped)).
		{"python -c write into abi", "python -c \"open('fak/internal/abi/kernel.go','w').write('x')\"", true},
		{"python3 -c write into kernel", "python3 -c \"open('fak/internal/kernel/x.go','w').write('x')\"", true},
		{"node -e write into adjudicator", "node -e \"require('fs').writeFileSync('fak/internal/adjudicator/decide.go','x')\"", true},
		{"node --eval write into dos.toml", "node --eval \"require('fs').writeFileSync('dos.toml','x')\"", true},
		{"ruby -e write into adjudicator", "ruby -e 'File.write(\"fak/internal/adjudicator/decide.go\", x)'", true},
		{"ruby --eval write into kernel", "ruby --eval 'IO.write(\"fak/internal/kernel/x.go\", x)'", true},
		// `awk -i inplace`/`gawk -i inplace` in-place edits into a guarded tree →
		// SELF_MODIFY: GNU awk's in-place flag rewrites the file exactly as `sed -i`/
		// `perl -i`/`ruby -i` do (#172 Hole 1 residual, the sed -i family one tool out —
		// `awk` carries no `-i` token in shellWriteVerbs). A read-only `awk '{print}'`
		// has no `-i inplace` and stays allowed (the negative case below).
		{"gawk -i inplace into abi", "gawk -i inplace '{gsub(/x/,\"y\")}1' fak/internal/abi/kernel.go", true},
		{"awk -i inplace into kernel", "awk -i inplace '{print}' fak/internal/kernel/x.go", true},
		// `ed`/`ex` scripted line editors that WRITE their file operand into a guarded
		// tree → SELF_MODIFY: a write idiom carrying no redirect and no caught verb
		// (`ex -s -c wq file`, `ed -s file`). Matched only in COMMAND POSITION so they
		// never trip on `sed`/`indexed`/`expr` (the negatives below).
		{"ex scripted write into abi", "ex -s -c 'wq' fak/internal/abi/kernel.go", true},
		{"ed scripted write into kernel", "ed -s fak/internal/kernel/x.go", true},
		{"ed after a separator into shipgate", "true; ed internal/shipgate/gate.go", true},
		// Archive EXTRACTION / sync INTO a guarded tree → SELF_MODIFY: an extractor
		// writes files into its destination with none of the tokens above (#172 Hole 1
		// residual). tar EXTRACT mode (-x / bundled x), unzip's default (non-list) mode,
		// and rsync's destination all overwrite the guarded tree. The READ idioms over
		// the same tree (tar LIST/CREATE, unzip LIST/test) stay allowed — the negatives.
		{"tar extract into abi", "tar -xf evil.tar -C fak/internal/abi", true},
		{"tar bundled extract into kernel", "tar xzf evil.tgz -C fak/internal/kernel", true},
		{"tar --extract into abi", "tar --extract -f e.tar -C fak/internal/abi", true},
		{"unzip into abi", "unzip -o evil.zip -d fak/internal/abi", true},
		{"unzip default into kernel", "unzip evil.zip -d fak/internal/kernel", true},
		{"rsync into abi", "rsync -a evil/ fak/internal/abi/", true},
		// A read-only `git -C <guarded> status`/`log` → NOT a write, stays allowed.
		{"git -C status read", "git -C fak/internal/abi status", false},
		{"git -C log read", "git -C fak/internal/kernel log --oneline", false},
		// An interpreter run WITHOUT an inline-eval flag → not matched as a write by
		// the interpreter rung (`python score.py` / `node app.js` over a guarded tree
		// is a script run, not an inline self-edit; a real write inside it would still
		// have to use a redirect/verb the other rungs catch). Stays allowed here.
		{"python script run reads guarded", "python fak/internal/abi/gen.py", false},
		{"node script run reads guarded", "node fak/internal/kernel/build.js", false},
		// An inline interpreter program that touches NO guarded tree → allowed (the
		// guard only fires once a guarded glob is named in the command).
		{"python -c write outside guarded", "python3 -c \"open('README.md','w').write('x')\"", false},
		{"ruby -e write outside guarded", "ruby -e 'File.write(\"README.md\", x)'", false},
		{"ruby -e pure compute, no path", "ruby -e 'puts 1+1'", false},
		// A read-only `find` over a guarded tree (no -delete/-exec) → NOT a write,
		// stays allowed by this guard (mirrors the cat/grep read cases above).
		{"find read-only in abi", "find fak/internal/abi -name '*.go'", false},
		// A read-only `awk`/`gawk` over a guarded tree (no `-i inplace`) → NOT a write,
		// stays allowed (only the in-place flag is the self-edit).
		{"awk read-only in abi", "awk '{print $1}' fak/internal/abi/kernel.go", false},
		{"gawk read-only in kernel", "gawk '{print}' fak/internal/kernel/x.go", false},
		// The `ed`/`ex` command-position guard must NOT trip on a substring or an
		// argument: `sed`/`indexed`/`expr` contain the letters but are not the editor,
		// and `grep ed <guarded>` searches for the word `ed` as an argument — all reads.
		{"sed without -i is not ed", "sed -n '1p' fak/internal/abi/kernel.go", false},
		{"grep for the word ed is not ed", "grep ed fak/internal/abi/kernel.go", false},
		{"expr substring not ex", "expr 1 + 1 # fak/internal/abi/kernel.go", false},
		// Archive READS over a guarded tree → NOT a write, stay allowed. tar LIST (-t)
		// and CREATE (-c, which reads the tree into an archive elsewhere) and unzip
		// LIST/test (-l/-t) are reads; and `tar`/`unzip` as a SUBSTRING of another
		// command name (`star`, `mostar`) must not trip the command-leading guard.
		{"tar list reads abi", "tar -tf archive.tar fak/internal/abi", false},
		{"tar create reads abi", "tar -czf out.tgz fak/internal/abi", false},
		{"tar create bundled reads abi", "tar czf out.tgz fak/internal/abi", false},
		{"unzip list reads abi", "unzip -l evil.zip fak/internal/abi", false},
		{"unzip test reads abi", "unzip -t evil.zip fak/internal/abi", false},
		{"tar substring star not tar", "star fak/internal/abi", false},
		{"tar substring mostar not tar", "mostar -x fak/internal/abi", false},
		// Shell READS of a guarded file → NOT a self-modify (stays allowed by this
		// guard; a separate arg-deny rule may still apply, but not SELF_MODIFY here).
		{"cat a guarded file", "cat fak/internal/abi/kernel.go", false},
		{"grep a guarded tree", "grep -r foo fak/internal/kernel/", false},
		// Writes OUTSIDE the guarded trees → allowed by this guard.
		{"write a normal file", "sed -i 's/a/b/' README.md", false},
		{"redirect to a normal path", "echo hi > fak/docs/notes.md", false},
		// No command arg at all → guard does not fire.
		{"no command arg", "", false},
	}
	for _, c := range cases {
		args := `{"command":"` + jsonEscape(c.command) + `"}`
		if c.command == "" {
			args = `{}`
		}
		v := a.Adjudicate(context.Background(), inlineCall("Bash", args))
		gotDup := v.Kind == abi.VerdictDeny && v.Reason == abi.ReasonSelfModify
		if gotDup != c.wantDup {
			t.Errorf("%s: command %q got %v/%s, wantSelfModify=%v",
				c.name, c.command, v.Kind, abi.ReasonName(v.Reason), c.wantDup)
		}
	}
}

// jsonEscape minimally escapes a command string for inline JSON test args (only
// the characters the test commands actually contain: backslash and double-quote;
// the single-quotes and redirects are JSON-safe as-is).
func jsonEscape(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	return strings.ReplaceAll(s, `"`, `\"`)
}

// Units 21,22: BenchmarkDecide times Default.Adjudicate on an allowed call.
func BenchmarkDecide(b *testing.B) {
	ctx := context.Background()
	call := inlineCall("get_user_details", `{}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := Default.Adjudicate(ctx, call)
		if v.Kind != abi.VerdictAllow {
			b.Fatalf("got Kind=%v, want VerdictAllow", v.Kind)
		}
	}
}
