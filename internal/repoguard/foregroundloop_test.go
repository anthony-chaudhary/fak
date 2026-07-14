package repoguard

import (
	"strings"
	"testing"
)

func TestClassifyForegroundNetworkLoopFlagsPerItemFanout(t *testing.T) {
	cases := []struct {
		cmd    string
		wantOp string
	}{
		// The #4595 audit form: a for-loop viewing N issues one API call at a time.
		{"for n in 3009 3010 3011; do gh issue view $n; done", "gh"},
		// `do` glued to the body command in one segment.
		{"for n in 1 2 3; do gh issue view $n --json state; done", "gh"},
		// The list-generating gh in the HEADER is fine; only the body call fires.
		{"for pr in $(gh pr list --json number -q '.[].number'); do gh pr view $pr; done", "gh"},
		// curl per item.
		{"for u in a b c; do curl -s https://x/$u; done", "curl"},
		// wget per item.
		{"for f in 1 2; do wget https://x/$f; done", "wget"},
		// git network subcommand per item.
		{"for r in a b c; do git fetch $r; done", "git fetch"},
		{"for r in a b; do git push origin $r; done", "git push"},
		// env-prefixed body verb still reaches the network call.
		{"for n in 1 2; do GH_TOKEN=x gh issue view $n; done", "gh"},
		// C-style for header.
		{"for ((i=0;i<10;i++)); do curl -s host/$i; done", "curl"},
	}
	for _, tc := range cases {
		vs := ClassifyForegroundNetworkLoop(tc.cmd)
		if len(vs) != 1 {
			t.Errorf("ClassifyForegroundNetworkLoop(%q) = %d violations, want 1: %+v", tc.cmd, len(vs), vs)
			continue
		}
		if vs[0].Reason != ReasonForegroundNetworkLoop {
			t.Errorf("ClassifyForegroundNetworkLoop(%q) reason = %q, want %q", tc.cmd, vs[0].Reason, ReasonForegroundNetworkLoop)
		}
		if vs[0].Op != tc.wantOp {
			t.Errorf("ClassifyForegroundNetworkLoop(%q) op = %q, want %q", tc.cmd, vs[0].Op, tc.wantOp)
		}
		if !strings.Contains(vs[0].Fix, "run_in_background") {
			t.Errorf("ClassifyForegroundNetworkLoop(%q) fix should name the background alternative: %q", tc.cmd, vs[0].Fix)
		}
	}
}

func TestClassifyForegroundNetworkLoopPasses(t *testing.T) {
	cases := []string{
		// A single batched call — the form we steer toward — is not a loop.
		"gh issue list --json number,title --limit 100",
		"gh api --paginate repos/o/r/issues",
		// A for-loop with no network call in the body.
		"for f in *.go; do gofmt -w $f; done",
		"for n in 1 2 3; do echo $n; done",
		// A git loop over LOCAL subcommands is not a network round trip.
		"for c in a b c; do git show --stat $c; done",
		// The network call is BEFORE the loop, not in its body.
		"gh issue list --json number; for n in 1 2; do echo $n; done",
		// Backgrounded loop does not hold the turn.
		"for n in 1 2 3; do gh issue view $n; done &",
		// A quoted loop-looking string inside a local command is not a loop.
		"grep -r 'for n in 1 2; do gh issue view' .",
		"echo 'for x in a; do gh view; done'",
		// while/until deliberately out of scope (retry/backoff/stream, not a fan-out).
		"until gh api rate_limit; do sleep 5; done",
		"while read n; do gh issue view $n; done < list.txt",
	}
	for _, cmd := range cases {
		if vs := ClassifyForegroundNetworkLoop(cmd); len(vs) != 0 {
			t.Errorf("ClassifyForegroundNetworkLoop(%q) = %+v, want none", cmd, vs)
		}
	}
}

func TestEvaluateWiresForegroundNetworkLoopForBashOnly(t *testing.T) {
	cmd := "for n in 1 2 3; do gh issue view $n; done"
	vs := Evaluate("Bash", map[string]any{"command": cmd}, "C:/w/fak", nil)
	found := false
	for _, v := range vs {
		if v.Reason == ReasonForegroundNetworkLoop {
			found = true
		}
	}
	if !found {
		t.Errorf("Evaluate(Bash, %q) = %+v, want a FOREGROUND_NETWORK_LOOP finding", cmd, vs)
	}
	// PowerShell parses foreach differently and only gets the sleep rung, so the
	// POSIX for-loop classifier must not fire there.
	if vs := Evaluate("PowerShell", map[string]any{"command": cmd}, "C:/w/fak", nil); len(vs) != 0 {
		t.Errorf("Evaluate(PowerShell, %q) = %+v, want none (Bash-only rung)", cmd, vs)
	}
}

func TestRenderReasonIncludesNetworkLoopBlock(t *testing.T) {
	vs := ClassifyForegroundNetworkLoop("for n in 1 2 3; do gh issue view $n; done")
	reason := RenderReason(vs)
	if !strings.Contains(reason, ReasonForegroundNetworkLoop) || !strings.Contains(reason, "run_in_background") {
		t.Fatalf("RenderReason(network loop) = %q, want the FOREGROUND_NETWORK_LOOP block with the fix", reason)
	}
}

func TestDefaultSeverityForegroundNetworkLoopWarns(t *testing.T) {
	if got := DefaultSeverity(ReasonForegroundNetworkLoop); got != SeverityWarn {
		t.Errorf("DefaultSeverity(%s) = %v, want warn", ReasonForegroundNetworkLoop, got)
	}
}
