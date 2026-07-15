package adjudicator

import "testing"

// Regression witnesses for fak-private#21: the reversibility preview-confirm gate
// kept re-issuing a FRESH one-shot token for an explicitly-approved outward-facing
// publish instead of letting the byte-identical command execute. #2777 removed the
// free-text "description" and cosmetic whitespace from the token hash; the RESIDUAL
// axis this issue reported is the client SUPERVISION knobs (how long the caller waits,
// whether it detaches), which a model routinely nudges between attempts — nudging the
// timeout after a slow/timed-out publish over a flaky remote bridge drew a new token,
// so the operator's repeated "go" never became executable.
//
// The three arms map to the issue's done-conditions: one-time binding (the token
// tracks exactly the effect-bearing command, so an incidental re-propose converges),
// expiry/mismatch (a confirm for one call is refused on any materially different call
// and a forged/stale token never confirms), and the successful explicitly-approved
// publication command executing on re-propose.

// outwardPublishPost is an outward-facing publication command that reaches the gate (a
// curl POST to a remote page API — the http-write family). The host is a placeholder:
// the family matches on the write method, not the destination, so no real internal
// endpoint is embedded. The gate holds it for a preview the operator acknowledges.
const outwardPublishPost = "curl -X POST -H \"Authorization: Bearer $PUBLISH_TOKEN\" " +
	"https://pages.internal.example/rest/api/content --data-binary @page.json"

// pagePush is the git-push step of a publish path (commit the page and push to origin);
// the push is the outward-facing step the gate holds.
const pagePush = "git push origin main"

func mustHoldP21(t *testing.T, tool string, args map[string]any) ReversibilityEnvelope {
	t.Helper()
	env, ok := ReversibilityConfirmed(tool, args)
	if ok {
		t.Fatalf("expected an outward-facing HOLD, but the call was confirmed unprompted: %+v", env)
	}
	if env.Class == ReversibilityReversible || env.ConfirmToken == "" {
		t.Fatalf("expected a gated class with a confirm token, got %+v", env)
	}
	return env
}

// --- ARM 1: one-time binding — an incidental supervision re-propose converges. ---

func TestP21_SupervisionKnobsDoNotRotateToken(t *testing.T) {
	for _, tc := range []struct {
		name       string
		tool       string
		first      map[string]any
		reproposed map[string]any // same command, drifted supervision knob(s)
	}{
		{
			name:       "curl publish: timeout bumped after a slow first attempt",
			tool:       "shell_command",
			first:      map[string]any{"command": outwardPublishPost, "timeout": float64(120000)},
			reproposed: map[string]any{"command": outwardPublishPost, "timeout": float64(600000)},
		},
		{
			name:       "curl publish: timeout_ms spelling (Codex shell_command) drifts",
			tool:       "shell_command",
			first:      map[string]any{"command": outwardPublishPost, "timeout_ms": float64(120000)},
			reproposed: map[string]any{"command": outwardPublishPost, "timeout_ms": float64(300000)},
		},
		{
			name:       "page push: timeout dropped entirely on the re-proposal",
			tool:       "shell_command",
			first:      map[string]any{"command": pagePush, "timeout": float64(120000)},
			reproposed: map[string]any{"command": pagePush},
		},
		{
			name:       "page push: flipped to background to escape the foreground cap",
			tool:       "shell_command",
			first:      map[string]any{"command": pagePush, "run_in_background": false},
			reproposed: map[string]any{"command": pagePush, "run_in_background": true},
		},
		{
			name:       "page push: description AND timeout drift together (real Claude Code shape)",
			tool:       "Bash",
			first:      map[string]any{"command": pagePush, "description": "publish the page", "timeout": float64(120000)},
			reproposed: map[string]any{"command": pagePush, "description": "push the page now", "timeout": float64(600000)},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t1 := ReversibilityConfirmToken(ReversibilityOutwardFacing, tc.tool, tc.first)
			t2 := ReversibilityConfirmToken(ReversibilityOutwardFacing, tc.tool, tc.reproposed)
			if t1 != t2 {
				t.Fatalf("confirm token rotated on a supervision knob (fak-private#21): %q != %q", t1, t2)
			}
			// End to end: first proposal is held and issues a token; echoing it on the
			// drifted re-proposal must CONFIRM, not draw a fresh refusal.
			env := mustHoldP21(t, tc.tool, tc.first)
			retry := map[string]any{ReversibilityConfirmArg: env.ConfirmToken}
			for k, v := range tc.reproposed {
				retry[k] = v
			}
			if _, ok := ReversibilityConfirmed(tc.tool, retry); !ok {
				t.Fatalf("supervision-drifted re-propose was refused — the fak-private#21 loop is not fixed")
			}
		})
	}
}

// --- ARM 2: expiry / mismatch — a confirm never transplants onto another call. ---

func TestP21_TokenBindsEffectBearingArgsAndRejectsMismatch(t *testing.T) {
	base := map[string]any{"command": pagePush, "timeout": float64(120000)}
	baseToken := ReversibilityConfirmToken(ReversibilityOutwardFacing, "shell_command", base)

	for _, tc := range []struct {
		name string
		tool string
		args map[string]any
	}{
		{"force flag added", "shell_command",
			map[string]any{"command": "git push --force origin main", "timeout": float64(120000)}},
		{"explicit refspec added", "shell_command",
			map[string]any{"command": "git push origin HEAD:main", "timeout": float64(120000)}},
		{"a different remote", "shell_command",
			map[string]any{"command": "git push public main", "timeout": float64(120000)}},
		{"a different outward command entirely", "shell_command",
			map[string]any{"command": outwardPublishPost, "timeout": float64(120000)}},
		{"sandbox disabled — effect-bearing, not supervision", "shell_command",
			map[string]any{"command": pagePush, "timeout": float64(120000), "dangerouslyDisableSandbox": true}},
		{"a different tool proposing identical text", "Bash",
			map[string]any{"command": pagePush, "timeout": float64(120000)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := ReversibilityConfirmToken(ReversibilityOutwardFacing, tc.tool, tc.args); got == baseToken {
				t.Fatalf("token failed to bind (%s): a confirm for %q would transplant onto it", tc.name, pagePush)
			}
		})
	}

	// A structured MCP git_push must still bind its danger flags: the supervision
	// exclusion must not launder a force push past a confirm taken for a bare one.
	bare := map[string]any{"timeout": float64(120000)}
	forced := map[string]any{"timeout": float64(120000), "force": true}
	if ReversibilityConfirmToken(ReversibilityOutwardFacing, "git_push", bare) ==
		ReversibilityConfirmToken(ReversibilityOutwardFacing, "git_push", forced) {
		t.Fatalf("token ignored an MCP force flag — binding weakened")
	}

	// End to end: a token issued for the bare push is REFUSED on a force push, a
	// forged token never confirms, and the matching token still confirms (a pause,
	// not a wall).
	env := mustHoldP21(t, "shell_command", map[string]any{"command": pagePush})
	if _, ok := ReversibilityConfirmed("shell_command", map[string]any{
		"command": "git push --force origin main", ReversibilityConfirmArg: env.ConfirmToken,
	}); ok {
		t.Fatalf("a confirm issued for %q was transplanted onto a force push", pagePush)
	}
	if _, ok := ReversibilityConfirmed("shell_command", map[string]any{
		"command": pagePush, ReversibilityConfirmArg: "fak-0000000000000000",
	}); ok {
		t.Fatalf("a forged confirm token was accepted")
	}
	if _, ok := ReversibilityConfirmed("shell_command", map[string]any{
		"command": pagePush, ReversibilityConfirmArg: env.ConfirmToken,
	}); !ok {
		t.Fatalf("the matching confirm token was refused — the gate became a wall")
	}
}

// --- ARM 3: the explicitly-approved outward publication executes on re-propose. ---

func TestP21_ApprovedOutwardPublishBecomesExecutable(t *testing.T) {
	// The operator approved the publish; the model issues the curl POST with a first
	// timeout. The gate holds it and hands back a token.
	first := map[string]any{"command": outwardPublishPost, "timeout": float64(120000)}
	env := mustHoldP21(t, "shell_command", first)
	if env.Class != ReversibilityOutwardFacing {
		t.Fatalf("outward publish should preview as outward-facing, got %q", env.Class)
	}

	// The publish over the flaky bridge is slow, so on the operator's repeated "go"
	// the model re-proposes the byte-identical command but bumps the timeout and
	// detaches it — and echoes the issued token. Before the fix this drew a fresh
	// refusal (the loop); now it must confirm and become executable.
	approved := map[string]any{
		"command":               outwardPublishPost,
		"timeout":               float64(600000),
		"run_in_background":     true,
		ReversibilityConfirmArg: env.ConfirmToken,
	}
	if _, ok := ReversibilityConfirmed("shell_command", approved); !ok {
		t.Fatalf("explicitly-approved outward publish did not become executable on re-propose (fak-private#21)")
	}
}
