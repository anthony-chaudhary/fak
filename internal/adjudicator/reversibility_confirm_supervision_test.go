package adjudicator

import "testing"

// The DA33 Confluence publication path (fak-private#21) publishes a page by
// committing it under confluence-helpers/pages/ and pushing to origin; the push is
// the step the outward-facing gate holds. These are the regression witnesses for the
// residual axis of the confirm-loop that blocked it.

// da33PublishPush is the effect-bearing command of the gated step in the DA33
// Confluence publication path: the page push whose preview the operator acknowledges.
const da33PublishPush = "git push origin main"

// TestReversibilityConfirmTokenIgnoresClientSupervisionDrift is the regression
// witness for fak-private#21. The gate advertises "re-propose it byte-identical —
// same tool and same command text; the free-text description need not match"
// (internal/gateway/reversibility_note.go), but the token hashed EVERY non-annotation
// argument — including the client supervision knobs a model routinely nudges between
// attempts. Re-proposing the identical command with a bumped `timeout` (the ordinary
// reaction to a slow call) drew a FRESH token, so the advertised recovery could not
// converge and the operator's repeated approval never became executable.
//
// A supervision knob says how the CALLER WAITS, not what the call DOES: the same
// commits reach the same remote at any timeout, foreground or background. So it must
// not rotate the token.
func TestReversibilityConfirmTokenIgnoresClientSupervisionDrift(t *testing.T) {
	for _, tc := range []struct {
		name  string
		first map[string]any
		again map[string]any
	}{
		{
			name:  "timeout bumped after a slow first attempt",
			first: map[string]any{"command": da33PublishPush, "timeout": float64(120000)},
			again: map[string]any{"command": da33PublishPush, "timeout": float64(180000)},
		},
		{
			name:  "timeout dropped entirely on the re-proposal",
			first: map[string]any{"command": da33PublishPush, "timeout": float64(120000)},
			again: map[string]any{"command": da33PublishPush},
		},
		{
			name:  "run_in_background flipped to escape the foreground cap",
			first: map[string]any{"command": da33PublishPush, "run_in_background": false},
			again: map[string]any{"command": da33PublishPush, "run_in_background": true},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t1 := ReversibilityConfirmToken(ReversibilityOutwardFacing, "shell_command", tc.first)
			t2 := ReversibilityConfirmToken(ReversibilityOutwardFacing, "shell_command", tc.again)
			if t1 != t2 {
				t.Fatalf("confirm token rotated on a client supervision knob: %q != %q", t1, t2)
			}

			// End to end: the first proposal is held and issues a token; the
			// re-proposal echoing it under drifted supervision must CONFIRM, not draw
			// a fresh refusal — the fak-private#21 loop.
			env, ok := ReversibilityConfirmed("shell_command", tc.first)
			if ok {
				t.Fatalf("unconfirmed outward-facing push was allowed: %+v", env)
			}
			if env.ConfirmToken == "" {
				t.Fatalf("gate issued no confirm token: %+v", env)
			}
			reproposed := map[string]any{ReversibilityConfirmArg: env.ConfirmToken}
			for k, v := range tc.again {
				reproposed[k] = v
			}
			if _, ok := ReversibilityConfirmed("shell_command", reproposed); !ok {
				t.Fatalf("confirm token rotated on supervision drift — the fak-private#21 loop is not fixed")
			}
		})
	}
}

// TestReversibilityConfirmTokenBindsEffectBearingArgs is the other half of the fix:
// excluding the supervision knobs must not let a confirm be transplanted onto a
// materially different call. Every argument that STEERS the effect still binds, so a
// token acknowledged for one action cannot approve another. This is the test that
// fails if a future exclusion is drawn too wide.
func TestReversibilityConfirmTokenBindsEffectBearingArgs(t *testing.T) {
	base := map[string]any{"command": da33PublishPush, "timeout": float64(120000)}
	baseToken := ReversibilityConfirmToken(ReversibilityOutwardFacing, "shell_command", base)

	for _, tc := range []struct {
		name string
		tool string
		args map[string]any
	}{
		{
			name: "a different command text",
			tool: "shell_command",
			args: map[string]any{"command": "git push --force origin main", "timeout": float64(120000)},
		},
		{
			name: "a different remote",
			tool: "shell_command",
			args: map[string]any{"command": "git push public main", "timeout": float64(120000)},
		},
		{
			name: "sandbox disabled — an effect-bearing knob, not a supervision one",
			tool: "shell_command",
			args: map[string]any{"command": da33PublishPush, "timeout": float64(120000), "dangerouslyDisableSandbox": true},
		},
		{
			name: "a different tool proposing the same text",
			tool: "Bash",
			args: map[string]any{"command": da33PublishPush, "timeout": float64(120000)},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := ReversibilityConfirmToken(ReversibilityOutwardFacing, tc.tool, tc.args); got == baseToken {
				t.Fatalf("confirm token failed to bind (%s): token transplantable from %q", tc.name, da33PublishPush)
			}
		})
	}

	// A structured MCP git_push keeps binding to its danger flags: the supervision
	// exclusion must not launder a force push past a confirm taken for a bare one.
	bare := map[string]any{"timeout": float64(120000)}
	forced := map[string]any{"timeout": float64(120000), "force": true}
	if ReversibilityConfirmToken(ReversibilityOutwardFacing, "git_push", bare) ==
		ReversibilityConfirmToken(ReversibilityOutwardFacing, "git_push", forced) {
		t.Fatalf("confirm token ignored an MCP force flag — binding weakened")
	}
}

// TestReversibilityConfirmTokenMismatchIsRefused pins the mismatch arm end to end: a
// token issued for one call must be REFUSED when echoed on another, and a forged or
// stale token must never confirm. The gate stays a real acknowledgement of the
// previewed action rather than a key any string satisfies.
func TestReversibilityConfirmTokenMismatchIsRefused(t *testing.T) {
	env, ok := ReversibilityConfirmed("shell_command", map[string]any{"command": da33PublishPush})
	if ok || env.ConfirmToken == "" {
		t.Fatalf("expected a held push with a token, got ok=%v env=%+v", ok, env)
	}

	// The token from the page push must not confirm a DIFFERENT outward-facing call.
	if _, ok := ReversibilityConfirmed("shell_command", map[string]any{
		"command":               "git push --force origin main",
		ReversibilityConfirmArg: env.ConfirmToken,
	}); ok {
		t.Fatalf("a confirm issued for %q was transplanted onto a force push", da33PublishPush)
	}

	// A forged token is refused.
	if _, ok := ReversibilityConfirmed("shell_command", map[string]any{
		"command":               da33PublishPush,
		ReversibilityConfirmArg: "fak-0000000000000000",
	}); ok {
		t.Fatalf("a forged confirm token was accepted")
	}

	// The matching token still confirms — the gate is a pause, not a wall.
	if _, ok := ReversibilityConfirmed("shell_command", map[string]any{
		"command":               da33PublishPush,
		ReversibilityConfirmArg: env.ConfirmToken,
	}); !ok {
		t.Fatalf("the matching confirm token was refused")
	}
}
