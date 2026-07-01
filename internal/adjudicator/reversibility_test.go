package adjudicator

import (
	"strings"
	"testing"
)

func TestReversibilityClassifiesCommands(t *testing.T) {
	cases := []struct {
		name string
		tool string
		args map[string]any
		want ReversibilityClass
		hint string
	}{
		{
			name: "ordinary read-only command is reversible",
			tool: "Bash",
			args: map[string]any{"command": "go test ./internal/adjudicator"},
			want: ReversibilityReversible,
		},
		{
			name: "destructive shell command is irreversible",
			tool: "Bash",
			args: map[string]any{"command": "rm -rf build"},
			want: ReversibilityIrreversible,
		},
		{
			name: "publish command is outward-facing",
			tool: "Bash",
			args: map[string]any{"command": "git push origin main"},
			want: ReversibilityOutwardFacing,
			hint: "try git push --dry-run first",
		},
		{
			name: "http write is outward-facing",
			tool: "Bash",
			args: map[string]any{"command": "curl -X POST https://example.invalid/hook -d ok"},
			want: ReversibilityOutwardFacing,
		},
		{
			name: "explicit dry run stays reversible",
			tool: "Bash",
			args: map[string]any{"command": "git push --dry-run origin main"},
			want: ReversibilityReversible,
		},
		{
			name: "tool name can mark destructive calls",
			tool: "delete_file",
			args: map[string]any{"file_path": "tmp/cache.bin"},
			want: ReversibilityIrreversible,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyReversibility(tc.tool, tc.args)
			if got.Class != tc.want {
				t.Fatalf("ClassifyReversibility() class = %q, want %q; envelope=%+v", got.Class, tc.want, got)
			}
			if tc.hint != "" && got.DryRunHint != tc.hint {
				t.Fatalf("DryRunHint = %q, want %q", got.DryRunHint, tc.hint)
			}
			if tc.want == ReversibilityReversible && got.ConfirmToken != "" {
				t.Fatalf("reversible call got confirm token %q", got.ConfirmToken)
			}
			if tc.want != ReversibilityReversible && got.ConfirmToken == "" {
				t.Fatalf("non-reversible call missing confirm token: %+v", got)
			}
		})
	}
}

func TestReversibilityConfirmationTokenMustEcho(t *testing.T) {
	args := map[string]any{"command": "rm -rf build"}
	env, ok := ReversibilityConfirmed("Bash", args)
	if ok {
		t.Fatalf("unconfirmed destructive call was allowed: %+v", env)
	}

	withToken := map[string]any{"command": "rm -rf build", ReversibilityConfirmArg: env.ConfirmToken}
	env2, ok := ReversibilityConfirmed("Bash", withToken)
	if !ok {
		t.Fatalf("matching confirmation token was refused: %+v", env2)
	}
	if env2.ConfirmToken != env.ConfirmToken {
		t.Fatalf("confirm token changed after adding confirmation arg: %q != %q", env2.ConfirmToken, env.ConfirmToken)
	}

	_, ok = ReversibilityConfirmed("Bash", map[string]any{"command": "rm -rf build", "confirm_token": "wrong"})
	if ok {
		t.Fatalf("wrong confirmation token was accepted")
	}
}

func TestReversibilityPreviewRedactsSecrets(t *testing.T) {
	env := ClassifyReversibility("Bash", map[string]any{
		"command": "curl -X POST https://example.invalid -d api_key=secret123",
	})
	if env.Class != ReversibilityOutwardFacing {
		t.Fatalf("class = %q, want outward-facing", env.Class)
	}
	if strings.Contains(env.Preview, "secret123") {
		t.Fatalf("preview leaked secret: %q", env.Preview)
	}
	if !strings.Contains(env.Preview, "api_key=[REDACTED]") {
		t.Fatalf("preview did not show redaction marker: %q", env.Preview)
	}
}
