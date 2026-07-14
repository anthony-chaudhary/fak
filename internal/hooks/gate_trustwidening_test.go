package hooks

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTrustWideningPolarity(t *testing.T) {
	cases := []struct {
		name    string
		file    string
		content string
		add     []string
		want    int
	}{
		{
			name:    "guard allow member added",
			file:    "cmd/fak/guard-default-policy.json",
			content: "{\n  \"allow\": [\n    \"mcp__new__admin\",\n  ]\n}\n",
			add:     []string{"{", `  "allow": [`, `    "mcp__new__admin",`},
			want:    1,
		},
		{
			name:    "self modify glob added",
			file:    "examples/dev-agent-policy.json",
			content: "{\n  \"self_modify_globs\": [\n    \"internal/policy/\",\n  ]\n}\n",
			add:     []string{"{", `  "self_modify_globs": [`, `    "internal/policy/",`},
			want:    1,
		},
		{
			name:    "claude grant added compact",
			file:    ".claude/settings.json",
			content: `{"allowedTools": ["Bash(git push:*)"]}`,
			add:     []string{`{"allowedTools": ["Bash(git push:*)"]}`},
			want:    1,
		},
		{
			name:    "deny member is not a grant",
			file:    "cmd/fak/guard-default-policy.json",
			content: "{\n  \"deny_regex\": [\n    \"dangerous_new_pattern\",\n  ]\n}\n",
			add:     []string{"{", `  "deny_regex": [`, `    "dangerous_new_pattern",`},
			want:    0,
		},
		{
			name:    "array marker alone is not a grant",
			file:    "cmd/fak/guard-default-policy.json",
			content: "{\n  \"allow\": [\n  ]\n}\n",
			add:     []string{"{", `  "allow": [`},
			want:    0,
		},
		{
			name: "removal only has no added lines",
			file: "cmd/fak/guard-default-policy.json",
			want: 0,
		},
		{
			name:    "unrelated json string is out of scope",
			file:    "docs/example.json",
			content: `"Bash"`,
			add:     []string{`"Bash"`},
			want:    0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			if tc.content != "" {
				full := filepath.Join(root, filepath.FromSlash(tc.file))
				if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(full, []byte(tc.content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			d := diffOf(root, map[string][]string{tc.file: tc.add})
			got, err := gateTrustWidening(d)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != tc.want {
				t.Fatalf("findings=%d want %d: %+v", len(got), tc.want, got)
			}
			if tc.want > 0 && !hasFindingFor(got, "TRUST_WIDENING", "ESCALATE") {
				t.Fatalf("finding lacks typed escalation: %+v", got)
			}
		})
	}
}

func TestTrustWideningRegisteredAdvisory(t *testing.T) {
	for _, gate := range PreCommitGates() {
		if gate.Name != "TRUST_WIDENING" {
			continue
		}
		if gate.DefaultMode != "warn" {
			t.Fatalf("DefaultMode=%q want warn", gate.DefaultMode)
		}
		return
	}
	t.Fatal("TRUST_WIDENING is not registered in PreCommitGates")
}
