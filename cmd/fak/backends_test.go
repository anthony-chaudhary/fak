package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// TestBackendsOutput is the release-gate witness for the `fak backends` verb: the
// registry surface a published asset exposes must be non-empty (the cpu-ref
// Reference floor self-registers in EVERY build) and must name cpu-ref, so the
// gate can fail loud when a shipped binary somehow carries no registered backends.
func TestBackendsOutput(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr bool
		check   func(t *testing.T, stdout, stderr string)
	}{
		{
			name: "plain lists one name per line and includes cpu-ref",
			args: nil,
			check: func(t *testing.T, stdout, stderr string) {
				if stderr != "" {
					t.Errorf("stderr = %q, want empty", stderr)
				}
				lines := strings.Split(strings.TrimRight(stdout, "\n"), "\n")
				if len(lines) == 0 || lines[0] == "" {
					t.Fatalf("stdout = %q, want at least one backend name", stdout)
				}
				found := false
				for _, l := range lines {
					if strings.Contains(l, "\n") {
						t.Errorf("unexpected embedded newline in line %q", l)
					}
					if strings.TrimSpace(l) == "" {
						t.Errorf("blank line in backend listing: %q", stdout)
					}
					if l == "cpu-ref" {
						found = true
					}
				}
				if !found {
					t.Errorf("backend listing %q does not contain the always-present cpu-ref floor", stdout)
				}
			},
		},
		{
			name: "--json emits a parseable JSON array containing cpu-ref",
			args: []string{"--json"},
			check: func(t *testing.T, stdout, stderr string) {
				var got []string
				if err := json.Unmarshal([]byte(stdout), &got); err != nil {
					t.Fatalf("--json output does not parse as a JSON array: %v\n%s", err, stdout)
				}
				if len(got) == 0 {
					t.Fatalf("--json array is empty: %q", stdout)
				}
				hasCPURef := false
				for _, name := range got {
					if name == "cpu-ref" {
						hasCPURef = true
					}
				}
				if !hasCPURef {
					t.Errorf("--json array %v does not contain cpu-ref", got)
				}
			},
		},
		{
			name: "-json short flag form parses identically",
			args: []string{"-json"},
			check: func(t *testing.T, stdout, stderr string) {
				var got []string
				if err := json.Unmarshal([]byte(stdout), &got); err != nil {
					t.Fatalf("-json output does not parse: %v\n%s", err, stdout)
				}
				if len(got) == 0 {
					t.Errorf("-json array is empty")
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out, errb bytes.Buffer
			code := runBackends(&out, &errb, tt.args)
			if tt.wantErr && code != 1 {
				t.Fatalf("exit code = %d, want 1", code)
			}
			if !tt.wantErr && code != 0 {
				t.Fatalf("exit code = %d, want 0 (stderr: %s)", code, errb.String())
			}
			tt.check(t, out.String(), errb.String())
		})
	}
}

// TestBackendsJSONRequested pins the flag scan so a future flag addition cannot
// silently break the release probe's machine-readable path (it greps for
// `fak backends --json` shape, not for Go internals).
func TestBackendsJSONRequested(t *testing.T) {
	if backendsJSONRequested(nil) {
		t.Error("no args: JSONRequested = true, want false")
	}
	if backendsJSONRequested([]string{"plain"}) {
		t.Error("non-flag arg: JSONRequested = true, want false")
	}
	if !backendsJSONRequested([]string{"--json"}) {
		t.Error("--json: JSONRequested = false, want true")
	}
	if !backendsJSONRequested([]string{"-json"}) {
		t.Error("-json: JSONRequested = false, want true")
	}
	if !backendsJSONRequested([]string{"extra", "--json", "words"}) {
		t.Error("--json mid-args: JSONRequested = false, want true")
	}
}
