package main

import (
	"slices"
	"testing"
)

// OpenCode v2 binds a flag to the command it follows: `opencode --model m run`
// exits "Unrecognized flag: --model in command opencode", and `run` has no
// --pure or --dir. The child argv must lead with `run` so the guard's
// --standalone insertion applies.
func TestOpencodeLauncherChildArgvIsOpenCodeV2RunShape(t *testing.T) {
	for _, tc := range []struct {
		name           string
		opts           opencodeLaunchOptions
		wantChild      []string
		wantStandalone []string
	}{
		{
			name:           "probe",
			opts:           opencodeLaunchOptions{model: "m1", probePrompt: "do it", auto: true},
			wantChild:      []string{"opencode", "run", "--model", "fak/m1", "--format", "json", "--auto", "do it"},
			wantStandalone: []string{"opencode", "run", "--standalone", "--model", "fak/m1", "--format", "json", "--auto", "do it"},
		},
		{
			name:           "passthrough run",
			opts:           opencodeLaunchOptions{model: "fak/m1", passthrough: []string{"run", "status"}},
			wantChild:      []string{"opencode", "run", "--model", "fak/m1", "status"},
			wantStandalone: []string{"opencode", "run", "--standalone", "--model", "fak/m1", "status"},
		},
		{
			name:           "interactive",
			opts:           opencodeLaunchOptions{model: "m1", passthrough: []string{"--continue"}},
			wantChild:      []string{"opencode", "--continue"},
			wantStandalone: []string{"opencode", "--continue"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			argv := buildOpencodeLaunchArgv("fak", tc.opts)
			sep := slices.Index(argv, "--")
			if sep < 0 {
				t.Fatalf("guard argv has no child boundary: %q", argv)
			}
			child := argv[sep+1:]
			if !slices.Equal(child, tc.wantChild) {
				t.Fatalf("guarded child argv = %q, want %q", child, tc.wantChild)
			}
			if direct := buildOpencodeDirectArgv(tc.opts); !slices.Equal(direct, tc.wantChild) {
				t.Fatalf("direct argv = %q, want %q", direct, tc.wantChild)
			}
			if got := guardOpenCodeStandaloneCommand(child); !slices.Equal(got, tc.wantStandalone) {
				t.Fatalf("guard standalone command = %q, want %q", got, tc.wantStandalone)
			}
		})
	}
}
