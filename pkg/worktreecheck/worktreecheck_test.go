package worktreecheck_test

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/worktreecheck"
)

// fak-test:runtime fast est=10ms
func TestTopologyCandidateParentsCoversMixedHistory(t *testing.T) {
	// A workspace can switch between local-only and sibling modules. Retaining
	// both candidate parents is what lets a later sweep cover that mixed history.
	for _, tc := range []struct {
		name         string
		workspace    string
		tempIsParent bool
		wantSibling  bool
	}{
		{name: "no_workspace"},
		{name: "local_workspace", workspace: "go 1.26\nuse .\n"},
		{name: "sibling_workspace", workspace: "go 1.26\nuse (\n.\n../companion\n)\n", wantSibling: true},
		{name: "coincident_parents", workspace: "go 1.26\nuse ../companion\n", tempIsParent: true, wantSibling: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parent := t.TempDir()
			root := filepath.Join(parent, "repo")
			if err := os.Mkdir(root, 0o755); err != nil {
				t.Fatal(err)
			}
			if tc.workspace != "" {
				if err := os.WriteFile(filepath.Join(root, "go.work"), []byte(tc.workspace), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			temp := t.TempDir()
			if tc.tempIsParent {
				temp = parent
			}
			// os.TempDir uses different environment names across supported hosts.
			// These subtests stay serial because the environment is process-wide.
			for _, key := range []string{"TMPDIR", "TMP", "TEMP", "SystemTemp"} {
				t.Setenv(key, temp)
			}
			if got := os.TempDir(); got != temp {
				t.Fatalf("temporary-directory setup: got %q, want %q", got, temp)
			}
			want := []string{temp}
			if tc.wantSibling && !tc.tempIsParent {
				want = []string{parent, temp}
			}
			if got := worktreecheck.TopologyCandidateParents(root); !slices.Equal(got, want) {
				t.Fatalf("candidate parents = %q, want %q", got, want)
			}
		})
	}
}
