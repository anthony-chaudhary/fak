package scorecard

import (
	"path/filepath"
	"testing"
)

func TestWorkspaceRootMatchesFilepathAbs(t *testing.T) {
	want, err := filepath.Abs(".")
	if err != nil {
		t.Fatal(err)
	}
	if got := WorkspaceRoot("."); got != want {
		t.Fatalf("WorkspaceRoot(.) = %q, want %q", got, want)
	}
}
