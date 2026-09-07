package hooks

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrePushKnownRedAndUnknownBlockByDefault(t *testing.T) {
	path := filepath.Join("..", "..", "tools", "githooks", "pre-push")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	s := string(raw)
	for _, want := range []string{
		`build_mode="${FLEET_BUILD_GUARD:-block}"`,
		`if [ "$build_status" -eq 1 ]`,
		`if [ "$build_mode" = "block" ]`,
		`elif [ "$build_status" -ne 0 ]`,
		`build gate could not run; push refused (fail closed)`,
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("pre-push policy missing %q", want)
		}
	}
	known := strings.Index(s, `if [ "$build_status" -eq 1 ]`)
	unknown := strings.Index(s, `elif [ "$build_status" -ne 0 ]`)
	if known < 0 || unknown <= known {
		t.Fatalf("KNOWN_RED and UNKNOWN branches are not separately ordered")
	}
	knownBranch := s[known:unknown]
	if !strings.Contains(knownBranch, "exit 1") {
		t.Fatalf("known-red default branch does not refuse the push")
	}
	unknownBranch := s[unknown:]
	if !strings.Contains(strings.SplitN(unknownBranch, "else", 2)[0], "exit 1") {
		t.Fatalf("unknown infrastructure branch must block (fail closed) under block mode")
	}
}
