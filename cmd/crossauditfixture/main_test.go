package main

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestFixtureCommandPassesMatchingCandidateAndFailsMismatch(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "crossauditfixture")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build fixture: %v\n%s", err, out)
	}
	for _, tc := range []struct {
		name, candidate string
		wantFail        bool
	}{{"clean", "contract", false}, {"corrupt", "different", true}} {
		t.Run(tc.name, func(t *testing.T) {
			cmd := exec.Command(bin, "--contract-base64", "Y29udHJhY3Q=")
			cmd.Stdin = strings.NewReader(tc.candidate)
			out, err := cmd.CombinedOutput()
			if (err != nil) != tc.wantFail {
				t.Fatalf("err=%v output=%q", err, out)
			}
			want := "PASS"
			if tc.wantFail {
				want = "FAIL"
			}
			if strings.TrimSpace(string(out)) != want {
				t.Fatalf("output=%q want=%q", out, want)
			}
		})
	}
}
