package taskrun

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestAgentBenchVisibleTestsOutsideWritableRoots(t *testing.T) {
	if err := SandboxAvailable(); err != nil {
		t.Skip(err)
	}
	root := t.TempDir()
	candidate, visible, oracle := filepath.Join(root, "candidate"), filepath.Join(root, "visible"), filepath.Join(root, "oracle")
	for _, dir := range []string{candidate, visible, oracle} {
		if err := os.Mkdir(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}
	visiblePath := filepath.Join(visible, "visible_test.go")
	visibleSource := "package fixture\nimport \"testing\"\nfunc TestAnswer(t *testing.T) { if Answer() != 42 { t.Fatal(\"wrong answer\") } }\n"
	if err := os.WriteFile(visiblePath, []byte(visibleSource), 0600); err != nil {
		t.Fatal(err)
	}
	target := "target.go"
	malicious := fmt.Sprintf(`package fixture
import "os"
func init() {
	if err := os.WriteFile(%q, []byte("package fixture"), 0600); err == nil { panic("visible input was writable") }
	if err := os.WriteFile("target.go", []byte("package fixture"), 0600); err == nil { panic("assembled candidate was writable") }
}
func Answer() int { return 42 }
`, visiblePath)
	if err := os.WriteFile(filepath.Join(candidate, target), []byte(malicious), 0600); err != nil {
		t.Fatal(err)
	}
	result, err := runCandidateTests(context.Background(), candidate, target, visible, oracle)
	visibleAfter, visibleErr := os.ReadFile(visiblePath)
	candidateAfter, candidateErr := os.ReadFile(filepath.Join(candidate, target))
	if err != nil || result.ExitCode != 0 {
		t.Fatalf("sandboxed mutation probes did not run through the real visible test: result=%+v err=%v", result, err)
	}
	if visibleErr != nil || candidateErr != nil || string(visibleAfter) != visibleSource || string(candidateAfter) != malicious {
		t.Fatalf("agent test execution mutated protected inputs: visibleErr=%v candidateErr=%v", visibleErr, candidateErr)
	}

	valid := filepath.Join(root, "valid-candidate")
	if err := os.Mkdir(valid, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(valid, target), []byte("package fixture\nfunc Answer() int { return 42 }\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if result, err := runCandidateTests(context.Background(), valid, target, visible, oracle); err != nil || result.ExitCode != 0 {
		t.Fatalf("known valid candidate did not execute external visible test: result=%+v err=%v", result, err)
	}
	if _, err := runCandidateTests(context.Background(), visible, target, visible, oracle); err == nil {
		t.Fatal("overlapping candidate and visible roots were admitted")
	}
}
