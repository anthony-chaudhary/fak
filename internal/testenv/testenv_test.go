package testenv

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestWithoutCredentialsUsesRepositorySecretRegistry(t *testing.T) {
	t.Parallel()
	got := WithoutCredentials([]string{
		"PATH=/bin",
		"ANTHROPIC_API_KEY=must-not-escape",
		"CLAUDE_CODE_OAUTH_TOKEN=must-not-escape",
		"FAK_TEST_MODE=1",
	})
	joined := strings.Join(got, "\n")
	for _, forbidden := range []string{"ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "must-not-escape"} {
		if strings.Contains(joined, forbidden) {
			t.Fatalf("credential survived fence: %s in %q", forbidden, joined)
		}
	}
	for _, want := range []string{"PATH=/bin", "FAK_TEST_MODE=1"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("non-credential environment removed: want %q in %q", want, joined)
		}
	}
}

func TestRunStripsCredentialsAtChildBoundary(t *testing.T) {
	if os.Getenv("FAK_TESTENV_HELPER") == "1" {
		for _, name := range []string{"ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"} {
			if _, ok := os.LookupEnv(name); ok {
				os.Exit(41)
			}
		}
		os.Exit(0)
	}

	t.Setenv("ANTHROPIC_API_KEY", "bogus-anthropic")
	t.Setenv("CLAUDE_CODE_OAUTH_TOKEN", "bogus-oauth")
	t.Setenv("FAK_TESTENV_HELPER", "1")
	if err := Run([]string{os.Args[0], "-test.run=^TestRunStripsCredentialsAtChildBoundary$"}); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			t.Fatalf("child saw a provider credential (exit %d): %v", exitErr.ExitCode(), err)
		}
		t.Fatal(err)
	}
}
