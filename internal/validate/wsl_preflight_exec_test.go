package validate

import (
	"context"
	"testing"
)

func TestValidateWSLPreflightExecsBashWithoutDefaultShell(t *testing.T) {
	resetValidateWSLCapabilityCacheForTest()
	oldLookPath, oldCommand := validateWSLLookPath, validateWSLCommand
	validateWSLLookPath = func(string) (string, error) { return `C:\Windows\System32\wsl.exe`, nil }
	var argv []string
	validateWSLCommand = func(_ context.Context, args ...string) ([]byte, error) {
		argv = args
		return []byte(validateWSLDistroPrefix + "Ubuntu-24.04\n" + validateWSLPathPrefix + "/usr/bin\n"), nil
	}
	t.Cleanup(func() {
		validateWSLLookPath, validateWSLCommand = oldLookPath, oldCommand
		resetValidateWSLCapabilityCacheForTest()
	})

	preflightValidateWSLCapabilitiesWithin(context.Background())
	if len(argv) < 3 || argv[0] != "-e" || argv[1] != "bash" {
		t.Fatalf("probe argv = %q, want -e bash ... so wsl.exe skips the default-shell re-parse", argv)
	}
}
