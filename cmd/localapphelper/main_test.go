package main

import (
	"os"
	"testing"
)

func TestRequiresCapability(t *testing.T) {
	old := os.Args
	defer func() { os.Args = old }()
	os.Args = []string{"localapphelper", "--team-id", "T", "--bundle-id", "B", "--install-id", "I", "--revision", "R"}
	os.Unsetenv("FAK_APP_CAPABILITY")
	if got := run(); got != 2 {
		t.Fatalf("run=%d", got)
	}
}
