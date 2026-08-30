//go:build darwin

package modelperfobs

import (
	"context"
	"strings"
	"testing"
)

func TestDarwinNativeCommandRejectsExecutableOutsideStaticSet(t *testing.T) {
	_, err := runDarwinNativeCommand(context.Background(), "/tmp/operator-selected-command")
	if err == nil || !strings.Contains(err.Error(), "unsupported Darwin native command") {
		t.Fatalf("dynamic Darwin executable error = %v, want static-set refusal", err)
	}
}
