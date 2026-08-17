package main

import (
	"os"
	"strings"
	"testing"
)

func TestGuardDefaultsToolcallControlToShadow(t *testing.T) {
	data, err := os.ReadFile("guard.go")
	if err != nil {
		t.Fatal(err)
	}
	src := string(data)
	if !strings.Contains(src, `fs.String("toolcall-control", "shadow"`) {
		t.Fatal("guard must default avoidable-call control to shadow")
	}
	if !strings.Contains(src, `"FAK_TOOLCALL_CONTROL_MODE", string(mode)`) {
		t.Fatal("guard must inject selected mode into toolproc hooks")
	}
}
