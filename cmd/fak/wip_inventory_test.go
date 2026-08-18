package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestWIPInventoryUsage(t *testing.T) {
	var out bytes.Buffer
	wipUsage(&out)
	if !strings.Contains(out.String(), "fak wip inventory [--json]") {
		t.Fatalf("usage missing inventory: %s", out.String())
	}
}
