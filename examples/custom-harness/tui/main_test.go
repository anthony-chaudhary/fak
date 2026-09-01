package main

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestScriptedSessionCapturesTerminalRender(t *testing.T) {
	a, err := newApp()
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := a.run(context.Background(), strings.NewReader("teach me the seam\n/quit\n"), &out); err != nil {
		t.Fatal(err)
	}
	got := out.String()
	for _, want := range []string{
		"fak custom harness — terminal example",
		">   turn.started",
		"teach me the seam",
		"model.response",
		"tool.requested",
		"tool.completed",
		"turn.completed",
		"> bye",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("transcript missing %q:\n%s", want, got)
		}
	}
}
