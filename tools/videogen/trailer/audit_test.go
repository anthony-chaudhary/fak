package main

import (
	"strings"
	"testing"
)

func goodConfig() Config {
	return Config{Width: 1280, Height: 720, FPS: 30, Scenes: []Scene{{Kind: "hook", Secs: 4, Title: "Your agent can act."}, {Kind: "checkpoint", Secs: 4, Title: "Before the tool runs."}, {Kind: "proof", Secs: 3, Title: "Dangerous call stopped."}, {Kind: "cta", Secs: 8, Title: "Try one guarded session.", Command: "fak guard -- claude"}}}
}
func TestAuditAcceptsFocusedReadableTrailer(t *testing.T) {
	a, e := audit(goodConfig())
	if e != nil {
		t.Fatal(e)
	}
	if a.Duration != 19 || a.CTAStart != 11 || a.CTAHold != 8 {
		t.Fatalf("audit=%+v", a)
	}
}
func TestAuditRefusesLateCTA(t *testing.T) {
	c := goodConfig()
	c.Scenes[0].Secs = 7
	if _, e := audit(c); e == nil || !strings.Contains(e.Error(), "CTA starts") {
		t.Fatalf("error=%v", e)
	}
}
func TestAuditRefusesDenseCopy(t *testing.T) {
	c := goodConfig()
	c.Scenes[0].Title = "one two three four five six seven eight nine"
	if _, e := audit(c); e == nil || !strings.Contains(e.Error(), "max 8") {
		t.Fatalf("error=%v", e)
	}
}
func TestAuditRefusesUnreadablyBriefScene(t *testing.T) {
	c := goodConfig()
	c.Scenes[1].Secs = 1
	if _, e := audit(c); e == nil || !strings.Contains(e.Error(), "too brief") {
		t.Fatalf("error=%v", e)
	}
}
