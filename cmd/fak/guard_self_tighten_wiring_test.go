package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestGuardSelfTightenGateHasALiveCaller is the dead-code witness for #5411, and it is
// deliberately written against nothing but the package's own source bytes so it can be
// run at the PRE-FIX commit: at HEAD it FAILS (admitSelfTightenOverlay was called only
// from guard_self_tighten_test.go — the gate shipped as a decision half with no seam),
// and it passes once the gate is wired into a production file.
//
// A behavioural test cannot carry this claim on its own. A gate that is never called
// still classifies correctly when a unit test calls it directly, so every existing test
// in guard_self_tighten_test.go passed for as long as the gate was dead. Only a check
// that a NON-test file names it can tell "decides correctly" apart from "decides
// anything at all" — which is the exact confusion the file's own header called out.
func TestGuardSelfTightenGateHasALiveCaller(t *testing.T) {
	callers := guardNonTestFilesNaming(t, "admitSelfTightenOverlay(")
	// The definition itself lives in guard_self_tighten.go; a call from anywhere else
	// is what proves the gate is reachable in production.
	var live []string
	for _, f := range callers {
		if f != "guard_self_tighten.go" {
			live = append(live, f)
		}
	}
	if len(live) == 0 {
		t.Fatal("admitSelfTightenOverlay has NO non-test caller: the self-tighten admission gate is dead code — no live overlay is admitted or refused by it (#5411)")
	}
	t.Logf("self-tighten gate is called from: %s", strings.Join(live, ", "))
}

// TestGuardSelfTightenIsWiredIntoTheFloorAssembly pins the SEAM, not merely the call:
// the gate must run inside loadGuardCapabilityFloor, after the two operator overlays
// (so the agent tightens the floor an operator actually configured) and before
// protectGuardPolicyConfig (so the agent's own overlay file is not swept into
// SelfModifyGlobs, which would make it unwritable by the agent that authors it).
func TestGuardSelfTightenIsWiredIntoTheFloorAssembly(t *testing.T) {
	src, err := os.ReadFile("guard_startup.go")
	if err != nil {
		t.Fatal(err)
	}
	floorAt := bytes.Index(src, []byte("func loadGuardCapabilityFloor("))
	if floorAt < 0 {
		t.Fatal("cmd/fak/guard_startup.go no longer defines loadGuardCapabilityFloor")
	}
	body := src[floorAt:]
	applyAt := bytes.Index(body, []byte("guardApplySelfTightenOverlay(&rt,"))
	if applyAt < 0 {
		t.Fatal("loadGuardCapabilityFloor does not apply the self-tighten overlay: the #5181 gate is not on the launch path (#5411)")
	}
	denyAt := bytes.Index(body, []byte("guardApplyDenyOverlay(&rt,"))
	if denyAt < 0 {
		t.Fatal("loadGuardCapabilityFloor no longer applies the operator deny overlay")
	}
	if applyAt < denyAt {
		t.Fatal("the self-tighten overlay must be unioned AFTER the operator overlays, or it tightens a floor the operator has not finished assembling")
	}
	protectAt := bytes.Index(body, []byte("protectGuardPolicyConfig(rt,"))
	if protectAt < 0 {
		t.Fatal("loadGuardCapabilityFloor no longer write-protects the policy config")
	}
	if applyAt > protectAt {
		t.Fatal("the self-tighten overlay must be applied BEFORE protectGuardPolicyConfig")
	}
	// The agent's own overlay path must NOT be handed to protectGuardPolicyConfig; if it
	// were, the file would join SelfModifyGlobs and the agent could no longer author it.
	protectCall := body[protectAt:]
	if end := bytes.IndexByte(protectCall, '\n'); end > 0 {
		protectCall = protectCall[:end]
	}
	if bytes.Contains(protectCall, []byte("selfTightenPath")) {
		t.Fatal("the self-tighten overlay path was added to the self-modify protection set — the wrapped agent could then never write its own tightenings")
	}
}

// guardNonTestFilesNaming returns the non-test .go files in this package whose source
// contains needle.
func guardNonTestFilesNaming(t *testing.T, needle string) []string {
	t.Helper()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || filepath.Ext(name) != ".go" || strings.HasSuffix(name, "_test.go") {
			continue
		}
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(b, []byte(needle)) {
			out = append(out, name)
		}
	}
	return out
}
