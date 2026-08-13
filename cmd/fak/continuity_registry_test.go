package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestContinuityRegistrySelfcheckCapturedLifecycle(t *testing.T) {
	var out, errout bytes.Buffer
	if code := runContinuityRegistrySelfcheck(&out, &errout, false); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	for _, want := range []string{"PASS public registry lifecycle", "dry-run by default", "provenance+dependencies+sensitivity+compatibility+signature verified", "install: inactive; activation: explicit", "permissions previewed", "breaking changes+migration", "quarantined; evidence retained", "denied before activation (TAMPERED"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q in %s", want, out.String())
		}
	}
}
func TestContinuityRegistrySelfcheckJSON(t *testing.T) {
	var out, errout bytes.Buffer
	if code := runContinuityRegistrySelfcheck(&out, &errout, true); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	for _, want := range []string{`"result": "PASS"`, `"publish_default": "dry-run"`, `"install": "inactive"`, `"revocation": "quarantined; evidence retained"`} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q", want)
		}
	}
}
