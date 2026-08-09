package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// A real reconciliation failure must come out of this leaf wearing the token dos.toml declares
// for it. This drives the refusal from an actually-corrupt artifact rather than a synthetic
// error, so the test fails if the verifier stops refusing OR if the refusal stops naming the
// reason — the two ways [reasons.MICROCONTEXT_LEDGER_REFUSED] goes back to promising a code no
// consumer can receive.
func TestLedgerRefusalNamesTheDeclaredReason(t *testing.T) {
	path := filepath.Join(t.TempDir(), "quality.json")
	// An accounting failure, not a parse failure: the ledger is well-formed JSON whose
	// useful-work rate does not reconcile, which is exactly what the summary describes.
	body, err := json.Marshal(map[string]any{"claim_families": map[string]any{}})
	if err != nil {
		t.Fatalf("marshal fixture: %v", err)
	}
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	verr := verifyQualityLedgerArtifact(path)
	if verr == nil {
		t.Fatal("verifyQualityLedgerArtifact accepted an unreconciled ledger; the refusal this reason names no longer happens")
	}

	line := ledgerRefusal("verify-quality", path, verr)
	if !strings.HasPrefix(line, LedgerRefusedReason+":") {
		t.Errorf("refusal must LEAD with the reason code so a consumer can route on the first field, got %q", line)
	}
	if LedgerRefusedReason != "MICROCONTEXT_LEDGER_REFUSED" {
		t.Errorf("reason code drifted from the dos.toml declaration: %q", LedgerRefusedReason)
	}
	for _, want := range []string{"-verify-quality", path, verr.Error()} {
		if !strings.Contains(line, want) {
			t.Errorf("refusal %q does not carry %q — the operator cannot tell which artifact to regenerate", line, want)
		}
	}
}

var verifyFlagRE = regexp.MustCompile(`StringVar\(&?\w+, "(verify[a-z0-9-]*)"`)

// Every -verify-* flag must refuse through runVerify. This is the ratchet: naming the reason
// once would leave the NEXT verify path free to refuse anonymously, which is the drift
// internal/architest.TestEveryDeclaredReasonHasAnEmitter caught in the first place.
func TestEveryVerifyFlagRefusesThroughTheNamedDoor(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	var sources []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		b, rerr := os.ReadFile(name)
		if rerr != nil {
			t.Fatalf("read %s: %v", name, rerr)
		}
		sources = append(sources, string(b))
	}
	joined := strings.Join(sources, "\n")

	var flags []string
	for _, m := range verifyFlagRE.FindAllStringSubmatch(joined, -1) {
		flags = append(flags, m[1])
	}
	if len(flags) == 0 {
		t.Fatal("found no -verify* flags; the scan would pass vacuously")
	}
	for _, name := range flags {
		if !strings.Contains(joined, `runVerify("`+name+`"`) {
			t.Errorf("-%s does not dispatch through runVerify, so its refusal carries no reason code: "+
				"a consumer routing on %s cannot attribute it. Route it through runVerify.", name, LedgerRefusedReason)
		}
	}
}
