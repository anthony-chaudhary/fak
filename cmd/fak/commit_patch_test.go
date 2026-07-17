package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/patchcommit"
)

func TestCommitPatchRequiresExplicitInputs(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runCommitPatch(&stdout, &stderr, []string{"--json"}); code != 1 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr.String())
	}
	var got patchcommit.Result
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v; output=%q", err, stdout.String())
	}
	if got.Reason != patchcommit.ReasonPatchInvalid {
		t.Fatalf("reason = %q, want %q", got.Reason, patchcommit.ReasonPatchInvalid)
	}
}
