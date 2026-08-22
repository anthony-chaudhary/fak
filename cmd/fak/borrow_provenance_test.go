package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/borrowprovenance"
)

func TestBorrowProvenancePinVerifyAndDrift(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "upstream.h")
	manifestPath := filepath.Join(dir, "pin.json")
	if err := os.WriteFile(sourcePath, []byte("typedef int copied;\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	var pinOut bytes.Buffer
	if err := runBorrowProvenance([]string{"pin", "--source", sourcePath, "--url", "https://example.test/upstream", "--ref", "abc123", "--source-path", "include/upstream.h", "--license", "Apache-2.0"}, &pinOut); err != nil {
		t.Fatal(err)
	}
	var record borrowprovenance.Record
	if err := json.Unmarshal(pinOut.Bytes(), &record); err != nil {
		t.Fatal(err)
	}
	if record.SourceSHA256 == "" || record.SourcePath != "include/upstream.h" {
		t.Fatalf("incomplete pin: %+v", record)
	}
	if err := os.WriteFile(manifestPath, pinOut.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	var verifyOut bytes.Buffer
	if err := runBorrowProvenance([]string{"verify", "--manifest", manifestPath, "--source", sourcePath}, &verifyOut); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(sourcePath, []byte("typedef long copied;\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	verifyOut.Reset()
	if err := runBorrowProvenance([]string{"verify", "--manifest", manifestPath, "--source", sourcePath}, &verifyOut); err == nil {
		t.Fatal("changed upstream source passed verification")
	}
	var result borrowprovenance.Verification
	if err := json.Unmarshal(verifyOut.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Match || result.ExpectedSHA256 == result.ActualSHA256 {
		t.Fatalf("drift evidence missing: %+v", result)
	}
}
