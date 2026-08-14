package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"
)

func TestHarnessInitCLI(t *testing.T) {
	var out, errb bytes.Buffer
	root := filepath.Join(t.TempDir(), "product")
	code := runHarness(&out, &errb, []string{"init", "--dir", root, "--module", "example.test/product", "--json"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var got struct {
		ContractVersion string   `json:"contract_version"`
		Created         []string `json:"created"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.ContractVersion != "v1alpha1" || len(got.Created) == 0 {
		t.Fatalf("result=%s", out.String())
	}
}
