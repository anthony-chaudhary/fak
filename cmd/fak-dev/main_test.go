package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

func TestHelpIdentifiesIndependentDevelopmentArtifact(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"help"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"fak-dev — repository-development tooling", "index ownership", "separately buildable 'fak' artifact"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("help missing %q:\n%s", want, out.String())
		}
	}
}

func TestOwnershipCommandUsesInventory(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"index", "ownership", "--json", "--root", devindex.FindRoot(".")}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got devindex.OwnershipReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, command := range got.Commands {
		if command.Name == "index" && command.Owner == devindex.OwnerDev && command.DispatchTarget == "fak-dev" {
			found = true
		}
	}
	if !found {
		t.Fatal("ownership inventory does not authorize index on fak-dev")
	}
}
