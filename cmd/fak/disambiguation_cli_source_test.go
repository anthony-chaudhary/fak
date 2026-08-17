package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
)

func TestDisambiguationCLISourceUsesPublicHelp(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runDisambiguation(&out, &errb, []string{"cli-source", "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	var got disambiguation.CLISourceReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != disambiguation.CLISourceSchemaVersion || !cliSourceHasTerm(got.Terms, "serve") {
		t.Fatalf("report = %+v", got)
	}
}

func TestDisambiguationCLISourceSelfTestAddsAndStales(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runDisambiguation(&out, &errb, []string{"cli-source", "--self-test", "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	var got disambiguation.CLISourceReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !cliSourceHasTerm(got.Terms, "added-fixture") || !cliSourceHasTerm(got.Stale, "removed-fixture") {
		t.Fatalf("self-test report = %+v", got)
	}
}
