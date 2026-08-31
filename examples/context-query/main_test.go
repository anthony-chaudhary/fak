package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/contextq"
)

func TestSelfcheck(t *testing.T) {
	t.Run("deterministic payload-free receipt", testSelfcheckReceipt)
	t.Run("CLI misuse", testContextQueryRequiresSelfcheck)
}

func testSelfcheckReceipt(t *testing.T) {
	var firstOut, firstErr bytes.Buffer
	if rc := run(&firstOut, &firstErr, []string{"-selfcheck"}); rc != 0 {
		t.Fatalf("run rc=%d stderr=%s", rc, firstErr.String())
	}
	var secondOut, secondErr bytes.Buffer
	if rc := run(&secondOut, &secondErr, []string{"-selfcheck"}); rc != 0 {
		t.Fatalf("replay rc=%d stderr=%s", rc, secondErr.String())
	}
	if !bytes.Equal(firstOut.Bytes(), secondOut.Bytes()) {
		t.Fatalf("selfcheck is not byte-deterministic:\nfirst  %s\nsecond %s", firstOut.Bytes(), secondOut.Bytes())
	}
	if strings.Contains(firstOut.String(), "source-only-detail") {
		t.Fatal("receipt leaked raw source records")
	}
	if !strings.Contains(firstOut.String(), `"derivation_digest":`) || strings.Contains(firstOut.String(), `"plan_digest":`) {
		t.Fatalf("receipt must expose the derivation lineage name: %s", firstOut.Bytes())
	}

	var receipt selfcheckReceipt
	if err := json.Unmarshal(firstOut.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt: %v\n%s", err, firstOut.Bytes())
	}
	if receipt.Schema != "fak.context-query-selfcheck/1" || receipt.View.Schema != contextq.DerivedRecordViewSchema {
		t.Fatalf("schemas = %q / %q", receipt.Schema, receipt.View.Schema)
	}
	want := `[{"group":"alice","count":2},{"group":"bob","count":2},{"group":"carol","count":2}]`
	if string(receipt.Result) != want {
		t.Fatalf("result = %s, want %s", receipt.Result, want)
	}
	a := receipt.View.Accounting
	if a.RecordsRead != 30 || a.RecordsMatch != 6 || a.RecordsOutput != 3 || a.WorkUnits != 66 {
		t.Fatalf("accounting = %#v", a)
	}
	if a.OutputBytes >= a.SourceBytes {
		t.Fatalf("derived view bytes not smaller than source: %#v", a)
	}
	if receipt.View.SourceDigest == "" || receipt.View.DerivationDigest == "" || receipt.View.OutputDigest == "" {
		t.Fatalf("lineage incomplete: %#v", receipt.View)
	}
	if receipt.View.Truncated || receipt.ModelRoundTrips != 0 || receipt.NetworkCalls != 0 {
		t.Fatalf("truncated/model/network calls = %v/%d/%d", receipt.View.Truncated, receipt.ModelRoundTrips, receipt.NetworkCalls)
	}
}

func testContextQueryRequiresSelfcheck(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if rc := run(&stdout, &stderr, nil); rc != 2 {
		t.Fatalf("run rc=%d, want 2", rc)
	}
	if !strings.Contains(stderr.String(), "usage: context-query -selfcheck") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}
