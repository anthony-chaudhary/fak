package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestTokenProfileHaloCapturedOutput(t *testing.T) {
	var out, errOut bytes.Buffer
	if rc := runTokenProfile(&out, &errOut, []string{"--halo"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	want := "HALO equal_total=102000 scalar=ADMIT (both) class_budget=50000\n" +
		"PROFILE_A cache-heavy cost=$0.077000 load=40500 ADMIT  [cost #---------] [load #---------]\n" +
		"PROFILE_B decode-heavy cost=$1.012999 load=405000 REFUSE_CLASS_LOAD  [cost ##########] [load ##########]\n" +
		"SHIFT LEFT: preserve cache affinity; cap or route decode-heavy output\n" +
		"BOUNDARY: forecast only; reconcile provider-observed usage after completion\n"
	if got := out.String(); got != want {
		t.Fatalf("captured halo:\n%s\nwant:\n%s", got, want)
	}
}

func TestTokenProfileHaloJSONContract(t *testing.T) {
	var out, errOut bytes.Buffer
	if rc := runTokenProfile(&out, &errOut, []string{"--halo", "--json"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	var got tokenProfileHalo
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak.token-profile-halo.v1" || got.EqualTotal != 102000 || got.ProfileA.TotalTokens != got.ProfileB.TotalTokens || got.VerdictA != "ADMIT" || got.VerdictB != "REFUSE_CLASS_LOAD" {
		t.Fatalf("halo=%+v", got)
	}
}

func TestTokenProfileHaloSelfcheck(t *testing.T) {
	var out, errOut bytes.Buffer
	if rc := runTokenProfile(&out, &errOut, []string{"--selfcheck"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	if got, want := out.String(), "SELFCHECK OK: equal totals, unequal cost/load, different class outcomes\n"; got != want {
		t.Fatalf("selfcheck=%q want=%q", got, want)
	}
	if strings.TrimSpace(errOut.String()) != "" {
		t.Fatalf("stderr=%q", errOut.String())
	}
}
