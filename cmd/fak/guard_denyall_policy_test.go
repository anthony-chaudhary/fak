package main

import (
	"reflect"
	"testing"
)

func TestDenyAllRetrySettingsConsolidateLimits(t *testing.T) {
	p := newDenyAllRetrySettings()
	if err := p.Set("shadow,warn=3,final=7,max=9,same-stop=8"); err != nil {
		t.Fatal(err)
	}
	want := denyAllRetrySettings{Mode: "shadow", Warn: 3, Final: 7, Max: 9, SameStop: 8}
	if p != want {
		t.Fatalf("policy = %+v, want %+v", p, want)
	}
}

func TestRewriteLegacyDenyAllArgs(t *testing.T) {
	got := rewriteLegacyDenyAllArgs([]string{
		"--deny-all-continue=shadow", "--deny-all-warn", "3", "--deny-all-final=7",
		"--deny-all-max", "9", "--same-stop=8", "--", "claude", "--same-stop", "99",
	})
	want := []string{
		"--deny-all-continue=shadow", "--deny-all-continue=warn=3",
		"--deny-all-continue=final=7", "--deny-all-continue=max=9",
		"--deny-all-continue=same-stop=8", "--", "claude", "--same-stop", "99",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("rewrite = %#v, want %#v", got, want)
	}
}

func TestDenyAllRetrySettingsRejectUnknownSettings(t *testing.T) {
	p := newDenyAllRetrySettings()
	for _, raw := range []string{"maybe", "max=nope", "mystery=3", "enforce,,max=4"} {
		if err := p.Set(raw); err == nil {
			t.Errorf("Set(%q) accepted an invalid policy", raw)
		}
	}
}
