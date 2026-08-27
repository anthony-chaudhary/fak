package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCompareProfilesKeepAndReject(t *testing.T) {
	r := compareProfiles([]float64{100, 100, 100}, []float64{80, 84, 82}, 15, true)
	if r.Verdict != "KEEP" {
		t.Fatal(r)
	}
	r = compareProfiles([]float64{100, 100, 100}, []float64{80, 101, 79}, 15, true)
	if r.Verdict != "REJECT" {
		t.Fatal(r)
	}
}
func TestLoadPrefillRequiresExactNativeP32T64(t *testing.T) {
	p := filepath.Join(t.TempDir(), "p.json")
	os.WriteFile(p, []byte(`{"schema":"fak-native-performance-profile/1","prompt_tokens":32,"completion_tokens":64,"prefill_ms":10,"engine":"fak-native","fallback":"none","artifact":"sha256:x"}`), 0600)
	if _, err := loadPrefill(p); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(p, []byte(`{"schema":"fak-native-performance-profile/1","prompt_tokens":31}`), 0600)
	if _, err := loadPrefill(p); err == nil {
		t.Fatal("wrong profile accepted")
	}
}
