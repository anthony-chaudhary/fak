package model

import "testing"

// fak-test:runtime fast est=1ms
func TestDraftKVDtypeInheritance(t *testing.T) {
	t.Parallel()
	override := KVPrecisionFP16
	empty := KVPrecision("")
	for _, tc := range []struct {
		name     string
		target   KVPrecision
		override *KVPrecision
		usesKV   bool
		want     KVPrecision
	}{
		{"model inherits target", KVPrecisionQ8_0, nil, true, KVPrecisionQ8_0},
		{"model explicit override", KVPrecisionQ8_0, &override, true, KVPrecisionFP16},
		{"explicit empty override", KVPrecisionQ8_0, &empty, true, ""},
		{"unset target stays unset", "", nil, true, ""},
		{"ngram has no KV", KVPrecisionQ8_0, &override, false, ""},
		{"suffix has no KV", KVPrecisionBF16, nil, false, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := ResolveDraftKVDtype(tc.target, tc.override, tc.usesKV); got != tc.want {
				t.Fatalf("ResolveDraftKVDtype = %q, want %q", got, tc.want)
			}
		})
	}
	if override != KVPrecisionFP16 || empty != "" {
		t.Fatal("resolution mutated an override")
	}
}
