package referenceoracle_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/referenceoracle"
)

// TestPickBinaryDigestDerivesFromReceiptRepo locks the adapter's binary-digest
// resolution through the public Parse seam: the preferred keys come from the
// engine the receipt itself names in reference_engine.repo, never from a
// hardcoded engine list.
func TestPickBinaryDigestDerivesFromReceiptRepo(t *testing.T) {
	doc := func(repo, sha256 string) []byte {
		return []byte(`{"schema":"fak.v41_reference_oracle.v1",
			"reference_engine":{"repo":` + repo + `,"commit":"c","sha256":` + sha256 + `},
			"model":{"repo_id":"m"},
			"prompts":{"p":{"input_token_ids":[1],"greedy_token_ids":[2]}}}`)
	}
	cases := map[string]struct {
		repo   string
		sha256 string
		want   string
	}{
		"server preferred":    {`"https://github.com/vcruz305/llama.cpp"`, `{"llama-cli":"b","llama-server":"s"}`, "s"},
		"cli when no server":  {`"https://github.com/vcruz305/llama.cpp"`, `{"llama-cli":"b"}`, "b"},
		"bare repo name":      {`"llama.cpp"`, `{"other":"x","llama-server":"s"}`, "s"},
		"git suffix":          {`"https://github.com/foo/bar.cpp.git"`, `{"bar-cli":"c","bar-server":"s"}`, "s"},
		"trailing slash url":  {`"https://github.com/vcruz305/llama.cpp/"`, `{"llama-cli":"b","llama-server":"s"}`, "s"},
		"trailing slash bare": {`"llama.cpp/"`, `{"llama-cli":"b","llama-server":"s"}`, "s"},
		"other engine named":  {`"https://github.com/foo/engine.x"`, `{"engine.x-cli":"c","engine.x-server":"s"}`, "s"},
		// Blank repo: trims to no engine name, so no preferred keys derive.
		// (A truly empty repo is rejected by Parse validation.)
		"blank repo single": {`" "`, `{"anything":"x"}`, "x"},
		"blank repo multi":  {`" "`, `{"a":"1","b":"2"}`, ""},
		"unrelated entries": {`"https://github.com/foo/bar"`, `{"x":"1","y":"2"}`, ""},
		"uppercase repo":    {`"https://github.com/foo/Llama.CPP"`, `{"llama-cli":"b"}`, "b"},
	}
	for name, tc := range cases {
		o, err := referenceoracle.Parse(doc(tc.repo, tc.sha256))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if o.Reference.BinarySHA256 != tc.want {
			t.Errorf("%s: binary_sha256=%q, want %q", name, o.Reference.BinarySHA256, tc.want)
		}
	}
}

// TestExplicitBinarySHA256Wins locks that an explicit binary_sha256 field is
// always honored over any derived lookup.
func TestExplicitBinarySHA256Wins(t *testing.T) {
	data := []byte(`{"schema":"fak.v41_reference_oracle.v1",
		"reference_engine":{"repo":"https://github.com/vcruz305/llama.cpp","commit":"c",
			"binary_sha256":"explicit","sha256":{"llama-server":"s"}},
		"model":{"repo_id":"m"},
		"prompts":{"p":{"input_token_ids":[1],"greedy_token_ids":[2]}}}`)
	o, err := referenceoracle.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	if o.Reference.BinarySHA256 != "explicit" {
		t.Fatalf("binary_sha256=%q, want explicit", o.Reference.BinarySHA256)
	}
}
