package referenceoracle_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/referenceoracle"
)

// The fixture is the fak-private 376a40633 V4.1 oracle receipt with host and
// lease detail removed; prompts, ids and log-probabilities are unchanged.
func TestV41OracleLoadsThroughAdapter(t *testing.T) {
	o, err := referenceoracle.ReadFile("testdata/v41-reference-oracle.json")
	if err != nil {
		t.Fatal(err)
	}
	if o.Schema != referenceoracle.Schema || o.SourceSchema != referenceoracle.LegacySchemaV41 {
		t.Fatalf("schema=%q source=%q", o.Schema, o.SourceSchema)
	}
	if o.Reference.SHA != "5210c7c5ed61dddaee6ed476623abf4b63093d16" || o.Reference.BinarySHA256 != "9e30ee6e8e6673b08beb21b514c0e05c3f98a4e3f67d3d4dd62c2bf539f85a6a" {
		t.Fatalf("reference=%+v", o.Reference)
	}
	if o.Artifact.Rev != "543d86fd975044332d4366204eadd6da1b0cbeda" || o.Artifact.ShardCount != 7 || len(o.Artifact.Shards) != 1 {
		t.Fatalf("artifact=%+v", o.Artifact)
	}
	rb := o.Artifact.Rebind
	if rb == nil || rb.FromRev != "543d86fd975044332d4366204eadd6da1b0cbeda" || rb.ToRev != "c4a085541cb53f67ee5e57b63d255e80cef286e7" || rb.WitnessedUTC != "2026-10-27" {
		t.Fatalf("rebind=%+v", rb)
	}
	if !slices.Equal(rb.Matched, []string{"shard_count", "shard1_sha256", "all_shard_sha256", "total_bytes"}) {
		t.Fatalf("rebind matched=%v", rb.Matched)
	}
	// The rebind must record the full seven-shard byte identity re-verified
	// against the live revision, so an independent verifier needs nothing
	// outside this document.
	digests := []string{
		"0bcee934bd4e8350c54410681d9300b0321a300fcb1df1437a20703394d08ee0",
		"124ffa15b6b7ec9630715ad18752a4c3f371749647837b8c9d5f7eb52c4f73c0",
		"4dd35b0b086cc0fb19d3a72afd8aed331317950e2257f37b9672702edc3f2453",
		"d24832f4c2f42340d5d4bf4d9b3277c861e0a0a9ccd25e94bda5581c5351f960",
		"34714c3880bde310207e3d12fa488d8d676f338ff635157fded8ec73e133efad",
		"08dc941d26687cb51f736a4f1c3a2c4843fb1d79573164c645106711081f5faf",
		"550bbdb94a69142abfa4b2f2db86ffe3dd6d4ca45eafbb6728f4fe40506fa6b9",
	}
	sizes := []int64{43072989056, 44805480352, 14935107904, 44217904672, 44804598176, 44803713696, 27875485600}
	if len(rb.Shards) != 7 {
		t.Fatalf("rebind shards=%d, want 7", len(rb.Shards))
	}
	var total int64
	for i, s := range rb.Shards {
		want := fmt.Sprintf("DeepSeek-V4.1-Flash-Q2_K-%05d-of-00007.gguf", i+1)
		if s.Name != want || s.SHA256 != digests[i] || s.SizeBytes != sizes[i] {
			t.Fatalf("rebind shard %d = %+v, want name %q sha %s size %d", i, s, want, digests[i], sizes[i])
		}
		total += s.SizeBytes
	}
	if total != 264515279456 {
		t.Fatalf("rebind shard bytes sum to %d, want 264515279456", total)
	}
	wantTexts := []string{"The chemical symbol for gold is", "The capital of France is", "Say hello.", "What is 17 times 3?"}
	var texts []string
	for _, p := range o.Prompts {
		texts = append(texts, p.Text)
		if len(p.GreedyIDs) != 16 || len(p.Top5) != 16 || len(p.Top5[0]) != 5 {
			t.Fatalf("%q: greedy=%d top5=%d", p.Text, len(p.GreedyIDs), len(p.Top5))
		}
	}
	if !slices.Equal(texts, wantTexts) {
		t.Fatalf("prompt order %q", texts)
	}
	gold := o.Prompts[0]
	if !slices.Equal(gold.InputTokenIDs, []int{0, 671, 7658, 7380, 362, 8495, 344}) || gold.GreedyIDs[1] != 48865 || gold.Top5[0][0].Logprob != -1.103766 {
		t.Fatalf("gold prompt=%+v", gold.InputTokenIDs)
	}
	if !o.SelfParity.Checked || !o.SelfParity.Identical {
		t.Fatalf("self parity=%+v", o.SelfParity)
	}
}

func TestNativeOracleRoundTrip(t *testing.T) {
	o, err := referenceoracle.ReadFile("testdata/v41-reference-oracle.json")
	if err != nil {
		t.Fatal(err)
	}
	o.SourceSchema = ""
	o.TraceRefs = []referenceoracle.TraceRef{{Prompt: 1, Path: "r0/raw/p1.trace.json", StageMap: "deepseek41", Layers: 43, Tokens: 8}}
	data, err := json.Marshal(o)
	if err != nil {
		t.Fatal(err)
	}
	back, err := referenceoracle.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	if ref, ok := back.TraceRef(1); !ok || ref.Path != "r0/raw/p1.trace.json" || ref.Layers != 43 {
		t.Fatalf("trace ref=%+v ok=%t", ref, ok)
	}
	if _, ok := back.TraceRef(0); ok {
		t.Fatal("prompt 0 has no trace ref")
	}
	if back.Artifact.Rebind == nil || back.Artifact.Rebind.ToRev != "c4a085541cb53f67ee5e57b63d255e80cef286e7" {
		t.Fatalf("rebind lost in native round trip: %+v", back.Artifact.Rebind)
	}
	if got := back.Artifact.Rebind.Shards; len(got) != 7 || got[0].SHA256 != "0bcee934bd4e8350c54410681d9300b0321a300fcb1df1437a20703394d08ee0" || got[6].SHA256 != "550bbdb94a69142abfa4b2f2db86ffe3dd6d4ca45eafbb6728f4fe40506fa6b9" {
		t.Fatalf("rebind shard table lost in native round trip: %+v", got)
	}
	if !referenceoracle.IsOracleSchema("fak.qwen38_reference_oracle.v1") || referenceoracle.IsOracleSchema("fak.compute_trace.v1") {
		t.Fatal("IsOracleSchema")
	}
}

func TestOracleRefusals(t *testing.T) {
	base := func() referenceoracle.Oracle {
		return referenceoracle.Oracle{
			Schema:    referenceoracle.Schema,
			Reference: referenceoracle.Reference{Repo: "r", SHA: "s"},
			Artifact:  referenceoracle.Artifact{Repo: "a"},
			Prompts: []referenceoracle.Prompt{{
				Text: "p", InputTokenIDs: []int{1}, GreedyIDs: []int{7},
				Top5: [][]referenceoracle.TokenProb{{{ID: 7, Logprob: -0.1}, {ID: 8, Logprob: -2}}},
			}},
		}
	}
	if err := base().Validate(); err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(*referenceoracle.Oracle){
		"no reference sha":       func(o *referenceoracle.Oracle) { o.Reference.SHA = "" },
		"no prompts":             func(o *referenceoracle.Oracle) { o.Prompts = nil },
		"top5 not led by greedy": func(o *referenceoracle.Oracle) { o.Prompts[0].GreedyIDs = []int{8} },
		"top5 ascending":         func(o *referenceoracle.Oracle) { o.Prompts[0].Top5[0][1].Logprob = -0.01 },
		"positive logprob":       func(o *referenceoracle.Oracle) { o.Prompts[0].Top5[0][0].Logprob = 0.5 },
		"top5 rows short":        func(o *referenceoracle.Oracle) { o.Prompts[0].GreedyIDs = []int{7, 9} },
		"trace ref out of range": func(o *referenceoracle.Oracle) { o.TraceRefs = []referenceoracle.TraceRef{{Prompt: 3, Path: "t.json"}} },
		"trace ref escapes":      func(o *referenceoracle.Oracle) { o.TraceRefs = []referenceoracle.TraceRef{{Path: "../t.json"}} },
		"duplicate trace ref": func(o *referenceoracle.Oracle) {
			o.TraceRefs = []referenceoracle.TraceRef{{Path: "a.json"}, {Path: "b.json"}}
		},
		"rebind missing to rev": func(o *referenceoracle.Oracle) {
			o.Artifact.Rebind = &referenceoracle.Rebind{FromRev: "a", WitnessedUTC: "2026-10-27"}
		},
		"rebind same rev": func(o *referenceoracle.Oracle) {
			o.Artifact.Rebind = &referenceoracle.Rebind{FromRev: "a", ToRev: "a", WitnessedUTC: "2026-10-27"}
		},
		"rebind shards short": func(o *referenceoracle.Oracle) {
			o.Artifact.ShardCount = 7
			o.Artifact.Rebind = &referenceoracle.Rebind{FromRev: "a", ToRev: "b", WitnessedUTC: "2026-10-27",
				Shards: []referenceoracle.Shard{{Name: "s1.gguf", SHA256: strings.Repeat("ab", 32)}}}
		},
		"rebind shard empty name": func(o *referenceoracle.Oracle) {
			o.Artifact.ShardCount = 1
			o.Artifact.Rebind = &referenceoracle.Rebind{FromRev: "a", ToRev: "b", WitnessedUTC: "2026-10-27",
				Shards: []referenceoracle.Shard{{SHA256: strings.Repeat("cd", 32)}}}
		},
		"rebind shard duplicate name": func(o *referenceoracle.Oracle) {
			o.Artifact.ShardCount = 2
			o.Artifact.Rebind = &referenceoracle.Rebind{FromRev: "a", ToRev: "b", WitnessedUTC: "2026-10-27",
				Shards: []referenceoracle.Shard{{Name: "s.gguf", SHA256: strings.Repeat("ab", 32)}, {Name: "s.gguf", SHA256: strings.Repeat("cd", 32)}}}
		},
		"rebind shard uppercase digest": func(o *referenceoracle.Oracle) {
			o.Artifact.ShardCount = 1
			o.Artifact.Rebind = &referenceoracle.Rebind{FromRev: "a", ToRev: "b", WitnessedUTC: "2026-10-27",
				Shards: []referenceoracle.Shard{{Name: "s.gguf", SHA256: strings.ToUpper(strings.Repeat("ab", 32))}}}
		},
		"rebind shard negative size": func(o *referenceoracle.Oracle) {
			o.Artifact.ShardCount = 1
			o.Artifact.Rebind = &referenceoracle.Rebind{FromRev: "a", ToRev: "b", WitnessedUTC: "2026-10-27",
				Shards: []referenceoracle.Shard{{Name: "s.gguf", SHA256: strings.Repeat("ef", 32), SizeBytes: -1}}}
		},
	}
	for name, mutate := range cases {
		o := base()
		mutate(&o)
		if err := o.Validate(); !errors.Is(err, referenceoracle.ErrInvalid) {
			t.Errorf("%s: err=%v, want ErrInvalid", name, err)
		}
	}
	// A complete, well-formed table must pass: fail-closed means rejecting the
	// bad, not rejecting everything.
	ok := base()
	ok.Artifact.ShardCount = 1
	ok.Artifact.Rebind = &referenceoracle.Rebind{FromRev: "a", ToRev: "b", WitnessedUTC: "2026-10-27",
		Shards: []referenceoracle.Shard{{Name: "s.gguf", SHA256: strings.Repeat("ab", 32), SizeBytes: 10}}}
	if err := ok.Validate(); err != nil {
		t.Errorf("valid populated shard table rejected: %v", err)
	}
	if _, err := referenceoracle.Decode(strings.NewReader(`{"schema":"fak.compute_trace.v1"}`)); !errors.Is(err, referenceoracle.ErrUnknownSchema) {
		t.Fatalf("unknown schema err=%v", err)
	}
	if _, err := referenceoracle.Parse([]byte(`{"schema":"fak.v41_reference_oracle.v1","prompts":[1]}`)); !errors.Is(err, referenceoracle.ErrInvalid) {
		t.Fatalf("array prompts err=%v", err)
	}
}

func TestLegacySelfParityNeedsEveryPrompt(t *testing.T) {
	doc := func(results string) []byte {
		return []byte(`{"schema":"fak.qwen38_reference_oracle.v1",
			"reference_engine":{"repo":"r","commit":"c"},"model":{"repo_id":"m","shards":3},
			"determinism":{"method":"rerun","results":` + results + `},
			"prompts":{"a":{"input_token_ids":[1],"greedy_token_ids":[2]},"b":{"input_token_ids":[3],"greedy_token_ids":[4]}}}`)
	}
	for name, tc := range map[string]struct {
		results            string
		checked, identical bool
	}{
		"all prompts":      {`[{"prompt":"a","greedy_tokens_identical":true,"top5_logprobs_identical_1e-6":true},{"prompt":"b","greedy_tokens_identical":true,"top5_logprobs_identical_1e-6":true}]`, true, true},
		"one differs":      {`[{"prompt":"a","greedy_tokens_identical":true,"top5_logprobs_identical_1e-6":true},{"prompt":"b","greedy_tokens_identical":false,"top5_logprobs_identical_1e-6":true}]`, true, false},
		"partial":          {`[{"prompt":"a","greedy_tokens_identical":true,"top5_logprobs_identical_1e-6":true}]`, false, false},
		"duplicate prompt": {`[{"prompt":"a","greedy_tokens_identical":true,"top5_logprobs_identical_1e-6":true},{"prompt":"a","greedy_tokens_identical":true,"top5_logprobs_identical_1e-6":true}]`, false, false},
		"unknown prompt":   {`[{"prompt":"a","greedy_tokens_identical":true,"top5_logprobs_identical_1e-6":true},{"prompt":"z","greedy_tokens_identical":true,"top5_logprobs_identical_1e-6":true}]`, false, false},
	} {
		o, err := referenceoracle.Parse(doc(tc.results))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if o.SelfParity.Checked != tc.checked || o.SelfParity.Identical != tc.identical || o.Artifact.ShardCount != 3 {
			t.Errorf("%s: self parity=%+v shards=%d", name, o.SelfParity, o.Artifact.ShardCount)
		}
	}
}
