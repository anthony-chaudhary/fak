package referenceoracle_test

import (
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/referenceoracle"
)

// The fixture is the retained V4.1 gold-prefill trace pair from the 2026-10-12
// layer-parity capture, with host detail scrubbed; digests, builds, coverage
// and caveats are unchanged.
func TestV41GoldPrefillTracePairPinsRetainedChain(t *testing.T) {
	tp, err := referenceoracle.ReadTracePair("testdata/v41-gold-prefill-trace-pair.json")
	if err != nil {
		t.Fatal(err)
	}
	if tp.Label != "v41-gold-prefill" || tp.CapturedUTC != "2026-10-12" {
		t.Fatalf("label=%q captured=%q", tp.Label, tp.CapturedUTC)
	}
	r := tp.Reference
	if r.Repo != "vcruz305/llama.cpp" || r.Branch != "runtime/deepseek41" ||
		r.Commit != "5210c7c5ed61dddaee6ed476623abf4b63093d16" ||
		r.LibSHA256 != "f619ee177c6d8cd7" || r.TracerSHA256 != "80a2cbf7e5346277" {
		t.Fatalf("reference=%+v", r)
	}
	n := tp.Native
	if n.Repo != "anthony-chaudhary/fak" ||
		n.Commit != "b98e317bb188d5c8370fe8e840885a5e959dea35" ||
		!slices.Equal(n.ExtraCommits, []string{"edcd285ae19"}) {
		t.Fatalf("native=%+v", n)
	}
	if tp.ForkRun.Nodes != 769 || tp.ForkRun.DurationS != 74 || tp.ForkRun.ExitCode != 0 ||
		tp.ForkRun.LeaseSuffix != "01464bcf6ebd" || tp.ForkRun.RunName != "run-gold-20261012T052134Z" {
		t.Fatalf("fork_run=%+v", tp.ForkRun)
	}
	if tp.Prompt.Text != "The chemical symbol for gold is" ||
		!slices.Equal(tp.Prompt.InputTokenIDs, []int{0, 671, 7658, 7380, 362, 8495, 344}) {
		t.Fatalf("prompt=%+v", tp.Prompt)
	}
	c := tp.Coverage
	if c.Layers != 40 || c.Stages != 5 || c.Tokens != 7 || c.SamplesPerEvent != 8 ||
		c.Events != 1400 || c.Samples != 11200 ||
		!slices.Equal(c.ExcludedSampleIndexes, []int{448}) || c.NonfiniteRows != 16 {
		t.Fatalf("coverage=%+v", c)
	}
	wantDigests := map[string]string{
		"index.jsonl":             "19700554d28f89ae",
		"fak-1_gold_prefill.json": "e525bffccb7074ff",
		"ref.json":                "dad90189af9bacf9",
		"profile.json":            "767d93e793c8e2fa",
	}
	if len(tp.Digests) != len(wantDigests) {
		t.Fatalf("digests=%+v", tp.Digests)
	}
	for _, d := range tp.Digests {
		if wantDigests[d.Name] != d.SHA256 {
			t.Fatalf("digest %s=%q role=%q", d.Name, d.SHA256, d.Role)
		}
	}
	if !tp.AdoptedDigests || tp.DigestProvenance == "" || len(tp.Caveats) == 0 {
		t.Fatalf("provenance=%q caveats=%d adopted=%t", tp.DigestProvenance, len(tp.Caveats), tp.AdoptedDigests)
	}
}

func TestTracePairRoundTrip(t *testing.T) {
	tp, err := referenceoracle.ReadTracePair("testdata/v41-gold-prefill-trace-pair.json")
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(tp)
	if err != nil {
		t.Fatal(err)
	}
	back, err := referenceoracle.ParseTracePair(data)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(back, tp) {
		t.Fatalf("round trip differs:\n%+v\n%+v", back, tp)
	}
}

func TestTracePairTamperingFailsClosed(t *testing.T) {
	base, err := referenceoracle.ReadTracePair("testdata/v41-gold-prefill-trace-pair.json")
	if err != nil {
		t.Fatal(err)
	}
	parsed := func(mutate func(*referenceoracle.TracePair)) error {
		doc := base
		mutate(&doc)
		data, err := json.Marshal(doc)
		if err != nil {
			t.Fatal(err)
		}
		_, err = referenceoracle.ParseTracePair(data)
		return err
	}
	cases := map[string]func(*referenceoracle.TracePair){
		"events off by one":       func(tp *referenceoracle.TracePair) { tp.Coverage.Events = 1401 },
		"samples off by one":      func(tp *referenceoracle.TracePair) { tp.Coverage.Samples = 11201 },
		"zero layers":             func(tp *referenceoracle.TracePair) { tp.Coverage.Layers = 0 },
		"excluded index at bound": func(tp *referenceoracle.TracePair) { tp.Coverage.ExcludedSampleIndexes = []int{11200} },
		"negative excluded index": func(tp *referenceoracle.TracePair) { tp.Coverage.ExcludedSampleIndexes = []int{-1} },
		"nonfinite rows too many": func(tp *referenceoracle.TracePair) { tp.Coverage.NonfiniteRows = 1401 },
		"digest not 16-hex":       func(tp *referenceoracle.TracePair) { tp.Digests[0].SHA256 = "zzz" },
		"digest full length": func(tp *referenceoracle.TracePair) {
			tp.Digests[0].SHA256 = "19700554d28f89ae19700554d28f89ae19700554d28f89ae19700554d28f89ae"
		},
		"duplicate digest name":      func(tp *referenceoracle.TracePair) { tp.Digests[1].Name = tp.Digests[0].Name },
		"digest missing role":        func(tp *referenceoracle.TracePair) { tp.Digests[2].Role = "" },
		"no digests":                 func(tp *referenceoracle.TracePair) { tp.Digests = nil },
		"adopted without provenance": func(tp *referenceoracle.TracePair) { tp.DigestProvenance = "" },
		"adopted without caveats":    func(tp *referenceoracle.TracePair) { tp.Caveats = nil },
		"missing native commit":      func(tp *referenceoracle.TracePair) { tp.Native.Commit = "" },
		"empty prompt ids":           func(tp *referenceoracle.TracePair) { tp.Prompt.InputTokenIDs = nil },
	}
	for name, mutate := range cases {
		if err := parsed(mutate); !errors.Is(err, referenceoracle.ErrInvalidTracePair) {
			t.Errorf("%s: err=%v, want ErrInvalidTracePair", name, err)
		}
	}
	if _, err := referenceoracle.ParseTracePair([]byte(`{"schema":"fak.reference_oracle.v1"}`)); !errors.Is(err, referenceoracle.ErrInvalidTracePair) {
		t.Fatalf("foreign schema err=%v", err)
	}
}
