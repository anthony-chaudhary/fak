package hwgatelint

import "sort"

// FixtureCase is one labeled sample of agent final-output text and what the
// scanner must say about it: whether it is fleet-clean, and (when not) which
// Classes must appear. The corpus spans honest dispatched-to-the-lab outputs,
// the "already redirected" suppression cases, and the hardware-gate shapes
// across every Class.
type FixtureCase struct {
	Name   string  `json:"name"`
	Text   string  `json:"text"`
	Clean  bool    `json:"clean"`
	Expect []Class `json:"expect,omitempty"`
}

// FixtureOutcome is the graded result of one FixtureCase.
type FixtureOutcome struct {
	Name    string  `json:"name"`
	Clean   bool    `json:"expect_clean"`
	Got     string  `json:"got_verdict"`
	Classes []Class `json:"got_classes"`
	OK      bool    `json:"ok"`
}

// Fixture is the built-in corpus, asserted by hwgatelint_test.go and runnable
// from the CLI via `fak hwgate-lint --self-test` — the scanner's own
// re-derivable witness.
func Fixture() []FixtureCase {
	return []FixtureCase{
		// --- fleet-clean: dispatched the work, or never gated on hardware ---
		{
			Name:  "clean-dispatched-dgx",
			Text:  "The device-GEMM witness needs a GPU, which this laptop lacks — so I dispatched it to the DGX via dgxbridge and recorded the result.",
			Clean: true,
		},
		{
			Name:  "clean-gpu-node-ran-it",
			Text:  "Ran the CUDA acceptance suite on fak-realmodel (L4 sm_89); Q8 device-GEMM matches the reference within tolerance.",
			Clean: true,
		},
		{
			Name:  "clean-handed-operator-command",
			Text:  "No credential session for the GPU node is live here. Produced the ready-to-run command sequence and handed it to the operator to run on GCP.",
			Clean: true,
		},
		{
			Name:  "clean-no-hardware-mention",
			Text:  "Implemented the loader and committed as abc123. Tests pass (go test ./internal/loader).",
			Clean: true,
		},

		// --- hardware-gated: one shape per Class ---
		{
			Name:   "gate-no-local-gpu",
			Text:   "Not yet — there is no GPU on this host, so I can't run the CUDA device witness.",
			Expect: []Class{NoLocalGPU},
		},
		{
			Name:   "gate-requires-accelerator",
			Text:   "This benchmark requires an NVIDIA accelerator and none is available on the machine.",
			Expect: []Class{NoLocalGPU},
		},
		{
			Name:   "gate-no-node-runtime",
			Text:   "Blocked: the web harness needs Node.js and npm, which are not installed here.",
			Expect: []Class{NoLocalRuntime},
		},
		{
			Name:   "gate-local-boundary",
			Text:   "Can't reproduce the throughput number on this laptop — not enough memory to load the model.",
			Expect: []Class{LocalBoundary},
		},
	}
}

// RunFixture grades the corpus: a case passes when its verdict matches Clean and
// (when dirty) every expected Class appears. Returns the per-case outcomes and
// the number that passed.
func RunFixture() (cases []FixtureOutcome, passed int) {
	for _, fc := range Fixture() {
		rep := Scan(fc.Text)
		got := make([]Class, 0, len(rep.Classes))
		for c := range rep.Classes {
			got = append(got, c)
		}
		sort.Slice(got, func(i, j int) bool { return got[i] < got[j] })

		ok := (rep.Verdict == Clean) == fc.Clean
		if ok && !fc.Clean {
			for _, want := range fc.Expect {
				if rep.Classes[want] == 0 {
					ok = false
					break
				}
			}
		}
		cases = append(cases, FixtureOutcome{
			Name:    fc.Name,
			Clean:   fc.Clean,
			Got:     rep.Verdict,
			Classes: got,
			OK:      ok,
		})
		if ok {
			passed++
		}
	}
	return cases, passed
}
