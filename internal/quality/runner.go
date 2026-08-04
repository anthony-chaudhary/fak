package quality

// Runner is the reference-runner adapter (#4518): the seam a differential test
// plugs a decode path into. The spine judges an ENGINE runner's trace against a
// REFERENCE runner's trace, so both a golden implementation and the path under
// test satisfy the same interface. Real adapters — a llama.cpp reference, a fak
// engine mode — wire in here behind Run without changing the harness.
type Runner interface {
	Name() string
	Run(c QualityCase) (Trace, error)
}

// ReferenceRunner is the golden path: it returns the case's declared reference
// trace verbatim, stamped with its own runner name. It is the baseline every
// differential oracle names — the "what the case says correct looks like" side.
type ReferenceRunner struct{}

func (ReferenceRunner) Name() string { return "reference" }

func (r ReferenceRunner) Run(c QualityCase) (Trace, error) {
	t := c.Reference
	t.Runner = r.Name()
	return t, nil
}

// ScriptedRunner is an engine-path adapter that replays a fixed trace. It models a
// specific engine build/mode for hermetic tests and for the CLI demo, and is the
// shape real engine adapters take: capture the engine's decode into a Trace and
// return it. The spine never trusts the runner's self-description — only its
// emitted tokens/text, judged against the reference.
type ScriptedRunner struct {
	Label string
	Trace Trace
}

func (s ScriptedRunner) Name() string {
	if s.Label != "" {
		return s.Label
	}
	return "engine"
}

func (s ScriptedRunner) Run(_ QualityCase) (Trace, error) {
	t := s.Trace
	t.Runner = s.Name()
	return t, nil
}

// DemoCase is the built-in spine case: a tiny greedy executive-report decode with a
// grounded reference and a rubric requiring the two material claims. `fak quality
// run` uses it so the spine is runnable end-to-end with no corpus on disk, and the
// epic Witness ("intentionally injected decode and report-quality defects each trip
// the expected gate") exercises it via DemoEngine.
func DemoCase() QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      "spine-demo-exec-report",
		Version: 1,
		Prompt:  "Summarize this week's throughput for the executive rollup.",
		Params:  SamplingParams{Temperature: 0, MaxTokens: 8},
		Reference: Trace{
			Tokens: []string{"Throughput", "increased", "12", "%", "week", "over", "week", "."},
			Text:   "Throughput increased 12% week over week.",
		},
		Oracles: []string{"greedy-token-diff", "grounding-rubric"},
		Rubric: RubricSpec{
			Required:  []string{"throughput", "12%"},
			Forbidden: []string{"decreased"},
			MinScore:  1,
		},
	}
}

// DemoEngine returns an engine runner for the demo case with an optional injected
// defect: "" reproduces the reference (clean pass); "decode" flips one token so the
// greedy differential oracle fails at that index; "stop" decodes past the reference's
// last token so the failure localizes to the stop decision; "report" corrupts the text
// so the grounding rubric fails on a forbidden/omitted claim. This is the deterministic
// mutant source the spine test and CLI use to prove each gate trips.
func DemoEngine(defect string) ScriptedRunner {
	ref := DemoCase().Reference
	switch defect {
	case "decode":
		// Flip token 1 "increased" -> "decreased": fluent text, wrong direction.
		toks := append([]string(nil), ref.Tokens...)
		toks[1] = "decreased"
		return ScriptedRunner{
			Label: "engine-decode-defect",
			Trace: Trace{Tokens: toks, Text: "Throughput decreased 12% week over week."},
		}
	case "stop":
		// The stop token is not honored: the engine reproduces the reference and
		// then keeps decoding past it. Every shared token still agrees, so the
		// only thing that differs is where the stream ended — the planted defect
		// that localizes to the "stops" stage rather than to the decode (#4520).
		toks := append(append([]string(nil), ref.Tokens...), "Also", ",", "revenue", "rose", ".")
		return ScriptedRunner{
			Label: "engine-stop-defect",
			Trace: Trace{Tokens: toks, Text: ref.Text + " Also, revenue rose."},
		}
	case "report":
		// Tokens match, but the assembled text drops the required "12%" figure and
		// asserts a forbidden direction — the "reads fine, wrong facts" defect.
		return ScriptedRunner{
			Label: "engine-report-defect",
			Trace: Trace{Tokens: ref.Tokens, Text: "Throughput decreased this week."},
		}
	default:
		return ScriptedRunner{Label: "engine-clean", Trace: ref}
	}
}
