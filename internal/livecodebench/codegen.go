package livecodebench

import "fmt"

// codegen.go implements the LiveCodeBench code-generation scenario end to end
// against the fak adapter (#2096, epic #2085): given the n completions sampled
// per problem, it extracts the gradeable code from each sample (ExtractCode),
// grades each via an injected grader, tallies the passing samples, and computes
// the per-problem and summary pass@1 / pass@5 with the unbiased Chen et al.
// estimator (PassAtK / MeanPassAtK). Extraction and grading are pure/injected,
// so the whole scenario is unit-tested without a network or a code sandbox; the
// CLI plugs the real gateway sampler (raw arm) and the official lcb_runner
// checker as the grader. The scorer carries NO evidence class and NO result
// claim — the honesty fence stays with the caller, so a pass rate is claimable
// only when the injected grader is the official one.

// Upstream lcb_runner code-generation sampling defaults. A fak codegen run
// mirrors them so a fak run and a raw `python -m lcb_runner.runner.main` run
// sample the same way unless explicitly overridden.
const (
	CodegenDefaultN           = 10
	CodegenDefaultTemperature = 0.2
)

// CodegenConfig pins the codegen scenario's recorded sampling parameters. N and
// Temperature are recorded verbatim in the report so a reader can trace which
// sampling regime produced the pass rates; Scenario is carried for the report
// header (it is always ScenarioCodeGeneration for this scorer).
type CodegenConfig struct {
	Scenario    Scenario
	N           int
	Temperature float64
}

// CodegenGrader grades ONE extracted candidate program for a problem, returning
// whether it passes the problem's tests. It is injected (like RawArmSampler) so
// the scenario is unit-tested without executing untrusted code and the CLI can
// plug the official lcb_runner checker or a sandbox. A NoCode extraction is
// never handed to the grader — it is scored an automatic miss.
type CodegenGrader func(p Problem, code string) (pass bool, err error)

// CodegenProblemScore is one problem's per-problem codegen result: n samples of
// which Correct passed grading (Extracted of them yielded gradeable code at
// all), and the pass@1 / pass@5 that (n, Correct) implies.
type CodegenProblemScore struct {
	QuestionID string  `json:"question_id"`
	Samples    int     `json:"samples"`   // n: completions graded for this problem
	Extracted  int     `json:"extracted"` // samples that yielded gradeable code
	Correct    int     `json:"correct"`   // c: samples that passed grading
	Pass1      float64 `json:"pass_1"`
	Pass5      float64 `json:"pass_5"`
}

// CodegenSummary is the benchmark-level fold: the mean of the per-problem
// pass@1 / pass@5 across every scored problem, matching how lcb_runner
// aggregates a scenario.
type CodegenSummary struct {
	Problems    int     `json:"problems"`
	Generations int     `json:"generations"`
	Graded      int     `json:"graded"`
	Pass1       float64 `json:"pass_1"`
	Pass5       float64 `json:"pass_5"`
}

// CodegenReport is the codegen scenario's end-to-end result: the recorded
// sampling config, one per-problem row, and the summary pass@1 / pass@5.
type CodegenReport struct {
	Scenario    Scenario              `json:"scenario"`
	N           int                   `json:"n"`
	Temperature float64               `json:"temperature"`
	Problems    []CodegenProblemScore `json:"problems"`
	Summary     CodegenSummary        `json:"summary"`
}

// ScoreCodegen runs the code-generation scenario end to end over the completions
// sampled for each problem. completions[i] holds the samples generated for
// problems[i]; the two slices must be the same length and in the same order.
// For each sample it extracts the gradeable code (ExtractCode with the problem's
// starter_code), scores a NoCode extraction an automatic miss, and otherwise
// hands the extracted program to the injected grader; the passing count c and
// the sample count n feed the unbiased PassAtK estimator for per-problem pass@1
// / pass@5, and the summary is their mean over problems (MeanPassAtK). cfg.N and
// cfg.Temperature are recorded verbatim.
//
// Every problem must carry at least five samples so pass@5 is well-defined
// (upstream samples n=10); the grader is required and its first error aborts the
// run with problem/sample context.
func ScoreCodegen(cfg CodegenConfig, problems []Problem, completions [][]string, grade CodegenGrader) (CodegenReport, error) {
	if grade == nil {
		return CodegenReport{}, fmt.Errorf("livecodebench codegen: grader is required")
	}
	if len(problems) == 0 {
		return CodegenReport{}, fmt.Errorf("livecodebench codegen: at least one problem is required")
	}
	if len(completions) != len(problems) {
		return CodegenReport{}, fmt.Errorf("livecodebench codegen: %d completion sets for %d problems", len(completions), len(problems))
	}

	scen := cfg.Scenario
	if scen == "" {
		scen = ScenarioCodeGeneration
	}
	report := CodegenReport{
		Scenario:    scen,
		N:           cfg.N,
		Temperature: cfg.Temperature,
		Problems:    make([]CodegenProblemScore, 0, len(problems)),
	}

	tallies1 := make([]SampleTally, 0, len(problems))
	tallies5 := make([]SampleTally, 0, len(problems))
	for pi := range problems {
		p := problems[pi]
		samples := completions[pi]
		n := len(samples)
		if n < 5 {
			return CodegenReport{}, fmt.Errorf("livecodebench codegen: problem %q has %d sample(s), pass@5 needs at least 5", p.QuestionID, n)
		}
		extracted, correct := 0, 0
		for si, raw := range samples {
			ex := ExtractCode(raw, p.StarterCode)
			if ex.NoCode {
				continue
			}
			extracted++
			pass, err := grade(p, ex.Code)
			if err != nil {
				return CodegenReport{}, fmt.Errorf("livecodebench codegen: problem %q sample %d: %w", p.QuestionID, si, err)
			}
			if pass {
				correct++
			}
		}
		pass1, err := PassAtK(n, correct, 1)
		if err != nil {
			return CodegenReport{}, fmt.Errorf("livecodebench codegen: problem %q: %w", p.QuestionID, err)
		}
		pass5, err := PassAtK(n, correct, 5)
		if err != nil {
			return CodegenReport{}, fmt.Errorf("livecodebench codegen: problem %q: %w", p.QuestionID, err)
		}
		report.Problems = append(report.Problems, CodegenProblemScore{
			QuestionID: p.QuestionID,
			Samples:    n,
			Extracted:  extracted,
			Correct:    correct,
			Pass1:      pass1,
			Pass5:      pass5,
		})
		tallies1 = append(tallies1, SampleTally{QuestionID: p.QuestionID, Samples: n, Correct: correct})
		tallies5 = append(tallies5, SampleTally{QuestionID: p.QuestionID, Samples: n, Correct: correct})
		report.Summary.Generations += n
		report.Summary.Graded += extracted
	}

	meanPass1, err := MeanPassAtK(tallies1, 1)
	if err != nil {
		return CodegenReport{}, fmt.Errorf("livecodebench codegen: summary pass@1: %w", err)
	}
	meanPass5, err := MeanPassAtK(tallies5, 5)
	if err != nil {
		return CodegenReport{}, fmt.Errorf("livecodebench codegen: summary pass@5: %w", err)
	}
	report.Summary.Problems = len(report.Problems)
	report.Summary.Pass1 = meanPass1
	report.Summary.Pass5 = meanPass5
	return report, nil
}
