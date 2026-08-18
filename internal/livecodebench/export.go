package livecodebench

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// CustomEvalItem is one entry of the input the upstream
// lcb_runner.runner.custom_evaluator consumes: a question_id paired with the
// model's code generations for that problem. The JSON shape is exactly
// [{"question_id": ..., "code_list": [...]}], the format the OFFICIAL checker
// grades — this is the bridge that turns local generations into a real,
// claimable result.
type CustomEvalItem struct {
	QuestionID string   `json:"question_id"`
	CodeList   []string `json:"code_list"`
}

// CustomEvaluatorItems projects a fixture's items into custom_evaluator input,
// preserving fixture order so the emitted slice lines up with the benchmark
// problems, and passing each question_id through unchanged. An item with no
// code_list is rejected: the custom evaluator has nothing to grade for it, and
// silently emitting an empty code_list would fabricate an ungradeable entry.
func CustomEvaluatorItems(f Fixture) ([]CustomEvalItem, error) {
	items := make([]CustomEvalItem, 0, len(f.Items))
	for i, it := range f.Items {
		if len(it.CodeList) == 0 {
			return nil, fmt.Errorf("livecodebench custom-evaluator: item %d (%q) has no code_list generations to grade", i, it.QuestionID)
		}
		items = append(items, CustomEvalItem{QuestionID: it.QuestionID, CodeList: it.CodeList})
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("livecodebench custom-evaluator: fixture has no items to export")
	}
	return items, nil
}

// WriteCustomEvaluatorInput writes the custom_evaluator input JSON for a fixture
// to w, ordered to match the benchmark problems. Grade the emitted file with the
// official checker:
//
//	python -m lcb_runner.runner.custom_evaluator \
//	    --custom_output_file <input.json> \
//	    --release_version <release_vN>
//
// The result is only claimable once that official run produces its grading —
// a local export can never promote itself into a score.
func WriteCustomEvaluatorInput(w io.Writer, f Fixture) error {
	items, err := CustomEvaluatorItems(f)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(items); err != nil {
		return fmt.Errorf("livecodebench custom-evaluator: encode: %w", err)
	}
	return nil
}

// LoadArmReportFixture converts a raw or fak generation report into the exact
// fixture shape consumed by the official custom evaluator. This is the
// deterministic report-to-grader seam: generation output is read back from
// disk, validated, and projected without another model call.
func LoadArmReportFixture(path string) (Fixture, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Fixture{}, fmt.Errorf("livecodebench arm report: read: %w", err)
	}
	var envelope struct {
		Arm     string `json:"arm"`
		Release string `json:"release"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return Fixture{}, fmt.Errorf("livecodebench arm report: decode: %w", err)
	}
	arm := strings.TrimSpace(envelope.Arm)
	if arm != "raw" && arm != "fak" {
		return Fixture{}, fmt.Errorf("livecodebench arm report: arm = %q, want raw or fak", arm)
	}
	if strings.TrimSpace(envelope.Release) == "" {
		return Fixture{}, fmt.Errorf("livecodebench arm report: release is required for official grading")
	}
	var report struct {
		Problems []RawArmProblem `json:"problems"`
	}
	if err := json.Unmarshal(data, &report); err != nil {
		return Fixture{}, fmt.Errorf("livecodebench arm report: decode problems: %w", err)
	}
	if len(report.Problems) == 0 {
		return Fixture{}, fmt.Errorf("livecodebench arm report: no problems")
	}
	fixture := Fixture{Schema: FixtureSchema, ReleaseVersion: envelope.Release}
	seen := make(map[string]struct{}, len(report.Problems))
	for i, problem := range report.Problems {
		id := strings.TrimSpace(problem.QuestionID)
		if id == "" {
			return Fixture{}, fmt.Errorf("livecodebench arm report: problem %d has no question_id", i)
		}
		if _, ok := seen[id]; ok {
			return Fixture{}, fmt.Errorf("livecodebench arm report: duplicate question_id %q", id)
		}
		seen[id] = struct{}{}
		if len(problem.Completions) == 0 {
			return Fixture{}, fmt.Errorf("livecodebench arm report: problem %q has no completions", id)
		}
		for j, completion := range problem.Completions {
			if strings.TrimSpace(completion) == "" {
				return Fixture{}, fmt.Errorf("livecodebench arm report: problem %q completion %d is empty", id, j)
			}
		}
		fixture.Items = append(fixture.Items, FixtureItem{
			QuestionID: id,
			Scenario:   string(ScenarioCodeGeneration),
			CodeList:   append([]string(nil), problem.Completions...),
		})
	}
	return fixture, nil
}
