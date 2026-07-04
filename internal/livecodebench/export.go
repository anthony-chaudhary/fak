package livecodebench

import (
	"encoding/json"
	"fmt"
	"io"
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
