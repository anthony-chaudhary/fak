package taskrun

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskfixture"
)

type PrehistoryReceipt struct {
	Passed       bool   `json:"passed"`
	PriorTaskID  string `json:"prior_task_id"`
	BeforeSHA256 string `json:"before_sha256"`
	AfterSHA256  string `json:"after_sha256"`
	DiffSHA256   string `json:"diff_sha256"`
	TestSHA256   string `json:"test_sha256"`
}

func applyFrozenPriorDiff(before, raw string) (string, error) {
	var edit struct {
		Old string `json:"old_string"`
		New string `json:"new_string"`
	}
	if json.Unmarshal([]byte(raw), &edit) != nil || edit.Old == "" || strings.Count(before, edit.Old) != 1 {
		return "", errors.New("frozen prior diff is not one exact edit")
	}
	return strings.Replace(before, edit.Old, edit.New, 1), nil
}

func validateFrozenPrehistory(ctx context.Context, f taskfixture.Fixture, artifact string) (PrehistoryReceipt, error) {
	w := f.Workflow
	receipt := PrehistoryReceipt{PriorTaskID: w.PriorTaskID, BeforeSHA256: digestString(w.PriorBeforeSource), AfterSHA256: digestString(w.PriorAfterSource), DiffSHA256: digestString(w.PriorDiff), TestSHA256: digestString(w.PriorTestSource)}
	after, err := applyFrozenPriorDiff(w.PriorBeforeSource, w.PriorDiff)
	if err != nil || after != w.PriorAfterSource || after != f.BrokenSource || digestString(after) != w.PriorStateSHA256 {
		return receipt, errors.New("frozen prior source/diff identity mismatch")
	}
	root, err := os.MkdirTemp("", "agentbench-prehistory-")
	if err != nil {
		return receipt, err
	}
	defer os.RemoveAll(root)
	candidate, visible, oracle := filepath.Join(root, "candidate"), filepath.Join(root, "visible"), filepath.Join(root, "oracle")
	for _, dir := range []string{candidate, visible, oracle} {
		if err := os.Mkdir(dir, 0700); err != nil {
			return receipt, err
		}
	}
	if err := os.WriteFile(filepath.Join(candidate, f.TargetFile), []byte(after), 0400); err != nil {
		return receipt, err
	}
	if err := os.WriteFile(filepath.Join(visible, "visible_test.go"), []byte(w.PriorTestSource), 0400); err != nil {
		return receipt, err
	}
	result, err := runCandidateTests(ctx, candidate, f.TargetFile, visible, oracle)
	if err != nil || result.ExitCode != 0 {
		return receipt, errors.New("frozen prior test record did not pass")
	}
	receipt.Passed = true
	if err := writeJSON(artifact, receipt); err != nil {
		return receipt, err
	}
	return receipt, nil
}
