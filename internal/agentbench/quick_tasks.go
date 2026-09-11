package agentbench

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

func resolvedRequestedModel(summary runSummary, requested string) bool {
	if requested == "" || len(summary.ObservedModels) != 1 {
		return false
	}
	return summary.ObservedModels[0] == requested
}

func runQuickTask(ctx context.Context, out, endpoint, model string) (*taskrun.Receipt, string, error) {
	executable, err := os.Executable()
	if err != nil {
		return nil, "", err
	}
	taskOut := filepath.Join(out, "task")
	if err := os.Mkdir(taskOut, 0700); err != nil {
		return nil, "", err
	}
	if err := os.Chmod(taskOut, 0700); err != nil {
		return nil, "", err
	}
	receipt, runErr := taskrun.Run(ctx, taskrun.Options{
		Executable:  executable,
		Endpoint:    taskPlannerBaseURL(endpoint),
		Model:       model,
		OutDir:      taskOut,
		Concurrency: 1,
		TaskLimit:   1,
	})
	path := filepath.Join(taskOut, "receipt.json")
	if _, statErr := os.Stat(path); statErr != nil {
		if runErr != nil {
			return &receipt, "", runErr
		}
		return &receipt, "", statErr
	}
	if runErr != nil {
		return &receipt, path, runErr
	}
	if !quickTaskAccepted(&receipt) {
		return &receipt, path, errors.New("quick task external verifier rejected candidate")
	}
	return &receipt, path, nil
}

func taskPlannerBaseURL(endpoint string) string {
	return strings.TrimSuffix(endpoint, "/chat/completions")
}

func quickTaskAccepted(receipt *taskrun.Receipt) bool {
	return receipt != nil && len(receipt.Tasks) == 1 && receipt.Tasks[0].Accepted && receipt.Tasks[0].ExternalPassed
}
