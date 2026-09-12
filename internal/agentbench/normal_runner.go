package agentbench

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/profileplan"
	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

type normalRunReceipt struct {
	Schema               string                  `json:"schema"`
	Status               string                  `json:"status"`
	OverallVerdict       string                  `json:"overall_verdict"`
	RequestedModel       string                  `json:"requested_model"`
	ResolvedModel        string                  `json:"resolved_model"`
	StartedAt            time.Time               `json:"started_at"`
	CompletedAt          time.Time               `json:"completed_at"`
	Duration             time.Duration           `json:"duration"`
	RequestCeiling       int                     `json:"request_ceiling"`
	ServiceQualification normalQualification     `json:"service_qualification"`
	CapacitySelection    normalCapacitySelection `json:"capacity_selection"`
	SharedC2Tasks        *taskrun.Receipt        `json:"shared_c2_tasks,omitempty"`
	OwnCapacityTasks     *taskrun.Receipt        `json:"own_capacity_tasks,omitempty"`
	Unknowns             []string                `json:"unknowns"`
	Artifacts            []string                `json:"artifacts"`
	ArtifactSHA256       map[string]string       `json:"artifact_sha256"`
	Phase                string                  `json:"phase"`
}

func runNormal(ctx context.Context, progress io.Writer, repo, endpoint, requestedModel, out string) (normalRunReceipt, error) {
	started := time.Now().UTC()
	receipt := normalRunReceipt{Schema: "fak.agentbench.normal-run.v1", Status: "incomplete", OverallVerdict: "INCOMPLETE", RequestedModel: requestedModel, StartedAt: started, Unknowns: []string{"model pricing and cost", "internal GPU occupancy and physical cache residency"}}
	runCtx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()
	deadline := started.Add(30 * time.Minute)
	model, err := resolveNormalModel(runCtx, endpoint, requestedModel)
	if err != nil {
		return finishNormalReceipt(receipt, out, err)
	}
	receipt.ResolvedModel = model
	if err := os.MkdirAll(out, 0700); err != nil {
		return receipt, err
	}
	manifestPath := filepath.Join(out, "manifest.json")
	manifest := map[string]any{"schema": "fak.agentbench.manifest.v1", "profile": "normal", "repository": repo, "endpoint": endpoint, "requested_model": requestedModel, "resolved_model": model, "deadline": "30m", "scheduled_replay_requests": 184, "priming_requests": 4, "task_concurrency_shared": 2, "overall_verdict": "INCOMPLETE", "unknowns": receipt.Unknowns}
	if err := writeExclusiveJSON(manifestPath, manifest); err != nil {
		return receipt, err
	}
	receipt.Artifacts = append(receipt.Artifacts, manifestPath)
	progressState, err := startNormalProgress(runCtx, filepath.Join(out, "summary.json"), started, deadline, progress, cancel)
	if err != nil {
		return receipt, err
	}
	finish := func(runErr error) (normalRunReceipt, error) {
		if progressErr := progressState.Close(); progressErr != nil && runErr == nil {
			runErr = progressErr
		}
		if receipt.Phase == "" {
			receipt.Phase = progressState.Phase()
		}
		return finishNormalReceipt(receipt, out, runErr)
	}
	if err := progressState.SetPhase("context-freeze"); err != nil {
		return finish(err)
	}
	repoInput, err := freezeRepositoryInput(repo)
	if err != nil {
		return finish(err)
	}
	replay, err := runNormalReplay(runCtx, progress, endpoint, model, out, newNormalInputs(repo, repoInput))
	if replay != nil {
		receipt.ServiceQualification = replay.qualification
		receipt.CapacitySelection = selectNormalCapacity(mustNormalPlan(), replay.events)
	}
	receipt.Artifacts = append(receipt.Artifacts, filepath.Join(out, "normal-context-manifest.json"), filepath.Join(out, "normal-events.jsonl"))
	if err != nil {
		return finish(fmt.Errorf("normal replay: %w", err))
	}
	if err := progressState.SetPhase("service-replay-complete"); err != nil {
		return finish(err)
	}
	selected := replay.selectedConcurrency
	receipt.RequestCeiling = normalRunCeiling(selected)
	exe, err := os.Executable()
	if err != nil {
		return finish(err)
	}
	base, baseErr := referenceBaseURL(endpoint)
	if baseErr != nil {
		return finish(baseErr)
	}
	taskEndpoint := base.String()
	if err := progressState.SetPhase("tasks-c2"); err != nil {
		return finish(err)
	}
	shared, taskErr := taskrun.Run(runCtx, taskrun.Options{Executable: exe, Endpoint: taskEndpoint, Model: model, OutDir: filepath.Join(out, "tasks-c2"), Concurrency: 2, TaskLimit: 6})
	receipt.SharedC2Tasks = &shared
	receipt.Artifacts = append(receipt.Artifacts, filepath.Join(out, "tasks-c2", "receipt.json"))
	if taskErr != nil || !acceptedTaskRun(shared) {
		if taskErr == nil {
			taskErr = errors.New("shared C2 task cohort was not accepted")
		}
		return finish(taskErr)
	}
	if selected != 2 {
		if err := progressState.SetPhase("tasks-selected"); err != nil {
			return finish(err)
		}
		limit := 6
		heldout := false
		if selected == 8 {
			limit = 8
			heldout = true
		}
		own, ownErr := taskrun.Run(runCtx, taskrun.Options{Executable: exe, Endpoint: taskEndpoint, Model: model, OutDir: filepath.Join(out, "tasks-selected"), Concurrency: selected, TaskLimit: limit, IncludeHeldout: heldout})
		receipt.OwnCapacityTasks = &own
		receipt.Artifacts = append(receipt.Artifacts, filepath.Join(out, "tasks-selected", "receipt.json"))
		if ownErr != nil || !acceptedTaskRun(own) {
			if ownErr == nil {
				ownErr = errors.New("selected-capacity task cohort was not accepted")
			}
			return finish(ownErr)
		}
	} else {
		receipt.OwnCapacityTasks = &shared
	}
	receipt.Status = "PASSED"
	receipt.OverallVerdict = "NORMAL_PASSED"
	receipt.Phase = "complete"
	return finish(nil)
}

func mustNormalPlan() profileplan.Plan {
	p, _ := profileplan.Build(profileplan.Options{Profile: "normal", PreferredConcurrency: 2})
	return p
}
func normalRunCeiling(c int) int {
	switch c {
	case 1:
		return 344
	case 2:
		return 272
	case 4:
		return 344
	case 8:
		return 368
	}
	return 0
}
func acceptedTaskRun(r taskrun.Receipt) bool {
	if len(r.Tasks) == 0 || !r.ConcurrencyQualified {
		return false
	}
	for _, task := range r.Tasks {
		if !task.Accepted || task.Model.Requests != task.Model.UpstreamRequests || task.Model.UpstreamRequests != task.Model.SuccessfulRequests {
			return false
		}
	}
	return true
}
func finishNormalReceipt(r normalRunReceipt, out string, runErr error) (normalRunReceipt, error) {
	r.CompletedAt = time.Now().UTC()
	r.Duration = r.CompletedAt.Sub(r.StartedAt)
	if runErr != nil {
		r.Status = "incomplete"
		r.OverallVerdict = "INCOMPLETE"
	}
	if out == "" {
		return r, runErr
	}
	summary := filepath.Join(out, "summary.json")
	if err := writeNormalProgressJSON(summary, r); err != nil {
		return r, err
	}
	r.Artifacts = append(r.Artifacts, summary)
	existing := make([]string, 0, len(r.Artifacts))
	r.ArtifactSHA256 = make(map[string]string, len(r.Artifacts))
	for _, path := range r.Artifacts {
		if info, err := os.Stat(path); err == nil && info.Mode().IsRegular() {
			existing = append(existing, path)
			name, relErr := filepath.Rel(out, path)
			digest, hashErr := hashFile(path)
			if relErr != nil || hashErr != nil {
				if relErr != nil {
					return r, relErr
				}
				return r, hashErr
			}
			r.ArtifactSHA256[filepath.ToSlash(name)] = digest
		}
	}
	if err := writeReceiptAtomic(filepath.Join(out, "receipt.json"), out, "normal", existing); err != nil {
		return r, err
	}
	return r, runErr
}

func resolveNormalModel(ctx context.Context, endpoint, requested string) (string, error) {
	if requested != "" {
		return requested, nil
	}
	base, err := referenceBaseURL(endpoint)
	if err != nil {
		return "", err
	}
	modelsURL := base.ResolveReference(&url.URL{Path: "/v1/models"})
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	client := &http.Client{Timeout: requestTimeout, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	if err := referenceJSON(ctx, client, http.MethodGet, modelsURL.String(), nil, &payload); err != nil {
		return "", err
	}
	if len(payload.Data) != 1 || payload.Data[0].ID == "" {
		return "", errors.New("normal requires --model when /v1/models does not advertise exactly one model")
	}
	return payload.Data[0].ID, nil
}

func runNormalCLICommand(ctx context.Context, stdout, stderr io.Writer, opts options) int {
	repo, err := resolveRepository(opts.repo)
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: repository: %v\n", err)
		return 2
	}
	if opts.endpoint == "" || opts.out == "" {
		fmt.Fprintln(stderr, "agentbench: --endpoint and --out are required")
		return 2
	}
	endpoint, err := normalizeEndpoint(opts.endpoint)
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: endpoint: %v\n", err)
		return 2
	}
	out, err := resolveOutput(repo, opts.out)
	if err != nil {
		fmt.Fprintf(stderr, "agentbench: output: %v\n", err)
		return 2
	}
	receipt, runErr := runNormal(ctx, stderr, repo, endpoint, opts.model, out)
	if err := printValue(stdout, receipt, opts.json); err != nil && runErr == nil {
		runErr = err
	}
	if runErr != nil {
		fmt.Fprintf(stderr, "agentbench: normal incomplete: %v\n", runErr)
		return 1
	}
	return 0
}
