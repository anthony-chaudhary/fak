package agentbench

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/taskrun"
)

type burnInEpochConfig struct {
	Client      *http.Client
	Endpoint    string
	Model       string
	OutDir      string
	Corpus      *burnInCorpus
	Reference   *nativeReference
	Sink        *eventSink
	Now         func() time.Time
	RunTasks    func(context.Context, taskrun.Options) (taskrun.Receipt, error)
	TaskOptions taskrun.Options
}

func runBurnInEpoch(ctx context.Context, cfg burnInEpochConfig, spec burnInEpochSpec, cutoff time.Time) burnInEpochReceipt {
	receipt := burnInEpochReceipt{Ordinal: spec.Ordinal, Partial: true, PressureStatus: "unknown", PressureCohorts: []int{50, 80, 100, 120}}
	now := time.Now
	if cfg.Now != nil {
		now = cfg.Now
	}
	if !now().Before(cutoff) {
		receipt.CompletedAt = now()
		return receipt
	}
	if cfg.Client == nil || cfg.Corpus == nil || cfg.Reference == nil || len(cfg.Corpus.Sessions) != spec.Sessions || cfg.RunTasks == nil || cfg.Endpoint == "" || cfg.Model == "" || cfg.Reference.ModelID != cfg.Model || cfg.Reference.ContextWindowTokens < 32768 {
		receipt.Errors = append(receipt.Errors, "burn-in epoch configuration incomplete")
		return receipt
	}
	if err := os.MkdirAll(cfg.OutDir, 0700); err != nil {
		receipt.Errors = append(receipt.Errors, err.Error())
		return receipt
	}
	eventsPath := filepath.Join(cfg.OutDir, fmt.Sprintf("burn-epoch-%03d.jsonl", spec.Ordinal))
	file, err := os.OpenFile(eventsPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		receipt.Errors = append(receipt.Errors, err.Error())
		return receipt
	}
	writer := bufio.NewWriter(file)
	var eventMu sync.Mutex
	var eventErr error
	receipt.EventsPath = eventsPath
	writeEvent := func(stage, kind string, sequence int, area string) {
		eventMu.Lock()
		defer eventMu.Unlock()
		if eventErr == nil {
			eventErr = json.NewEncoder(writer).Encode(map[string]any{"stage": stage, "event": kind, "sequence": sequence, "area": area, "epoch": spec.Ordinal})
			if eventErr == nil {
				eventErr = writer.Flush()
			}
		}
	}
	sequence := 0
	var tasks taskrun.Receipt
	var taskErr error
	type steadyJob struct {
		sequence int
		area     string
		request  referenceRequest
	}
	var steadyJobs []steadyJob
	for _, session := range cfg.Corpus.Sessions {
		area := session.Area
		before := burnRequestDigest(session.Turns[0].Request)
		after := burnRequestDigest(session.Turns[len(session.Turns)-1].Request)
		receipt.AreaTransitions = append(receipt.AreaTransitions, burnInAreaTransition{Area: area, BeforeSHA256: before, AfterSHA256: after})
		for _, turn := range session.Turns {
			sequence++
			steadyJobs = append(steadyJobs, steadyJob{sequence: sequence, area: area, request: turn.Request})
		}
	}
	jobs := make(chan steadyJob)
	results := make(chan error, len(steadyJobs))
	var workers sync.WaitGroup
	for worker := 0; worker < spec.Concurrency; worker++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for job := range jobs {
				if !now().Before(cutoff) {
					results <- context.DeadlineExceeded
					continue
				}
				writeEvent("dispatched", "steady", job.sequence, job.area)
				err := dispatchBurnInRequest(ctx, cfg, job.request, job.sequence, job.area, nil)
				writeEvent("terminal", "steady", job.sequence, job.area)
				results <- err
			}
		}()
	}
	for _, job := range steadyJobs {
		jobs <- job
	}
	close(jobs)
	workers.Wait()
	close(results)
	for result := range results {
		if result == context.DeadlineExceeded {
			continue
		}
		receipt.SteadyRequests++
		if result != nil {
			receipt.Errors = append(receipt.Errors, result.Error())
		}
	}
	for control := 1; control <= spec.Controls; control++ {
		if !now().Before(cutoff) {
			goto finish
		}
		sequence++
		request := cfg.Corpus.Sessions[(control-1)%len(cfg.Corpus.Sessions)].Turns[31].Request
		kind := "control"
		controlArea := "control"
		if control == 1 {
			kind = "intentional_cancel"
			cancelCtx, cancel := context.WithCancel(ctx)
			cancelIssued := false
			writeEvent("dispatched", kind, sequence, "cancel")
			_ = dispatchBurnInRequest(cancelCtx, cfg, request, sequence, "cancel", func() { cancelIssued = true; cancel() })
			writeEvent("terminal", kind, sequence, "cancel")
			cancel()
			receipt.IntentionalCancellationObserved = cancelIssued
			if !cancelIssued {
				receipt.Errors = append(receipt.Errors, "intentional cancellation was not injected after first output")
			}
		} else {
			switch control {
			case 2:
				kind, controlArea = "recovery", "recovery"
			case 3:
				kind, controlArea = "source_edit", "source-edit"
			case 4:
				kind, controlArea = "source_reread", "source-reread"
			default:
				if control >= 5 && control <= 8 {
					cohort := receipt.PressureCohorts[control-5]
					kind, controlArea = "fixture_pressure", fmt.Sprintf("pressure-%d", cohort)
				}
			}
			request = burnControlRequest(request, control, receipt.PressureCohorts)
			writeEvent("dispatched", kind, sequence, controlArea)
			err := dispatchBurnInRequest(ctx, cfg, request, sequence, controlArea, nil)
			writeEvent("terminal", kind, sequence, controlArea)
			if err != nil {
				receipt.Errors = append(receipt.Errors, err.Error())
			} else {
				switch control {
				case 2:
					receipt.RecoveryPassed = true
				case 4:
					receipt.SourceEditRereadPassed = true
				}
			}
		}
		receipt.ControlRequests++
	}
	if !now().Before(cutoff) {
		goto finish
	}
	cfg.TaskOptions.Concurrency = spec.Concurrency
	cfg.TaskOptions.TaskLimit = spec.Tasks
	tasks, taskErr = cfg.RunTasks(ctx, cfg.TaskOptions)
	receipt.TaskAttempts = len(tasks.Tasks)
	for _, task := range tasks.Tasks {
		if task.Accepted && task.Passed && task.ExternalPassed {
			receipt.AcceptedTasks++
		}
	}
	if taskErr != nil {
		receipt.Errors = append(receipt.Errors, taskErr.Error())
	}

finish:
	receipt.AreasVisited = distinctBurnAreas(receipt.AreaTransitions)
	receipt.CompletedAt = now()
	receipt.Complete = receipt.SteadyRequests == spec.Sessions*spec.TurnsPerSession && receipt.ControlRequests == spec.Controls && receipt.TaskAttempts == spec.Tasks && receipt.AcceptedTasks == spec.Tasks && receipt.IntentionalCancellationObserved && receipt.RecoveryPassed && receipt.AreasVisited == spec.Areas && len(receipt.Errors) == 0
	receipt.Partial = !receipt.Complete
	if err := writer.Flush(); err != nil && eventErr == nil {
		eventErr = err
	}
	if err := file.Close(); err != nil && eventErr == nil {
		eventErr = err
	}
	if eventErr != nil {
		receipt.Errors = append(receipt.Errors, "persist epoch lifecycle: "+eventErr.Error())
		receipt.Complete, receipt.Partial = false, true
	}
	if body, readErr := os.ReadFile(eventsPath); readErr == nil {
		sum := sha256.Sum256(body)
		receipt.EventsDigest = hex.EncodeToString(sum[:])
	}
	return receipt
}

func burnControlRequest(base referenceRequest, control int, cohorts []int) referenceRequest {
	request := referenceRequest{Messages: append([]chatMessage(nil), base.Messages...), Tools: append([]json.RawMessage(nil), base.Tools...), OutputTokens: base.OutputTokens}
	if control == 3 || control == 4 {
		changed := "func frozenBurnSource() string { return \"after\" }"
		version := burnDigest(changed)
		request.Messages = append(request.Messages, chatMessage{Role: "user", Content: "burn-source-version:" + version + "\n" + changed})
	}
	if control >= 5 && control <= 8 {
		volume := cohorts[control-5]
		material := ""
		for _, message := range base.Messages {
			material += message.Content
		}
		keep := len(material) * volume / 100
		expanded := material
		for len(expanded) < keep {
			expanded += material
		}
		request.Messages = append(request.Messages, chatMessage{Role: "user", Content: expanded[:keep]})
	}
	return request
}

func dispatchBurnInRequest(ctx context.Context, cfg burnInEpochConfig, request referenceRequest, sequence int, area string, cancel context.CancelFunc) error {
	_, err := dispatchWithProgress(ctx, cfg.Client, cfg.Endpoint, requestEnvelope{Model: cfg.Model, Stream: true, StreamOptions: streamOptions{IncludeUsage: true}, Messages: request.Messages, Tools: request.Tools, MaxTokens: request.OutputTokens, Metadata: requestMetadata{Benchmark: "agentbench", Profile: "burn-in", Rung: area, Session: fmt.Sprintf("epoch-%d", sequence), Turn: sequence, Sequence: sequence, ScoredRequest: true}}, cfg.Model, func(time.Time) error {
		if cancel != nil {
			cancel()
		}
		return nil
	}, nil, nil)
	return err
}

func distinctBurnAreas(transitions []burnInAreaTransition) int {
	seen := map[string]bool{}
	for _, transition := range transitions {
		seen[transition.Area] = true
	}
	return len(seen)
}
