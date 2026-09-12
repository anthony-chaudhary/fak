package agentbench

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

type quickReplayState struct {
	runID, manifestPath, eventsPath, summaryPath string
	started                                      time.Time
	runCtx                                       context.Context
	cancel                                       context.CancelFunc
	summary                                      runSummary
	writeErr                                     error
	inputPaths                                   []string
	sink                                         *eventSink
	progressDone                                 chan struct{}
	progressWG                                   sync.WaitGroup
	progressPhase                                atomic.Value
	progressSummaryMu                            sync.Mutex
	out                                          string
	closeOnce                                    sync.Once
	closeSummary                                 *runSummary
	closeErr                                     error
}

func runQuickReplay(ctx context.Context, progress io.Writer, repo, endpoint, requestedModel, out string, inputs frozenInputs, plan replayPlan) (*quickReplayState, error) {
	if err := prepareOutput(out); err != nil {
		return nil, err
	}
	started := time.Now().UTC()
	runID := fmt.Sprintf("quick-%d", started.UnixNano())
	m := manifest{
		Schema: "fak.agentbench.manifest.v1", RunID: runID,
		Mode: "replay-plus-task", Profile: "quick", Repository: repo,
		Endpoint: endpoint, RequestedModel: requestedModel,
		ModelIdentityPolicy: plan.ModelIdentity, Rungs: plan.Rungs,
		ScoredRequests: plan.ScoredRequests, MaxInFlight: plan.MaxInFlight,
		ParentTimeoutMillis: parentTimeout.Milliseconds(), RequestTimeoutMillis: requestTimeout.Milliseconds(),
		MaxStreamBytes: maxStreamBytes, MaxResponseBytes: maxResponseBytes,
		InputArtifacts: plan.InputArtifacts, ReferenceTokenizer: plan.ReferenceTokenizer,
		UsagePolicy:    "provider-reported complete and internally consistent usage only; otherwise unknown",
		CachePolicy:    "provider-reported cached prompt tokens only; absence is unknown, never zero",
		OverallVerdict: "INCOMPLETE", TaskVerdict: "NOT_EVALUATED",
		StartedAt: started.Format(time.RFC3339Nano), TaskFixtures: plan.TaskFixtures,
		MaxTaskTurns: plan.MaxTaskTurns, RequestCeiling: plan.RequestCeiling,
	}
	manifestPath := filepath.Join(out, "manifest.json")
	if err := writeExclusiveJSON(manifestPath, m); err != nil {
		return nil, fmt.Errorf("write manifest: %w", err)
	}
	inputPaths, err := writeInputArtifacts(out, inputs)
	if err != nil {
		return nil, fmt.Errorf("write frozen input artifacts: %w", err)
	}
	eventsPath, summaryPath := filepath.Join(out, "events.jsonl"), filepath.Join(out, "summary.json")
	eventsFile, err := os.OpenFile(eventsPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("create events log: %w", err)
	}
	sink := &eventSink{file: eventsFile, terminal: make(map[int]lifecycleEvent, quickRequestCount), inFlight: map[string]int{"C1": 0, "C2": 0}, peak: map[string]int{"C1": 0, "C2": 0}}
	runCtx, cancel := context.WithTimeout(ctx, parentTimeout)
	s := &quickReplayState{runID: runID, started: started, runCtx: runCtx, cancel: cancel, manifestPath: manifestPath, eventsPath: eventsPath, summaryPath: summaryPath, inputPaths: inputPaths, sink: sink, progressDone: make(chan struct{}), out: out}
	s.progressPhase.Store("replay")
	s.progressWG.Add(1)
	go s.reportProgress(progress, requestedModel, plan)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxConnsPerHost = quickMaxInFlight
	transport.MaxIdleConnsPerHost = quickMaxInFlight
	client := &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	defer transport.CloseIdleConnections()
	requests := plannedRequests()
	runRung(runCtx, client, endpoint, requestedModel, inputs, sink, requests[:8], 1)
	if runCtx.Err() == nil {
		runRung(runCtx, client, endpoint, requestedModel, inputs, sink, requests[8:], 2)
	}
	missingStatus, missingError := "not_dispatched", "scheduler ended before dispatch"
	if runCtx.Err() != nil {
		missingStatus, missingError = "canceled", runCtx.Err().Error()
	}
	sink.terminalizeMissing(requests, requestedModel, missingStatus, missingError)
	s.writeErr = sink.err()
	if err := eventsFile.Close(); err != nil && s.writeErr == nil {
		s.writeErr = fmt.Errorf("close events log: %w", err)
	}
	s.summary = summarize(runID, started, requestedModel, sink.terminals(), sink.peaks(), plan.ReferenceTokenizer, runCtx.Err(), s.writeErr)
	return s, nil
}

func (s *quickReplayState) reportProgress(progress io.Writer, model string, plan replayPlan) {
	defer s.progressWG.Done()
	ticker := time.NewTicker(progressInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			terminal, peaks := s.sink.progress()
			fmt.Fprintf(progress, "agentbench quick progress: terminal=%d/%d c1-peak=%d c2-peak=%d\n", terminal, quickRequestCount, peaks["C1"], peaks["C2"])
			s.progressSummaryMu.Lock()
			checkpoint := summarize(s.runID, s.started, model, s.sink.terminals(), peaks, plan.ReferenceTokenizer, nil, s.sink.err())
			checkpoint.Status = "in_progress"
			checkpoint.Phase = s.progressPhase.Load().(string)
			checkpoint.ServiceVerdict = "IN_PROGRESS"
			checkpoint.OverallVerdict = "INCOMPLETE"
			if checkpoint.Phase == "task" {
				checkpoint.TaskVerdict = "IN_PROGRESS"
			}
			checkpoint.CompletedAt = ""
			checkpoint.Error = ""
			err := writeAtomicJSON(s.summaryPath, checkpoint)
			s.progressSummaryMu.Unlock()
			if err != nil {
				s.sink.recordError(fmt.Errorf("write progress summary: %w", err))
			}
		case <-s.progressDone:
			return
		}
	}
}

func (s *quickReplayState) setPhase(name string) error {
	s.progressSummaryMu.Lock()
	defer s.progressSummaryMu.Unlock()
	s.progressPhase.Store(name)
	cp := s.summary
	cp.Status = "in_progress"
	cp.Phase = name
	cp.TaskVerdict = "IN_PROGRESS"
	cp.OverallVerdict = "INCOMPLETE"
	cp.CompletedAt = ""
	cp.Error = ""
	if err := writeAtomicJSON(s.summaryPath, cp); err != nil {
		s.sink.recordError(fmt.Errorf("write %s phase summary: %w", name, err))
		s.writeErr = s.sink.err()
		return err
	}
	return nil
}

func (s *quickReplayState) Close(summary runSummary, extraArtifacts ...string) (*runSummary, error) {
	s.closeOnce.Do(func() {
		parentErr := s.runCtx.Err()
		close(s.progressDone)
		s.progressWG.Wait()
		if parentErr != nil {
			summary.Status = "incomplete"
			summary.OverallVerdict = "INCOMPLETE"
			summary.ServiceVerdict = "CANCELED"
			summary.Error = parentErr.Error()
		}
		if late := s.sink.err(); late != nil {
			s.writeErr = late
			summary.Status = "incomplete"
			summary.TaskVerdict = "REJECTED"
			summary.OverallVerdict = "INCOMPLETE"
			summary.ServiceVerdict = "ARTIFACT_FAILURE"
			summary.Error = late.Error()
		}
		summary.Phase = "terminal"
		summary.CompletedAt = time.Now().UTC().Format(time.RFC3339Nano)
		if err := writeAtomicJSON(s.summaryPath, summary); err != nil {
			s.closeErr = fmt.Errorf("write atomic summary: %w", err)
		} else {
			arts := append([]string{s.manifestPath}, s.inputPaths...)
			arts = append(arts, s.eventsPath, s.summaryPath)
			arts = append(arts, extraArtifacts...)
			if err := writeReceiptAtomic(filepath.Join(s.out, "receipt.json"), s.out, s.runID, arts); err != nil {
				s.closeErr = fmt.Errorf("write hashed receipt: %w", err)
			}
		}
		if s.closeErr == nil && parentErr != nil {
			s.closeErr = errCanceled
		}
		if s.closeErr == nil && s.writeErr != nil {
			s.closeErr = s.writeErr
		}
		if s.closeErr == nil && summary.FailedRequests != 0 {
			s.closeErr = fmt.Errorf("%d replay requests did not complete", summary.FailedRequests)
		}
		s.cancel()
		copy := summary
		s.closeSummary = &copy
	})
	if s.closeSummary == nil {
		return nil, errors.New("quick replay close produced no summary")
	}
	copy := *s.closeSummary
	return &copy, s.closeErr
}
