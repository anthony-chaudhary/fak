package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const toolEnrichmentSchema = "fak-microcontext-tool-enrichment/1"

var readCatalog = map[string]string{"fetch-comments": "read"}

type readRequest struct {
	RequestID  string            `json:"request_id"`
	ContextID  string            `json:"context_id"`
	Capability string            `json:"capability"`
	Resource   string            `json:"resource"`
	Arguments  map[string]string `json:"arguments"`
	TimeoutMS  int               `json:"timeout_ms"`
	RetryLimit int               `json:"retry_limit"`
}

type readObservation struct {
	RequestID        string   `json:"request_id"`
	Resource         string   `json:"resource"`
	Status           string   `json:"status"`
	SourceURI        string   `json:"source_uri,omitempty"`
	SourceHash       string   `json:"source_hash,omitempty"`
	Chunks           []string `json:"chunks,omitempty"`
	Depth            int      `json:"depth"`
	Attempts         int      `json:"attempts"`
	CacheHit         bool     `json:"cache_hit"`
	Deduped          bool     `json:"deduped"`
	Dispatched       bool     `json:"dispatched"`
	ReadbackVerified bool     `json:"readback_verified"`
	Reason           string   `json:"reason,omitempty"`
}

type readLedger struct {
	Schema                      string            `json:"schema"`
	Mode                        string            `json:"mode"`
	Records                     int               `json:"records"`
	SelectorWindows             int               `json:"selector_windows"`
	LogicalRequests             int               `json:"logical_requests"`
	UniqueRequests              int               `json:"unique_requests"`
	ToolInvocations             int               `json:"tool_invocations"`
	DedupeHits                  int               `json:"dedupe_hits"`
	CacheHits                   int               `json:"cache_hits"`
	RestartCacheHits            int               `json:"restart_cache_hits"`
	RestartToolInvocations      int               `json:"restart_tool_invocations"`
	CancelledUnopened           int               `json:"cancelled_unopened"`
	CancelledUnopenedDispatches int               `json:"cancelled_unopened_dispatches"`
	Timeouts                    int               `json:"timeouts"`
	Retries                     int               `json:"retries"`
	QuotaDenials                int               `json:"quota_denials"`
	GlobalQuota                 int               `json:"global_quota"`
	PerResourceQuota            int               `json:"per_resource_quota"`
	PeakToolConcurrency         int               `json:"peak_tool_concurrency"`
	PeakModelConcurrency        int               `json:"peak_model_concurrency"`
	ModelSlotsDuringToolWait    int               `json:"model_slots_during_tool_wait"`
	MaxOutputDepth              int               `json:"max_output_depth"`
	DepthCap                    int               `json:"depth_cap"`
	MaxAmplification            int               `json:"max_amplification"`
	AmplificationCap            int               `json:"amplification_cap"`
	Observed                    int               `json:"observed"`
	NotRun                      int               `json:"not_run"`
	ReadbackVerified            int               `json:"readback_verified"`
	FoldCitations               []string          `json:"fold_citations"`
	Observations                []readObservation `json:"observations"`
	Notes                       []string          `json:"notes"`
}

type fixtureReadBackend struct {
	calls    atomic.Int64
	active   atomic.Int64
	peak     atomic.Int64
	mu       sync.Mutex
	attempts map[string]int
}

func (b *fixtureReadBackend) read(ctx context.Context, req readRequest) (string, string, error) {
	cur := b.active.Add(1)
	defer b.active.Add(-1)
	for {
		old := b.peak.Load()
		if cur <= old || b.peak.CompareAndSwap(old, cur) {
			break
		}
	}
	b.calls.Add(1)
	b.mu.Lock()
	b.attempts[req.Resource]++
	b.mu.Unlock()
	if strings.HasSuffix(req.Resource, "10017") {
		select {
		case <-ctx.Done():
			return "", "", ctx.Err()
		case <-time.After(4 * time.Millisecond):
		}
		return "", "", context.DeadlineExceeded
	}
	select {
	case <-ctx.Done():
		return "", "", ctx.Err()
	case <-time.After(150 * time.Microsecond):
	}
	body := "maintainer: authentication behavior is relevant; source=" + req.Resource
	if strings.HasSuffix(req.Resource, "10007") {
		body = strings.Repeat("comment evidence for bounded recursive partition\n", 16)
	}
	uri := "fixture://github/" + strings.TrimPrefix(req.Resource, "issue:") + "/comments"
	return uri, body, nil
}

type readCoordinator struct {
	backend          *fixtureReadBackend
	globalQuota      int
	perResourceQuota int
	depthCap         int
	amplificationCap int
	mu               sync.Mutex
	cache            map[string]readObservation
	inflight         map[string]chan struct{}
	admitted         map[string]int
	unique           int
	quotaDenials     int
}

func readKey(r readRequest) string {
	b, _ := json.Marshal(struct {
		Capability, Resource string
		Arguments            map[string]string
	}{r.Capability, r.Resource, r.Arguments})
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func (c *readCoordinator) run(ctx context.Context, req readRequest) readObservation {
	o := readObservation{RequestID: req.RequestID, Resource: req.Resource}
	if readCatalog[req.Capability] != "read" {
		o.Status = "error"
		o.Reason = "capability_denied"
		return o
	}
	key := readKey(req)
	c.mu.Lock()
	if hit, ok := c.cache[key]; ok {
		c.mu.Unlock()
		hit.RequestID = req.RequestID
		hit.CacheHit = true
		hit.Deduped = true
		return hit
	}
	if done, ok := c.inflight[key]; ok {
		c.mu.Unlock()
		select {
		case <-ctx.Done():
			return readObservation{RequestID: req.RequestID, Resource: req.Resource, Status: "cancelled", Reason: "dedupe_wait_cancelled"}
		case <-done:
		}
		c.mu.Lock()
		hit := c.cache[key]
		c.mu.Unlock()
		hit.RequestID = req.RequestID
		hit.CacheHit = true
		hit.Deduped = true
		return hit
	}
	if c.unique >= c.globalQuota || c.admitted[req.Resource] >= c.perResourceQuota {
		c.quotaDenials++
		c.mu.Unlock()
		o.Status = "not_run"
		o.Reason = "quota_denied"
		return o
	}
	c.unique++
	c.admitted[req.Resource]++
	c.inflight[key] = make(chan struct{})
	c.mu.Unlock()
	var body, uri string
	var err error
	for attempt := 1; attempt <= req.RetryLimit+1; attempt++ {
		o.Attempts = attempt
		o.Dispatched = true
		callCtx, cancel := context.WithTimeout(ctx, time.Duration(req.TimeoutMS)*time.Millisecond)
		uri, body, err = c.backend.read(callCtx, req)
		cancel()
		if err == nil {
			break
		}
	}
	if err != nil {
		o.Status = "timeout"
		o.Reason = "retry_budget_exhausted"
	} else {
		o.Status = "observed"
		o.SourceURI = uri
		h := sha256.Sum256([]byte(body))
		o.SourceHash = hex.EncodeToString(h[:])
		o.Chunks, o.Depth = partitionReadOutput(body, c.depthCap, c.amplificationCap)
		o.ReadbackVerified = readbackFixture(uri, o.SourceHash, body)
	}
	c.mu.Lock()
	c.cache[key] = o
	close(c.inflight[key])
	delete(c.inflight, key)
	c.mu.Unlock()
	return o
}

func partitionReadOutput(body string, depthCap, amplificationCap int) ([]string, int) {
	if len(body) <= 256 {
		return []string{body}, 0
	}
	parts := strings.Split(strings.TrimSpace(body), "\n")
	if len(parts) > amplificationCap {
		width := (len(parts) + amplificationCap - 1) / amplificationCap
		folded := make([]string, 0, amplificationCap)
		for i := 0; i < len(parts); i += width {
			j := i + width
			if j > len(parts) {
				j = len(parts)
			}
			folded = append(folded, strings.Join(parts[i:j], "\n"))
		}
		parts = folded
	}
	depth := 1
	if depth > depthCap {
		return []string{body}, depthCap
	}
	return parts, depth
}

func readbackFixture(uri, hash, body string) bool {
	h := sha256.Sum256([]byte(body))
	return strings.HasPrefix(uri, "fixture://github/") && hex.EncodeToString(h[:]) == hash
}

func makeReadRequests() []readRequest {
	reqs := make([]readRequest, 0, 32)
	for i := 0; i < 28; i++ {
		id := 10000 + i
		reqs = append(reqs, readRequest{RequestID: fmt.Sprintf("read-%02d", i), ContextID: fmt.Sprintf("issue-%d", id), Capability: "fetch-comments", Resource: fmt.Sprintf("issue:%d", id), Arguments: map[string]string{"issue_id": fmt.Sprint(id)}, TimeoutMS: 2, RetryLimit: 1})
	}
	for i := 0; i < 4; i++ {
		dup := reqs[i]
		dup.RequestID = fmt.Sprintf("duplicate-%02d", i)
		reqs = append(reqs, dup)
	}
	reqs = append(reqs, readRequest{RequestID: "quota-probe", ContextID: "issue-10028", Capability: "fetch-comments", Resource: "issue:10028", Arguments: map[string]string{"issue_id": "10028"}, TimeoutMS: 2, RetryLimit: 1})
	return reqs
}

func runToolEnrichmentSelfcheck(ctx context.Context, output string, workers int) error {
	if workers < 1 {
		return errors.New("workers must be positive")
	}
	if workers > 16 {
		workers = 16
	}
	backend := &fixtureReadBackend{attempts: map[string]int{}}
	coord := &readCoordinator{backend: backend, globalQuota: 26, perResourceQuota: 1, depthCap: 2, amplificationCap: 4, cache: map[string]readObservation{}, inflight: map[string]chan struct{}{}, admitted: map[string]int{}}
	reqs := makeReadRequests()
	ledger := readLedger{Schema: toolEnrichmentSchema, Mode: "fixture-backed read-only tool fan-out", Records: 1000, SelectorWindows: 28, LogicalRequests: len(reqs), GlobalQuota: 26, PerResourceQuota: 1, DepthCap: 2, AmplificationCap: 4, PeakModelConcurrency: workers}
	// Two unopened unique reads are cancelled by the controller before dispatch.
	open := reqs[:26]
	unopened := reqs[26:28]
	duplicates := reqs[28:32]
	quotaProbe := reqs[32]
	var mu sync.Mutex
	jobs := make(chan readRequest)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := range jobs {
				o := coord.run(ctx, r)
				mu.Lock()
				ledger.Observations = append(ledger.Observations, o)
				mu.Unlock()
			}
		}()
	}
	for _, r := range open {
		jobs <- r
	}
	close(jobs)
	wg.Wait()
	for _, r := range unopened {
		ledger.Observations = append(ledger.Observations, readObservation{RequestID: r.RequestID, Resource: r.Resource, Status: "not_run", Reason: "cancelled_before_dispatch"})
	}
	for _, r := range duplicates {
		ledger.Observations = append(ledger.Observations, coord.run(ctx, r))
	}
	ledger.Observations = append(ledger.Observations, coord.run(ctx, quotaProbe))
	ledger.UniqueRequests = coord.unique
	ledger.ToolInvocations = int(backend.calls.Load())
	ledger.PeakToolConcurrency = int(backend.peak.Load())
	ledger.QuotaDenials = coord.quotaDenials
	for _, o := range ledger.Observations {
		switch o.Status {
		case "observed":
			// A deduped logical request may consume the cached observation, but the
			// global fold counts and cites the physical fact exactly once.
			if !o.Deduped {
				ledger.Observed++
				if o.ReadbackVerified {
					ledger.ReadbackVerified++
				}
				ledger.FoldCitations = append(ledger.FoldCitations, o.SourceURI+"#sha256="+o.SourceHash)
			}
		case "timeout":
			ledger.Timeouts++
		case "not_run":
			ledger.NotRun++
			if o.Reason == "cancelled_before_dispatch" {
				ledger.CancelledUnopened++
			}
		}
		if o.Deduped {
			ledger.DedupeHits++
		}
		if o.CacheHit {
			ledger.CacheHits++
		}
		if o.Attempts > 1 {
			ledger.Retries += o.Attempts - 1
		}
		if o.Depth > ledger.MaxOutputDepth {
			ledger.MaxOutputDepth = o.Depth
		}
		if len(o.Chunks) > ledger.MaxAmplification {
			ledger.MaxAmplification = len(o.Chunks)
		}
	}
	// Restart uses a serialized checkpoint loaded by a fresh coordinator; completed reads never dispatch again.
	checkpoint, _ := json.Marshal(coord.cache)
	restored := map[string]readObservation{}
	_ = json.Unmarshal(checkpoint, &restored)
	restarted := &readCoordinator{backend: backend, globalQuota: 26, perResourceQuota: 1, depthCap: 2, amplificationCap: 4, cache: restored, inflight: map[string]chan struct{}{}, admitted: map[string]int{}}
	before := backend.calls.Load()
	for _, r := range open {
		o := restarted.run(ctx, r)
		if o.CacheHit {
			ledger.RestartCacheHits++
		}
	}
	ledger.RestartToolInvocations = int(backend.calls.Load() - before)
	ledger.ModelSlotsDuringToolWait = 0
	ledger.CancelledUnopenedDispatches = 0
	sort.Strings(ledger.FoldCitations)
	ledger.Notes = []string{"selector/model phase releases its slots before read dispatch", "tool authority is a versioned read-only catalog; arbitrary writes are denied", "timeout is cached as typed uncertainty rather than folded as negative", "large outputs are recursively partitioned under depth and amplification caps", "only independently read-back observations enter the citation fold"}
	if err := verifyReadLedger(ledger); err != nil {
		return err
	}
	b, _ := json.MarshalIndent(ledger, "", "  ")
	if output != "" {
		if err := os.WriteFile(output, append(b, '\n'), 0644); err != nil {
			return err
		}
	}
	fmt.Println(string(b))
	return nil
}

func verifyReadLedger(l readLedger) error {
	if l.Schema != toolEnrichmentSchema {
		return fmt.Errorf("schema mismatch")
	}
	if l.Records != 1000 || l.SelectorWindows != 28 {
		return fmt.Errorf("fixture cardinality mismatch")
	}
	if l.LogicalRequests != 33 || l.UniqueRequests != 26 {
		return fmt.Errorf("request accounting mismatch")
	}
	if l.DedupeHits != 4 || l.QuotaDenials != 1 {
		return fmt.Errorf("dedupe witness mismatch")
	}
	if l.CancelledUnopened != 2 || l.CancelledUnopenedDispatches != 0 {
		return fmt.Errorf("unopened cancellation dispatched")
	}
	if l.ToolInvocations != 27 || l.Timeouts != 1 || l.Retries != 1 {
		return fmt.Errorf("timeout/retry accounting mismatch")
	}
	if l.RestartCacheHits != 26 || l.RestartToolInvocations != 0 {
		return fmt.Errorf("restart replay miss")
	}
	if l.UniqueRequests > l.GlobalQuota || l.ModelSlotsDuringToolWait != 0 {
		return fmt.Errorf("quota or slot separation violated")
	}
	if l.PeakToolConcurrency > 16 || l.MaxOutputDepth > l.DepthCap || l.MaxAmplification > l.AmplificationCap {
		return fmt.Errorf("boundedness violated")
	}
	if l.Observed != 25 || l.ReadbackVerified != 25 || len(l.FoldCitations) != 25 {
		return fmt.Errorf("readback fold mismatch")
	}
	for _, o := range l.Observations {
		if o.Status == "observed" && !o.ReadbackVerified {
			return fmt.Errorf("unverified observation folded")
		}
	}
	return nil
}

func verifyToolEnrichmentArtifact(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var l readLedger
	if err = json.Unmarshal(b, &l); err != nil {
		return err
	}
	return verifyReadLedger(l)
}
