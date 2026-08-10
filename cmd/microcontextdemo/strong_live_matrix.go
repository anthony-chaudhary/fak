package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

const strongLiveMatrixSchema = "fak-microcontext-strong-live-matrix/1"

type strongLiveReport struct {
	Schema                                 string `json:"schema"`
	CreatedAt, PacketSHA256, GoldSHA256    string
	Endpoint                               endpointProvenance `json:"endpoint"`
	Trials, Workers, ChunkSize, RetrievalK int
	Pipelines                              []strongPipeline `json:"pipelines"`
	Limits                                 []string         `json:"limits"`
}
type strongPipeline struct {
	Pipeline       string                `json:"pipeline"`
	Calibration    abstentionCalibration `json:"calibration"`
	RetrievalTrace []retrievalSelection  `json:"retrieval_trace,omitempty"`
	TrialResults   []liveTrial           `json:"trial_results"`
	Aggregate      liveAggregate         `json:"aggregate"`
	Grade          semanticGrade         `json:"grade"`
}
type abstentionCalibration struct {
	Candidates  []float64      `json:"candidates"`
	Selected    float64        `json:"selected"`
	TuneExact   map[string]int `json:"tune_exact"`
	TuneRecords int            `json:"tune_records"`
	Rule        string         `json:"rule"`
}
type retrievalSelection struct {
	QueryID     string    `json:"query_id"`
	SelectedIDs []string  `json:"selected_ids"`
	Scores      []float64 `json:"scores"`
}

func words(s string) map[string]bool {
	r := strings.NewReplacer("/", " ", "-", " ", "_", " ", ".", " ", ",", " ", "(", " ", ")", " ", "`", " ")
	m := map[string]bool{}
	for _, x := range strings.Fields(strings.ToLower(r.Replace(s))) {
		if len(x) > 2 {
			m[x] = true
		}
	}
	return m
}
func similarity(a, b semanticRecord) float64 {
	x, y := words(a.Title+" "+a.Body), words(b.Title+" "+b.Body)
	hit := 0
	for k := range x {
		if y[k] {
			hit++
		}
	}
	if len(x)+len(y) == 0 {
		return 0
	}
	return 2 * float64(hit) / float64(len(x)+len(y))
}
func topKExamples(q semanticRecord, tuneRecords []semanticRecord, gold map[string]semanticConsensus, k int) ([]semanticConsensus, retrievalSelection) {
	type pair struct {
		i int
		s float64
	}
	var ps []pair
	for i, x := range tuneRecords {
		if x.ID == q.ID {
			continue
		} // leave-one-out during tune calibration
		ps = append(ps, pair{i, similarity(q, x)})
	}
	sort.Slice(ps, func(i, j int) bool {
		if ps[i].s == ps[j].s {
			return tuneRecords[ps[i].i].ID < tuneRecords[ps[j].i].ID
		}
		return ps[i].s > ps[j].s
	})
	if k > len(ps) {
		k = len(ps)
	}
	var ex []semanticConsensus
	tr := retrievalSelection{QueryID: q.ID}
	for _, p := range ps[:k] {
		id := tuneRecords[p.i].ID
		ex = append(ex, gold[id])
		tr.SelectedIDs = append(tr.SelectedIDs, id)
		tr.Scores = append(tr.Scores, p.s)
	}
	return ex, tr
}
func applyThreshold(xs []semanticConsensus, th float64) []semanticConsensus {
	out := append([]semanticConsensus(nil), xs...)
	for i := range out {
		c := out[i].Confidence
		if c == nil {
			c = map[string]float64{}
		}
		if c["semantic_need"] < th {
			out[i].SemanticNeed = "abstain"
		}
		if c["tool_need"] < th {
			out[i].ToolNeed = "abstain"
		}
		if c["actionability"] < th {
			out[i].Actionability = "abstain"
		}
	}
	return out
}
func exactCount(gold map[string]semanticConsensus, pred []semanticConsensus) int {
	n := 0
	for _, x := range pred {
		g, ok := gold[x.ID]
		if ok && g.SemanticNeed == x.SemanticNeed && g.ToolNeed == x.ToolNeed && g.Actionability == x.Actionability {
			n++
		}
	}
	return n
}
func strongCalls(ctx context.Context, c *liveMatrixClient, name string, records, tuneRecords []semanticRecord, tuneGold map[string]semanticConsensus, k, workers, chunk int) ([]liveCall, []retrievalSelection) {
	switch name {
	case "long-context":
		return []liveCall{c.call(ctx, livePrompt(records, nil, name+" calibrated"))}, nil
	case "chunk-map-reduce":
		var groups [][]semanticRecord
		for i := 0; i < len(records); i += chunk {
			j := i + chunk
			if j > len(records) {
				j = len(records)
			}
			groups = append(groups, records[i:j])
		}
		sem := make(chan struct{}, workers)
		out := make(chan liveCall, len(groups))
		var wg sync.WaitGroup
		for _, g := range groups {
			g := g
			wg.Add(1)
			go func() {
				defer wg.Done()
				sem <- struct{}{}
				v := c.call(ctx, livePrompt(g, nil, name+" parallel typed reduce"))
				<-sem
				out <- v
			}()
		}
		wg.Wait()
		close(out)
		var calls []liveCall
		for v := range out {
			calls = append(calls, v)
		}
		return calls, nil
	default:
		sem := make(chan struct{}, workers)
		out := make(chan struct {
			v  liveCall
			tr retrievalSelection
		}, len(records))
		var wg sync.WaitGroup
		for _, r := range records {
			r := r
			wg.Add(1)
			go func() {
				defer wg.Done()
				var ex []semanticConsensus
				var tr retrievalSelection
				if name == "retrieval-rerank" {
					ex, tr = topKExamples(r, tuneRecords, tuneGold, k)
				}
				sem <- struct{}{}
				v := c.call(ctx, livePrompt([]semanticRecord{r}, ex, name+" calibrated"))
				<-sem
				out <- struct {
					v  liveCall
					tr retrievalSelection
				}{v, tr}
			}()
		}
		wg.Wait()
		close(out)
		var calls []liveCall
		var trs []retrievalSelection
		for x := range out {
			calls = append(calls, x.v)
			if x.tr.QueryID != "" {
				trs = append(trs, x.tr)
			}
		}
		sort.Slice(trs, func(i, j int) bool { return trs[i].QueryID < trs[j].QueryID })
		return calls, trs
	}
}
func foldCalls(calls []liveCall, records []semanticRecord, trial int, th float64, start time.Time) liveTrial {
	t := liveTrial{Trial: trial}
	var tt, lat []time.Duration
	m := map[string]semanticConsensus{}
	for _, x := range calls {
		t.Requests++
		t.Retries += x.retry
		if x.err != nil {
			t.Failed++
			t.Errors = append(t.Errors, x.err.Error())
			continue
		}
		t.Successful++
		t.PromptTokens += x.prompt
		t.CompletionTokens += x.completion
		t.CachedTokens += x.cached
		tt = append(tt, x.ttft)
		lat = append(lat, x.latency)
		for _, a := range applyThreshold(x.answers, th) {
			m[a.ID] = a
		}
	}
	for _, r := range records {
		if a, ok := m[r.ID]; ok {
			t.Answers = append(t.Answers, a)
		}
	}
	t.WallMS = float64(time.Since(start)) / float64(time.Millisecond)
	t.TTFTP50MS = pctDur(tt, 50)
	t.TTFTP95MS = pctDur(tt, 95)
	t.LatencyP50MS = pctDur(lat, 50)
	t.LatencyP95MS = pctDur(lat, 95)
	return t
}
func runStrongLiveMatrix(packetPath, goldPath, out, endpoint, key, model, class, hardware, batch, cache, pricing string, trials, workers, k, chunk int) error {
	pb, e := os.ReadFile(packetPath)
	if e != nil {
		return e
	}
	gb, e := os.ReadFile(goldPath)
	if e != nil {
		return e
	}
	var p semanticPacket
	var g semanticGold
	if e = json.Unmarshal(pb, &p); e != nil {
		return e
	}
	if e = json.Unmarshal(gb, &g); e != nil {
		return e
	}
	var tune, test []semanticRecord
	for _, x := range p.Records {
		if x.Split == "tune" {
			tune = append(tune, x)
		} else if x.Split == "test" {
			test = append(test, x)
		}
	}
	gm := map[string]semanticConsensus{}
	for _, x := range g.Answers {
		gm[x.ID] = x
	}
	c := &liveMatrixClient{endpoint: endpoint, key: key, model: model, client: &http.Client{Timeout: 12 * time.Minute}}
	r := strongLiveReport{Schema: strongLiveMatrixSchema, CreatedAt: time.Now().UTC().Format(time.RFC3339), PacketSHA256: shaHex(pb), GoldSHA256: shaHex(gb), Endpoint: endpointProvenance{Class: class, Model: model, Hardware: hardware, NativeBatch: batch, PrefixCache: cache, PricingSnapshot: pricing}, Trials: trials, Workers: workers, RetrievalK: k, ChunkSize: chunk}
	thresholds := []float64{0, 0.5, 0.7, 0.85, 0.95, 1.01}
	for _, name := range []string{"retrieval-rerank", "long-context", "chunk-map-reduce", "micro-context"} {
		sp := strongPipeline{Pipeline: name, Calibration: abstentionCalibration{Candidates: thresholds, TuneExact: map[string]int{}, TuneRecords: len(tune), Rule: "maximize exact tune records; ties choose lower threshold"}}
		calls, tr := strongCalls(context.Background(), c, name, tune, tune, gm, k, workers, chunk)
		sp.RetrievalTrace = tr
		var raw []semanticConsensus
		for _, v := range calls {
			raw = append(raw, v.answers...)
		}
		best := -1
		for _, th := range thresholds {
			n := exactCount(gm, applyThreshold(raw, th))
			sp.Calibration.TuneExact[fmt.Sprintf("%.2f", th)] = n
			if n > best {
				best = n
				sp.Calibration.Selected = th
			}
		}
		for i := 1; i <= trials; i++ {
			start := time.Now()
			cs, trace := strongCalls(context.Background(), c, name, test, tune, gm, k, workers, chunk)
			if len(trace) > 0 {
				sp.RetrievalTrace = trace
			}
			sp.TrialResults = append(sp.TrialResults, foldCalls(cs, test, i, sp.Calibration.Selected, start))
		}
		sub := semanticSubmission{Schema: "fak-microcontext-semantic-submission/1", Answers: sp.TrialResults[0].Answers}
		tmp := out + ".tmp-sub.json"
		grade := out + ".tmp-grade.json"
		_ = writeJSONFile(tmp, sub)
		_ = gradeSemanticFiles(goldPath, tmp, grade, "test")
		b, _ := os.ReadFile(grade)
		_ = json.Unmarshal(b, &sp.Grade)
		_ = os.Remove(tmp)
		_ = os.Remove(grade)
		legacy := livePipelineResult{TrialResults: sp.TrialResults}
		aggregateLive(&legacy)
		sp.Aggregate = legacy.Aggregate
		r.Pipelines = append(r.Pipelines, sp)
	}
	r.Limits = []string{"Tune-only threshold calibration; held-out answers never select configuration.", "Dollar price unavailable for this exact route.", "Typed chunk reduce concatenates validated per-record facts; no extra model call.", "Tool evidence execution remains outside this matrix."}
	return writeJSONFile(out, r)
}
func verifyStrongLiveMatrix(path string) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	var r strongLiveReport
	if e = json.Unmarshal(b, &r); e != nil {
		return e
	}
	if r.Schema != strongLiveMatrixSchema || r.Trials < 2 || r.Workers < 2 || r.RetrievalK < 1 || len(r.Pipelines) != 4 {
		return fmt.Errorf("strong matrix envelope incomplete")
	}
	for _, p := range r.Pipelines {
		if p.Calibration.TuneRecords != 16 || len(p.Calibration.Candidates) < 4 || len(p.TrialResults) != r.Trials || p.Aggregate.PromptTokens == 0 || p.Grade.Records != 16 {
			return fmt.Errorf("%s incomplete", p.Pipeline)
		}
		if p.Pipeline == "retrieval-rerank" && len(p.RetrievalTrace) != 16 {
			return fmt.Errorf("retrieval trace incomplete")
		}
	}
	return nil
}
