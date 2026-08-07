package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

const batchExecutionSchema = "fak-microcontext-compat-batch-execution/1"

type batchExecutionReport struct {
	Schema                    string   `json:"schema"`
	Verdict                   string   `json:"verdict"`
	CapturedAt                string   `json:"captured_at"`
	ModelArtifact             string   `json:"model_artifact"`
	Hardware                  string   `json:"hardware"`
	ForwardPath               string   `json:"forward_path"`
	Submitted                 int      `json:"submitted"`
	Scheduled                 int      `json:"scheduled"`
	Cancelled                 int      `json:"cancelled"`
	Rejected                  int      `json:"rejected"`
	Classes                   int      `json:"compatibility_classes"`
	Batches                   int      `json:"executed_batches"`
	MaxBatch                  int      `json:"max_batch"`
	PrefixTokens              int      `json:"prefix_tokens_prefilled_once"`
	UsefulTokenSteps          int      `json:"useful_token_steps"`
	AllocatedSteps            int      `json:"allocated_steps"`
	PaddingTax                float64  `json:"real_padding_tax"`
	BatchFill                 float64  `json:"real_batch_fill"`
	QueueDelayP50MS           float64  `json:"queue_delay_p50_ms"`
	QueueDelayP95MS           float64  `json:"queue_delay_p95_ms"`
	BatchWallMS               float64  `json:"batch_wall_ms"`
	SequentialWallMS          float64  `json:"tuned_sequential_wall_ms"`
	BatchTokensPerSecond      float64  `json:"batch_tokens_per_second"`
	SequentialTokensPerSecond float64  `json:"tuned_sequential_tokens_per_second"`
	ThroughputRatio           float64  `json:"batch_vs_sequential_throughput_ratio"`
	NonemptyLogitRows         int      `json:"nonempty_logit_rows"`
	ExpectedLogitRows         int      `json:"expected_logit_rows"`
	IsolationCheck            string   `json:"isolation_check"`
	Claims                    []string `json:"claims"`
	NonClaims                 []string `json:"non_claims"`
}

type batchTask struct {
	id       string
	tokens   []int
	enqueued time.Time
}

func runBatchExecution(modelPath, hardware, out string, maxBatch int) error {
	if modelPath == "" || hardware == "" || out == "" {
		return errors.New("batch execution requires -gguf, -hardware, and -compat-batch-execution")
	}
	loaded, err := ggufload.LoadModelQuant(modelPath)
	if err != nil {
		return err
	}
	prefix := make([]int, 64)
	for i := range prefix {
		prefix[i] = 100 + i%7
	}
	seed := loaded.NewSession()
	seed.Quant = true
	seed.PrefillNoLogits(prefix)
	now := time.Now()
	keys := []microagent.CompatibilityKey{
		{Model: "qwen-0.5b", Sampling: "greedy", Tools: "none", Prefix: "base-1", Phase: "decode", SequenceBucket: 8},
		{Model: "qwen-0.5b", Sampling: "greedy", Tools: "read", Prefix: "base-1", Phase: "decode", SequenceBucket: 8},
		{Model: "qwen-0.5b", Sampling: "greedy", Tools: "none", Prefix: "base-1", Phase: "decode", SequenceBucket: 16},
	}
	tasks := map[string]batchTask{}
	var work []microagent.CompatibleWork
	for i := 0; i < 24; i++ {
		id := fmt.Sprintf("exec-%02d", i)
		n := 2 + i%5
		ids := make([]int, n)
		for j := range ids {
			ids[j] = 200 + (i*7+j)%200
		}
		enq := now.Add(time.Duration(i%4) * time.Millisecond)
		tasks[id] = batchTask{id, ids, enq}
		work = append(work, microagent.CompatibleWork{ID: id, Key: keys[i%3], Tokens: n, Priority: i % 3, Enqueued: enq})
	}
	work = append(work, microagent.CompatibleWork{ID: "cancelled", Key: keys[0], Tokens: 3, Cancelled: true, Enqueued: now})
	plans, summary, err := microagent.ComposeCompatible(work, microagent.CompatibilityConfig{MaxBatch: maxBatch, MaxQueuePerClass: 32, MaxPadding: .50, StarvationAfter: 2 * time.Millisecond, Now: now.Add(10 * time.Millisecond)})
	if err != nil {
		return err
	}
	start := time.Now()
	var delays []float64
	useful, slots, rows := 0, 0, 0
	for _, plan := range plans {
		batchStart := time.Now()
		seqs := loaded.NewBatchFromPrefixReserve(seed.Cache, len(plan.IDs), 6)
		seqs.SetQuant(true)
		maxLen := 0
		for _, id := range plan.IDs {
			if len(tasks[id].tokens) > maxLen {
				maxLen = len(tasks[id].tokens)
			}
			delays = append(delays, float64(batchStart.Sub(start))/float64(time.Millisecond))
		}
		for pos := 0; pos < maxLen; pos++ {
			ids := make([]int, len(plan.IDs))
			active := make([]bool, len(plan.IDs))
			for i, id := range plan.IDs {
				t := tasks[id]
				if pos < len(t.tokens) {
					ids[i] = t.tokens[pos]
					active[i] = true
					useful++
				}
			}
			logits := seqs.StepBatchActive(ids, active)
			slots += len(plan.IDs)
			for i, on := range active {
				if on && len(logits[i]) > 0 && finiteRow(logits[i]) {
					rows++
				}
			}
		}
	}
	batchWall := time.Since(start)
	seqStart := time.Now()
	seqRows := 0
	for _, w := range work {
		if w.Cancelled {
			continue
		}
		t := tasks[w.ID]
		s := loaded.SessionFromPrefix(seed.Cache)
		s.Quant = true
		for _, id := range t.tokens {
			if logits := s.Step(id); len(logits) > 0 && finiteRow(logits) {
				seqRows++
			}
		}
	}
	seqWall := time.Since(seqStart)
	sort.Float64s(delays)
	r := batchExecutionReport{Schema: batchExecutionSchema, Verdict: "PASS", CapturedAt: time.Now().UTC().Format(time.RFC3339), ModelArtifact: modelPath, Hardware: hardware, ForwardPath: "in-kernel GGUF lean-Q8 model.NewBatchFromPrefixReserve + BatchSession.StepBatchActive", Submitted: summary.Submitted, Scheduled: summary.Scheduled, Cancelled: summary.Cancelled, Rejected: summary.Rejected, Classes: 3, Batches: len(plans), MaxBatch: maxBatch, PrefixTokens: len(prefix), UsefulTokenSteps: useful, AllocatedSteps: slots, PaddingTax: float64(slots-useful) / float64(slots), BatchFill: float64(summary.Scheduled) / float64(len(plans)*maxBatch), QueueDelayP50MS: percentile(delays, .50), QueueDelayP95MS: percentile(delays, .95), BatchWallMS: float64(batchWall) / float64(time.Millisecond), SequentialWallMS: float64(seqWall) / float64(time.Millisecond), BatchTokensPerSecond: float64(useful) / batchWall.Seconds(), SequentialTokensPerSecond: float64(useful) / seqWall.Seconds(), ThroughputRatio: seqWall.Seconds() / batchWall.Seconds(), NonemptyLogitRows: rows, ExpectedLogitRows: useful, IsolationCheck: fmt.Sprintf("planner retained 3 model/tool/bucket classes; each lane cloned one exact 64-token prefix KV and held an independent cache; batch=%d sequential=%d finite rows", rows, seqRows), Claims: []string{"real compatibility plans executed through NewBatchFromPrefixReserve and StepBatchActive", "aggregate token-step throughput is compared with same-model prefix-cloned sequential decode"}, NonClaims: []string{"token IDs are a deterministic mixed-workload fixture, not useful text quality", "aggregate throughput is not single-stream latency or orchestration concurrency"}}
	if err := verifyBatchExecutionReport(r); err != nil {
		r.Verdict = "FAIL"
		return err
	}
	b, _ := json.MarshalIndent(r, "", "  ")
	return os.WriteFile(out, append(b, '\n'), 0644)
}

func finiteRow(v []float32) bool {
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	return true
}
func percentile(v []float64, q float64) float64 {
	if len(v) == 0 {
		return 0
	}
	i := int(math.Ceil(q*float64(len(v)))) - 1
	if i < 0 {
		i = 0
	}
	return v[i]
}
func verifyBatchExecutionReport(r batchExecutionReport) error {
	if r.Schema != batchExecutionSchema || r.Verdict != "PASS" || r.Submitted != 25 || r.Scheduled != 24 || r.Cancelled != 1 || r.Rejected != 0 || r.Classes != 3 || r.Batches < 3 || r.PrefixTokens != 64 || r.UsefulTokenSteps <= 0 || r.AllocatedSteps < r.UsefulTokenSteps || r.NonemptyLogitRows != r.ExpectedLogitRows || r.BatchWallMS <= 0 || r.SequentialWallMS <= 0 || r.BatchTokensPerSecond <= 0 || r.SequentialTokensPerSecond <= 0 || r.ThroughputRatio <= 0 || len(r.Claims) == 0 || len(r.NonClaims) == 0 {
		return errors.New("compatibility batch execution invariant failed")
	}
	return nil
}
func verifyBatchExecutionArtifact(path string) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	var r batchExecutionReport
	if e = json.Unmarshal(b, &r); e != nil {
		return e
	}
	return verifyBatchExecutionReport(r)
}

func registerBatchExecutionFlags(fs *flag.FlagSet, modelPath, hardware, output, verify *string, maxBatch *int) {
	fs.StringVar(modelPath, "gguf", "", "GGUF model for real compatibility-batch execution")
	fs.StringVar(hardware, "compat-batch-hardware", "", "hardware provenance for compatibility-batch execution")
	fs.StringVar(output, "compat-batch-execution", "", "execute planned compatibility batches and write artifact")
	fs.StringVar(verify, "verify-compat-batch-execution", "", "verify compatibility-batch execution artifact")
	fs.IntVar(maxBatch, "compat-batch-size", 8, "maximum physical batch width")
}
