package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

const tailPolicySchema = "fak-microcontext-tail-policy-matrix/1"

type tailPolicyReport struct {
	Schema, CreatedAt, PacketSHA256, GoldSHA256 string
	Endpoint                                    endpointProvenance
	Trials, Workers                             int
	WindowDeadlineMS, TaskDeadlineMS            int64
	Policies                                    []tailPolicyResult
	Limits                                      []string
}
type tailPolicyResult struct {
	Policy      string
	Description string
	Trials      []tailTrial
	Aggregate   tailAggregate
	Grade       semanticGrade
}
type tailTrial struct {
	Trial                                                                                                        int
	WallMS                                                                                                       float64
	Opened, Completed, TimedOut, CancelledBeforeOpen, CancelledInFlight, HedgesOpened, HedgeWins, LogicalRecords int
	PromptTokens, CompletionTokens, CachedTokens                                                                 int64
	Retries                                                                                                      int
	WastedPromptTokens, WastedCompletionTokens                                                                   int64
	CancelledBilled                                                                                              string
	Receipts                                                                                                     []windowReceipt
	Answers                                                                                                      []semanticConsensus
}
type windowReceipt struct {
	ID, Status, Winner                           string
	Opened, Hedged, ReadbackVerified             bool
	LatencyMS                                    float64
	PromptTokens, CompletionTokens, CachedTokens int64
	Reason                                       string
}
type tailAggregate struct {
	MeanWallMS                                                                                   float64
	Opened, Completed, TimedOut, CancelledBeforeOpen, CancelledInFlight, HedgesOpened, HedgeWins int
	PromptTokens, CompletionTokens, CachedTokens, WastedTokens                                   int64
}
type windowResult struct {
	id             string
	primary, hedge liveCall
	winner         liveCall
	hedged         bool
	status         string
	reason         string
}

func abstentionFor(id string) semanticConsensus {
	return semanticConsensus{ID: id, SemanticNeed: "abstain", ToolNeed: "abstain", Actionability: "abstain"}
}
func runWindow(ctx context.Context, c *liveMatrixClient, r semanticRecord, policy string, window time.Duration, hedgeAfter time.Duration) windowResult {
	wctx, cancel := context.WithTimeout(ctx, window)
	defer cancel()
	prompt := livePrompt([]semanticRecord{r}, nil, "micro-context tail policy "+policy)
	if policy != "bounded-hedge" {
		v := c.call(wctx, prompt)
		x := windowResult{id: r.ID, primary: v, winner: v}
		if v.err != nil {
			x.status = "timed_out"
			x.reason = v.err.Error()
		} else {
			x.status = "confirmed"
		}
		return x
	}
	ch := make(chan struct {
		which string
		v     liveCall
	}, 2)
	go func() {
		ch <- struct {
			which string
			v     liveCall
		}{"primary", c.call(wctx, prompt)}
	}()
	timer := time.NewTimer(hedgeAfter)
	defer timer.Stop()
	var first *struct {
		which string
		v     liveCall
	}
	select {
	case x := <-ch:
		first = &x
	case <-timer.C:
	}
	if first != nil && first.v.err == nil {
		return windowResult{id: r.ID, primary: first.v, winner: first.v, status: "confirmed"}
	}
	go func() {
		ch <- struct {
			which string
			v     liveCall
		}{"hedge", c.call(wctx, prompt)}
	}()
	x := windowResult{id: r.ID, hedged: true}
	if first != nil {
		if first.which == "primary" {
			x.primary = first.v
		} else {
			x.hedge = first.v
		}
	}
	for i := 0; i < 2; i++ {
		select {
		case z := <-ch:
			if z.which == "primary" {
				x.primary = z.v
			} else {
				x.hedge = z.v
			}
			if z.v.err == nil {
				cancel()
				x.winner = z.v
				x.status = "confirmed"
				x.reason = z.which
				return x
			}
		case <-wctx.Done():
			x.status = "timed_out"
			x.reason = wctx.Err().Error()
			return x
		}
	}
	x.status = "timed_out"
	x.reason = "both attempts failed"
	return x
}
func executeTailPolicy(c *liveMatrixClient, records []semanticRecord, policy string, trial, workers int, window, task, hedgeAfter time.Duration, sufficiency int) tailTrial {
	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), task)
	defer cancel()
	t := tailTrial{Trial: trial, LogicalRecords: len(records), CancelledBilled: "unknown: endpoint does not expose post-cancel billing"}
	jobs := make(chan semanticRecord)
	out := make(chan windowResult, len(records))
	var wg sync.WaitGroup
	var stop atomic.Bool
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := range jobs {
				if stop.Load() || ctx.Err() != nil {
					out <- windowResult{id: r.ID, status: "cancelled_before_open", reason: "task stop"}
					continue
				}
				out <- runWindow(ctx, c, r, policy, window, hedgeAfter)
			}
		}()
	}
	go func() {
		for _, r := range records {
			jobs <- r
		}
		close(jobs)
		wg.Wait()
		close(out)
	}()
	answers := map[string]semanticConsensus{}
	confirmed := 0
	for x := range out {
		rec := windowReceipt{ID: x.id, Status: x.status, Hedged: x.hedged, Reason: x.reason, Winner: x.reason}
		if x.status == "confirmed" {
			confirmed++
			rec.Opened = true
			rec.ReadbackVerified = true
			rec.LatencyMS = float64(x.winner.latency) / float64(time.Millisecond)
			rec.PromptTokens = x.winner.prompt
			rec.CompletionTokens = x.winner.completion
			rec.CachedTokens = x.winner.cached
			t.Completed++
			t.PromptTokens += x.winner.prompt
			t.CompletionTokens += x.winner.completion
			t.CachedTokens += x.winner.cached
			t.Retries += x.winner.retry
			for _, a := range applyThreshold(x.winner.answers, .95) {
				answers[a.ID] = a
			}
			if policy == "sufficiency-stop" && confirmed >= sufficiency {
				stop.Store(true)
				cancel()
			}
		} else if x.status == "cancelled_before_open" {
			t.CancelledBeforeOpen++
			answers[x.id] = abstentionFor(x.id)
		} else {
			t.TimedOut++
			t.CancelledInFlight++
			answers[x.id] = abstentionFor(x.id)
		}
		if x.hedged {
			t.HedgesOpened++
			loser := x.primary
			if x.reason == "primary" {
				loser = x.hedge
			} else if x.reason == "hedge" {
				t.HedgeWins++
				loser = x.primary
			}
			t.WastedPromptTokens += loser.prompt
			t.WastedCompletionTokens += loser.completion
		}
		t.Receipts = append(t.Receipts, rec)
	}
	for _, r := range records {
		if a, ok := answers[r.ID]; ok {
			t.Answers = append(t.Answers, a)
		} else {
			t.Answers = append(t.Answers, abstentionFor(r.ID))
		}
	}
	sort.Slice(t.Receipts, func(i, j int) bool { return t.Receipts[i].ID < t.Receipts[j].ID })
	t.Opened = t.Completed + t.TimedOut
	t.WallMS = float64(time.Since(start)) / float64(time.Millisecond)
	return t
}
func runTailPolicyMatrix(packetPath, goldPath, out, endpoint, key, model, class, hardware string, trials, workers int, windowMS, taskMS, hedgeMS int64, sufficiency int) error {
	pb, e := os.ReadFile(packetPath)
	if e != nil {
		return e
	}
	gb, e := os.ReadFile(goldPath)
	if e != nil {
		return e
	}
	var p semanticPacket
	if e = json.Unmarshal(pb, &p); e != nil {
		return e
	}
	var test []semanticRecord
	for _, x := range p.Records {
		if x.Split == "test" {
			test = append(test, x)
		}
	}
	c := &liveMatrixClient{endpoint: endpoint, key: key, model: model, client: &http.Client{Timeout: 12 * time.Minute}}
	r := tailPolicyReport{Schema: tailPolicySchema, CreatedAt: time.Now().UTC().Format(time.RFC3339), PacketSHA256: shaHex(pb), GoldSHA256: shaHex(gb), Endpoint: endpointProvenance{Class: class, Model: model, Hardware: hardware, NativeBatch: "unsupported-by-chat-route", PrefixCache: "usage-observed-only", PricingSnapshot: "unavailable"}, Trials: trials, Workers: workers, WindowDeadlineMS: windowMS, TaskDeadlineMS: taskMS}
	for _, pol := range []string{"wait-all", "deadline-abstain", "sufficiency-stop", "bounded-hedge"} {
		x := tailPolicyResult{Policy: pol, Description: map[string]string{"wait-all": "no policy deadline beyond task cap", "deadline-abstain": "per-window deadline folds timeout as abstention", "sufficiency-stop": "cancel after configured confirmed-fact count", "bounded-hedge": "launch one duplicate after hedge delay; first valid result wins"}[pol]}
		w := time.Duration(windowMS) * time.Millisecond
		if pol == "wait-all" {
			w = time.Duration(taskMS) * time.Millisecond
		}
		for i := 1; i <= trials; i++ {
			x.Trials = append(x.Trials, executeTailPolicy(c, test, pol, i, workers, w, time.Duration(taskMS)*time.Millisecond, time.Duration(hedgeMS)*time.Millisecond, sufficiency))
		}
		sub := semanticSubmission{Schema: "fak-microcontext-semantic-submission/1", Answers: x.Trials[0].Answers}
		tmp := out + ".sub.tmp"
		gr := out + ".grade.tmp"
		_ = writeJSONFile(tmp, sub)
		_ = gradeSemanticFiles(goldPath, tmp, gr, "test")
		b, _ := os.ReadFile(gr)
		_ = json.Unmarshal(b, &x.Grade)
		_ = os.Remove(tmp)
		_ = os.Remove(gr)
		aggregateTail(&x)
		r.Policies = append(r.Policies, x)
	}
	r.Limits = []string{"Cancelled-but-billed is unknown because the endpoint exposes usage only on completed streams.", "Quality uses first-trial fold; all trials preserve receipts and usage.", "Sufficiency is a configured count proxy, not a learned task-completion oracle.", "No dollar claim: exact route pricing is unavailable."}
	return writeJSONFile(out, r)
}
func aggregateTail(x *tailPolicyResult) {
	n := float64(len(x.Trials))
	for _, t := range x.Trials {
		x.Aggregate.MeanWallMS += t.WallMS
		x.Aggregate.Opened += t.Opened
		x.Aggregate.Completed += t.Completed
		x.Aggregate.TimedOut += t.TimedOut
		x.Aggregate.CancelledBeforeOpen += t.CancelledBeforeOpen
		x.Aggregate.CancelledInFlight += t.CancelledInFlight
		x.Aggregate.HedgesOpened += t.HedgesOpened
		x.Aggregate.HedgeWins += t.HedgeWins
		x.Aggregate.PromptTokens += t.PromptTokens
		x.Aggregate.CompletionTokens += t.CompletionTokens
		x.Aggregate.CachedTokens += t.CachedTokens
		x.Aggregate.WastedTokens += t.WastedPromptTokens + t.WastedCompletionTokens
	}
	x.Aggregate.MeanWallMS /= n
}
func verifyTailPolicyMatrix(path string) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	var r tailPolicyReport
	if e = json.Unmarshal(b, &r); e != nil {
		return e
	}
	if r.Schema != tailPolicySchema || r.Trials < 2 || len(r.Policies) != 4 || r.WindowDeadlineMS <= 0 || r.TaskDeadlineMS <= r.WindowDeadlineMS {
		return fmt.Errorf("tail matrix envelope incomplete")
	}
	seenCancel, seenHedge := false, false
	for _, p := range r.Policies {
		if len(p.Trials) != r.Trials || p.Grade.Records != 16 {
			return fmt.Errorf("%s incomplete", p.Policy)
		}
		for _, t := range p.Trials {
			if len(t.Receipts) != 16 {
				return fmt.Errorf("%s receipt count", p.Policy)
			}
			if t.CancelledBeforeOpen+t.CancelledInFlight > 0 {
				seenCancel = true
			}
			if t.HedgesOpened > 0 {
				seenHedge = true
			}
		}
	}
	if !seenCancel || !seenHedge {
		return fmt.Errorf("matrix did not exercise cancellation and hedging")
	}
	return nil
}
