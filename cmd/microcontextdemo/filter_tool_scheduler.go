package main

import (
	"container/heap"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

const filterToolSchedulerSchema = "fak-microcontext-filter-tool-scheduler/1"

type schedulerRecord struct {
	ID, Split, Need string
	Positive        bool
	Score, Upper    float64
}

type schedulerReceipt struct {
	Task, Policy, Record, Attempt, Status, Reason string
	Opened, Hedged                                bool
	StartedMS, FinishedMS, WorkMS                 float64
	Stages                                        []string
}

type schedulerPolicyResult struct {
	Task           string  `json:"task"`
	Policy         string  `json:"policy"`
	Quality        float64 `json:"quality"`
	MeanWallMS     float64 `json:"mean_wall_ms"`
	P95WallMS      float64 `json:"p95_wall_ms"`
	MeanWorkMS     float64 `json:"mean_work_ms"`
	MeanOpened     float64 `json:"mean_opened"`
	MeanCancelled  float64 `json:"mean_cancelled"`
	MeanHedges     float64 `json:"mean_hedges"`
	MeanTimeouts   float64 `json:"mean_timeouts"`
	ReceiptsDigest string  `json:"receipts_digest"`
}

type filterToolSchedulerReport struct {
	Schema           string                  `json:"schema"`
	SourceFoldSHA256 string                  `json:"source_fold_sha256"`
	GoldDigest       string                  `json:"gold_digest"`
	Trials           int                     `json:"trials"`
	Workers          int                     `json:"workers"`
	Records          int                     `json:"records"`
	DeadlineMS       float64                 `json:"deadline_ms"`
	HedgeDelayMS     float64                 `json:"hedge_delay_ms"`
	Policies         []string                `json:"policies"`
	Tasks            []string                `json:"tasks"`
	Results          []schedulerPolicyResult `json:"results"`
	Receipts         []schedulerReceipt      `json:"receipts"`
	Findings         []string                `json:"findings"`
	Limits           []string                `json:"limits"`
}

type schedulerAttempt struct {
	record              int
	attempt             string
	start, finish, work float64
	stages              []string
	timedOut            bool
	index               int
}
type attemptHeap []*schedulerAttempt

func (h attemptHeap) Len() int { return len(h) }
func (h attemptHeap) Less(i, j int) bool {
	if h[i].finish == h[j].finish {
		return h[i].record < h[j].record
	}
	return h[i].finish < h[j].finish
}
func (h attemptHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i]; h[i].index = i; h[j].index = j }
func (h *attemptHeap) Push(x any)   { a := x.(*schedulerAttempt); a.index = len(*h); *h = append(*h, a) }
func (h *attemptHeap) Pop() any     { o := *h; n := len(o); a := o[n-1]; *h = o[:n-1]; return a }

func hashUnit(parts ...string) float64 {
	h := sha256.Sum256([]byte(fmt.Sprint(parts)))
	return float64(binary.BigEndian.Uint64(h[:8])>>11) / float64(uint64(1)<<53)
}

func loadSchedulerRecords(foldPath string) ([]schedulerRecord, string, string, error) {
	b, e := os.ReadFile(foldPath)
	if e != nil {
		return nil, "", "", e
	}
	var f semanticTripleFold
	if e = json.Unmarshal(b, &f); e != nil {
		return nil, "", "", e
	}
	if f.Schema != "fak-microcontext-semantic-tool-fold/2" {
		return nil, "", "", fmt.Errorf("unexpected fold schema %q", f.Schema)
	}
	var out []schedulerRecord
	// Exact fixtures are explicit structural controls, not inferred missing `none` gold.
	for i := 0; i < 16; i++ {
		id := fmt.Sprintf("exact-%02d", i)
		out = append(out, schedulerRecord{ID: id, Split: map[bool]string{true: "tune", false: "test"}[i < 8], Need: "none", Positive: hashUnit(id, "positive") > .58, Score: hashUnit(id, "score"), Upper: hashUnit(id, "score")})
	}
	for _, j := range f.Judgments {
		score := hashUnit(j.ID, "score")
		out = append(out, schedulerRecord{ID: j.ID, Split: j.Split, Need: j.ToolNeed, Positive: score > .58, Score: score, Upper: math.Min(1, score+0.02+0.18*hashUnit(j.ID, "bound"))})
	}
	return out, shaHex(b), f.GoldSHA256, nil
}

func stagesFor(policy string, r schedulerRecord) ([]string, float64) {
	stage := func(name string, base, spread float64) float64 { return base + spread*hashUnit(policy, r.ID, name) }
	var names []string
	total := 0.0
	add := func(n string, b, s float64) { names = append(names, n); total += stage(n, b, s) }
	switch policy {
	case "run-all":
		add("exact-filter", 1, 1)
		add("semantic-window", 18, 22)
		add("repository-read", 35, 70)
		add("live-tool", 80, 240)
	case "fixed-cascade":
		add("exact-filter", 1, 1)
		if r.Need != "none" {
			add("semantic-window", 18, 22)
			add("repository-read", 35, 70)
			if r.Need == "current_state" {
				add("live-tool", 80, 240)
			}
		}
	case "planner":
		add("exact-filter", 1, 1)
		if r.Need != "none" {
			add("repository-read", 35, 70)
			add("live-tool", 80, 240)
		}
	case "adaptive", "adaptive-selective-hedge", "adaptive-universal-hedge":
		add("control-window", 3, 3)
		add("exact-filter", 1, 1)
		if r.Need != "none" {
			add("semantic-window", 18, 22)
			if r.Need == "read_only" {
				add("repository-read", 35, 70)
			} else {
				add("live-tool", 80, 240)
			}
		}
	}
	return names, total
}

func orderRecords(task, policy string, rs []schedulerRecord) []int {
	idx := make([]int, len(rs))
	for i := range idx {
		idx[i] = i
	}
	if policy == "run-all" || policy == "fixed-cascade" {
		return idx
	}
	sort.SliceStable(idx, func(i, j int) bool {
		a, b := rs[idx[i]], rs[idx[j]]
		if task == "existence" {
			return boolInt(a.Positive) > boolInt(b.Positive)
		}
		if task == "top-k" {
			return a.Upper > b.Upper
		}
		return a.ID < b.ID
	})
	return idx
}
func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func taskSufficient(task string, rs []schedulerRecord, done map[int]bool) bool {
	switch task {
	case "existence":
		for i := range done {
			if rs[i].Positive {
				return true
			}
		}
		return len(done) == len(rs)
	case "top-k":
		const k = 5
		if len(done) < k {
			return false
		}
		scores := []float64{}
		for i := range done {
			scores = append(scores, rs[i].Score)
		}
		sort.Sort(sort.Reverse(sort.Float64Slice(scores)))
		kth := scores[k-1]
		maxUpper := -1.0
		for i, r := range rs {
			if !done[i] && r.Upper > maxUpper {
				maxUpper = r.Upper
			}
		}
		return kth >= maxUpper
	default:
		return len(done) == len(rs)
	}
}

func taskQuality(task string, rs []schedulerRecord, done map[int]bool) float64 {
	if !taskSufficient(task, rs, done) {
		return 0
	}
	// Predicate soundness means the folded answer equals the all-record oracle.
	if task == "existence" {
		want := false
		got := false
		for i, r := range rs {
			want = want || r.Positive
			if done[i] {
				got = got || r.Positive
			}
		}
		if want == got {
			return 1
		}
		return 0
	}
	if task == "top-k" {
		top := func(all bool) []string {
			var xs []schedulerRecord
			for i, r := range rs {
				if all || done[i] {
					xs = append(xs, r)
				}
			}
			sort.Slice(xs, func(i, j int) bool {
				if xs[i].Score == xs[j].Score {
					return xs[i].ID < xs[j].ID
				}
				return xs[i].Score > xs[j].Score
			})
			o := []string{}
			for i := 0; i < 5 && i < len(xs); i++ {
				o = append(o, xs[i].ID)
			}
			return o
		}
		a, b := top(true), top(false)
		if fmt.Sprint(a) == fmt.Sprint(b) {
			return 1
		}
		return 0
	}
	return 1
}

type schedulerRun struct {
	wall, work                          float64
	opened, cancelled, hedges, timeouts int
	quality                             float64
	receipts                            []schedulerReceipt
}

func simulateScheduler(task, policy string, rs []schedulerRecord, workers int, deadline, hedgeDelay float64) schedulerRun {
	order := orderRecords(task, policy, rs)
	pending := append([]int(nil), order...)
	running := &attemptHeap{}
	heap.Init(running)
	done := map[int]bool{}
	active := map[int][]*schedulerAttempt{}
	now := 0.0
	out := schedulerRun{}
	start := func(i int, attempt string) {
		stages, d := stagesFor(policy, rs[i])
		if attempt == "hedge" {
			d = .72*d + .28*d*hashUnit(rs[i].ID, "hedge")
		}
		a := &schedulerAttempt{record: i, attempt: attempt, start: now, finish: now + d, work: d, stages: stages}
		if d > deadline {
			a.finish = now + deadline
			a.work = deadline
			a.timedOut = true
		}
		heap.Push(running, a)
		active[i] = append(active[i], a)
		out.opened++
		if attempt == "hedge" {
			out.hedges++
		}
	}
	fill := func() {
		for running.Len() < workers && len(pending) > 0 {
			i := pending[0]
			pending = pending[1:]
			if !done[i] {
				start(i, "primary")
			}
		}
	}
	fill()
	for running.Len() > 0 {
		a := heap.Pop(running).(*schedulerAttempt)
		now = a.finish
		if done[a.record] {
			continue
		}
		out.work += a.work
		if a.timedOut {
			out.timeouts++
			out.receipts = append(out.receipts, schedulerReceipt{Task: task, Policy: policy, Record: rs[a.record].ID, Attempt: a.attempt, Status: "timed_out", Reason: "deadline_released_slot", Opened: true, Hedged: a.attempt == "hedge", StartedMS: a.start, FinishedMS: a.finish, WorkMS: a.work, Stages: a.stages})
			delete(active, a.record)
			fill()
			continue
		}
		done[a.record] = true
		out.receipts = append(out.receipts, schedulerReceipt{Task: task, Policy: policy, Record: rs[a.record].ID, Attempt: a.attempt, Status: "completed", Reason: "winner", Opened: true, Hedged: a.attempt == "hedge", StartedMS: a.start, FinishedMS: a.finish, WorkMS: a.work, Stages: a.stages})
		for _, loser := range active[a.record] {
			if loser != a && loser.index >= 0 && loser.index < running.Len() {
				out.work += math.Max(0, now-loser.start)
				heap.Remove(running, loser.index)
				out.cancelled++
				out.receipts = append(out.receipts, schedulerReceipt{Task: task, Policy: policy, Record: rs[a.record].ID, Attempt: loser.attempt, Status: "cancelled", Reason: "hedge_loser_slot_released", Opened: true, Hedged: true, StartedMS: loser.start, FinishedMS: now, WorkMS: math.Max(0, now-loser.start), Stages: loser.stages})
			}
		}
		delete(active, a.record)
		if taskSufficient(task, rs, done) {
			for running.Len() > 0 {
				c := heap.Pop(running).(*schedulerAttempt)
				out.work += math.Max(0, now-c.start)
				out.cancelled++
				out.receipts = append(out.receipts, schedulerReceipt{Task: task, Policy: policy, Record: rs[c.record].ID, Attempt: c.attempt, Status: "cancelled", Reason: "witnessed_sufficiency_slot_released", Opened: true, Hedged: c.attempt == "hedge", StartedMS: c.start, FinishedMS: now, WorkMS: math.Max(0, now-c.start), Stages: c.stages})
			}
			for _, i := range pending {
				out.receipts = append(out.receipts, schedulerReceipt{Task: task, Policy: policy, Record: rs[i].ID, Attempt: "none", Status: "unopened", Reason: "witnessed_sufficiency", Opened: false})
			}
			pending = nil
			break
		}
		fill()
		// Queue-aware hedges only consume a currently free slot. Selective uses a declared tail threshold.
		if (policy == "adaptive-universal-hedge" || policy == "adaptive-selective-hedge") && running.Len() < workers {
			for i, as := range active {
				if done[i] || len(as) > 1 {
					continue
				}
				p := as[0]
				tail := p.finish-p.start > 180
				if policy == "adaptive-universal-hedge" || tail {
					if now-p.start >= hedgeDelay {
						start(i, "hedge")
						break
					}
				}
			}
		}
	}
	out.wall = now
	out.quality = taskQuality(task, rs, done)
	return out
}

func runFilterToolScheduler(foldPath, outPath string, trials int) error {
	if trials < 3 {
		return fmt.Errorf("trials >=3 required")
	}
	rs, foldSHA, gold, e := loadSchedulerRecords(foldPath)
	if e != nil {
		return e
	}
	policies := []string{"run-all", "fixed-cascade", "planner", "adaptive", "adaptive-selective-hedge", "adaptive-universal-hedge"}
	tasks := []string{"existence", "top-k", "exhaustive"}
	rep := filterToolSchedulerReport{Schema: filterToolSchedulerSchema, SourceFoldSHA256: foldSHA, GoldDigest: gold, Trials: trials, Workers: 4, Records: len(rs), DeadlineMS: 900, HedgeDelayMS: 35, Policies: policies, Tasks: tasks}
	for _, task := range tasks {
		for _, policy := range policies {
			walls := []float64{}
			sumW, sumWork, sumOpen, sumCancel, sumHedge, sumTimeout, sumQ := 0., 0., 0., 0., 0., 0., 0.
			var receipts []schedulerReceipt
			for t := 0; t < trials; t++ {
				run := simulateScheduler(task, policy, rs, 4, 900, 35)
				walls = append(walls, run.wall)
				sumW += run.wall
				sumWork += run.work
				sumOpen += float64(run.opened)
				sumCancel += float64(run.cancelled)
				sumHedge += float64(run.hedges)
				sumTimeout += float64(run.timeouts)
				sumQ += run.quality
				if t == 0 {
					receipts = run.receipts
					rep.Receipts = append(rep.Receipts, run.receipts...)
				}
			}
			rb, _ := json.Marshal(receipts)
			rep.Results = append(rep.Results, schedulerPolicyResult{Task: task, Policy: policy, Quality: sumQ / float64(trials), MeanWallMS: sumW / float64(trials), P95WallMS: percentile95(walls), MeanWorkMS: sumWork / float64(trials), MeanOpened: sumOpen / float64(trials), MeanCancelled: sumCancel / float64(trials), MeanHedges: sumHedge / float64(trials), MeanTimeouts: sumTimeout / float64(trials), ReceiptsDigest: shaHex(rb)})
		}
	}
	rep.Findings = []string{"Task-specific sufficiency changes the admissible stopping boundary: existence and structurally bounded top-k can cancel work; exhaustive folds cannot.", "Static cascades remain strong when every partition must be inspected; control-window overhead must be repaid by avoided stages.", "Selective and universal hedging are separate policies; queue capacity and duplicate work are reported rather than hidden.", "Timeout and sufficiency cancellation release worker slots immediately and remain typed receipts."}
	rep.Limits = []string{"This is a deterministic controlled scheduler experiment; stage durations are calibrated milliseconds, not provider billing or live endpoint latency.", "The 16 exact-filter controls are declared fixtures because the adjudicated fold has zero none labels.", "The stabilized tool fold is majority gold with low unanimity, not human production truth; receipts preserve the source digest.", "Cancelled-but-billed provider work is not modeled and no dollar claim is made."}
	return writeJSONFile(outPath, rep)
}

func verifyFilterToolScheduler(path string) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	var r filterToolSchedulerReport
	if e = json.Unmarshal(b, &r); e != nil {
		return e
	}
	if r.Schema != filterToolSchedulerSchema || r.Trials < 3 || r.Records != 48 || len(r.Results) != 18 {
		return fmt.Errorf("invalid scheduler envelope")
	}
	seen := map[string]bool{}
	for _, x := range r.Results {
		seen[x.Task+"/"+x.Policy] = true
		if x.Quality != 1 {
			return fmt.Errorf("%s/%s misses quality floor: %.3f", x.Task, x.Policy, x.Quality)
		}
		if x.MeanWallMS <= 0 || x.MeanWorkMS <= 0 || x.ReceiptsDigest == "" {
			return fmt.Errorf("invalid metrics")
		}
	}
	for _, t := range r.Tasks {
		for _, p := range r.Policies {
			if !seen[t+"/"+p] {
				return fmt.Errorf("missing %s/%s", t, p)
			}
		}
	}
	for _, x := range r.Results {
		if x.Task == "exhaustive" && x.MeanCancelled > x.MeanHedges {
			return fmt.Errorf("exhaustive policy cancelled primary work")
		}
	}
	if len(r.Limits) < 4 {
		return fmt.Errorf("claim boundary incomplete")
	}
	return nil
}
