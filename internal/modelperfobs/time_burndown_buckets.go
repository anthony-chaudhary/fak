package modelperfobs

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

// TimeBurndownBucketSchema is the additive campaign view: the /1 report plus
// fixed buckets normalized per token and an optional baseline delta.
const TimeBurndownBucketSchema = "fak-time-burndown/2"

// Campaign buckets. Every millisecond of included API wall lands in exactly one.
const (
	BucketCPUWork     = "cpu_work"
	BucketMemoryCopy  = "memory_copy"
	BucketWeightLoad  = "weight_load"
	BucketGPUKernel   = "gpu_kernel"
	BucketSync        = "sync"
	BucketUnexplained = "unexplained"
)

// Stages and their per-unit denominators.
const (
	StagePrefill  = "prefill"  // ms per prompt token
	StageDecode   = "decode"   // ms per completion token
	StageUnstaged = "unstaged" // ms per request
)

// Default campaign targets in ms per token.
const (
	DefaultPrefillTargetMS = 4.0
	DefaultDecodeTargetMS  = 20.0
)

// BurndownBuckets lists the closed bucket vocabulary in report order.
func BurndownBuckets() []string {
	return []string{BucketCPUWork, BucketMemoryCopy, BucketWeightLoad, BucketGPUKernel, BucketSync, BucketUnexplained}
}

// BurndownStages lists the closed stage vocabulary in report order.
func BurndownStages() []string { return []string{StagePrefill, StageDecode, StageUnstaged} }

// burndownPhaseBucket is the closed native-phase -> bucket map. A phase marked
// split books Exclusive.Active to cpu_work and Exclusive.Wait to unexplained
// (waiting on something no bounded phase names); otherwise its whole exclusive
// wall goes to the named bucket.
//
//	phase                  bucket
//	kernel                 gpu_kernel
//	synchronization        sync
//	host_device_upload     memory_copy
//	host_device_download   memory_copy
//	kv_restore             memory_copy
//	model_load             weight_load
//	tokenization           cpu_work active / unexplained wait
//	sampling               cpu_work active / unexplained wait
//	queue_admission        cpu_work active / unexplained wait
//	prefill (exclusive)    cpu_work active / unexplained wait
//	decode (exclusive)     cpu_work active / unexplained wait
//	kv_lookup              cpu_work active / unexplained wait
//	kv_evict               cpu_work active / unexplained wait
//	output                 cpu_work active / unexplained wait
//	native wall - phases   unexplained
//	API wall - native wall unexplained (stage unstaged)
//	API-only sample wall   unexplained (stage unstaged)
var burndownPhaseBucket = map[nativeperf.Phase]struct {
	bucket string
	split  bool
}{
	nativeperf.PhaseKernel:          {BucketGPUKernel, false},
	nativeperf.PhaseSynchronization: {BucketSync, false},
	nativeperf.PhaseHostUpload:      {BucketMemoryCopy, false},
	nativeperf.PhaseHostDownload:    {BucketMemoryCopy, false},
	nativeperf.PhaseKVRestore:       {BucketMemoryCopy, false},
	nativeperf.PhaseModelLoad:       {BucketWeightLoad, false},
	nativeperf.PhaseTokenization:    {BucketCPUWork, true},
	nativeperf.PhaseSampling:        {BucketCPUWork, true},
	nativeperf.PhaseQueueAdmission:  {BucketCPUWork, true},
	nativeperf.PhasePrefill:         {BucketCPUWork, true},
	nativeperf.PhaseDecode:          {BucketCPUWork, true},
	nativeperf.PhaseKVLookup:        {BucketCPUWork, true},
	nativeperf.PhaseKVEvict:         {BucketCPUWork, true},
	nativeperf.PhaseOutput:          {BucketCPUWork, true},
}

// burndownPhaseStage: prefill/decode own their stage; a child phase takes its
// receipt parent's stage; an ambiguous parent (the recorder clears it when a
// phase ran under both) is unstaged rather than guessed.
func burndownPhaseStage(p nativeperf.PhaseAccounting) string {
	switch {
	case p.Phase == nativeperf.PhasePrefill || p.Parent == nativeperf.PhasePrefill:
		return StagePrefill
	case p.Phase == nativeperf.PhaseDecode || p.Parent == nativeperf.PhaseDecode:
		return StageDecode
	default:
		return StageUnstaged
	}
}

// BurndownBucketRow is one (stage, bucket) cell.
type BurndownBucketRow struct {
	Stage           string   `json:"stage"`
	Bucket          string   `json:"bucket"`
	TotalMS         float64  `json:"total_ms"`
	MSPerUnit       *float64 `json:"ms_per_unit"`
	BeforeMSPerUnit *float64 `json:"before_ms_per_unit,omitempty"`
	DeltaPct        *float64 `json:"delta_pct,omitempty"`
}

// BurndownStageTotal sums a stage's buckets. Remaining = after - target; a
// negative value is headroom under target.
type BurndownStageTotal struct {
	Stage              string   `json:"stage"`
	Unit               string   `json:"unit"`
	Units              int64    `json:"units"`
	UnitsUnavailable   string   `json:"units_unavailable,omitempty"`
	TotalMS            float64  `json:"total_ms"`
	MSPerUnit          *float64 `json:"ms_per_unit"`
	BeforeMSPerUnit    *float64 `json:"before_ms_per_unit,omitempty"`
	DeltaPct           *float64 `json:"delta_pct,omitempty"`
	TargetMSPerUnit    *float64 `json:"target_ms_per_unit,omitempty"`
	RemainingMSPerUnit *float64 `json:"remaining_ms_per_unit,omitempty"`
}

type TimeBurndownBucketGroup struct {
	Backend    string  `json:"backend"`
	Model      string  `json:"model"`
	CacheState string  `json:"cache_state"`
	APIWallMS  float64 `json:"api_wall_ms"`
	BucketedMS float64 `json:"bucketed_ms"`
	// ProducerUnattributedMS is the envelope unattributed_ms total: the part of
	// unexplained the producer itself observed outside its receipt. It is a
	// subset of unexplained, never added to it.
	ProducerUnattributedMS *float64             `json:"producer_unattributed_ms,omitempty"`
	Baseline               string               `json:"baseline"`
	Stages                 []BurndownStageTotal `json:"stages"`
	Buckets                []BurndownBucketRow  `json:"buckets"`
}

// GoRuntimeLine is process-level Go runtime cost over the run. Its denominator
// is the process interval, not requests, so it is never merged into buckets.
type GoRuntimeLine struct {
	IntervalSeconds float64        `json:"interval_seconds"`
	GCCycles        runtimeobs.U64 `json:"gc_cycles"`
	GCCPUSeconds    runtimeobs.F64 `json:"gc_cpu_seconds"`
	GCPauseSeconds  runtimeobs.F64 `json:"gc_pause_cpu_seconds"`
	GCCPUFraction   runtimeobs.F64 `json:"gc_cpu_fraction"`
}

type BurndownTargets struct {
	PrefillMS float64
	DecodeMS  float64
}

// TimeBurndownV2 embeds the /1 report under the /2 schema.
type TimeBurndownV2 struct {
	TimeBurndown
	PrefillTargetMS float64                   `json:"prefill_target_ms"`
	DecodeTargetMS  float64                   `json:"decode_target_ms"`
	BucketGroups    []TimeBurndownBucketGroup `json:"bucket_groups"`
	GoRuntime       *GoRuntimeLine            `json:"go_runtime,omitempty"`
}

type bucketGroupAcc struct {
	out          TimeBurndownBucketGroup
	cells        map[[2]string]float64
	promptTokens int64
	outputTokens int64
	nativeN      int
	promptN      int
	outputN      int
	samples      int
}

// ReduceTimeBurndownBuckets builds the /2 report. Inputs are validated by
// ReduceTimeBurndown first, so this only partitions already-admitted time.
func ReduceTimeBurndownBuckets(rows []Observation, native []RequestPhaseReceipt, targets BurndownTargets, baseline *TimeBurndownV2) (TimeBurndownV2, error) {
	if !finiteNonnegative(targets.PrefillMS) || !finiteNonnegative(targets.DecodeMS) {
		return TimeBurndownV2{}, fmt.Errorf("%w: targets must be finite and nonnegative", ErrInvalidTimeBurndown)
	}
	base, err := ReduceTimeBurndown(rows, native)
	if err != nil {
		return TimeBurndownV2{}, err
	}
	report := TimeBurndownV2{TimeBurndown: base, PrefillTargetMS: targets.PrefillMS, DecodeTargetMS: targets.DecodeMS, BucketGroups: []TimeBurndownBucketGroup{}}
	report.Schema = TimeBurndownBucketSchema
	joins := make(map[string]RequestPhaseReceipt, len(native))
	for _, row := range native {
		joins[row.RequestID] = row
	}
	type groupKey struct{ backend, model, cache string }
	groups := make(map[groupKey]*bucketGroupAcc)
	for _, row := range rows {
		if row.Warmup || row.Error != "" || row.Status < 200 || row.Status >= 400 {
			continue
		}
		state, _ := ParseCacheState(row.CacheState)
		key := groupKey{row.Backend, row.Model, state}
		acc := groups[key]
		if acc == nil {
			acc = &bucketGroupAcc{out: TimeBurndownBucketGroup{Backend: key.backend, Model: key.model, CacheState: key.cache}, cells: make(map[[2]string]float64)}
			groups[key] = acc
		}
		acc.samples++
		acc.out.APIWallMS += row.DurationMS
		env, ok := joins[row.RequestID]
		if !ok {
			acc.cells[[2]string{StageUnstaged, BucketUnexplained}] += row.DurationMS
			continue
		}
		acc.nativeN++
		if row.PromptTokens > 0 {
			acc.promptN++
			acc.promptTokens += row.PromptTokens
		}
		if row.CompletionTokens > 0 {
			acc.outputN++
			acc.outputTokens += row.CompletionTokens
		}
		if env.UnattributedMS != nil {
			if acc.out.ProducerUnattributedMS == nil {
				acc.out.ProducerUnattributedMS = new(float64)
			}
			*acc.out.ProducerUnattributedMS += *env.UnattributedMS
		}
		nativeWall := millis(env.Receipt.Wall)
		var phased float64
		for _, phase := range env.Receipt.Phases {
			rule, known := burndownPhaseBucket[phase.Phase]
			if !known {
				return TimeBurndownV2{}, fmt.Errorf("%w: phase %q has no burndown bucket", ErrInvalidTimeBurndown, phase.Phase)
			}
			stage := burndownPhaseStage(phase)
			wall := millis(phase.Exclusive.Wall)
			phased += wall
			if rule.split {
				acc.cells[[2]string{stage, BucketCPUWork}] += millis(phase.Exclusive.Active)
				acc.cells[[2]string{stage, BucketUnexplained}] += millis(phase.Exclusive.Wait)
				continue
			}
			acc.cells[[2]string{stage, rule.bucket}] += wall
		}
		acc.cells[[2]string{StageUnstaged, BucketUnexplained}] += math.Max(0, nativeWall-phased) + (row.DurationMS - nativeWall)
	}
	baseGroups := make(map[groupKey]TimeBurndownBucketGroup)
	if baseline != nil {
		for _, g := range baseline.BucketGroups {
			baseGroups[groupKey{g.Backend, g.Model, g.CacheState}] = g
		}
	}
	for key, acc := range groups {
		g := &acc.out
		g.Baseline = "absent"
		prior, hasPrior := baseGroups[key]
		if hasPrior {
			g.Baseline = "matched"
		}
		units := map[string]int64{StagePrefill: acc.promptTokens, StageDecode: acc.outputTokens, StageUnstaged: int64(acc.samples)}
		unitNames := map[string]string{StagePrefill: "prompt_token", StageDecode: "completion_token", StageUnstaged: "request"}
		unavailable := map[string]string{}
		if acc.nativeN == 0 {
			unavailable[StagePrefill], unavailable[StageDecode] = "no-native-receipts", "no-native-receipts"
		} else {
			if acc.promptN != acc.nativeN {
				unavailable[StagePrefill] = "prompt-tokens-missing"
			}
			if acc.outputN != acc.nativeN {
				unavailable[StageDecode] = "completion-tokens-missing"
			}
		}
		perUnit := func(stage string, ms float64) *float64 {
			if unavailable[stage] != "" || units[stage] == 0 {
				return nil
			}
			v := ms / float64(units[stage])
			return &v
		}
		stageTotals := map[string]float64{}
		for _, stage := range BurndownStages() {
			for _, bucket := range BurndownBuckets() {
				ms := acc.cells[[2]string{stage, bucket}]
				if !finiteNonnegative(ms) {
					return TimeBurndownV2{}, fmt.Errorf("%w: bucket duration overflow", ErrInvalidTimeBurndown)
				}
				stageTotals[stage] += ms
				g.BucketedMS += ms
				row := BurndownBucketRow{Stage: stage, Bucket: bucket, TotalMS: ms, MSPerUnit: perUnit(stage, ms)}
				if hasPrior {
					for _, p := range prior.Buckets {
						if p.Stage == stage && p.Bucket == bucket {
							row.BeforeMSPerUnit = p.MSPerUnit
						}
					}
					row.DeltaPct = deltaPct(row.BeforeMSPerUnit, row.MSPerUnit)
				}
				g.Buckets = append(g.Buckets, row)
			}
		}
		for _, stage := range BurndownStages() {
			total := BurndownStageTotal{Stage: stage, Unit: unitNames[stage], Units: units[stage], UnitsUnavailable: unavailable[stage], TotalMS: stageTotals[stage], MSPerUnit: perUnit(stage, stageTotals[stage])}
			var target *float64
			switch stage {
			case StagePrefill:
				target = &report.PrefillTargetMS
			case StageDecode:
				target = &report.DecodeTargetMS
			}
			if target != nil {
				t := *target
				total.TargetMSPerUnit = &t
				if total.MSPerUnit != nil {
					r := *total.MSPerUnit - t
					total.RemainingMSPerUnit = &r
				}
			}
			if hasPrior {
				for _, p := range prior.Stages {
					if p.Stage == stage {
						total.BeforeMSPerUnit = p.MSPerUnit
					}
				}
				total.DeltaPct = deltaPct(total.BeforeMSPerUnit, total.MSPerUnit)
			}
			g.Stages = append(g.Stages, total)
		}
		report.BucketGroups = append(report.BucketGroups, *g)
	}
	sort.Slice(report.BucketGroups, func(i, j int) bool {
		a, b := report.BucketGroups[i], report.BucketGroups[j]
		if a.Backend != b.Backend {
			return a.Backend < b.Backend
		}
		if a.Model != b.Model {
			return a.Model < b.Model
		}
		return a.CacheState < b.CacheState
	})
	return report, nil
}

func deltaPct(before, after *float64) *float64 {
	if before == nil || after == nil || *before == 0 {
		return nil
	}
	v := (*after - *before) / *before * 100
	return &v
}

// ReadTimeBurndownBaseline admits only a prior /2 report: a /1 report has no
// buckets to diff against.
func ReadTimeBurndownBaseline(r io.Reader) (*TimeBurndownV2, error) {
	var prior TimeBurndownV2
	if err := json.NewDecoder(r).Decode(&prior); err != nil {
		return nil, fmt.Errorf("%w: baseline: %v", ErrInvalidTimeBurndown, err)
	}
	if prior.Schema != TimeBurndownBucketSchema {
		return nil, fmt.Errorf("%w: baseline schema %q, want %s", ErrInvalidTimeBurndown, prior.Schema, TimeBurndownBucketSchema)
	}
	return &prior, nil
}

// ReadGoRuntimeReceipt accepts a /debug/vars scrape (go_runtime key) or a bare
// runtime receipt.
func ReadGoRuntimeReceipt(r io.Reader) (runtimeobs.Receipt, error) {
	var vars struct {
		GoRuntime *runtimeobs.Receipt `json:"go_runtime"`
		Schema    string              `json:"schema"`
	}
	raw, err := io.ReadAll(r)
	if err != nil {
		return runtimeobs.Receipt{}, err
	}
	if err := json.Unmarshal(raw, &vars); err != nil {
		return runtimeobs.Receipt{}, fmt.Errorf("%w: go runtime scrape: %v", ErrInvalidTimeBurndown, err)
	}
	if vars.GoRuntime != nil {
		return *vars.GoRuntime, nil
	}
	var receipt runtimeobs.Receipt
	if err := json.Unmarshal(raw, &receipt); err != nil {
		return runtimeobs.Receipt{}, fmt.Errorf("%w: go runtime scrape: %v", ErrInvalidTimeBurndown, err)
	}
	return receipt, nil
}

// GoRuntimeLineFrom diffs two scrapes of one process lifetime.
func GoRuntimeLineFrom(before, after runtimeobs.Receipt) (*GoRuntimeLine, error) {
	d, err := runtimeobs.Diff(before, after)
	if err != nil {
		return nil, fmt.Errorf("%w: go runtime diff: %v", ErrInvalidTimeBurndown, err)
	}
	return &GoRuntimeLine{IntervalSeconds: d.IntervalSeconds, GCCycles: d.GCCycles, GCCPUSeconds: d.CPUGCSeconds, GCPauseSeconds: d.CPUGCPauseSeconds, GCCPUFraction: d.GCCPUFraction}, nil
}

func WriteTimeBurndownBucketsMarkdown(w io.Writer, report TimeBurndownV2) error {
	if err := WriteTimeBurndownMarkdown(w, report.TimeBurndown); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "\n# Campaign buckets\n\nTargets: prefill %.3f ms/prompt token; decode %.3f ms/completion token. Remaining = after - target (negative is headroom).\n", report.PrefillTargetMS, report.DecodeTargetMS); err != nil {
		return err
	}
	for _, g := range report.BucketGroups {
		if _, err := fmt.Fprintf(w, "\n## %s / %s / %s\n\nAPI wall %.3f ms = bucketed %.3f ms. Baseline: %s.\n\n| Stage | Unit | Before ms/unit | After ms/unit | Delta | Target | Remaining |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n", markdownCell(g.Backend), markdownCell(g.Model), g.CacheState, g.APIWallMS, g.BucketedMS, g.Baseline); err != nil {
			return err
		}
		for _, s := range g.Stages {
			unit := s.Unit
			if s.UnitsUnavailable != "" {
				unit += " (" + s.UnitsUnavailable + ")"
			}
			if _, err := fmt.Fprintf(w, "| %s | %s | %s | %s | %s | %s | %s |\n", s.Stage, unit, optMS(s.BeforeMSPerUnit), optMS(s.MSPerUnit), optPct(s.DeltaPct), optMS(s.TargetMSPerUnit), optMS(s.RemainingMSPerUnit)); err != nil {
				return err
			}
		}
		if _, err := fmt.Fprintln(w, "\n| Stage | Bucket | Total ms | Before ms/unit | After ms/unit | Delta |\n| --- | --- | ---: | ---: | ---: | ---: |"); err != nil {
			return err
		}
		for _, b := range g.Buckets {
			if _, err := fmt.Fprintf(w, "| %s | %s | %.3f | %s | %s | %s |\n", b.Stage, b.Bucket, b.TotalMS, optMS(b.BeforeMSPerUnit), optMS(b.MSPerUnit), optPct(b.DeltaPct)); err != nil {
				return err
			}
		}
	}
	if report.GoRuntime != nil {
		rt := report.GoRuntime
		if _, err := fmt.Fprintf(w, "\nGo runtime (process-level, over %.3f s; not per request): GC CPU %s s, GC pause CPU %s s, GC CPU fraction %s.\n", rt.IntervalSeconds, optF64(rt.GCCPUSeconds), optF64(rt.GCPauseSeconds), optF64(rt.GCCPUFraction)); err != nil {
			return err
		}
	}
	return nil
}

func optMS(v *float64) string {
	if v == nil {
		return "unavailable"
	}
	return fmt.Sprintf("%.4f", *v)
}

func optPct(v *float64) string {
	if v == nil {
		return "n/a"
	}
	return fmt.Sprintf("%+.2f%%", *v)
}

func optF64(v runtimeobs.F64) string {
	if !v.OK {
		return "unavailable"
	}
	return fmt.Sprintf("%.6g", v.V)
}
