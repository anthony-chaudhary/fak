package armbench

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// ReportSchema tags the rolled-up report artifact.
const ReportSchema = "fak.armbench.report/2"

// ArmSummary is one arm's rollup. Input and output tokens stay in separate
// columns and the total is explicitly labelled, because an unlabelled "tokens"
// number is the shape a context-compression claim hides inside: an arm can cut
// output tokens sharply while raising input tokens, and one blended figure calls
// that a win.
type ArmSummary struct {
	ArmID        string   `json:"arm_id"`
	Kind         ArmKind  `json:"kind"`
	Capabilities []string `json:"capabilities,omitempty"`

	Trials   int `json:"trials"`
	Resumed  int `json:"resumed"`
	Failures int `json:"failures"`
	Retries  int `json:"retries"`

	// Graded is the number of trials that reached the judge. Rates below are
	// over Graded, never over Trials — dividing passes by attempted trials
	// silently converts a provider outage into a quality regression.
	Graded    int     `json:"graded"`
	Passes    int     `json:"passes"`
	PassRate  float64 `json:"pass_rate"`
	MeanScore float64 `json:"mean_score"`

	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	TotalTokens  int     `json:"total_tokens"`
	CostUSD      float64 `json:"cost_usd"`

	MeanWallMS float64 `json:"mean_wall_ms"`
	// TTFT/inter-token means are reported only over the trials that actually
	// measured them, with the count and an availability bit alongside, so an
	// unavailable timing never averages in as a zero.
	MeanTTFTMS        float64 `json:"mean_ttft_ms"`
	TTFTSamples       int     `json:"ttft_samples"`
	TTFTAvailable     bool    `json:"ttft_available"`
	MeanInterTokenMS  float64 `json:"mean_inter_token_ms"`
	InterTokenSamples int     `json:"inter_token_samples"`
	InterTokenAvail   bool    `json:"inter_token_available"`

	CacheReadTokens  int `json:"cache_read_tokens"`
	CacheWriteTokens int `json:"cache_write_tokens"`
	CacheHits        int `json:"cache_hits"`
	CacheMisses      int `json:"cache_misses"`
	// Accounting is the canonical publishable total. The numeric fields above
	// are compatibility projections for callers that already consume this Go
	// type; availability and claim gates always come from this receipt.
	Accounting AccountingReceipt `json:"accounting"`
	accounting []AccountingReceipt

	Setup SetupCost `json:"setup"`
	// SetupAmortizedWallMS / SetupAmortizedCostUSD spread the one-time setup
	// across the arm's graded trials. A per-turn saving that never repays its
	// setup is not a saving, and this is the column that shows it.
	SetupAmortizedWallMS  float64 `json:"setup_amortized_wall_ms"`
	SetupAmortizedCostUSD float64 `json:"setup_amortized_cost_usd"`
}

// Report is the rolled-up, publishable view of one run.
type Report struct {
	Schema           string       `json:"schema"`
	ManifestIdentity string       `json:"manifest_identity"`
	ManifestID       string       `json:"manifest_id"`
	Model            Model        `json:"model"`
	Corpus           Corpus       `json:"corpus"`
	Judge            Judge        `json:"judge"`
	Trials           Trials       `json:"trials"`
	Environment      Environment  `json:"environment"`
	Sources          []Source     `json:"sources"`
	Arms             []ArmSummary `json:"arms"`

	TotalTrials   int `json:"total_trials"`
	ExecutedCount int `json:"executed"`
	ResumedCount  int `json:"resumed"`
	FailureCount  int `json:"failures"`
	MaxParallel   int `json:"max_parallel,omitempty"`
}

// Summarize folds a raw run into per-arm rollups. It re-checks the evidence
// fence on every row: a run artifact can arrive from disk, and a report is the
// thing that gets published, so the last gate before publication re-asks the
// question rather than trusting that the producer asked it.
func Summarize(r *Run) (*Report, error) {
	if r == nil || r.Manifest == nil {
		return nil, refuse(ReasonManifestInvalid, "run or its manifest is nil")
	}
	if err := r.Manifest.Validate(); err != nil {
		return nil, err
	}
	if want := r.Manifest.Identity(); r.ManifestIdentity != want {
		return nil, refuse(ReasonIncomparableManifest, "run declares identity %s but its embedded manifest hashes to %s — the manifest was edited after the run", r.ManifestIdentity, want)
	}
	if err := validateRunLedger(r); err != nil {
		return nil, err
	}

	acc := map[string]*ArmSummary{}
	for _, arm := range r.Manifest.Arms {
		acc[arm.ID] = &ArmSummary{
			ArmID:        arm.ID,
			Kind:         arm.Kind,
			Capabilities: arm.Capabilities,
			Setup:        r.Setup[arm.ID],
		}
	}

	failures := 0
	for _, t := range r.Trials {
		s, ok := acc[t.ArmID]
		if !ok {
			return nil, refuse(ReasonManifestInvalid, "trial row names arm %q which the manifest does not declare", t.ArmID)
		}
		if err := checkEvidence(t.Response, t.Judgment); err != nil {
			return nil, fmt.Errorf("trial %s: %w", t.Key(), err)
		}
		accumulate(s, t)
		if t.Response.Failure != "" {
			failures++
		}
	}

	arms := make([]ArmSummary, 0, len(acc))
	for _, s := range acc {
		finalize(s)
		arms = append(arms, *s)
	}
	sort.Slice(arms, func(i, j int) bool { return arms[i].ArmID < arms[j].ArmID })

	m := r.Manifest
	return &Report{
		Schema:           ReportSchema,
		ManifestIdentity: r.ManifestIdentity,
		ManifestID:       m.ID,
		Model:            m.Model,
		Corpus:           m.Corpus,
		Judge:            m.Judge,
		Trials:           m.Trials,
		Environment:      m.Environment,
		Sources:          m.Sources,
		Arms:             arms,
		TotalTrials:      len(r.Trials),
		ExecutedCount:    r.Executed,
		ResumedCount:     r.ResumedCount,
		FailureCount:     failures,
		MaxParallel:      r.MaxParallel,
	}, nil
}

// validateRunLedger prevents a hand-edited or partially-written artifact from
// becoming a plausible report. Raw evidence alone is insufficient if a row was
// duplicated, dropped, attributed to the wrong arm, or had its resume counters
// rewritten after execution.
func validateRunLedger(r *Run) error {
	maxParallel := effectiveRunMaxParallel(r)
	if maxParallel < 1 {
		return refuse(ReasonManifestInvalid, "run max_parallel is %d, want >= 1", r.MaxParallel)
	}
	if err := validateParallelAssignments(r.Manifest, maxParallel); err != nil {
		return err
	}
	expectedPerArm := r.Manifest.Corpus.TaskCount * r.Manifest.Trials.Count
	expectedTotal := expectedPerArm * len(r.Manifest.Arms)
	if len(r.Trials) != expectedTotal {
		return refuse(ReasonManifestInvalid, "run contains %d trial rows, want %d (%d tasks x %d trials x %d arms)", len(r.Trials), expectedTotal, r.Manifest.Corpus.TaskCount, r.Manifest.Trials.Count, len(r.Manifest.Arms))
	}
	seen := make(map[string]bool, len(r.Trials))
	tasks := map[string]bool{}
	perArm := map[string]int{}
	resumed := 0
	for _, t := range r.Trials {
		if t.ManifestIdentity != r.ManifestIdentity {
			return refuse(ReasonIncomparableManifest, "trial %s declares identity %s, run declares %s", t.Key(), t.ManifestIdentity, r.ManifestIdentity)
		}
		arm, ok := r.Manifest.ArmByID(t.ArmID)
		if !ok || arm.Kind != t.ArmKind {
			return refuse(ReasonManifestInvalid, "trial %s names undeclared arm/kind %q/%q", t.Key(), t.ArmID, t.ArmKind)
		}
		if t.Trial < 0 || t.Trial >= r.Manifest.Trials.Count || t.Position < 0 || t.Position >= len(r.Manifest.Arms) {
			return refuse(ReasonManifestInvalid, "trial %s has trial/position outside the manifest plan", t.Key())
		}
		if t.Launch == nil {
			if r.MaxParallel != 0 && !t.Resumed {
				return refuse(ReasonManifestInvalid, "trial %s has no launch receipt", t.Key())
			}
		} else {
			if err := validateLaunchReceipt(*t.Launch); err != nil {
				return fmt.Errorf("trial %s: %w", t.Key(), err)
			}
			wantWave := t.Position/maxParallel + 1
			if t.Launch.Wave != wantWave {
				return refuse(ReasonManifestInvalid, "trial %s records wave %d, want %d for position %d at max_parallel %d", t.Key(), t.Launch.Wave, wantWave, t.Position, maxParallel)
			}
			if !sameInt(t.Launch.GPUIndex, arm.GPUIndex) {
				return refuse(ReasonManifestInvalid, "trial %s launch gpu_index does not match its arm assignment", t.Key())
			}
			wantCUDA := ""
			if arm.GPUIndex != nil {
				wantCUDA = fmt.Sprint(*arm.GPUIndex)
			}
			if t.Launch.CUDAVisibleDevices != wantCUDA {
				return refuse(ReasonManifestInvalid, "trial %s records CUDA_VISIBLE_DEVICES=%q, want %q", t.Key(), t.Launch.CUDAVisibleDevices, wantCUDA)
			}
		}
		key := t.Key()
		if seen[key] {
			return refuse(ReasonDuplicateTrial, "run contains trial key %s more than once", key)
		}
		seen[key] = true
		tasks[t.TaskID] = true
		perArm[t.ArmID]++
		if t.Resumed {
			resumed++
		}
		if err := checkEvidence(t.Response, t.Judgment); err != nil {
			return fmt.Errorf("trial %s: %w", key, err)
		}
	}
	if len(tasks) != r.Manifest.Corpus.TaskCount {
		return refuse(ReasonManifestInvalid, "run contains %d distinct task ids, manifest pins %d", len(tasks), r.Manifest.Corpus.TaskCount)
	}
	for _, arm := range r.Manifest.Arms {
		if perArm[arm.ID] != expectedPerArm {
			return refuse(ReasonManifestInvalid, "run contains %d rows for arm %s, want %d", perArm[arm.ID], arm.ID, expectedPerArm)
		}
		if _, ok := r.Setup[arm.ID]; !ok {
			return refuse(ReasonManifestInvalid, "run records no setup cost (including an explicit zero) for arm %s", arm.ID)
		}
	}
	if resumed != r.ResumedCount || r.Executed+r.ResumedCount != len(r.Trials) {
		return refuse(ReasonManifestInvalid, "run counters disagree with rows: executed=%d resumed=%d row_resumed=%d total=%d", r.Executed, r.ResumedCount, resumed, len(r.Trials))
	}
	return nil
}

func sameInt(a, b *int) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

func accumulate(s *ArmSummary, t TrialResult) {
	s.Trials++
	if t.Resumed {
		s.Resumed++
	}
	s.Retries += t.Response.Retries
	receipt := normalizedAccountingReceipt(t.Response.Accounting)
	s.accounting = append(s.accounting, receipt)
	if t.Response.Accounting.Schema == "" {
		// Old run artifacts retain their legacy projections, but their canonical
		// accounting receipt remains missing and therefore cannot support a gain.
		s.InputTokens += t.Response.Usage.InputTokens
		s.OutputTokens += t.Response.Usage.OutputTokens
		s.CostUSD += t.Response.Usage.CostUSD
		s.CacheReadTokens += t.Response.Cache.ReadTokens
		s.CacheWriteTokens += t.Response.Cache.WriteTokens
		s.CacheHits += t.Response.Cache.Hits
		s.CacheMisses += t.Response.Cache.Misses
	} else {
		s.InputTokens += accountingInt(receipt.InputTokens)
		s.OutputTokens += accountingInt(receipt.OutputTokens)
		s.CostUSD += accountingFloat(receipt.CostUSD)
		s.CacheReadTokens += accountingInt(receipt.CacheReadTokens)
		s.CacheWriteTokens += accountingInt(receipt.CacheWriteTokens)
		s.CacheHits += accountingInt(receipt.CacheHits)
		s.CacheMisses += accountingInt(receipt.CacheMisses)
	}
	s.MeanWallMS += t.Response.Latency.WallMS
	if t.Response.Latency.TTFTAvailable {
		s.MeanTTFTMS += t.Response.Latency.TTFTMS
		s.TTFTSamples++
	}
	if t.Response.Latency.InterTokenAvailable {
		s.MeanInterTokenMS += t.Response.Latency.InterTokenMS
		s.InterTokenSamples++
	}
	if t.Response.Failure != "" {
		s.Failures++
		return
	}
	s.Graded++
	s.MeanScore += t.Judgment.Score
	if t.Judgment.Pass {
		s.Passes++
	}
}

func finalize(s *ArmSummary) {
	s.Accounting = AggregateAccounting(s.accounting)
	s.TotalTokens = s.InputTokens + s.OutputTokens
	if s.Trials > 0 {
		s.MeanWallMS /= float64(s.Trials)
	}
	if s.TTFTSamples > 0 {
		s.MeanTTFTMS /= float64(s.TTFTSamples)
		s.TTFTAvailable = true
	} else {
		s.MeanTTFTMS = 0
	}
	if s.InterTokenSamples > 0 {
		s.MeanInterTokenMS /= float64(s.InterTokenSamples)
		s.InterTokenAvail = true
	} else {
		s.MeanInterTokenMS = 0
	}
	if s.Graded > 0 {
		s.PassRate = float64(s.Passes) / float64(s.Graded)
		s.MeanScore /= float64(s.Graded)
		s.SetupAmortizedWallMS = s.Setup.WallMS / float64(s.Graded)
		s.SetupAmortizedCostUSD = s.Setup.CostUSD / float64(s.Graded)
	}
}

// MarshalReport renders the report as strict, stable JSON.
func MarshalReport(rep *Report) ([]byte, error) {
	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

// Human renders the operator-facing summary. It leads with the provenance block
// (a table of numbers whose pins are off-screen invites exactly the comparison
// this package exists to prevent), then one row per arm with input and output
// tokens in separate columns.
func Human(rep *Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "armbench %s\n", rep.ManifestID)
	fmt.Fprintf(&b, "  identity     %s\n", rep.ManifestIdentity)
	fmt.Fprintf(&b, "  model        %s %s (%s) temp=%v top_p=%v max_tokens=%d\n",
		rep.Model.Provider, rep.Model.Snapshot, rep.Model.Region,
		rep.Model.Sampling.Temperature, rep.Model.Sampling.TopP, rep.Model.MaxTokens)
	fmt.Fprintf(&b, "  corpus       %s (%d tasks) %s\n", rep.Corpus.ID, rep.Corpus.TaskCount, rep.Corpus.Hash)
	fmt.Fprintf(&b, "  judge        %s %s\n", rep.Judge.ID, rep.Judge.Hash)
	fmt.Fprintf(&b, "  trials       %d per task, order=%s seed=%d concurrency=%d",
		rep.Trials.Count, rep.Trials.Order, rep.Trials.Seed, rep.Trials.Concurrency)
	if rep.MaxParallel > 0 {
		fmt.Fprintf(&b, " max_parallel=%d", rep.MaxParallel)
	}
	b.WriteByte('\n')
	fmt.Fprintf(&b, "  pricing date %s   env %s/%s %s\n",
		rep.Environment.PricingDate, rep.Environment.OS, rep.Environment.Arch, rep.Environment.FakVersion)
	for _, s := range rep.Sources {
		fmt.Fprintf(&b, "  source       %s %s@%s %s %s\n", s.Name, s.Repo, shortSHA(s.SHA), s.Path, s.ContentHash)
	}
	fmt.Fprintf(&b, "  rows         %d total (%d executed, %d resumed, %d failed)\n\n",
		rep.TotalTrials, rep.ExecutedCount, rep.ResumedCount, rep.FailureCount)

	fmt.Fprintf(&b, "  %-22s %-18s %5s %5s %6s %10s %10s %10s %9s %9s %9s\n",
		"ARM", "KIND", "N", "FAIL", "PASS%", "IN_TOK", "OUT_TOK", "TOTAL_TOK", "COST_USD", "WALL_MS", "TTFT_MS")
	for _, s := range rep.Arms {
		label := s.ArmID
		if len(s.Capabilities) == 1 {
			label += "[" + s.Capabilities[0] + "]"
		}
		ttft := "n/a"
		if s.TTFTAvailable {
			ttft = fmt.Sprintf("%.1f", s.MeanTTFTMS)
		}
		inTokens := accountingDisplay(s.Accounting.InputTokens, false)
		outTokens := accountingDisplay(s.Accounting.OutputTokens, false)
		totalTokens := accountingTotalDisplay(s.Accounting.InputTokens, s.Accounting.OutputTokens)
		cost := accountingDisplay(s.Accounting.CostUSD, true)
		fmt.Fprintf(&b, "  %-22s %-18s %5d %5d %5.1f%% %10s %10s %10s %9s %9.1f %9s\n",
			truncate(label, 22), s.Kind, s.Trials, s.Failures, s.PassRate*100,
			inTokens, outTokens, totalTokens, cost, s.MeanWallMS, ttft)
	}
	b.WriteString("\n  accounting provenance (only available fields with matching authority and coverage are comparable)\n")
	for _, s := range rep.Arms {
		for _, metric := range accountingMetrics {
			field, _ := s.Accounting.Field(metric)
			artifact := field.Artifact.Ref
			if artifact == "" {
				artifact = "n/a"
			}
			fmt.Fprintf(&b, "    %-22s %-19s %-9s %-20s %d/%d %-28s",
				truncate(s.ArmID, 22), metric, field.Availability, field.Authority,
				field.Coverage.Observed, field.Coverage.Expected, truncate(artifact, 28))
			if field.RefusalReason != "" {
				fmt.Fprintf(&b, "  %s", field.RefusalReason)
			}
			b.WriteByte('\n')
		}
	}
	b.WriteString("\n  setup cost (charged once per arm, amortized over its graded trials)\n")
	for _, s := range rep.Arms {
		fmt.Fprintf(&b, "    %-22s wall %8.1f ms  tokens %6d  cost %8.4f  ->  per-trial wall %7.2f ms  cost %8.5f\n",
			truncate(s.ArmID, 22), s.Setup.WallMS, s.Setup.Tokens, s.Setup.CostUSD,
			s.SetupAmortizedWallMS, s.SetupAmortizedCostUSD)
	}
	b.WriteString("\n  input and output tokens are never blended: read both columns before quoting a saving.\n")
	return b.String()
}

func accountingInt(field AccountingField) int {
	if field.Value == nil {
		return 0
	}
	return int(*field.Value)
}

func accountingFloat(field AccountingField) float64 {
	if field.Value == nil {
		return 0
	}
	return *field.Value
}

func accountingDisplay(field AccountingField, money bool) string {
	if field.Value == nil {
		return "n/a"
	}
	prefix := ""
	if field.Availability != AvailabilityAvailable {
		prefix = "~"
	}
	if money {
		return fmt.Sprintf("%s%.4f", prefix, *field.Value)
	}
	return fmt.Sprintf("%s%.0f", prefix, *field.Value)
}

func accountingTotalDisplay(input, output AccountingField) string {
	if input.Value == nil || output.Value == nil {
		return "n/a"
	}
	prefix := ""
	if input.Availability != AvailabilityAvailable || output.Availability != AvailabilityAvailable {
		prefix = "~"
	}
	return fmt.Sprintf("%s%.0f", prefix, *input.Value+*output.Value)
}

func shortSHA(s string) string {
	if len(s) > 10 {
		return s[:10]
	}
	return s
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "~"
}
