package livecodebench

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// fakarm.go is the "fak" A/B arm (#2105, epic #2085): the SAME generation
// fan-out as the raw arm (#2104) — same problems, model, n, temperature,
// release — but routed through the fak gateway's ADJUDICATED path, with the
// per-response fak extension (tool-call adjudications, inbound result
// admissions, safe-resolve repairs) folded into the report as evidence. Like
// the raw arm, everything here is pure: the CLI (cmd/livecodebench fak)
// injects the sampler that touches the wire and extracts the evidence from
// the gateway's `fak` response extension. Neither arm grades — CompareArms
// asserts run-identity parity (SameProblemIDs / SamePromptHash) and reports
// token + adjudication deltas, and its pass-rate delta is pinned to the
// ungraded sentinel until the official evaluator grades both arms.

// FakArmName / RawArmName are the two arm identities a comparison binds.
const (
	RawArmName = "raw"
	FakArmName = "fak"
)

// FakSampleEvidence is the adjudication evidence for ONE sample, extracted by
// the sampler from the gateway's `fak` response extension. A zero value means
// the response carried no fak extension (Adjudicated=false): the request was
// served, but nothing on this sample witnessed the adjudicated path.
type FakSampleEvidence struct {
	Adjudicated      bool // the response carried the fak adjudication extension
	Adjudications    int  // tool-call adjudications recorded on the response
	Denied           int  // adjudications with admitted=false (policy denials)
	SafeResolves     int  // adjudications carrying repaired arguments (safe-resolve)
	ResultAdmissions int  // inbound result admissions recorded (evidence trail)
}

// FakArmSampler produces ONE completion for problem p at sample index i via the
// adjudicated gateway path, returning the sampler-normalized usage plus the
// adjudication evidence the response carried. Tests inject a deterministic stub.
type FakArmSampler func(ctx context.Context, p Problem, i int) (content string, u RawSampleUsage, ev FakSampleEvidence, err error)

// FakArmAdjudication is the run's adjudication evidence folded across every
// sample. It is evidence of the adjudicated path having been exercised — never
// a claim that adjudication changed any pass rate.
type FakArmAdjudication struct {
	AdjudicatedSamples int `json:"adjudicated_samples"`
	Adjudications      int `json:"adjudications"`
	Denied             int `json:"denied"`
	SafeResolves       int `json:"safe_resolves"`
	ResultAdmissions   int `json:"result_admissions"`
}

// FakArmReport is the machine-readable result of the fak arm. Its run-identity
// fields mirror RawArmReport exactly so CompareArms can assert the two arms ran
// the same problems / model / n / temperature / release.
type FakArmReport struct {
	Arm          string             `json:"arm"` // always "fak"
	Model        string             `json:"model"`
	Endpoint     string             `json:"endpoint"`
	N            int                `json:"n"`
	Temperature  float64            `json:"temperature"`
	Seed         int64              `json:"seed,omitempty"` // 0 = provider default, omitted
	Concurrency  int                `json:"concurrency"`
	MaxRetries   int                `json:"max_retries"` // per-sample retry budget the run honored (#2106)
	Release      string             `json:"release,omitempty"`
	Problems     []RawArmProblem    `json:"problems"`
	Usage        RawArmUsage        `json:"usage"`
	Adjudication FakArmAdjudication `json:"adjudication"`
}

// RunFakArm fans the sampler out over every (problem, sample) pair with the
// SAME fan-out semantics as RunRawArm (bounded concurrency, first error aborts,
// deterministic problem/sample ordering), folding the per-sample adjudication
// evidence into the report. The fan-out is delegated to RunRawArm so the two
// arms cannot drift.
func RunFakArm(ctx context.Context, cfg RawArmConfig, release string, problems []Problem, sample FakArmSampler) (FakArmReport, error) {
	if sample == nil {
		return FakArmReport{}, fmt.Errorf("livecodebench fak arm: sampler is required")
	}
	var mu sync.Mutex
	var adj FakArmAdjudication
	wrapped := func(ctx context.Context, p Problem, i int) (string, RawSampleUsage, error) {
		content, u, ev, err := sample(ctx, p, i)
		if err != nil {
			return "", RawSampleUsage{}, err
		}
		mu.Lock()
		if ev.Adjudicated {
			adj.AdjudicatedSamples++
		}
		adj.Adjudications += ev.Adjudications
		adj.Denied += ev.Denied
		adj.SafeResolves += ev.SafeResolves
		adj.ResultAdmissions += ev.ResultAdmissions
		mu.Unlock()
		return content, u, nil
	}
	raw, err := RunRawArm(ctx, cfg, problems, wrapped)
	if err != nil {
		return FakArmReport{}, fmt.Errorf("livecodebench fak arm: %w", err)
	}
	return FakArmReport{
		Arm:          FakArmName,
		Model:        raw.Model,
		Endpoint:     raw.Endpoint,
		N:            raw.N,
		Temperature:  raw.Temperature,
		Seed:         raw.Seed,
		Concurrency:  raw.Concurrency,
		MaxRetries:   raw.MaxRetries,
		Release:      strings.TrimSpace(release),
		Problems:     raw.Problems,
		Usage:        raw.Usage,
		Adjudication: adj,
	}, nil
}

// ABComparisonSchema identifies the raw-vs-fak arm comparison artifact.
const ABComparisonSchema = "fak.livecodebench-ab-comparison.v1"

// PassRateDeltaUngraded is the only pass-rate delta this package will ever
// state on its own authority: none. Only the official lcb_runner evaluator,
// grading the exact saved generations of BOTH arms, may back a claim that
// adjudication changed the pass rate (#2105 acceptance).
const PassRateDeltaUngraded = "ungraded: no pass-rate delta may be claimed until the official lcb_runner evaluator grades both arms' saved generations"

// ArmSummary is the per-arm run identity + usage the comparison emits for each arm.
type ArmSummary struct {
	Arm         string      `json:"arm"`
	Model       string      `json:"model"`
	Endpoint    string      `json:"endpoint"`
	N           int         `json:"n"`
	Temperature float64     `json:"temperature"`
	Release     string      `json:"release,omitempty"`
	Problems    int         `json:"problems"`
	Usage       RawArmUsage `json:"usage"`
}

// ArmUsageDelta is fak-minus-raw token accounting: the observable cost of the
// adjudicated path, stated without any quality claim.
type ArmUsageDelta struct {
	Samples            int `json:"samples"`
	PromptTokens       int `json:"prompt_tokens"`
	CompletionTokens   int `json:"completion_tokens"`
	CachedPromptTokens int `json:"cached_prompt_tokens"`
}

// ArmComparison is the two-arm A/B artifact (#2105): per-arm summaries, the
// cross-arm identity assertions the acceptance demands (SameProblemIDs /
// SamePromptHash, plus model / n / temperature / release), the usage delta,
// and the fak arm's adjudication evidence. ResultClaimAllowed is always false
// and PassRateDelta is always the ungraded sentinel — grading authority stays
// with the official evaluator.
type ArmComparison struct {
	Schema             string             `json:"schema"`
	Raw                ArmSummary         `json:"raw"`
	Fak                ArmSummary         `json:"fak"`
	SameProblemIDs     bool               `json:"same_problem_ids"`
	SamePromptHash     bool               `json:"same_prompt_hash"`
	SameModel          bool               `json:"same_model"`
	SameN              bool               `json:"same_n"`
	SameTemperature    bool               `json:"same_temperature"`
	SameRelease        bool               `json:"same_release"`
	Mismatches         []string           `json:"mismatches,omitempty"`
	UsageDelta         ArmUsageDelta      `json:"usage_delta"` // fak minus raw
	FakAdjudication    FakArmAdjudication `json:"fak_adjudication"`
	PassRateDelta      string             `json:"pass_rate_delta"`      // always PassRateDeltaUngraded here
	ResultClaimAllowed bool               `json:"result_claim_allowed"` // always false here
	ClaimBoundary      string             `json:"claim_boundary"`
}

// CompareArms builds the raw-vs-fak comparison from the two arm artifacts
// alone: every assertion is checked against what the reports actually recorded
// (never against shared inputs), so a stale or foreign report cannot pass. It
// is pure and never claims a pass-rate delta.
func CompareArms(raw RawArmReport, fak FakArmReport) ArmComparison {
	c := ArmComparison{
		Schema:          ABComparisonSchema,
		Raw:             armSummaryOf(raw.Arm, raw.Model, raw.Endpoint, raw.N, raw.Temperature, raw.Release, len(raw.Problems), raw.Usage),
		Fak:             armSummaryOf(fak.Arm, fak.Model, fak.Endpoint, fak.N, fak.Temperature, fak.Release, len(fak.Problems), fak.Usage),
		SameModel:       raw.Model != "" && raw.Model == fak.Model,
		SameN:           raw.N == fak.N,
		SameTemperature: raw.Temperature == fak.Temperature,
		SameRelease:     raw.Release != "" && raw.Release == fak.Release,
		UsageDelta: ArmUsageDelta{
			Samples:            fak.Usage.Samples - raw.Usage.Samples,
			PromptTokens:       fak.Usage.PromptTokens - raw.Usage.PromptTokens,
			CompletionTokens:   fak.Usage.CompletionTokens - raw.Usage.CompletionTokens,
			CachedPromptTokens: fak.Usage.CachedPromptTokens - raw.Usage.CachedPromptTokens,
		},
		FakAdjudication:    fak.Adjudication,
		PassRateDelta:      PassRateDeltaUngraded,
		ResultClaimAllowed: false,
		ClaimBoundary: "A/B generation comparison only: it asserts run-identity parity between the raw and fak arms and reports token + adjudication-evidence deltas. " +
			"Generations are ungraded here; no claim that adjudication changes the pass rate is permitted unless official graded evidence shows it (#2105).",
	}
	if !c.SameModel {
		c.Mismatches = append(c.Mismatches, fmt.Sprintf("model: raw=%q fak=%q", raw.Model, fak.Model))
	}
	if !c.SameN {
		c.Mismatches = append(c.Mismatches, fmt.Sprintf("n: raw=%d fak=%d", raw.N, fak.N))
	}
	if !c.SameTemperature {
		c.Mismatches = append(c.Mismatches, fmt.Sprintf("temperature: raw=%v fak=%v", raw.Temperature, fak.Temperature))
	}
	if !c.SameRelease {
		c.Mismatches = append(c.Mismatches, fmt.Sprintf("release: raw=%q fak=%q (both must record the same explicit release)", raw.Release, fak.Release))
	}

	rawIDs, rawHashes := problemIndex(raw.Problems)
	fakIDs, fakHashes := problemIndex(fak.Problems)
	c.SameProblemIDs = len(rawIDs) > 0 && slicesEqual(rawIDs, fakIDs)
	if !c.SameProblemIDs {
		c.Mismatches = append(c.Mismatches, problemIDMismatch(rawIDs, fakIDs))
	}

	// SamePromptHash: every question id present in both reports must carry a
	// non-empty, identical prompt hash. A missing hash is a mismatch, never a
	// silent pass — an old report without stamped hashes cannot assert parity.
	same := c.SameProblemIDs
	for _, id := range rawIDs {
		fh, ok := fakHashes[id]
		if !ok {
			continue // already reported by the problem-id mismatch
		}
		rh := rawHashes[id]
		switch {
		case rh == "" || fh == "":
			same = false
			c.Mismatches = append(c.Mismatches, fmt.Sprintf("prompt hash missing for %q (raw=%q fak=%q); regenerate with a runner that stamps prompt_sha256", id, rh, fh))
		case rh != fh:
			same = false
			c.Mismatches = append(c.Mismatches, fmt.Sprintf("prompt hash differs for %q: raw=%s fak=%s", id, rh, fh))
		}
	}
	c.SamePromptHash = same
	return c
}

func armSummaryOf(arm, model, endpoint string, n int, temperature float64, release string, problems int, usage RawArmUsage) ArmSummary {
	return ArmSummary{Arm: arm, Model: model, Endpoint: endpoint, N: n, Temperature: temperature, Release: release, Problems: problems, Usage: usage}
}

// problemIndex returns the sorted question ids and the id -> prompt hash map of
// one arm report.
func problemIndex(problems []RawArmProblem) ([]string, map[string]string) {
	ids := make([]string, 0, len(problems))
	hashes := make(map[string]string, len(problems))
	for _, p := range problems {
		ids = append(ids, p.QuestionID)
		hashes[p.QuestionID] = p.PromptSHA256
	}
	sort.Strings(ids)
	return ids, hashes
}

func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func problemIDMismatch(rawIDs, fakIDs []string) string {
	if len(rawIDs) == 0 && len(fakIDs) == 0 {
		return "problem ids: both reports are empty"
	}
	return fmt.Sprintf("problem ids: raw has %d (%s), fak has %d (%s)",
		len(rawIDs), strings.Join(rawIDs, ","), len(fakIDs), strings.Join(fakIDs, ","))
}
