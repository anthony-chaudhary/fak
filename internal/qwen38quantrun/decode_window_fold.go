package qwen38quantrun

import (
	"fmt"
	"math"
	"reflect"
)

const (
	LlamaClientDecodeTraceSchema     = "fak.llama-client-decode-trace/1"
	LlamaClientDecodeTraceProvenance = "client-http-body-arrival"
	MinimumLongDecodeTokens          = 2048
)

var LlamaClientDecodeContract = DecodeTraceContract{
	Schema: LlamaClientDecodeTraceSchema, Engine: "llama.cpp", Provenance: LlamaClientDecodeTraceProvenance,
	LeadingUntimedTokens: 1, AllowEqualElapsed: true,
}

// LlamaClientArrivalEvent is a caller-timestamped, already-decoded b9828
// /completion event. This package does not parse or count SSE fragments; the
// caller must retain the raw response and supply its explicit token fields.
type LlamaClientArrivalEvent struct {
	TokenIDs        []int `json:"tokens"`
	TokensPredicted int   `json:"tokens_predicted"`
	ElapsedNS       int64 `json:"elapsed_ns"`
}

type LlamaClientFinal struct {
	StopType        string `json:"stop_type"`
	TokensPredicted int    `json:"tokens_predicted"`
	TokenIDs        []int  `json:"tokens"`
}

// LlamaClientDecodeResult is the serializable raw comparator evidence for one
// repetition. Events and Final come from explicit b9828 fields, never text or
// SSE-fragment counts.
type LlamaClientDecodeResult struct {
	FixtureID string                    `json:"fixture_id"`
	Repeat    int                       `json:"repeat"`
	Events    []LlamaClientArrivalEvent `json:"events"`
	Final     LlamaClientFinal          `json:"final"`
}

type MatchedDecodeWindowSummary struct {
	Native     DecodeWindowSummary `json:"native"`
	Comparator DecodeWindowSummary `json:"comparator"`
	Verdict    string              `json:"verdict"`
	Failure    string              `json:"failure,omitempty"`
}

// BuildLlamaClientDecodeWindows adapts only the pinned explicit comparator
// contract. Its timestamps remain client-arrival observations and are never
// described as llama.cpp commit times.
func BuildLlamaClientDecodeWindows(events []LlamaClientArrivalEvent, final LlamaClientFinal) (DecodeTrace, DecodeWindowReport, error) {
	if final.StopType != "limit" {
		return DecodeTrace{}, DecodeWindowReport{}, fmt.Errorf("llama comparator stop_type=%q, want limit", final.StopType)
	}
	if final.TokensPredicted < MinimumLongDecodeTokens {
		return DecodeTrace{}, DecodeWindowReport{}, fmt.Errorf("llama comparator tokens_predicted=%d below %d", final.TokensPredicted, MinimumLongDecodeTokens)
	}
	if len(final.TokenIDs) != 0 {
		return DecodeTrace{}, DecodeWindowReport{}, fmt.Errorf("llama comparator final event tokens=%d, want empty", len(final.TokenIDs))
	}
	if len(events) != final.TokensPredicted {
		return DecodeTrace{}, DecodeWindowReport{}, fmt.Errorf("llama comparator event count=%d final tokens_predicted=%d", len(events), final.TokensPredicted)
	}
	if len(events) < 2 {
		return DecodeTrace{}, DecodeWindowReport{}, fmt.Errorf("llama comparator needs a first-token arrival baseline")
	}
	var previousArrival int64
	for i, event := range events {
		want := i + 1
		if event.TokensPredicted != want {
			return DecodeTrace{}, DecodeWindowReport{}, fmt.Errorf("llama comparator tokens_predicted at event %d=%d, want %d", i, event.TokensPredicted, want)
		}
		if len(event.TokenIDs) != 1 {
			return DecodeTrace{}, DecodeWindowReport{}, fmt.Errorf("llama comparator event %d tokens=%d, want singleton", i, len(event.TokenIDs))
		}
		if event.ElapsedNS < 0 || (i > 0 && event.ElapsedNS < previousArrival) {
			return DecodeTrace{}, DecodeWindowReport{}, fmt.Errorf("llama comparator elapsed_ns is non-monotonic at event %d: got %d after %d", i, event.ElapsedNS, previousArrival)
		}
		previousArrival = event.ElapsedNS
	}
	// The first token arrival is TTFT, not decode throughput. Retain its raw
	// cardinality in the report, but use it only as the zero point for the N-1
	// observable inter-token arrival intervals.
	baseline := events[0].ElapsedNS
	trace := DecodeTrace{Schema: LlamaClientDecodeContract.Schema, Engine: LlamaClientDecodeContract.Engine, Provenance: LlamaClientDecodeContract.Provenance, Events: make([]DecodeTraceEvent, len(events)-1)}
	for i := 1; i < len(events); i++ {
		tokenID := events[i].TokenIDs[0]
		trace.Events[i-1] = DecodeTraceEvent{TokenIndex: i, ElapsedNS: events[i].ElapsedNS - baseline, TokenID: &tokenID}
	}
	report, err := BuildDecodeWindows(trace, LlamaClientDecodeContract, final.TokensPredicted)
	if err != nil {
		return DecodeTrace{}, DecodeWindowReport{}, err
	}
	return trace, report, nil
}

// FoldMatchedDecodeCampaign produces the one issue-level PASS/HOLD from the
// later 3x2 campaign. The comparator is required, provenance-checked reference
// evidence; only fak-native's 0.85 ratio is the promotion threshold.
func FoldMatchedDecodeCampaign(native []Result, comparator []LlamaClientDecodeResult) (MatchedDecodeWindowSummary, error) {
	nativeReports, err := buildNativeDecodeReports(native)
	if err != nil {
		return MatchedDecodeWindowSummary{}, fmt.Errorf("native raw evidence: %w", err)
	}
	nativeSummary, err := FoldDecodeRepetitions(nativeReports)
	if err != nil {
		return MatchedDecodeWindowSummary{}, fmt.Errorf("native repetitions: %w", err)
	}
	comparatorReports, err := buildLlamaClientDecodeReports(comparator)
	if err != nil {
		return MatchedDecodeWindowSummary{}, fmt.Errorf("comparator raw evidence: %w", err)
	}
	if native[0].FixtureID != comparator[0].FixtureID {
		return MatchedDecodeWindowSummary{}, fmt.Errorf("matched decode fixture mismatch: native=%q comparator=%q", native[0].FixtureID, comparator[0].FixtureID)
	}
	comparatorSummary, err := FoldDecodeRepetitions(comparatorReports)
	if err != nil {
		return MatchedDecodeWindowSummary{}, fmt.Errorf("comparator repetitions: %w", err)
	}
	if nativeSummary.Contract != NativeDecodeContract || comparatorSummary.Contract != LlamaClientDecodeContract {
		return MatchedDecodeWindowSummary{}, fmt.Errorf("matched decode engine or provenance contract mismatch")
	}
	if nativeSummary.RawCompletionTokensPerRun < MinimumLongDecodeTokens || comparatorSummary.RawCompletionTokensPerRun < MinimumLongDecodeTokens {
		return MatchedDecodeWindowSummary{}, fmt.Errorf("matched decode campaign requires at least %d completion tokens per repetition", MinimumLongDecodeTokens)
	}
	if nativeSummary.RawCompletionTokensPerRun != comparatorSummary.RawCompletionTokensPerRun {
		return MatchedDecodeWindowSummary{}, fmt.Errorf("matched decode raw completion-token mismatch: native=%d comparator=%d", nativeSummary.RawCompletionTokensPerRun, comparatorSummary.RawCompletionTokensPerRun)
	}
	if nativeSummary.TimedTokensPerRun != nativeSummary.RawCompletionTokensPerRun || comparatorSummary.TimedTokensPerRun+1 != comparatorSummary.RawCompletionTokensPerRun {
		return MatchedDecodeWindowSummary{}, fmt.Errorf("matched decode timed-token cardinality mismatch: native=%d/%d comparator=%d/%d", nativeSummary.TimedTokensPerRun, nativeSummary.RawCompletionTokensPerRun, comparatorSummary.TimedTokensPerRun, comparatorSummary.RawCompletionTokensPerRun)
	}
	out := MatchedDecodeWindowSummary{Native: nativeSummary, Comparator: comparatorSummary, Verdict: "PASS"}
	if nativeSummary.Verdict != "PASS" {
		out.Verdict, out.Failure = "HOLD", nativeSummary.Failure
	}
	return out, nil
}

// FoldDecodeResults independently rebuilds every per-run window report from
// its retained raw trace before aggregating repetitions. Missing or tampered
// traces therefore produce HOLD evidence rather than a partial PASS.
func FoldDecodeResults(results []Result) DecodeWindowSummary {
	reports, err := buildNativeDecodeReports(results)
	if err != nil {
		return DecodeWindowSummary{Contract: NativeDecodeContract, Threshold: DecodeLateEarlyThreshold, Verdict: "HOLD", Failure: err.Error()}
	}
	summary, err := FoldDecodeRepetitions(reports)
	if err != nil {
		return DecodeWindowSummary{Contract: NativeDecodeContract, Threshold: DecodeLateEarlyThreshold, Verdict: "HOLD", Failure: err.Error()}
	}
	return summary
}

func buildNativeDecodeReports(results []Result) ([]DecodeWindowReport, error) {
	reports := make([]DecodeWindowReport, 0, len(results))
	var failure string
	fixtureID := ""
	seenRepetitions := make(map[int]bool, MinimumDecodeRepetitions)
	if len(results) != MinimumDecodeRepetitions {
		failure = fmt.Sprintf("decode campaign results=%d, want exactly %d repetitions", len(results), MinimumDecodeRepetitions)
	}
	for _, result := range results {
		if fixtureID == "" {
			fixtureID = result.FixtureID
		}
		if result.FixtureID == "" || result.FixtureID != fixtureID || result.Repeat < 1 || result.Repeat > MinimumDecodeRepetitions || seenRepetitions[result.Repeat] {
			failure = joinFailure(failure, fmt.Sprintf("%s/%d unmatched decode repetition identity", result.FixtureID, result.Repeat))
			continue
		}
		seenRepetitions[result.Repeat] = true
		if result.Quality != "PASS" || result.Failure != "" {
			failure = joinFailure(failure, fmt.Sprintf("%s/%d quality=%q failure=%q", result.FixtureID, result.Repeat, result.Quality, result.Failure))
			continue
		}
		if result.Usage["completion_tokens"] < MinimumLongDecodeTokens {
			failure = joinFailure(failure, fmt.Sprintf("%s/%d completion_tokens=%d below %d", result.FixtureID, result.Repeat, result.Usage["completion_tokens"], MinimumLongDecodeTokens))
			continue
		}
		if result.DecodeTrace == nil || result.DecodeWindows == nil {
			failure = joinFailure(failure, fmt.Sprintf("%s/%d missing native decode trace", result.FixtureID, result.Repeat))
			continue
		}
		rebuilt, err := BuildDecodeWindows(*result.DecodeTrace, NativeDecodeContract, result.Usage["completion_tokens"])
		if err != nil {
			failure = joinFailure(failure, fmt.Sprintf("%s/%d: %v", result.FixtureID, result.Repeat, err))
			continue
		}
		if !reflect.DeepEqual(rebuilt, *result.DecodeWindows) {
			failure = joinFailure(failure, fmt.Sprintf("%s/%d decode-window readback mismatch", result.FixtureID, result.Repeat))
			continue
		}
		reports = append(reports, rebuilt)
	}
	if failure != "" {
		return nil, fmt.Errorf("%s", failure)
	}
	return reports, nil
}

func buildLlamaClientDecodeReports(results []LlamaClientDecodeResult) ([]DecodeWindowReport, error) {
	if len(results) != MinimumDecodeRepetitions {
		return nil, fmt.Errorf("comparator results=%d, want exactly %d repetitions", len(results), MinimumDecodeRepetitions)
	}
	fixtureID := ""
	seen := make(map[int]bool, MinimumDecodeRepetitions)
	reports := make([]DecodeWindowReport, 0, len(results))
	for _, result := range results {
		if fixtureID == "" {
			fixtureID = result.FixtureID
		}
		if result.FixtureID == "" || result.FixtureID != fixtureID || result.Repeat < 1 || result.Repeat > MinimumDecodeRepetitions || seen[result.Repeat] {
			return nil, fmt.Errorf("%s/%d unmatched comparator repetition identity", result.FixtureID, result.Repeat)
		}
		seen[result.Repeat] = true
		_, report, err := BuildLlamaClientDecodeWindows(result.Events, result.Final)
		if err != nil {
			return nil, fmt.Errorf("%s/%d: %w", result.FixtureID, result.Repeat, err)
		}
		reports = append(reports, report)
	}
	return reports, nil
}

func validateDecodeWindowSummary(summary DecodeWindowSummary) error {
	return validateDecodeWindowSummaryContract(summary, NativeDecodeContract)
}

func validateDecodeWindowSummaryContract(summary DecodeWindowSummary, expected DecodeTraceContract) error {
	if summary.Verdict != "PASS" && summary.Verdict != "HOLD" {
		return fmt.Errorf("invalid decode-window verdict %q", summary.Verdict)
	}
	if summary.Threshold != DecodeLateEarlyThreshold || summary.Contract != expected {
		return fmt.Errorf("decode-window contract or threshold mismatch")
	}
	if summary.Verdict == "HOLD" {
		if summary.Failure == "" {
			return fmt.Errorf("decode-window HOLD lost its failure")
		}
		return nil
	}
	if summary.Failure != "" || summary.RawCompletionTokensPerRun < MinimumLongDecodeTokens || summary.TimedTokensPerRun+expected.LeadingUntimedTokens != summary.RawCompletionTokensPerRun || summary.Confidence.Repetitions != MinimumDecodeRepetitions || len(summary.Windows) != 3 || summary.LateEarlyRatio < DecodeLateEarlyThreshold || summary.ElapsedNS <= 0 || !decodeFinitePositive(summary.TokensPerSecond) {
		return fmt.Errorf("decode-window PASS lacks complete qualifying evidence")
	}
	sizes := [3]int{summary.TimedTokensPerRun / 3, summary.TimedTokensPerRun / 3, summary.TimedTokensPerRun / 3}
	for i := 0; i < summary.TimedTokensPerRun%3; i++ {
		sizes[i]++
	}
	names := [...]string{"early", "middle", "late"}
	for i, window := range summary.Windows {
		if window.Name != names[i] || window.FirstTokenIndex != 0 || window.LastTokenIndex != 0 || window.Tokens != sizes[i]*MinimumDecodeRepetitions || window.ElapsedNS <= 0 || !decodeFinitePositive(window.TokensPerSecond) {
			return fmt.Errorf("decode-window aggregate geometry is invalid")
		}
	}
	return nil
}

func validateMatchedDecodeWindowSummary(summary MatchedDecodeWindowSummary) error {
	if err := validateDecodeWindowSummaryContract(summary.Native, NativeDecodeContract); err != nil {
		return fmt.Errorf("matched native: %w", err)
	}
	if err := validateDecodeWindowSummaryContract(summary.Comparator, LlamaClientDecodeContract); err != nil {
		return fmt.Errorf("matched comparator: %w", err)
	}
	if summary.Native.RawCompletionTokensPerRun != summary.Comparator.RawCompletionTokensPerRun || summary.Native.TimedTokensPerRun != summary.Native.RawCompletionTokensPerRun || summary.Comparator.TimedTokensPerRun+1 != summary.Comparator.RawCompletionTokensPerRun {
		return fmt.Errorf("matched decode raw/timed cardinality mismatch")
	}
	wantVerdict, wantFailure := "PASS", ""
	if summary.Native.Verdict != "PASS" {
		wantVerdict, wantFailure = "HOLD", summary.Native.Failure
	}
	if summary.Verdict != wantVerdict || summary.Failure != wantFailure {
		return fmt.Errorf("matched decode verdict does not follow native evidence")
	}
	return nil
}

const (
	NativeDecodeTraceSchema     = "fak.native-decode-trace/1"
	NativeDecodeTraceEngine     = "fak-native"
	NativeDecodeTraceProvenance = "native-token-commit"
	MinimumDecodeRepetitions    = 3
	DecodeLateEarlyThreshold    = 0.85
)

// DecodeTraceContract binds a timestamp series to its explicit producer and
// timing semantics. Both producers may batch observations at one timestamp;
// only transport arrivals exclude their first-token baseline.
type DecodeTraceContract struct {
	Schema               string `json:"schema"`
	Engine               string `json:"engine"`
	Provenance           string `json:"provenance"`
	LeadingUntimedTokens int    `json:"leading_untimed_tokens,omitempty"`
	AllowEqualElapsed    bool   `json:"allow_equal_elapsed,omitempty"`
}

var NativeDecodeContract = DecodeTraceContract{
	Schema: NativeDecodeTraceSchema, Engine: NativeDecodeTraceEngine, Provenance: NativeDecodeTraceProvenance,
	AllowEqualElapsed: true,
}

type DecodeTrace struct {
	Schema     string             `json:"schema"`
	Engine     string             `json:"engine"`
	Provenance string             `json:"provenance,omitempty"`
	Events     []DecodeTraceEvent `json:"events"`
}

type DecodeTraceEvent struct {
	TokenIndex int   `json:"token_index"`
	ElapsedNS  int64 `json:"elapsed_ns"`
	TokenID    *int  `json:"token_id,omitempty"`
}

type DecodeWindow struct {
	Name            string  `json:"name"`
	FirstTokenIndex int     `json:"first_token_index"`
	LastTokenIndex  int     `json:"last_token_index"`
	Tokens          int     `json:"tokens"`
	ElapsedNS       int64   `json:"elapsed_ns"`
	TokensPerSecond float64 `json:"tokens_per_second"`
}

// DecodeWindowReport is one validated repetition. LinearSlope is the
// least-squares slope of window tok/s against token-position midpoint.
type DecodeWindowReport struct {
	Contract                DecodeTraceContract `json:"contract"`
	RawCompletionTokens     int                 `json:"raw_completion_tokens"`
	TimedTokens             int                 `json:"timed_tokens"`
	ElapsedNS               int64               `json:"elapsed_ns"`
	TokensPerSecond         float64             `json:"tokens_per_second"`
	Windows                 []DecodeWindow      `json:"windows"`
	LateEarlyRatio          float64             `json:"late_early_ratio"`
	LinearSlopeTokensPerSec float64             `json:"linear_slope_tokens_per_second_per_token"`
}

type DecodeRepetitionConfidence struct {
	Repetitions       int     `json:"repetitions"`
	LateEarlyMean     float64 `json:"late_early_mean"`
	LateEarlyMinimum  float64 `json:"late_early_minimum"`
	LateEarlyMaximum  float64 `json:"late_early_maximum"`
	LateEarlyStdDev   float64 `json:"late_early_stddev"`
	LinearSlopeMean   float64 `json:"linear_slope_mean"`
	LinearSlopeStdDev float64 `json:"linear_slope_stddev"`
}

type DecodeWindowSummary struct {
	Contract                  DecodeTraceContract        `json:"contract"`
	RawCompletionTokensPerRun int                        `json:"raw_completion_tokens_per_run"`
	TimedTokensPerRun         int                        `json:"timed_tokens_per_run"`
	ElapsedNS                 int64                      `json:"elapsed_ns"`
	TokensPerSecond           float64                    `json:"tokens_per_second"`
	Windows                   []DecodeWindow             `json:"windows"`
	LateEarlyRatio            float64                    `json:"late_early_ratio"`
	LinearSlopeTokensPerSec   float64                    `json:"linear_slope_tokens_per_second_per_token"`
	Confidence                DecodeRepetitionConfidence `json:"confidence"`
	Threshold                 float64                    `json:"threshold"`
	Verdict                   string                     `json:"verdict"`
	Failure                   string                     `json:"failure,omitempty"`
}

// BuildDecodeWindows validates an explicit token-index/elapsed trace and folds
// its timed tokens into deterministic early, middle, and late thirds. The
// contract declares whether leading raw tokens are excluded and whether equal
// elapsed observations are valid.
func BuildDecodeWindows(trace DecodeTrace, expected DecodeTraceContract, rawCompletionTokens int) (DecodeWindowReport, error) {
	if expected.Schema == "" || expected.Engine == "" || expected.Provenance == "" {
		return DecodeWindowReport{}, fmt.Errorf("decode trace expected contract is incomplete")
	}
	if trace.Schema != expected.Schema {
		return DecodeWindowReport{}, fmt.Errorf("decode trace schema mismatch: got %q want %q", trace.Schema, expected.Schema)
	}
	if trace.Engine != expected.Engine {
		return DecodeWindowReport{}, fmt.Errorf("decode trace engine mismatch: got %q want %q", trace.Engine, expected.Engine)
	}
	if trace.Provenance != expected.Provenance {
		return DecodeWindowReport{}, fmt.Errorf("decode trace provenance mismatch: got %q want %q", trace.Provenance, expected.Provenance)
	}
	if expected.LeadingUntimedTokens < 0 || expected.LeadingUntimedTokens > rawCompletionTokens {
		return DecodeWindowReport{}, fmt.Errorf("decode trace leading untimed tokens are invalid")
	}
	timedTokens := rawCompletionTokens - expected.LeadingUntimedTokens
	if timedTokens < 3 {
		return DecodeWindowReport{}, fmt.Errorf("decode trace timed tokens %d below three-window minimum", timedTokens)
	}
	if len(trace.Events) != timedTokens {
		return DecodeWindowReport{}, fmt.Errorf("decode trace completion-token mismatch: events=%d raw_completion_tokens=%d leading_untimed_tokens=%d", len(trace.Events), rawCompletionTokens, expected.LeadingUntimedTokens)
	}
	var previousElapsed int64
	for i, event := range trace.Events {
		want := i + 1
		if event.TokenIndex != want {
			reason := "gapped"
			if event.TokenIndex < want {
				reason = "duplicate or out-of-order"
			}
			return DecodeWindowReport{}, fmt.Errorf("decode trace %s token index at event %d: got %d want %d", reason, i, event.TokenIndex, want)
		}
		if event.ElapsedNS < previousElapsed || (!expected.AllowEqualElapsed && event.ElapsedNS == previousElapsed) {
			return DecodeWindowReport{}, fmt.Errorf("decode trace elapsed_ns is non-monotonic at token %d: got %d after %d", event.TokenIndex, event.ElapsedNS, previousElapsed)
		}
		previousElapsed = event.ElapsedNS
	}
	sizes := [3]int{timedTokens / 3, timedTokens / 3, timedTokens / 3}
	for i := 0; i < timedTokens%3; i++ {
		sizes[i]++
	}
	names := [...]string{"early", "middle", "late"}
	windows := make([]DecodeWindow, 0, 3)
	first, priorElapsed := 1, int64(0)
	for i, size := range sizes {
		last := first + size - 1
		elapsed := trace.Events[last-1].ElapsedNS - priorElapsed
		windows = append(windows, DecodeWindow{Name: names[i], FirstTokenIndex: first, LastTokenIndex: last, Tokens: size, ElapsedNS: elapsed, TokensPerSecond: decodeTokenRate(size, elapsed)})
		priorElapsed, first = trace.Events[last-1].ElapsedNS, last+1
	}
	report := DecodeWindowReport{Contract: expected, RawCompletionTokens: rawCompletionTokens, TimedTokens: timedTokens, ElapsedNS: previousElapsed, TokensPerSecond: decodeTokenRate(timedTokens, previousElapsed), Windows: windows}
	report.LateEarlyRatio = windows[2].TokensPerSecond / windows[0].TokensPerSecond
	report.LinearSlopeTokensPerSec = decodeWindowSlope(windows)
	if err := validateDecodeWindowReport(report); err != nil {
		return DecodeWindowReport{}, err
	}
	return report, nil
}

func FoldDecodeRepetitions(reports []DecodeWindowReport) (DecodeWindowSummary, error) {
	if len(reports) != MinimumDecodeRepetitions {
		return DecodeWindowSummary{}, fmt.Errorf("decode windows require exactly %d repetitions, got %d", MinimumDecodeRepetitions, len(reports))
	}
	first := reports[0]
	if err := validateDecodeWindowReport(first); err != nil {
		return DecodeWindowSummary{}, fmt.Errorf("repetition 1: %w", err)
	}
	summary := DecodeWindowSummary{Contract: first.Contract, RawCompletionTokensPerRun: first.RawCompletionTokens, TimedTokensPerRun: first.TimedTokens, Windows: make([]DecodeWindow, 3), Threshold: DecodeLateEarlyThreshold, Verdict: "PASS"}
	ratios, slopes := make([]float64, len(reports)), make([]float64, len(reports))
	for i, report := range reports {
		if err := validateDecodeWindowReport(report); err != nil {
			return DecodeWindowSummary{}, fmt.Errorf("repetition %d: %w", i+1, err)
		}
		if report.Contract != first.Contract || report.RawCompletionTokens != first.RawCompletionTokens || report.TimedTokens != first.TimedTokens {
			return DecodeWindowSummary{}, fmt.Errorf("repetition %d decode-window identity mismatch", i+1)
		}
		summary.ElapsedNS += report.ElapsedNS
		for j, window := range report.Windows {
			base := first.Windows[j]
			if window.Name != base.Name || window.FirstTokenIndex != base.FirstTokenIndex || window.LastTokenIndex != base.LastTokenIndex || window.Tokens != base.Tokens {
				return DecodeWindowSummary{}, fmt.Errorf("repetition %d window %d geometry mismatch", i+1, j)
			}
			summary.Windows[j].Name = window.Name
			summary.Windows[j].Tokens += window.Tokens
			summary.Windows[j].ElapsedNS += window.ElapsedNS
		}
		ratios[i], slopes[i] = report.LateEarlyRatio, report.LinearSlopeTokensPerSec
		if report.LateEarlyRatio < DecodeLateEarlyThreshold && summary.Failure == "" {
			summary.Verdict = "HOLD"
			summary.Failure = fmt.Sprintf("repetition %d late/early %.6f below %.2f", i+1, report.LateEarlyRatio, DecodeLateEarlyThreshold)
		}
	}
	for i := range summary.Windows {
		summary.Windows[i].TokensPerSecond = decodeTokenRate(summary.Windows[i].Tokens, summary.Windows[i].ElapsedNS)
	}
	summary.TokensPerSecond = decodeTokenRate(first.TimedTokens*len(reports), summary.ElapsedNS)
	summary.LateEarlyRatio = summary.Windows[2].TokensPerSecond / summary.Windows[0].TokensPerSecond
	slopeWindows := append([]DecodeWindow(nil), summary.Windows...)
	for i := range slopeWindows {
		slopeWindows[i].FirstTokenIndex = first.Windows[i].FirstTokenIndex
		slopeWindows[i].LastTokenIndex = first.Windows[i].LastTokenIndex
	}
	summary.LinearSlopeTokensPerSec = decodeWindowSlope(slopeWindows)
	summary.Confidence = decodeRepetitionConfidence(ratios, slopes)
	return summary, nil
}

func validateDecodeWindowReport(report DecodeWindowReport) error {
	if report.Contract.Schema == "" || report.Contract.Engine == "" || report.Contract.Provenance == "" || report.RawCompletionTokens < 3 || report.TimedTokens < 3 || report.ElapsedNS <= 0 || len(report.Windows) != 3 {
		return fmt.Errorf("decode-window report is incomplete")
	}
	if report.Contract.LeadingUntimedTokens < 0 || report.TimedTokens+report.Contract.LeadingUntimedTokens != report.RawCompletionTokens {
		return fmt.Errorf("decode-window raw/timed token cardinality mismatch")
	}
	if !decodeFinitePositive(report.TokensPerSecond) || !decodeFinitePositive(report.LateEarlyRatio) || math.IsNaN(report.LinearSlopeTokensPerSec) || math.IsInf(report.LinearSlopeTokensPerSec, 0) {
		return fmt.Errorf("decode-window report has non-finite metrics")
	}
	names := [...]string{"early", "middle", "late"}
	tokens, elapsed := 0, int64(0)
	for i, window := range report.Windows {
		if window.Name != names[i] || window.Tokens <= 0 || window.ElapsedNS <= 0 || !decodeFinitePositive(window.TokensPerSecond) {
			return fmt.Errorf("decode window %d is invalid", i)
		}
		tokens, elapsed = tokens+window.Tokens, elapsed+window.ElapsedNS
	}
	if tokens != report.TimedTokens || elapsed != report.ElapsedNS {
		return fmt.Errorf("decode-window totals do not match completion")
	}
	return nil
}

func decodeTokenRate(tokens int, elapsedNS int64) float64 {
	return float64(tokens) * 1e9 / float64(elapsedNS)
}

func decodeWindowSlope(windows []DecodeWindow) float64 {
	var sumX, sumY float64
	for _, window := range windows {
		sumX += float64(window.FirstTokenIndex+window.LastTokenIndex) / 2
		sumY += window.TokensPerSecond
	}
	meanX, meanY := sumX/float64(len(windows)), sumY/float64(len(windows))
	var numerator, denominator float64
	for _, window := range windows {
		x := float64(window.FirstTokenIndex+window.LastTokenIndex) / 2
		numerator += (x - meanX) * (window.TokensPerSecond - meanY)
		denominator += (x - meanX) * (x - meanX)
	}
	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}

func decodeRepetitionConfidence(ratios, slopes []float64) DecodeRepetitionConfidence {
	out := DecodeRepetitionConfidence{Repetitions: len(ratios), LateEarlyMinimum: ratios[0], LateEarlyMaximum: ratios[0]}
	for i := range ratios {
		out.LateEarlyMean += ratios[i]
		out.LinearSlopeMean += slopes[i]
		out.LateEarlyMinimum = math.Min(out.LateEarlyMinimum, ratios[i])
		out.LateEarlyMaximum = math.Max(out.LateEarlyMaximum, ratios[i])
	}
	out.LateEarlyMean /= float64(len(ratios))
	out.LinearSlopeMean /= float64(len(slopes))
	if len(ratios) > 1 {
		for i := range ratios {
			out.LateEarlyStdDev += math.Pow(ratios[i]-out.LateEarlyMean, 2)
			out.LinearSlopeStdDev += math.Pow(slopes[i]-out.LinearSlopeMean, 2)
		}
		out.LateEarlyStdDev = math.Sqrt(out.LateEarlyStdDev / float64(len(ratios)-1))
		out.LinearSlopeStdDev = math.Sqrt(out.LinearSlopeStdDev / float64(len(slopes)-1))
	}
	return out
}

func decodeFinitePositive(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}
