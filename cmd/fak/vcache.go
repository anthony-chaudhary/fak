package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcachechain"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

func cmdVCache(argv []string) {
	os.Exit(runVCache(os.Stdout, os.Stderr, argv))
}

func runVCache(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		vcacheUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "status":
		return runVCacheStatus(stdout, stderr, argv[1:])
	case "prove":
		return runVCacheProve(stdout, stderr, argv[1:])
	case "prove-telemetry":
		return runVCacheProveTelemetry(stdout, stderr, argv[1:])
	case "prove-recall":
		return runVCacheProveRecall(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		vcacheUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak vcache: unknown subcommand %q\n", argv[0])
		vcacheUsage(stderr)
		return 2
	}
}

func vcacheUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak vcache status [--json]
  fak vcache prove [--json] [--anchor-tokens N] [--suffix-tokens N] [--requests N]
                   [--min-prefix-tokens N] [--read-mult F] [--write-mult F]
                   [--content public|secret|regulated]
  fak vcache prove-telemetry --file FILE [--json]
                   [--read-mult F] [--write-5m-mult F] [--write-1h-mult F]
  fak vcache prove-recall [--json] [--prefix-tokens N] [--unit-tokens N]
                   [--read-mult F] [--siblings N]

status reports what is actually up: the M5 governor is a local, off-path policy
engine; the M4 chains & recall engine is off-path and gated OFF by default;
provider calibration/warming remain tracked by #716-#718, and Codex/OpenAI cached-
token telemetry remains tracked by #727.
prove runs the deterministic star-anchor token-savings proof. Exit 0 means PROVEN;
exit 1 means REFUTED; exit 2 means usage error.
prove-telemetry replays provider usage JSONL, such as Claude Code probe output or
OpenAI Responses/Chat usage objects, and proves realized savings from observed
cache counters.
prove-recall runs the deterministic M4 cost-gate proof (the §11.0 headline): a
single ~10-token unit recalled from a long warm prefix is almost always a net LOSS,
so the gate REFUSES it; rebuild wins only for amortized fan-out. Exit 0 = rebuild
allowed (PROVEN); exit 1 = refused (REFUTED); exit 2 = usage error.

`)
}

type vcacheStatusReport struct {
	Status         string                     `json:"status"`
	Governor       string                     `json:"governor"`
	Chains         string                     `json:"chains"`
	LiveProvider   string                     `json:"live_provider"`
	Proof          vcachegov.StarSavingsProof `json:"proof"`
	RecallProof    vcachechain.RecallProof    `json:"recall_proof"`
	CodexOpenAI    vcacheCodexOpenAIStatus    `json:"codex_openai"`
	M4Issue        string                     `json:"m4_issue"`
	M5Issue        string                     `json:"m5_issue"`
	Remaining      []vcacheRemainingIssue     `json:"remaining"`
	CorrectnessLaw string                     `json:"correctness_law"`
}

type vcacheRemainingIssue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	URL    string `json:"url"`
}

type vcacheCodexOpenAIStatus struct {
	Verifier            string                          `json:"verifier"`
	LiveTelemetry       string                          `json:"live_telemetry"`
	Reason              string                          `json:"reason"`
	OpenAIAPIKeyPresent bool                            `json:"openai_api_key_present"`
	CachedTokenFields   []string                        `json:"cached_token_fields"`
	Issue               string                          `json:"issue"`
	CachedSampleProof   vcachegov.TelemetrySavingsProof `json:"cached_sample_proof"`
	NoCacheRefutation   vcachegov.TelemetrySavingsProof `json:"no_cache_refutation"`
}

func runVCacheStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable status")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	rep := defaultVCacheStatus()
	if *asJSON {
		return writeJSON(stdout, rep)
	}
	fmt.Fprintf(stdout, "vCache status: %s\n", rep.Status)
	fmt.Fprintf(stdout, "vCache M5 governor: %s\n", rep.Governor)
	fmt.Fprintf(stdout, "vCache M4 chains & recall: %s\n", rep.Chains)
	fmt.Fprintf(stdout, "live provider loop: %s\n", rep.LiveProvider)
	fmt.Fprintf(stdout, "codex-like star proof: %s (%s)\n", rep.Proof.Status, rep.Proof.Reason)
	fmt.Fprintf(stdout, "token-equiv saved: %.1f / %.1f (%.1f%%)\n",
		rep.Proof.SavedTokenEquiv, rep.Proof.BaselineTokenEquiv, rep.Proof.SavedPct)
	fmt.Fprintf(stdout, "M4 recall cost-gate proof: %s — %s\n", rep.RecallProof.Status, rep.RecallProof.Decision)
	fmt.Fprintf(stdout, "M4 single-unit loss ratio: %.1fx (break-even %s siblings)\n",
		rep.RecallProof.LossRatio, formatBreakEven(rep.RecallProof.BreakEvenSiblings))
	fmt.Fprintf(stdout, "codex/openai verifier: %s\n", rep.CodexOpenAI.Verifier)
	fmt.Fprintf(stdout, "codex/openai live telemetry: %s (%s)\n",
		rep.CodexOpenAI.LiveTelemetry, rep.CodexOpenAI.Reason)
	fmt.Fprintf(stdout, "codex/openai cached-token sample: %s saved %.1f / %.1f (%.2f%%)\n",
		rep.CodexOpenAI.CachedSampleProof.Status,
		rep.CodexOpenAI.CachedSampleProof.SavedTokenEquiv,
		rep.CodexOpenAI.CachedSampleProof.BaselineTokenEquiv,
		rep.CodexOpenAI.CachedSampleProof.SavedPct)
	fmt.Fprintf(stdout, "codex/openai zero-cache sample: %s saved %.1f / %.1f (%.2f%%)\n",
		rep.CodexOpenAI.NoCacheRefutation.Status,
		rep.CodexOpenAI.NoCacheRefutation.SavedTokenEquiv,
		rep.CodexOpenAI.NoCacheRefutation.BaselineTokenEquiv,
		rep.CodexOpenAI.NoCacheRefutation.SavedPct)
	fmt.Fprintf(stdout, "correctness depends on cache hit: %v\n", rep.Proof.CorrectnessDependsOn)
	fmt.Fprintf(stdout, "m5 issue: %s\n", rep.M5Issue)
	fmt.Fprint(stdout, "remaining:")
	for _, issue := range rep.Remaining {
		fmt.Fprintf(stdout, " #%d", issue.Number)
	}
	fmt.Fprintln(stdout)
	return 0
}

func runVCacheProve(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache prove", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable proof")
	anchor := fs.Float64("anchor-tokens", 4096, "cacheable anchor size in input tokens")
	suffix := fs.Float64("suffix-tokens", 10, "fresh suffix tokens per sibling request")
	requests := fs.Int("requests", 7, "number of sibling requests sharing the anchor")
	minPrefix := fs.Float64("min-prefix-tokens", 1024, "provider minimum cacheable prefix")
	readMult := fs.Float64("read-mult", 0.1, "provider cached-read input-token multiplier")
	writeMult := fs.Float64("write-mult", vcachegov.WriteMult5Minutes, "provider cache-write input-token multiplier")
	content := fs.String("content", "public", "prefix content class: public, secret, regulated")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	proof := vcachegov.ProveStarSavings(vcachegov.StarSavingsInput{
		AnchorTokens:    *anchor,
		SuffixTokens:    *suffix,
		Requests:        *requests,
		MinPrefixTokens: *minPrefix,
		ReadMult:        *readMult,
		WriteMult:       *writeMult,
		Secret:          vcachegov.ClassifyPrefix(strings.ToLower(strings.TrimSpace(*content))),
	})
	if *asJSON {
		code := writeJSON(stdout, proof)
		if code != 0 {
			return code
		}
		return vcacheProofExit(proof.Status)
	}
	fmt.Fprintf(stdout, "status: %s\n", proof.Status)
	fmt.Fprintf(stdout, "reason: %s\n", proof.Reason)
	fmt.Fprintf(stdout, "requests: %d\n", proof.Requests)
	fmt.Fprintf(stdout, "anchor/suffix/min: %.0f / %.0f / %.0f tokens\n",
		proof.AnchorTokens, proof.SuffixTokens, proof.MinPrefixTokens)
	fmt.Fprintf(stdout, "read/write multipliers: %.3g / %.3g\n", proof.ReadMult, proof.WriteMult)
	fmt.Fprintf(stdout, "baseline token-equiv: %.1f\n", proof.BaselineTokenEquiv)
	fmt.Fprintf(stdout, "vcache token-equiv: %.1f\n", proof.VCacheTokenEquiv)
	fmt.Fprintf(stdout, "saved token-equiv: %.1f (%.1f%%)\n", proof.SavedTokenEquiv, proof.SavedPct)
	fmt.Fprintf(stdout, "break-even requests: %s\n", formatBreakEven(proof.BreakEvenRequests))
	fmt.Fprintf(stdout, "correctness depends on cache hit: %v\n", proof.CorrectnessDependsOn)
	return vcacheProofExit(proof.Status)
}

func runVCacheProveRecall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache prove-recall", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable proof")
	prefix := fs.Int64("prefix-tokens", 30000, "replayed warm prefix length in tokens (P)")
	unit := fs.Int64("unit-tokens", 10, "recalled unit fresh-prefill length in tokens (U)")
	readMult := fs.Float64("read-mult", 0.1, "provider cached-read token multiplier (r)")
	siblings := fs.Int("siblings", 1, "co-recalled sibling units sharing the prefix (S, the amortization)")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	proof := vcachechain.ProveRecall(vcachechain.ProveRecallInput{
		PrefixTokens: *prefix,
		UnitTokens:   *unit,
		ReadMult:     *readMult,
		Siblings:     *siblings,
	})
	if *asJSON {
		if code := writeJSON(stdout, proof); code != 0 {
			return code
		}
		return vcacheRecallProofExit(proof.Status)
	}
	fmt.Fprintf(stdout, "status: %s\n", proof.Status)
	fmt.Fprintf(stdout, "decision: %s\n", proof.Decision)
	fmt.Fprintf(stdout, "reason: %s\n", proof.Reason)
	fmt.Fprintf(stdout, "prefix/unit tokens: %d / %d\n", proof.PrefixTokens, proof.UnitTokens)
	fmt.Fprintf(stdout, "read multiplier: %.3g\n", proof.ReadMult)
	fmt.Fprintf(stdout, "siblings (amortization): %d\n", proof.Siblings)
	fmt.Fprintf(stdout, "replay cost (P·r): %.1f token-equiv\n", proof.ReplayCost)
	fmt.Fprintf(stdout, "fresh prefill (U): %.1f token-equiv\n", proof.FreshPrefillCost)
	fmt.Fprintf(stdout, "amortized savings (S·U): %.1f token-equiv\n", proof.AmortizedSavings)
	fmt.Fprintf(stdout, "single-unit loss ratio (P·r/U): %.1fx\n", proof.LossRatio)
	fmt.Fprintf(stdout, "break-even siblings: %s\n", formatBreakEven(proof.BreakEvenSiblings))
	fmt.Fprintf(stdout, "correctness depends on cache hit: %v\n", proof.CorrectnessDependsOn)
	return vcacheRecallProofExit(proof.Status)
}

func vcacheRecallProofExit(s vcachechain.ProofStatus) int {
	if s == vcachechain.ProofProven {
		return 0
	}
	return 1
}

func runVCacheProveTelemetry(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("vcache prove-telemetry", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable proof")
	file := fs.String("file", "", "provider telemetry JSONL file ('-' for stdin)")
	readMult := fs.Float64("read-mult", 0.1, "provider cached-read input-token multiplier")
	write5mMult := fs.Float64("write-5m-mult", vcachegov.WriteMult5Minutes, "5m cache-write input-token multiplier")
	write1hMult := fs.Float64("write-1h-mult", vcachegov.WriteMult1Hour, "1h cache-write input-token multiplier")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if strings.TrimSpace(*file) == "" {
		fmt.Fprintln(stderr, "fak vcache prove-telemetry: --file is required")
		return 2
	}

	rows, err := readVCacheTelemetry(*file, os.Stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak vcache prove-telemetry: %v\n", err)
		return 2
	}
	proof := vcachegov.ProveTelemetrySavings(vcachegov.TelemetrySavingsInput{
		Rows:        rows,
		ReadMult:    *readMult,
		Write5mMult: *write5mMult,
		Write1hMult: *write1hMult,
	})
	if *asJSON {
		code := writeJSON(stdout, proof)
		if code != 0 {
			return code
		}
		return vcacheProofExit(proof.Status)
	}
	fmt.Fprintf(stdout, "status: %s\n", proof.Status)
	fmt.Fprintf(stdout, "reason: %s\n", proof.Reason)
	fmt.Fprintf(stdout, "requests: %d\n", proof.Requests)
	fmt.Fprintf(stdout, "baseline token-equiv: %.1f\n", proof.BaselineTokenEquiv)
	fmt.Fprintf(stdout, "actual token-equiv: %.1f\n", proof.ActualTokenEquiv)
	fmt.Fprintf(stdout, "saved token-equiv: %.1f (%.2f%%)\n", proof.SavedTokenEquiv, proof.SavedPct)
	fmt.Fprintf(stdout, "cache read/write tokens: %.0f / %.0f\n", proof.CacheReadTokens, proof.CacheCreationTokens)
	fmt.Fprintf(stdout, "first positive request: %s\n", formatObservedPositive(proof.FirstPositiveRequest))
	fmt.Fprintf(stdout, "correctness depends on cache hit: %v\n", proof.CorrectnessDependsOn)
	return vcacheProofExit(proof.Status)
}

func defaultVCacheStatus() vcacheStatusReport {
	return vcacheStatusReport{
		Status:       "M5 governor up; M4 chains & recall up (gated OFF by default); full vCache provider loop not yet live",
		Governor:     "up (pin/lazy/evict, warm budget, affinity, secret gate)",
		Chains:       "up (prefix DAG, topological replay, cost-gated rebuild) — gated OFF by default; off-path",
		LiveProvider: "not wired; M1-M3 remain open; Codex/OpenAI telemetry probe #727 pending",
		Proof: vcachegov.ProveStarSavings(vcachegov.StarSavingsInput{
			AnchorTokens:    4096,
			SuffixTokens:    10,
			Requests:        7,
			MinPrefixTokens: 1024,
			ReadMult:        0.1,
			WriteMult:       vcachegov.WriteMult5Minutes,
			Secret:          vcachegov.Cacheable,
		}),
		RecallProof: vcachechain.ProveRecall(vcachechain.ProveRecallInput{
			PrefixTokens: 30000,
			UnitTokens:   10,
			ReadMult:     0.1,
			Siblings:     1,
		}),
		CodexOpenAI: defaultCodexOpenAIStatus(),
		M4Issue:     "https://github.com/anthony-chaudhary/fak/issues/719",
		M5Issue:     "https://github.com/anthony-chaudhary/fak/issues/720",
		Remaining: []vcacheRemainingIssue{
			{716, "M1 observe & calibrate", "https://github.com/anthony-chaudhary/fak/issues/716"},
			{717, "M2 star anchors", "https://github.com/anthony-chaudhary/fak/issues/717"},
			{718, "M3 dedicated warming", "https://github.com/anthony-chaudhary/fak/issues/718"},
			{727, "Codex/OpenAI telemetry probe", "https://github.com/anthony-chaudhary/fak/issues/727"},
		},
		CorrectnessLaw: "cost is budgeted at the uncached price; hits are realized rebates, never trust claims",
	}
}

func defaultCodexOpenAIStatus() vcacheCodexOpenAIStatus {
	hasKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")) != ""
	live := "unavailable"
	reason := "OPENAI_API_KEY not present; raw OpenAI API probe not run. Codex CLI session token_count JSONL can be passed to prove-telemetry."
	if hasKey {
		live = "not-run"
		reason = "OPENAI_API_KEY is present, but a provider-authored OpenAI usage JSONL file is still required for the raw API probe"
	}
	return vcacheCodexOpenAIStatus{
		Verifier:            "ready",
		LiveTelemetry:       live,
		Reason:              reason,
		OpenAIAPIKeyPresent: hasKey,
		CachedTokenFields: []string{
			"usage.input_tokens_details.cached_tokens",
			"usage.prompt_tokens_details.cached_tokens",
			"payload.info.last_token_usage.cached_input_tokens",
		},
		Issue: "https://github.com/anthony-chaudhary/fak/issues/727",
		CachedSampleProof: vcachegov.ProveTelemetrySavings(vcachegov.TelemetrySavingsInput{
			Rows:     []vcachegov.TelemetryRow{openAITelemetryRow(2006, 1920)},
			ReadMult: 0.1,
		}),
		NoCacheRefutation: vcachegov.ProveTelemetrySavings(vcachegov.TelemetrySavingsInput{
			Rows:     []vcachegov.TelemetryRow{openAITelemetryRow(2006, 0)},
			ReadMult: 0.1,
		}),
	}
}

func writeJSON(w io.Writer, v any) int {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return 2
	}
	fmt.Fprintln(w, string(b))
	return 0
}

func vcacheProofExit(s vcachegov.ProofStatus) int {
	if s == vcachegov.ProofProven {
		return 0
	}
	return 1
}

func formatBreakEven(n int) string {
	if n == int(^uint(0)>>1) {
		return "never"
	}
	return fmt.Sprintf("%d", n)
}

func formatObservedPositive(n int) string {
	if n <= 0 {
		return "never"
	}
	return fmt.Sprintf("%d", n)
}

type vcacheTelemetryJSONLRow struct {
	InputTokens              float64             `json:"input_tokens"`
	PromptTokens             float64             `json:"prompt_tokens"`
	CachedTokens             float64             `json:"cached_tokens"`
	CacheCreationInputTokens float64             `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     float64             `json:"cache_read_input_tokens"`
	Ephemeral1hInputTokens   float64             `json:"ephemeral_1h_input_tokens"`
	Ephemeral5mInputTokens   float64             `json:"ephemeral_5m_input_tokens"`
	Usage                    *vcacheOpenAIUsage  `json:"usage"`
	Type                     string              `json:"type"`
	Payload                  *vcacheCodexPayload `json:"payload"`
}

type vcacheOpenAIUsage struct {
	InputTokens         float64                   `json:"input_tokens"`
	PromptTokens        float64                   `json:"prompt_tokens"`
	InputTokensDetails  vcacheCachedTokensDetails `json:"input_tokens_details"`
	PromptTokensDetails vcacheCachedTokensDetails `json:"prompt_tokens_details"`
}

type vcacheCachedTokensDetails struct {
	CachedTokens float64 `json:"cached_tokens"`
}

type vcacheCodexPayload struct {
	Type string               `json:"type"`
	Info vcacheCodexTokenInfo `json:"info"`
}

type vcacheCodexTokenInfo struct {
	LastTokenUsage vcacheCodexTokenUsage `json:"last_token_usage"`
}

type vcacheCodexTokenUsage struct {
	InputTokens       float64 `json:"input_tokens"`
	CachedInputTokens float64 `json:"cached_input_tokens"`
}

func readVCacheTelemetry(path string, stdin io.Reader) ([]vcachegov.TelemetryRow, error) {
	var r io.Reader
	if path == "-" {
		r = stdin
	} else {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		r = f
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var rows []vcachegov.TelemetryRow
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var raw vcacheTelemetryJSONLRow
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		row, ok := raw.telemetryRow()
		if ok {
			rows = append(rows, row)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

func (r vcacheTelemetryJSONLRow) telemetryRow() (vcachegov.TelemetryRow, bool) {
	if r.hasClaudeCounters() {
		return vcachegov.TelemetryRow{
			InputTokens:              r.InputTokens,
			CacheCreationInputTokens: r.CacheCreationInputTokens,
			CacheReadInputTokens:     r.CacheReadInputTokens,
			Ephemeral1hInputTokens:   r.Ephemeral1hInputTokens,
			Ephemeral5mInputTokens:   r.Ephemeral5mInputTokens,
		}, true
	}
	if r.Usage != nil {
		total, cached := r.Usage.openAITokens()
		return openAITelemetryRow(total, cached), true
	}
	if r.Payload != nil && r.Type == "event_msg" && r.Payload.Type == "token_count" {
		usage := r.Payload.Info.LastTokenUsage
		if usage.InputTokens != 0 || usage.CachedInputTokens != 0 {
			return openAITelemetryRow(usage.InputTokens, usage.CachedInputTokens), true
		}
	}
	if r.InputTokens != 0 || r.PromptTokens != 0 || r.CachedTokens != 0 {
		return openAITelemetryRow(firstNonZero(r.InputTokens, r.PromptTokens), r.CachedTokens), true
	}
	return vcachegov.TelemetryRow{}, false
}

func (r vcacheTelemetryJSONLRow) hasClaudeCounters() bool {
	return r.CacheCreationInputTokens != 0 ||
		r.CacheReadInputTokens != 0 ||
		r.Ephemeral1hInputTokens != 0 ||
		r.Ephemeral5mInputTokens != 0
}

func (u vcacheOpenAIUsage) openAITokens() (float64, float64) {
	total := firstNonZero(u.InputTokens, u.PromptTokens)
	cached := firstNonZero(u.InputTokensDetails.CachedTokens, u.PromptTokensDetails.CachedTokens)
	return total, cached
}

func openAITelemetryRow(total, cached float64) vcachegov.TelemetryRow {
	if total < 0 {
		total = 0
	}
	if cached < 0 {
		cached = 0
	}
	if cached > total {
		cached = total
	}
	return vcachegov.TelemetryRow{
		InputTokens:          total - cached,
		CacheReadInputTokens: cached,
	}
}

func firstNonZero(values ...float64) float64 {
	for _, v := range values {
		if v != 0 {
			return v
		}
	}
	return 0
}
