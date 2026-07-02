package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcachesnapshot"
)

func defaultCodexOpenAIStatus() vcacheCodexOpenAIStatus {
	hasKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")) != ""
	live := "proven (Codex CLI replay artifact)"
	reason := "replay experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl with prove-telemetry; raw OpenAI API probe not run because OPENAI_API_KEY is not present"
	if hasKey {
		reason = "Codex CLI replay artifact is tracked; OPENAI_API_KEY is present, so tools/vcache_openai_probe.py can refresh the optional raw API probe"
	}
	return vcacheCodexOpenAIStatus{
		Verifier:            "ready",
		LiveTelemetry:       live,
		Reason:              reason,
		OpenAIAPIKeyPresent: hasKey,
		CachedTokenFields: []string{
			"usage.input_tokens_details.cached_tokens",
			"usage.prompt_tokens_details.cached_tokens",
			"usage.cached_input_tokens",
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

func defaultVCacheContextAPIStatus() vcacheContextAPIStatus {
	const fixture = "cmd/fak/testdata/guard-trace-context-e2e.json"
	snapshot := vcachesnapshot.DefaultContextPath()
	return vcacheContextAPIStatus{
		Verifier:            "ready",
		HTTP:                "GET /v1/fak/ctxvalue",
		MCPTool:             "fak_context_value",
		AdviceOnly:          true,
		Provenance:          []string{"OBSERVED tokens", "WITNESSED turns/context_events", "FORECAST turns_to_event", "DECISION step_advice"},
		ScoreIntegration:    "context events feed /v1/fak/vcache/score and persisted vcache snapshots only after a guard/serve context event fires",
		NoKeyReplayFixture:  fixture,
		NoKeyReplaySnapshot: snapshot,
		NoKeyReplayCommand:  "fak guard --replay-trace " + fixture + " --replay-wire openai",
		NoKeyWitnessCommand: "fak vcache context-witness",
		DefaultSnapshot:     snapshot,
		NoKeyScoreCommand:   "fak vcache score --json",
		Reason:              "managed-context value API is wired for running guard/serve sessions; it sizes next steps but does not itself prove a context-saving event",
	}
}

func defaultVCacheContextReplayFixturePath() string {
	const repoRootPath = "cmd/fak/testdata/guard-trace-context-e2e.json"
	if _, err := os.Stat(repoRootPath); err == nil {
		return repoRootPath
	}
	const packagePath = "testdata/guard-trace-context-e2e.json"
	if _, err := os.Stat(packagePath); err == nil {
		return packagePath
	}
	return repoRootPath
}

func writeJSON(w io.Writer, v any) int {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return 2
	}
	fmt.Fprintln(w, string(b))
	return 0
}

func writeJSONFile(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func vcacheProofExit(s vcachegov.ProofStatus) int {
	if s == vcachegov.ProofProven {
		return 0
	}
	return 1
}

func formatBreakEven(n int) string {
	switch n {
	case int(^uint(0) >> 1):
		return "never"
	default:
		return fmt.Sprintf("%d", n)
	}
}

func formatObservedPositive(n int) string {
	if n <= 0 {
		return "never"
	}
	return fmt.Sprintf("%d", n)
}

func passFail(ok bool) string {
	if ok {
		return "pass"
	}
	return "fail"
}
