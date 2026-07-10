package main

// trajctl_score.go — issue #2543, the W1 rung's operator door: `fak trajctl
// score` runs ONE registered scorer over ONE objective and appends its rows to
// the ledger. `--method judge` is the gateway-backed judge scorer
// (internal/trajctl/judgescorer.go): a pinned-schema, forced-tool-choice
// verdict call scoring current state against the objective statement, emitting
// a W1 row that ranks honestly below W2/W3 evidence.
//
// The impurities live here, at the call site, so the trajctl folds stay pure:
// the clock stamps the rows, the env supplies the gateway bearer, and the
// GatewayJudgeClient owns the network. The per-call token budget cap defaults
// to trajctl.DefaultJudgeMaxCallTokens and is enforced by the scorer on BOTH
// the request (max_tokens ceiling) and the returned usage (fail-closed).
//
// This subcommand never runs on the stop-hook path — a judge call costs real
// tokens, so scoring by judge is an explicit operator act (issue #2543 keeps
// stop-hook judging opt-in and out of scope).

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// trajctlJudgeMethodAlias is the operator-facing spelling of the judge method
// (`--method judge`); it resolves to the registry's stable method id, which is
// what every emitted row carries.
const trajctlJudgeMethodAlias = "judge"

func runTrajctlScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak trajctl score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	objective := fs.String("objective", "", "objective id to score (required)")
	method := fs.String("method", "", "scorer method: "+trajctlJudgeMethodAlias+" (or the full registry id)")
	state := fs.String("state", "", "current-state description handed to the judge (default: derived from the evidence window)")
	maxCallTokens := fs.Int("max-call-tokens", trajctl.DefaultJudgeMaxCallTokens, "per-call token budget cap, enforced on request and return")
	baseURL := fs.String("base-url", "", "OpenAI-compatible API root of the gateway (required for --method judge)")
	model := fs.String("model", "", "model id for the verdict call (empty: the gateway's default)")
	apiKeyEnv := fs.String("api-key-env", "FAK_GATEWAY_KEY", "env var NAMING the gateway bearer (never the secret itself)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+trajctl.DefaultLedgerRel+")")
	asJSON := fs.Bool("json", false, "emit the appended score rows as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *objective == "" {
		fmt.Fprintln(stderr, "fak trajctl score: --objective is required")
		return 2
	}
	if *method == "" {
		fmt.Fprintln(stderr, "fak trajctl score: --method is required (try --method "+trajctlJudgeMethodAlias+")")
		return 2
	}
	resolved := *method
	if resolved == trajctlJudgeMethodAlias {
		resolved = trajctl.JudgeScorerMethod
	}
	if resolved == trajctl.JudgeScorerMethod && strings.TrimSpace(*baseURL) == "" {
		fmt.Fprintln(stderr, "fak trajctl score: --base-url is required for --method judge (the verdict call needs a gateway)")
		return 2
	}

	reg := trajctl.NewRegistry()
	judge := trajctl.NewJudgeScorer(&trajctl.GatewayJudgeClient{
		BaseURL: *baseURL,
		APIKey:  os.Getenv(*apiKeyEnv),
		Model:   *model,
	}, *maxCallTokens)
	judge.State = *state
	if err := reg.Register(judge); err != nil {
		fmt.Fprintf(stderr, "fak trajctl score: %v\n", err)
		return 1
	}

	scorer, ok := reg.Get(resolved)
	if !ok {
		fmt.Fprintf(stderr, "fak trajctl score: unknown method %q (registered: %s)\n",
			*method, strings.Join(reg.Methods(), ", "))
		return 2
	}

	path := trajctlLedgerPath(*ledger)
	st := trajctl.Fold(trajctl.ReadLedgerFile(path))
	obj, found := st.Objectives[*objective]
	if !found {
		fmt.Fprintf(stderr, "fak trajctl score: unknown objective %q\n", *objective)
		return 1
	}

	win := trajctl.EvidenceWindow{
		PriorScores: st.Scores,
		UnixMillis:  time.Now().UnixMilli(),
	}
	rows := scorer.Score(obj, win)
	if len(rows) == 0 {
		fmt.Fprintf(stderr, "fak trajctl score: %s produced no row for %q (fail-closed: judge error, over-budget return, or closed objective credits no progress)\n",
			resolved, obj.ID)
		return 1
	}
	for _, row := range rows {
		if err := trajctl.Append(path, trajctl.ScoreRecord(row)); err != nil {
			fmt.Fprintf(stderr, "fak trajctl score: %v\n", err)
			return 1
		}
	}
	if *asJSON {
		return trajctlEmitJSON(stdout, stderr, rows)
	}
	for _, row := range rows {
		fmt.Fprintf(stdout, "%s %s=%.2f witness=%s (method=%s v%s)\n",
			row.ObjectiveID, "progress", row.Value, row.Witness, row.Method, row.Version)
	}
	return 0
}
