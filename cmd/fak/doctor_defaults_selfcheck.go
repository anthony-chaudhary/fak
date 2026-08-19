package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const defaultsSelfcheckSchema = "fak.defaults-selfcheck.v1"

type defaultsSelfcheckRow struct {
	Name     string `json:"name"`
	State    string `json:"state"`
	Evidence string `json:"evidence"`
}
type defaultsSelfcheckReport struct {
	Schema    string                 `json:"schema"`
	OK        bool                   `json:"ok"`
	Workspace string                 `json:"workspace"`
	NonFAK    bool                   `json:"non_fak"`
	Rows      []defaultsSelfcheckRow `json:"rows"`
	Postures  []launchPostureReport  `json:"postures"`
}

func runDoctorDefaultsSelfcheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak doctor defaults-selfcheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "existing workspace (default: create a disposable non-FAK repository)")
	asJSON := fs.Bool("json", false, "emit stable JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	root := strings.TrimSpace(*workspace)
	cleanup := func() {}
	if root == "" {
		var err error
		root, err = os.MkdirTemp("", "fak-defaults-nonfak-")
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
		cleanup = func() { _ = os.RemoveAll(root) }
	}
	defer cleanup()
	root, _ = filepath.Abs(root)
	outside, err := os.MkdirTemp("", "fak-defaults-outside-")
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}
	defer os.RemoveAll(outside)
	if err := os.MkdirAll(filepath.Join(root, ".git"), 0o755); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}

	tools, err := agent.RunCodeToolsSelfcheck(context.Background(), root, outside)
	if err != nil {
		fmt.Fprintf(stderr, "defaults selfcheck tools: %v\n", err)
		return 1
	}
	profiles := true
	for _, h := range []string{"claude", "codex"} {
		_, cap, err := injectGuardProfiles([]string{h, "run"}, agentDefaultOutputStyle, agentDefaultWorkProfile, false)
		profiles = profiles && err == nil && cap != nil && cap.OutputDigest != "" && cap.WorkDigest != "" && strings.Contains(cap.ActivationSeam, h)
	}
	gw, err := gateway.RunDefaultsSelfcheck()
	if err != nil {
		fmt.Fprintf(stderr, "defaults selfcheck gateway: %v\n", err)
		return 1
	}
	allTools := tools.ReadOK && tools.WriteOK && tools.EditOK && tools.BashOK && tools.GrepOK && tools.GlobOK && tools.TraversalDeny && tools.EngineCalls >= 6 && tools.Denies == 1
	rows := []defaultsSelfcheckRow{
		{"bounded-repository-tools", state(allTools), fmt.Sprintf("catalog=%v engine_calls=%d denies=%d", tools.Catalog, tools.EngineCalls, tools.Denies)},
		{"caveman-and-ponytail", state(profiles), "captured Claude append-system-prompt and Codex developer_instructions adapters"},
		{"compact-history", state(gw.CompactHistory), "Anthropic fixture request shortened through live compact transform"},
		{"stale-read-elision", state(gw.StaleReadElision), "superseded Read body removed through live transform"},
		{"cold-tool-deferral", state(gw.ColdToolDeferral), "cold custom tool deferred behind Anthropic ToolSearch"},
		{"vcache-anchor", state(gw.VCacheAnchor), "stable system prefix received cache_control"},
		{"minimum-prefix-steering", state(gw.MinimumPrefixGate), "measured floor suppressed uneconomic anchor authoring"},
		{"calibrated-read-pricing", state(gw.ReadPricing), "measured read multiplier changed spend"},
		{"calibrated-write-pricing", state(gw.WritePricing), "measured 5m and 1h multipliers changed spend independently"},
		{"ttl-tier-steering", state(gw.TTLTierSteering), "measured provider retention suppressed paid 1h tier"},
		{"cross-backend-vcache-signals", state(gw.VCacheSignals), "normalized cache-read usage reached live family window"},
		{"openai-cold-tool-deferral", "unsupported", gw.OpenAIColdTools},
	}
	postureOpts := []launchPostureOptions{
		{entrypoint: "agent", provider: "openai", workspace: root, nativeCodeTools: true, outputProfile: agentDefaultOutputStyle, workProfile: agentDefaultWorkProfile, compactHistory: gateway.DefaultCompactHistoryBudget, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true},
		{entrypoint: "guard", harness: "claude", provider: "anthropic", workspace: root, nativeCodeTools: true, outputProfile: agentDefaultOutputStyle, workProfile: agentDefaultWorkProfile, compactHistory: gateway.DefaultCompactHistoryBudget, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true},
		{entrypoint: "guard", harness: "codex", provider: "openai", workspace: root, nativeCodeTools: true, outputProfile: agentDefaultOutputStyle, workProfile: agentDefaultWorkProfile, compactHistory: gateway.DefaultCompactHistoryBudget, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true},
		{entrypoint: "serve", provider: "native", workspace: root, native: true, nativeCodeTools: true, outputProfile: agentDefaultOutputStyle, workProfile: agentDefaultWorkProfile, compactHistory: gateway.DefaultCompactHistoryBudget, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true},
		{entrypoint: "serve", provider: "openai", baseURL: "https://api.openai.com/v1", workspace: root, nativeCodeTools: true, outputProfile: "full", workProfile: "standard", compactHistory: gateway.DefaultCompactHistoryBudget, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true},
	}
	var postures []launchPostureReport
	for _, o := range postureOpts {
		r, e := deriveLaunchPosture(o)
		if e != nil {
			fmt.Fprintln(stderr, e)
			return 2
		}
		postures = append(postures, r)
	}
	ok := true
	for _, r := range rows {
		if r.State == "fail" {
			ok = false
		}
	}
	report := defaultsSelfcheckReport{Schema: defaultsSelfcheckSchema, OK: ok, Workspace: root, NonFAK: true, Rows: rows, Postures: postures}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(report)
	} else {
		renderDefaultsSelfcheck(stdout, report)
	}
	if ok {
		return 0
	}
	return 1
}
func state(ok bool) string {
	if ok {
		return "pass"
	}
	return "fail"
}
func renderDefaultsSelfcheck(w io.Writer, r defaultsSelfcheckReport) {
	fmt.Fprintf(w, "defaults selfcheck: %s workspace=%s\n", map[bool]string{true: "PASS", false: "FAIL"}[r.OK], r.Workspace)
	rows := append([]defaultsSelfcheckRow(nil), r.Rows...)
	sort.SliceStable(rows, func(i, j int) bool { return rows[i].Name < rows[j].Name })
	for _, x := range rows {
		fmt.Fprintf(w, "  %-32s %-11s %s\n", x.Name, strings.ToUpper(x.State), x.Evidence)
	}
	fmt.Fprintf(w, "postures: %d captured at %s\n", len(r.Postures), time.Now().UTC().Format(time.RFC3339))
}
