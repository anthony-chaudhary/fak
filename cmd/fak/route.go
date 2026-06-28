package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/maputil"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// fak route — the MODEL-ROUTING oracle. It is to model routing what `fak
// preflight` is to the capability floor: load a declarative routing manifest, then
// for ONE classified subject print which model — or which ensemble of models +
// reduction — the policy selects. Model routing is first-class at EVERY level: the
// routed unit is an ASPECT (the whole request, a single tool call, a sub-query, a
// planner state, a reasoning step), so one request routes different aspects to
// different models, and an ensemble is a first-class plan.
//
//	fak route --aspect tool_call --tool refund_payment        (uses the built-in manifest)
//	fak route --manifest FILE --aspect step --complexity high
//	fak route --manifest FILE --aspect tool_call --tool refund_payment --simulate "yes,no,yes"
//	fak route --dump            (emit the built-in manifest to edit)
//	fak route --check FILE      (validate a manifest)
//
// --simulate "<out>[@<score>],…" feeds STAND-IN member outputs through the chosen
// plan's reduction (vote / best-of / all-reduce / concat / first) and prints the
// rolled-up result — the ensemble half, proven end to end with no model in the loop.
func cmdRoute(argv []string) { os.Exit(runRoute(os.Stdout, os.Stderr, argv)) }

// runRoute is the testable core: it returns the process exit code (0 ok, 1 a
// manifest/load error, 2 a usage error) instead of calling os.Exit, and takes its
// streams explicitly.
func runRoute(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("route", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath := fs.String("manifest", "", "load the routing policy from a manifest (default: built-in DefaultManifest)")
	dump := fs.Bool("dump", false, "write the built-in DefaultManifest as a manifest to stdout")
	check := fs.String("check", "", "validate a manifest file and print the routing surface it admits")
	aspect := fs.String("aspect", string(modelroute.AspectRequest), "the aspect being routed: request|tool_call|query|state|step|scout|<custom>")
	tool := fs.String("tool", "", "tool name (when aspect=tool_call)")
	promptTokens := fs.Int("prompt-tokens", 0, "estimated prompt length in tokens")
	latency := fs.String("latency", "", "latency class: interactive|batch (empty = unconstrained)")
	complexity := fs.String("complexity", "", "complexity: low|medium|high (empty = unconstrained)")
	labels := fs.String("labels", "", "subject labels as k=v[,k=v...] (domain, lang, tenant, …)")
	simulate := fs.String("simulate", "", "stand-in member outputs '<out>[@score],…' to fold through the plan's reduction")
	frontier := fs.String("frontier", "", "SOTA baseline model for the rough usage estimate (default: an Opus-class frontier anchor, $3/$15 per Mtok)")
	prices := fs.String("prices", "", "override the rough price book: model=in/out[,model=N,...] (e.g. small=0.25/1.25,large=3/15)")
	accounts := fs.String("accounts", "", "load an account roster (the switcher) and bind each routed model id to a provider/account/upstream target")
	accountsDump := fs.Bool("accounts-dump", false, "write the built-in DefaultRoster (the account-switcher starter) to stdout")
	accountsCheck := fs.String("accounts-check", "", "validate an account roster and print the account + binding surface")
	asJSON := fs.Bool("json", false, "emit the decision (and any reduction) as JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	// The rough cost lens (usage saved vs the SOTA frontier baseline): the built-in
	// ladder, overlaid with any --prices the operator supplies. Built before the
	// switch so --check can show the surface's cost shape too.
	book := modelroute.DefaultPrices()
	if *prices != "" {
		over, err := modelroute.ParsePrices(*prices)
		if err != nil {
			fmt.Fprintln(stderr, "fak route:", err)
			return 2
		}
		book = book.Overlay(over)
	}

	switch {
	case *dump:
		stdout.Write(modelroute.DefaultManifest().JSON())
		return 0
	case *check != "":
		m, err := modelroute.LoadManifest(*check)
		if err != nil {
			fmt.Fprintln(stderr, "fak route:", err)
			return 1
		}
		fmt.Fprintf(stdout, "OK  %s  (manifest valid; %d rule(s), fail-closed default -> %s)\n\n%s",
			*check, len(m.Rules), m.Default.Primary(), routeSummary(m, book, *frontier))
		return 0
	case *accountsDump:
		stdout.Write(modelroute.DefaultRoster().JSON())
		return 0
	case *accountsCheck != "":
		r, err := modelroute.LoadRoster(*accountsCheck)
		if err != nil {
			fmt.Fprintln(stderr, "fak route:", err)
			return 1
		}
		fmt.Fprintf(stdout, "OK  %s  (roster valid; %d account(s), %d binding(s), default -> %s)\n\n%s",
			*accountsCheck, len(r.Accounts), len(r.Bindings), orDash(r.Default), accountSurface(r))
		return 0
	}

	// Resolve the manifest: an explicit file, else the built-in default.
	m := modelroute.DefaultManifest()
	if *manifestPath != "" {
		loaded, err := modelroute.LoadManifest(*manifestPath)
		if err != nil {
			fmt.Fprintln(stderr, "fak route:", err)
			return 1
		}
		m = loaded
		fmt.Fprintf(stderr, "fak: loaded routing policy from %s\n", *manifestPath)
	}

	subj := modelroute.Subject{
		Aspect:       modelroute.Aspect(*aspect),
		Tool:         *tool,
		PromptTokens: *promptTokens,
		Latency:      modelroute.Latency(*latency),
		Complexity:   modelroute.Complexity(*complexity),
		Labels:       parseLabels(*labels),
	}
	d := m.Route(subj)
	sav := modelroute.EstimateSavings(d, book, *frontier)

	var red *modelroute.Result
	if *simulate != "" {
		r, err := simulateReduce(d.Plan, *simulate)
		if err != nil {
			fmt.Fprintln(stderr, "fak route:", err)
			return 2
		}
		red = &r
	}

	// The account switcher: bind each routed model id (scout + every member) to the
	// concrete provider/account/upstream that serves it. This is the COMPOSITION:
	// the route picks the abstract model ids, the roster says whose account runs them.
	var bound *modelroute.ResolvedPlan
	if *accounts != "" {
		r, err := modelroute.LoadRoster(*accounts)
		if err != nil {
			fmt.Fprintln(stderr, "fak route:", err)
			return 1
		}
		fmt.Fprintf(stderr, "fak: loaded account roster from %s\n", *accounts)
		resolved, err := r.ResolveDecision(d)
		if err != nil {
			fmt.Fprintln(stderr, "fak route:", err)
			return 1
		}
		bound = &resolved
	}

	if *asJSON {
		fmt.Fprintln(stdout, routeJSON(d, red, sav, bound))
		return 0
	}
	printRoute(stdout, d, red, sav, bound)
	return 0
}

// parseLabels turns "k=v,k2=v2" into a map. Malformed pairs are skipped.
func parseLabels(s string) map[string]string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	out := map[string]string{}
	for _, pair := range strings.Split(s, ",") {
		kv := strings.SplitN(strings.TrimSpace(pair), "=", 2)
		if len(kv) == 2 && kv[0] != "" {
			out[strings.TrimSpace(kv[0])] = strings.TrimSpace(kv[1])
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// simulateReduce zips the comma-separated stand-in outputs onto the plan's members
// (synthesizing a member when there are more outputs than members) and folds them
// with the plan's reduction. A token "out@0.9" carries a best_of score. Member
// ORDER is preserved into the vote slice (the deterministic-reduce contract).
func simulateReduce(p modelroute.Plan, spec string) (modelroute.Result, error) {
	toks := strings.Split(spec, ",")
	votes := make([]modelroute.Vote, 0, len(toks))
	for i, t := range toks {
		out := strings.TrimSpace(t)
		var score float64
		if at := strings.LastIndex(out, "@"); at >= 0 {
			if f, err := strconv.ParseFloat(out[at+1:], 64); err == nil {
				score = f
				out = out[:at]
			}
		}
		var mem modelroute.Member
		if i < len(p.Members) {
			mem = p.Members[i]
		} else {
			mem = modelroute.Member{Model: fmt.Sprintf("member-%d", i)}
		}
		votes = append(votes, modelroute.Vote{Member: mem, Output: out, Score: score})
	}
	reduce := p.Reduce
	if !p.IsEnsemble() {
		reduce = modelroute.ReduceFirst // a single pick has nothing to fold
	}
	return modelroute.Combine(reduce, votes)
}

// printRoute renders the decision (and any simulated reduction) for a human.
func printRoute(w io.Writer, d modelroute.Decision, red *modelroute.Result, sav modelroute.Savings, bound *modelroute.ResolvedPlan) {
	s := d.Subject
	fmt.Fprintln(w, "== fak route ==")
	fmt.Fprintf(w, "subject     : aspect=%s tool=%s prompt_tokens=%d latency=%s complexity=%s%s\n",
		s.Aspect, orDash(s.Tool), s.PromptTokens, orDash(string(s.Latency)), orDash(string(s.Complexity)), labelStr(s.Labels))
	if d.Matched {
		fmt.Fprintf(w, "matched rule: %s\n", d.RuleName)
	} else {
		fmt.Fprintf(w, "matched rule: <none> -> fail-closed default\n")
	}
	if d.Plan.IsEnsemble() {
		fmt.Fprintf(w, "plan        : ENSEMBLE  reduce=%s%s\n", d.Plan.Reduce, scoutStr(d.Plan.Scout))
		for _, mem := range d.Plan.Members {
			fmt.Fprintf(w, "              - %-20s weight=%s%s\n", mem.Model, weightStr(mem.Weight), roleStr(mem.Role))
		}
	} else {
		fmt.Fprintf(w, "plan        : PICK -> %s%s   (the abi.ToolCall.Engine value)\n", d.Plan.Primary(), scoutStr(d.Plan.Scout))
	}
	if d.Plan.Reason != "" {
		fmt.Fprintf(w, "reason      : %s\n", d.Plan.Reason)
	}
	fmt.Fprintf(w, "%s\n", sav.Headline())
	if bound != nil {
		fmt.Fprintf(w, "\n-- account binding (the switcher: which of YOUR accounts serves each id) --\n")
		if bound.Scout != nil {
			printTarget(w, "scout", *bound.Scout)
		}
		for _, tg := range bound.Members {
			printTarget(w, "member", tg)
		}
	}
	if red != nil {
		fmt.Fprintf(w, "\n-- simulated reduction (stand-in member outputs) --\n")
		fmt.Fprintf(w, "reduce=%s  members=%d\n", red.Reduce, red.Members)
		if len(red.Tally) > 0 {
			for _, k := range sortedTally(red.Tally) {
				fmt.Fprintf(w, "  tally  %-20s %.2f\n", k, red.Tally[k])
			}
		}
		if red.Winner != "" {
			fmt.Fprintf(w, "winner : %s\n", red.Winner)
		}
		fmt.Fprintf(w, "output : %s\n", red.Output)
	}
}

// printTarget renders one resolved account binding (a routed id -> the concrete
// provider/account/upstream that serves it). It prints the credential ENV-VAR NAME,
// never a secret, and labels the route local (residency-exempt) or remote.
func printTarget(w io.Writer, role string, t modelroute.Target) {
	loc := "remote"
	if t.Local() {
		loc = "local "
	}
	cred := t.CredEnv
	if cred == "" {
		cred = "-"
	}
	fmt.Fprintf(w, "  %-7s %-14s -> %-14s %-16s upstream=%-16s key=$%-20s [%s] %s\n",
		role, t.Model, t.Account, string(t.Kind), t.UpstreamModel, cred, loc, t.EngineRoute())
}

// accountSurface renders a roster's accounts + bindings as an operator-readable table
// for `--accounts-check`, resolving each binding so the whole switch is visible at a
// glance. It never prints a secret - only the credential env-var name.
func accountSurface(r modelroute.Roster) string {
	var sb strings.Builder
	sb.WriteString("accounts:\n")
	for _, a := range r.Accounts {
		loc := "remote"
		if a.Kind == modelroute.KindLocal {
			loc = "local "
		}
		base := a.BaseURL
		if base == "" {
			base = modelroute.KindBaseURL(a.Kind)
		}
		cred := a.CredEnv
		if cred == "" {
			cred = "-"
		}
		sb.WriteString(fmt.Sprintf("  %-16s %-16s [%s] key=$%-20s %s\n", a.ID, string(a.Kind), loc, cred, base))
	}
	sb.WriteString("\nbindings (model id -> account / upstream wire model):\n")
	for _, b := range r.Bindings {
		t, err := r.Resolve(b.Model)
		if err != nil {
			sb.WriteString(fmt.Sprintf("  %-14s -> ERROR %v\n", b.Model, err))
			continue
		}
		sb.WriteString(fmt.Sprintf("  %-14s -> %-16s upstream=%-16s %s\n", b.Model, t.Account, t.UpstreamModel, t.EngineRoute()))
	}
	if r.Default != "" {
		sb.WriteString(fmt.Sprintf("\n  (any unbound id) -> default account %q\n", r.Default))
	}
	sb.WriteString("\nresidency: a 'local' route is residency-exempt (on-box); a 'remote' route is gated " +
		"by the engine-residency floor (a tenant/sensitive payload bound for it is denied). The credential is the\n" +
		"named env var, dereferenced only at dispatch - never stored here.\n")
	return sb.String()
}

// routeJSON renders the decision (and any reduction) as a stable JSON object.
func routeJSON(d modelroute.Decision, red *modelroute.Result, sav modelroute.Savings, bound *modelroute.ResolvedPlan) string {
	type memberJSON struct {
		Model  string  `json:"model"`
		Weight float64 `json:"weight,omitempty"`
		Role   string  `json:"role,omitempty"`
	}
	mems := make([]memberJSON, 0, len(d.Plan.Members))
	for _, mem := range d.Plan.Members {
		mems = append(mems, memberJSON{mem.Model, mem.Weight, mem.Role})
	}
	obj := map[string]any{
		"subject": map[string]any{
			"aspect":        string(d.Subject.Aspect),
			"tool":          d.Subject.Tool,
			"prompt_tokens": d.Subject.PromptTokens,
			"latency":       string(d.Subject.Latency),
			"complexity":    string(d.Subject.Complexity),
			"labels":        d.Subject.Labels,
		},
		"matched":  d.Matched,
		"rule":     d.RuleName,
		"ensemble": d.Plan.IsEnsemble(),
		"primary":  d.Plan.Primary(),
		"reduce":   string(d.Plan.Reduce),
		"scout":    d.Plan.Scout,
		"members":  mems,
		"reason":   d.Plan.Reason,
		"usage":    sav,
	}
	if red != nil {
		obj["reduction"] = map[string]any{
			"reduce":  string(red.Reduce),
			"output":  red.Output,
			"winner":  red.Winner,
			"tally":   red.Tally,
			"members": red.Members,
		}
	}
	if bound != nil {
		bj := map[string]any{"members": targetsJSON(bound.Members)}
		if bound.Scout != nil {
			bj["scout"] = targetJSON(*bound.Scout)
		}
		obj["binding"] = bj
	}
	b, _ := json.MarshalIndent(obj, "", "  ")
	return string(b)
}

// targetJSON renders one resolved account binding as a stable JSON object. It carries
// the credential ENV-VAR NAME (cred_env), never a secret, plus the local/remote flag
// and the abi.ToolCall.Engine route the dispatcher would write.
func targetJSON(t modelroute.Target) map[string]any {
	return map[string]any{
		"model":          t.Model,
		"account":        t.Account,
		"kind":           string(t.Kind),
		"base_url":       t.BaseURL,
		"cred_env":       t.CredEnv,
		"upstream_model": t.UpstreamModel,
		"local":          t.Local(),
		"engine_route":   t.EngineRoute(),
	}
}

func targetsJSON(ts []modelroute.Target) []map[string]any {
	out := make([]map[string]any, 0, len(ts))
	for _, t := range ts {
		out = append(out, targetJSON(t))
	}
	return out
}

// routeSummary renders a manifest's rules as an operator-readable table, with a
// rough cost tag per rule (cheaper / premium / baseline vs the SOTA frontier) so
// the whole policy's spend shape is visible at --check time.
func routeSummary(m modelroute.Manifest, book modelroute.PriceBook, frontier string) string {
	var sb strings.Builder
	sb.WriteString("routing surface:\n")
	row := func(name, match, plan string, p modelroute.Plan) {
		cost := costTag(modelroute.EstimateSavings(modelroute.Decision{Plan: p}, book, frontier))
		sb.WriteString(fmt.Sprintf("  %-22s %-26s %-44s [%s]\n", name, match, plan, cost))
	}
	for _, r := range m.Rules {
		plan := "PICK -> " + r.Plan.Primary()
		if r.Plan.IsEnsemble() {
			plan = fmt.Sprintf("ENSEMBLE(%s) -> %s", r.Plan.Reduce, strings.Join(r.Plan.Models(), "+"))
		}
		row(r.Name, matchStr(r.Match), plan, r.Plan)
	}
	row("(default)", "*", "PICK -> "+m.Default.Primary(), m.Default)
	fb := frontier
	if fb == "" {
		fb = "an Opus-class frontier ($3/$15 per Mtok)"
	}
	sb.WriteString(fmt.Sprintf("\ncost lens (rough, vs %s; overridable with --prices): "+
		"save=cheaper than baseline, premium=ensemble runs more compute, n/e=$0 baseline\n", fb))
	return sb.String()
}

// costTag is the compact per-rule cost label for the routeSummary table.
func costTag(s modelroute.Savings) string {
	if !s.Estimable {
		return "n/e"
	}
	switch f := s.SavedOutFrac; {
	case f > 0.005:
		return fmt.Sprintf("save ~%.0f%%", f*100)
	case f < -0.005:
		return fmt.Sprintf("premium +%.0f%%", -f*100)
	default:
		return "baseline"
	}
}

func matchStr(mt modelroute.Match) string {
	var parts []string
	if mt.Aspect != "" {
		parts = append(parts, string(mt.Aspect))
	}
	if mt.Tool != "" {
		parts = append(parts, "tool="+mt.Tool)
	}
	if mt.Latency != "" {
		parts = append(parts, string(mt.Latency))
	}
	if mt.MinComplexity != "" {
		parts = append(parts, ">="+string(mt.MinComplexity))
	}
	if mt.MaxPromptTokens != 0 || mt.MinPromptTokens != 0 {
		parts = append(parts, fmt.Sprintf("tok[%d,%d]", mt.MinPromptTokens, mt.MaxPromptTokens))
	}
	if len(parts) == 0 {
		return "*"
	}
	return strings.Join(parts, " ")
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func labelStr(m map[string]string) string {
	if len(m) == 0 {
		return ""
	}
	keys := maputil.SortedKeys(m)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+m[k])
	}
	return " labels[" + strings.Join(parts, ",") + "]"
}

func weightStr(w float64) string {
	if w <= 0 {
		return "1"
	}
	return strconv.FormatFloat(w, 'g', -1, 64)
}

func roleStr(role string) string {
	if role == "" {
		return ""
	}
	return " role=" + role
}

func scoutStr(scout string) string {
	if scout == "" {
		return ""
	}
	return " scout=" + scout
}

func sortedTally(m map[string]float64) []string {
	out := maputil.SortedKeys(m)
	return out
}
