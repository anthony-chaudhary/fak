package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
	"github.com/anthony-chaudhary/fak/internal/toollint"
)

// fak policy  -  author and validate the deployable capability floor. --dump emits
// the built-in DefaultPolicy as a manifest (the starting point an adopter edits);
// --check validates a manifest against the closed refusal vocabulary and prints
// the floor it admits, so a misconfigured policy is caught BEFORE it gates a run.
func cmdPolicy(argv []string) {
	fs := flag.NewFlagSet("policy", flag.ExitOnError)
	dump := fs.Bool("dump", false, "write the built-in DefaultPolicy as a manifest to stdout")
	check := fs.String("check", "", "validate a manifest file and print the floor it admits")
	_ = fs.Parse(argv)

	switch {
	case *dump:
		os.Stdout.Write(policy.FromPolicy(adjudicator.DefaultPolicy()).JSON())
	case *check != "":
		rt, err := policy.LoadRuntime(*check)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak policy:", err)
			os.Exit(1)
		}
		fmt.Printf("OK  %s  (manifest valid; every deny cites a closed-vocabulary reason)\n\n%s", *check, policy.SummaryRuntime(rt))
	default:
		fmt.Fprintln(os.Stderr, "fak policy: pass --dump (emit the default manifest) or --check FILE (validate one)")
		os.Exit(2)
	}
}

// fak lint  -  the STATIC tool linter. The kernel never trusts a tool's self-declared
// annotations: it re-checks them every call and silently does the safe thing (the
// vDSO overrides a lying readOnlyHint from the name, pre-flight re-validates args).
// This verb is the definition-time DUAL of those call-time re-checks: it runs once
// over the configured tool surface and says OUT LOUD what the runtime would only
// ever whisper to itself  -  a dead cache hint, an unreachable pure registration, a
// canned answer for a write-shaped tool, a schema the model is shown but the kernel
// never enforces. Exit 1 on an error-severity finding (or any finding with
// --strict), so it can gate a build.
func cmdLint(argv []string) {
	fs := flag.NewFlagSet("lint", flag.ExitOnError)
	asJSON := fs.Bool("json", false, "emit findings as JSON")
	strict := fs.Bool("strict", false, "exit non-zero on ANY finding (info/warn too), not just errors")
	kernelOnly := fs.Bool("kernel-only", false, "lint only the kernel registries (skip the agent hint classifier + model-facing catalog)")
	_ = fs.Parse(argv)

	var facts []toollint.ToolFacts
	if *kernelOnly {
		facts = toollint.FromKernel()
	} else {
		agent.Configure() // register the agent's schemas, grammar, and engine first
		facts = agent.LintFacts()
	}
	rep := toollint.Lint(facts)

	if *asJSON {
		type jf struct {
			Code      string `json:"code"`
			Severity  string `json:"severity"`
			Tool      string `json:"tool"`
			Message   string `json:"message"`
			Mechanism string `json:"mechanism"`
		}
		rows := make([]jf, 0, len(rep.Findings))
		for _, f := range rep.Findings {
			rows = append(rows, jf{string(f.Code), f.Severity.String(), f.Tool, f.Message, f.Mechanism})
		}
		b, _ := json.MarshalIndent(map[string]any{
			"tools":    len(facts),
			"findings": rows,
			"errors":   rep.Errors(),
			"warnings": rep.Warnings(),
			"infos":    rep.Infos(),
		}, "", "  ")
		fmt.Println(string(b))
	} else {
		for _, f := range rep.Findings {
			fmt.Printf("%s  %-5s  %-22s  %s\n          %s\n", f.Code, f.Severity.String(), f.Tool, f.Message, f.Mechanism)
		}
		if rep.Clean() {
			fmt.Printf("lint clean: %d tool(s), no findings\n", len(facts))
		} else {
			fmt.Printf("\n%d tool(s): %d error, %d warn, %d info\n", len(facts), rep.Errors(), rep.Warnings(), rep.Infos())
		}
	}

	if code := lintExitCode(rep, *strict); code != 0 {
		os.Exit(code)
	}
}

// lintExitCode is the PURE exit-code contract for `fak lint`, factored out so it is
// unit-testable without os.Exit: 1 on any error-severity finding, or  -  under
// --strict  -  on ANY finding at all (the "gate a build on a clean surface" mode the
// help text and cmdLint doc both promise). 0 otherwise.
func lintExitCode(rep toollint.Report, strict bool) int {
	if rep.Errors() > 0 || (strict && !rep.Clean()) {
		return 1
	}
	return 0
}

// fak serve  -  the GATEWAY. It fronts the kernel over an OpenAI-compatible HTTP
// surface and MCP so an agent in ANY language can route its tool calls through the
// in-process syscall boundary without writing Go. The gateway is Go and ON the
// request path (it adjudicates)  -  in-direction; non-Go CLIENTS live in the
// adopter's repo. Construction mirrors cmdAgent: registrations is already imported
// (so the resolver + full adjudicator chain are wired), the capability floor is
// installed fail-loud, and the kernel is built bound to a registered engine.
// resolveRequiredKey resolves a secret the operator REQUIRED by naming an env
// var via a --...-key-env flag. When the flag is unset (empty name) auth was not
// requested, so it returns ok=true with an empty key. But when the flag names an
// env var that is unset or empty, it returns ok=false: the operator asked for
// auth and the secret did not land (typo, un-propagated CI env, k8s Secret
// mis-mount, pod restarted without it). For an agent kernel the safe
// default is to fail CLOSED  -  refuse to start  -  not to warn and silently serve
// unauthenticated. The lookup is injected so the decision is unit-testable
// without touching process env. (issue #213-class fail-open fix; see #255.)
func resolveRequiredKey(envName string, lookup func(string) string) (key string, ok bool) {
	if envName == "" {
		return "", true // flag not set: auth not requested.
	}
	v := lookup(envName)
	if v == "" {
		return "", false // requested but missing: caller must fail closed.
	}
	return v, true
}

// fak hook  -  the spawned-hook decide transport (A/B baseline). Reads one call
// from stdin, folds the adjudicator chain, writes the verdict to stdout.
func cmdHook() {
	var c bench.Call
	if err := json.NewDecoder(os.Stdin).Decode(&c); err != nil {
		// an empty/invalid call still exercises the spawn+decide path
		c = bench.Call{Tool: "noop"}
	}
	res := abi.ActiveResolver()
	args := []byte(c.Args)
	if len(args) == 0 {
		args = []byte("{}")
	}
	ref, _ := res.Put(ctx(), args)
	tc := &abi.ToolCall{Tool: c.Tool, Args: ref, Meta: c.Meta}
	v := kernel.Fold(ctx(), abi.AdjudicatorsFor(tc), tc)
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"kind": verdictName(v.Kind), "reason": abi.ReasonName(v.Reason), "by": v.By,
	})
}

func printReport(rep *metrics.Report, path string) {
	fmt.Printf("== fak bench: %s ==\n", rep.Provenance.SliceID)
	fmt.Printf("in-process adjudication p50 : %d ns\n", rep.On.P50Ns)
	fmt.Printf("spawned-hook        p50     : %d ns (%.3f ms, n=%d)\n",
		rep.Baseline.P50Ns, float64(rep.Baseline.P50Ns)/1e6, rep.Baseline.Calls)
	if rep.Baseline.P50Ns > 0 && rep.On.P50Ns > 0 {
		fmt.Printf("fusion speedup (p50)        : %.0fx\n", float64(rep.Baseline.P50Ns)/float64(rep.On.P50Ns))
	}
	fmt.Printf("PRIMARY GATE                : %s  (%s)\n", rep.GatePrimary, rep.PrimaryDetail)
	fmt.Printf("secondary token delta       : %.2f%% (soft, never gates)\n", rep.TokenDeltaPct)
	fmt.Printf("vdso hit-rate               : %.3f   pollution-rate: %.3f\n",
		rep.KPIs.VDSOHitRate, rep.KPIs.ContextPollutionRate)
	fmt.Printf("workload hash               : %s   live seam: %s\n",
		rep.Provenance.WorkloadHash, rep.LiveSeam)
	fmt.Printf("report written              : %s\n", path)
}

func traceDir() string { return testdataDir("tau2") }

func turnTaxDir() string { return testdataDir("turntax") }

// testdataDir resolves the testdata/<name> directory relative to cwd first, then
// the executable dir; it falls back to the cwd-relative path when neither exists.
// testdata sits next to the module root.
func testdataDir(name string) string {
	if _, err := os.Stat(filepath.Join("testdata", name)); err == nil {
		return filepath.Join("testdata", name)
	}
	if exe, err := os.Executable(); err == nil {
		d := filepath.Join(filepath.Dir(exe), "testdata", name)
		if _, err := os.Stat(d); err == nil {
			return d
		}
	}
	return filepath.Join("testdata", name)
}

// resolveSuite turns a --suite NAME into its trace path under dir, but FAILS LOUD
// and actionable when the file is absent: a cold-start user (or agent) who follows
// the help's `--suite NAME` and guesses a name (e.g. "default") otherwise hits a raw
// `open testdata\...\NAME.json: cannot find the file specified` with no hint of what
// IS valid. Instead we list the available suites (the *.json basenames in dir) so the
// next command is obvious. An explicit --trace PATH bypasses this (the caller owns it).
// Returns the path unchanged when the file exists; exits 2 with the suite list when not.
func resolveSuite(dir, suite string) string {
	path := filepath.Join(dir, suite+".json")
	if _, err := os.Stat(path); err == nil {
		return path
	}
	avail := availableSuites(dir)
	if len(avail) == 0 {
		fmt.Fprintf(os.Stderr, "fak: unknown suite %q — no suites found under %s (pass --trace PATH to load a trace directly)\n", suite, dir)
	} else {
		fmt.Fprintf(os.Stderr, "fak: unknown suite %q — available: %s (or pass --trace PATH)\n", suite, strings.Join(avail, ", "))
	}
	os.Exit(2)
	return path // unreachable
}

// availableSuites lists the suite NAMES (the *.json basenames, extension stripped)
// in dir, sorted, so an unknown-suite error can name the real choices. Empty on a
// missing/unreadable dir (the caller reports that case).
func availableSuites(dir string) []string {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range ents {
		if name := e.Name(); !e.IsDir() && strings.HasSuffix(name, ".json") {
			out = append(out, strings.TrimSuffix(name, ".json"))
		}
	}
	sort.Strings(out)
	return out
}

func verdictName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	case abi.VerdictIndeterminate:
		return "INDETERMINATE"
	}
	return "K" + strconv.Itoa(int(k))
}

func statusName(s abi.Status) string {
	switch s {
	case abi.StatusOK:
		return "OK"
	case abi.StatusError:
		return "ERR"
	case abi.StatusPending:
		return "PEND"
	}
	return "?"
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak:", err)
		os.Exit(1)
	}
}

// embeddedGGUFTokenizer builds a tokenizer straight from the GGUF's own
// tokenizer.ggml.* metadata, mirroring cmd/simpledemo's embedded path. It lets
// `fak serve --gguf` serve real in-kernel chat without a separate tokenizer.json.
// Returns an error (not a panic) when the checkpoint embeds no usable BPE tokenizer,
// so the caller can fall back to the MockPlanner instead of aborting startup.
func embeddedGGUFTokenizer(ggufPath string) (*tokenizer.Tokenizer, error) {
	f, err := ggufload.Open(ggufPath)
	if err != nil {
		return nil, err
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		return nil, fmt.Errorf("no embedded BPE tokenizer in %s", filepath.Base(ggufPath))
	}
	return tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
}

func cmdNewModel(argv []string) {
	fs := flag.NewFlagSet("new-model", flag.ExitOnError)
	fs.SetOutput(os.Stderr)
	family := fs.String("family", "", "family name, lowercase (e.g. myfamily)")
	topology := fs.String("topology", "identity", "topology: prenorm, postnorm, parallel, or identity")
	dryRun := fs.Bool("dry-run", false, "print scaffold without writing files")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	if err := fs.Parse(argv); err != nil {
		os.Exit(2)
	}

	if *family == "" {
		fmt.Fprintln(os.Stderr, "fak new-model: --family is required")
		fmt.Fprintln(os.Stderr, "usage: fak new-model --family <name> [--topology <topology>] [--dry-run] [--json]")
		os.Exit(2)
	}

	res, err := newmodel.Run(newmodel.Scaffold{
		Family:   *family,
		Topology: *topology,
		DryRun:   *dryRun,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak new-model: %v\n", err)
		os.Exit(1)
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(res)
		return
	}

	fmt.Printf("=== Scaffolding model family '%s' (topology: %s) ===\n\n", res.Family, res.Topology)
	fmt.Println("Files to edit:")
	for _, e := range res.Edits {
		fmt.Printf("  - %s\n", e)
	}
	fmt.Println("\nNext steps:")
	for _, s := range res.NextSteps {
		fmt.Printf("%s\n", s)
	}
}
