package main

// armbench.go — `fak armbench`, the provenance-locked multi-arm benchmark
// runner (#6676, epic #6674). It is the one command every later benchmark issue
// in that epic consumes, so that a Caveman or Ponytail comparison is executed
// from ONE immutable manifest instead of a bespoke script per comparison.
//
//	fak armbench selfcheck [--json]                     # deterministic spine + every fail-closed proof
//	fak armbench emit-demo --dir <d>                    # write a runnable manifest+corpus pair
//	fak armbench import-fixtures [--suite all]           # pinned Caveman/Ponytail input importer
//	fak armbench validate --manifest <m> [--json]       # refuse an unpinned manifest
//	fak armbench identity --manifest <m> [--json]       # the sha256 that decides comparability
//	fak armbench run --manifest <m> --corpus <c> --out <run.json> [--resume <prior.json>]
//	fak armbench report --run <run.json> [--json]       # per-arm rollup, human or strict JSON
//	fak armbench compare --a <run.json> --b <run.json>  # refuse incomparable runs
//
// Exit codes are uniform across the subcommands: 0 ok, 1 runtime error, 2 usage,
// 3 a refusal from the closed vocabulary (an unpinned manifest, missing raw
// evidence, a bundled capability, an incomparable pair, a failed selfcheck). A
// script can therefore gate on 3 without parsing prose.
//
// WHY A VERB AND NOT A SCRIPT. Every term a benchmark comparison drifts on —
// model snapshot, sampling, corpus, judge, upstream SHA, capability set — is a
// field of the manifest, and the manifest hashes to an identity. Two numbers may
// be placed side by side only when their identities agree on the terms that
// decide what was measured. A per-comparison script cannot enforce that; this
// can, and refuses when it cannot.

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/armbench"
)

func cmdArmbench(argv []string) { os.Exit(runArmbench(os.Stdout, os.Stderr, argv)) }

const armbenchUsage = `usage: fak armbench <selfcheck|emit-demo|import-fixtures|ponytail|ponytail-managed|validate|identity|run|report|compare> [flags]

  selfcheck   run the deterministic fake-provider spine and every fail-closed proof
  emit-demo   write a runnable manifest + corpus pair to a directory
  import-fixtures
              fetch the pinned Caveman/Ponytail allowlists into an out-of-repo CAS
  validate    refuse a manifest that is not fully pinned
  identity    print the manifest identity that decides comparability
  run         execute the arms and write the raw trial ledger
  report      roll a ledger up per arm (human summary or strict JSON)
  compare     refuse two runs whose manifests disagree on what was measured

exit: 0 ok, 1 runtime error, 2 usage, 3 refusal`

// runArmbench is the testable core.
func runArmbench(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, armbenchUsage)
		return 2
	}
	switch argv[0] {
	case "selfcheck":
		return armbenchSelfcheck(stdout, stderr, argv[1:])
	case "emit-demo":
		return armbenchEmitDemo(stdout, stderr, argv[1:])
	case "import-fixtures":
		return armbenchImportFixtures(stdout, stderr, argv[1:])
	case "ponytail":
		return armbenchPonytail(stdout, stderr, argv[1:])
	case "ponytail-managed":
		return armbenchPonytailManaged(stdout, stderr, argv[1:])
	case "validate":
		return armbenchValidate(stdout, stderr, argv[1:])
	case "identity":
		return armbenchIdentity(stdout, stderr, argv[1:])
	case "run":
		return armbenchRun(stdout, stderr, argv[1:])
	case "report":
		return armbenchReport(stdout, stderr, argv[1:])
	case "compare":
		return armbenchCompare(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, armbenchUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak armbench: unknown subcommand %q\n%s\n", argv[0], armbenchUsage)
		return 2
	}
}

func armbenchImportFixtures(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("armbench import-fixtures", flag.ContinueOnError)
	fs.SetOutput(stderr)
	suite := fs.String("suite", string(armbench.FixtureSuiteAll), "fixture suite: caveman, ponytail, or all")
	store := fs.String("store", "", "out-of-repository content-addressed store (default: user cache)")
	asJSON := fs.Bool("json", false, "emit the import report as strict JSON")
	var licenseReviews stringListFlag
	fs.Var(&licenseReviews, "review-license", "repeatable revision-bound license review token")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak armbench import-fixtures: positional arguments are not accepted")
		return 2
	}
	storeRoot := *store
	if storeRoot == "" {
		var err error
		storeRoot, err = armbench.DefaultFixtureStore()
		if err != nil {
			fmt.Fprintln(stderr, "fak armbench import-fixtures:", err)
			return 1
		}
	}
	storeRoot, err := filepath.Abs(storeRoot)
	if err != nil {
		fmt.Fprintln(stderr, "fak armbench import-fixtures:", err)
		return 1
	}
	report, err := armbench.ImportFixtures(context.Background(), armbench.FixtureSuite(*suite), armbench.ImportOptions{
		StoreRoot:      storeRoot,
		WorkspaceRoot:  resolveRoot(""),
		FakVersion:     guardBannerVersion(),
		LicenseReviews: licenseReviews,
	})
	if err != nil {
		var refusal *armbench.RefusalError
		if errors.As(err, &refusal) {
			fmt.Fprintf(stderr, "refused (%s): %s\n", refusal.Reason, refusal)
			return 3
		}
		fmt.Fprintln(stderr, "fak armbench import-fixtures:", err)
		return 1
	}
	if *asJSON {
		blob, err := armbench.MarshalFixtureImportReport(report)
		if err != nil {
			fmt.Fprintln(stderr, "fak armbench import-fixtures:", err)
			return 1
		}
		_, _ = stdout.Write(blob)
		return 0
	}
	fmt.Fprintf(stdout, "store %s\n", storeRoot)
	for _, result := range report.Results {
		manifestPath := filepath.Join(storeRoot, filepath.FromSlash(result.ManifestPath))
		corpusPath := filepath.Join(storeRoot, filepath.FromSlash(result.CorpusPath))
		runPath := filepath.Join(storeRoot, filepath.FromSlash(result.InputDir), "run.json")
		fmt.Fprintf(stdout,
			"%s: %d pinned sources (%d bytes)\nmanifest %s\ncorpus %s\nnext: fak armbench run --manifest %q --corpus %q --out %q\n",
			result.Suite, result.SourceCount, result.SourceBytes,
			manifestPath, corpusPath, manifestPath, corpusPath, runPath)
	}
	return 0
}

func armbenchSelfcheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("armbench selfcheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the selfcheck artifact as strict JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	res, err := armbench.Selfcheck()
	if err != nil {
		fmt.Fprintln(stderr, "fak armbench selfcheck:", err)
		return 1
	}
	if *asJSON {
		b, err := armbench.MarshalSelfcheck(res)
		if err != nil {
			fmt.Fprintln(stderr, "fak armbench selfcheck:", err)
			return 1
		}
		_, _ = stdout.Write(b)
	} else {
		fmt.Fprint(stdout, armbench.HumanSelfcheck(res))
	}
	if !res.OK {
		return 3
	}
	return 0
}

func armbenchEmitDemo(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("armbench emit-demo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "directory to write manifest.json and corpus.json into (required)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *dir == "" {
		fmt.Fprintln(stderr, "fak armbench emit-demo: --dir is required")
		return 2
	}
	if err := os.MkdirAll(*dir, 0o755); err != nil {
		fmt.Fprintln(stderr, "fak armbench emit-demo:", err)
		return 1
	}
	man, err := armbench.MarshalManifest(armbench.DemoManifest())
	if err != nil {
		fmt.Fprintln(stderr, "fak armbench emit-demo:", err)
		return 1
	}
	corpus, err := armbench.MarshalCorpus(armbench.DemoCorpusFile())
	if err != nil {
		fmt.Fprintln(stderr, "fak armbench emit-demo:", err)
		return 1
	}
	manPath := filepath.Join(*dir, "manifest.json")
	corpusPath := filepath.Join(*dir, "corpus.json")
	if err := os.WriteFile(manPath, man, 0o644); err != nil {
		fmt.Fprintln(stderr, "fak armbench emit-demo:", err)
		return 1
	}
	if err := os.WriteFile(corpusPath, corpus, 0o644); err != nil {
		fmt.Fprintln(stderr, "fak armbench emit-demo:", err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote %s\nwrote %s\nnext: fak armbench run --manifest %s --corpus %s --out %s\n",
		manPath, corpusPath, manPath, corpusPath, filepath.Join(*dir, "run.json"))
	return 0
}

func armbenchValidate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("armbench validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifest := fs.String("manifest", "", "path to the manifest JSON (required)")
	asJSON := fs.Bool("json", false, "emit the verdict as strict JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	m, code := loadManifest(stderr, *manifest)
	if code != 0 {
		return code
	}
	err := m.Validate()
	if *asJSON {
		out := map[string]any{"ok": err == nil, "manifest_id": m.ID}
		if err == nil {
			out["identity"] = m.Identity()
		} else {
			out["reason"] = refusalReason(err)
			out["detail"] = err.Error()
		}
		writeArmbenchJSON(stdout, out)
	} else if err == nil {
		fmt.Fprintf(stdout, "ok: %s is fully pinned\nidentity %s\n", m.ID, m.Identity())
	} else {
		fmt.Fprintf(stderr, "refused (%s): %s\n", refusalReason(err), err)
	}
	if err != nil {
		return 3
	}
	return 0
}

func armbenchIdentity(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("armbench identity", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifest := fs.String("manifest", "", "path to the manifest JSON (required)")
	asJSON := fs.Bool("json", false, "emit the identity as strict JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	m, code := loadManifest(stderr, *manifest)
	if code != 0 {
		return code
	}
	if err := m.Validate(); err != nil {
		fmt.Fprintf(stderr, "refused (%s): %s\n", refusalReason(err), err)
		return 3
	}
	if *asJSON {
		writeArmbenchJSON(stdout, map[string]any{"manifest_id": m.ID, "identity": m.Identity()})
	} else {
		fmt.Fprintln(stdout, m.Identity())
	}
	return 0
}

func armbenchRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("armbench run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath := fs.String("manifest", "", "path to the manifest JSON (required)")
	corpusPath := fs.String("corpus", "", "path to the corpus JSON (required)")
	out := fs.String("out", "", "path to write the raw trial ledger (required)")
	resume := fs.String("resume", "", "path to a prior ledger to resume from (completed trials are carried, not re-run)")
	provider := fs.String("provider", "", "provider to execute (default: manifest.model.provider; an override must match it)")
	setupMS := fs.Float64("fake-setup-ms", 0, "one-time per-arm setup cost the fake provider charges")
	asJSON := fs.Bool("json", false, "emit the run rollup as strict JSON instead of the human summary")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *corpusPath == "" || *out == "" {
		fmt.Fprintln(stderr, "fak armbench run: --manifest, --corpus and --out are all required")
		return 2
	}
	m, code := loadManifest(stderr, *manifestPath)
	if code != 0 {
		return code
	}
	corpusBlob, err := os.ReadFile(*corpusPath)
	if err != nil {
		fmt.Fprintln(stderr, "fak armbench run:", err)
		return 1
	}
	corpus, err := armbench.UnmarshalCorpus(corpusBlob)
	if err != nil {
		fmt.Fprintf(stderr, "refused (%s): %s\n", refusalReason(err), err)
		return 3
	}
	if err := armbench.ValidateCorpus(m, corpus); err != nil {
		fmt.Fprintf(stderr, "refused (%s): %s\n", refusalReason(err), err)
		return 3
	}
	selectedProvider := *provider
	if selectedProvider == "" {
		selectedProvider = m.Model.Provider
	}
	if selectedProvider != m.Model.Provider {
		fmt.Fprintf(stderr, "refused (%s): --provider %q disagrees with manifest.model.provider %q\n", armbench.ReasonIncomparableManifest, selectedProvider, m.Model.Provider)
		return 3
	}
	// Only the deterministic fake provider is wired today. A live provider arm
	// is a named follow-on, and refusing by token beats silently running the
	// wrong thing under a familiar name.
	if selectedProvider != "fake" {
		fmt.Fprintf(stderr, "refused (%s): provider %q is not wired; the deterministic 'fake' spine is the only provider this runner registers today (live provider arms land under epic #6674)\n", armbench.ReasonProviderUnknown, selectedProvider)
		return 3
	}
	opts := armbench.Options{}
	if *resume != "" {
		prior, code := loadRun(stderr, *resume)
		if code != 0 {
			return code
		}
		opts.Resume = prior
	}
	run, err := armbench.Execute(context.Background(), m, corpus.Tasks,
		&armbench.FakeProvider{SetupWallMS: *setupMS}, &armbench.FakeGrader{}, opts)
	if err != nil {
		fmt.Fprintf(stderr, "refused (%s): %s\n", refusalReason(err), err)
		return 3
	}
	blob, err := armbench.MarshalRun(run)
	if err != nil {
		fmt.Fprintln(stderr, "fak armbench run:", err)
		return 1
	}
	if err := os.WriteFile(*out, blob, 0o644); err != nil {
		fmt.Fprintln(stderr, "fak armbench run:", err)
		return 1
	}
	return emitReport(stdout, stderr, run, *asJSON, fmt.Sprintf("ledger written to %s\n", *out))
}

func armbenchReport(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("armbench report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	runPath := fs.String("run", "", "path to a run ledger (required)")
	asJSON := fs.Bool("json", false, "emit the report as strict JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	run, code := loadRun(stderr, *runPath)
	if code != 0 {
		return code
	}
	return emitReport(stdout, stderr, run, *asJSON, "")
}

func armbenchCompare(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("armbench compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	aPath := fs.String("a", "", "path to the first run ledger (required)")
	bPath := fs.String("b", "", "path to the second run ledger (required)")
	asJSON := fs.Bool("json", false, "emit the verdict as strict JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	a, code := loadRun(stderr, *aPath)
	if code != 0 {
		return code
	}
	b, code := loadRun(stderr, *bPath)
	if code != 0 {
		return code
	}
	if _, err := armbench.Summarize(a); err != nil {
		fmt.Fprintf(stderr, "refused (%s): first run: %s\n", refusalReason(err), err)
		return 3
	}
	if _, err := armbench.Summarize(b); err != nil {
		fmt.Fprintf(stderr, "refused (%s): second run: %s\n", refusalReason(err), err)
		return 3
	}
	fields, err := armbench.CheckComparable(a.Manifest, b.Manifest)
	if *asJSON {
		out := map[string]any{
			"comparable": err == nil,
			"a_identity": a.ManifestIdentity,
			"b_identity": b.ManifestIdentity,
			"drift":      fields,
		}
		if err != nil {
			out["reason"] = refusalReason(err)
			out["detail"] = err.Error()
		}
		writeArmbenchJSON(stdout, out)
	} else if err == nil {
		fmt.Fprintf(stdout, "comparable: both runs measured the same question\n  a %s\n  b %s\n", a.ManifestIdentity, b.ManifestIdentity)
	} else {
		fmt.Fprintf(stderr, "refused (%s): %s\n", refusalReason(err), err)
	}
	if err != nil {
		return 3
	}
	return 0
}

func emitReport(stdout, stderr io.Writer, run *armbench.Run, asJSON bool, note string) int {
	rep, err := armbench.Summarize(run)
	if err != nil {
		fmt.Fprintf(stderr, "refused (%s): %s\n", refusalReason(err), err)
		return 3
	}
	if asJSON {
		b, err := armbench.MarshalReport(rep)
		if err != nil {
			fmt.Fprintln(stderr, "fak armbench:", err)
			return 1
		}
		_, _ = stdout.Write(b)
		return 0
	}
	fmt.Fprint(stdout, armbench.Human(rep))
	if note != "" {
		fmt.Fprint(stdout, "\n"+note)
	}
	return 0
}

func loadManifest(stderr io.Writer, path string) (*armbench.Manifest, int) {
	if path == "" {
		fmt.Fprintln(stderr, "fak armbench: --manifest is required")
		return nil, 2
	}
	blob, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintln(stderr, "fak armbench:", err)
		return nil, 1
	}
	m, err := armbench.UnmarshalManifest(blob)
	if err != nil {
		fmt.Fprintf(stderr, "refused (%s): %s\n", refusalReason(err), err)
		return nil, 3
	}
	return m, 0
}

func loadRun(stderr io.Writer, path string) (*armbench.Run, int) {
	if path == "" {
		fmt.Fprintln(stderr, "fak armbench: a run ledger path is required")
		return nil, 2
	}
	blob, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintln(stderr, "fak armbench:", err)
		return nil, 1
	}
	run, err := armbench.UnmarshalRun(blob)
	if err != nil {
		fmt.Fprintf(stderr, "refused (%s): %s\n", refusalReason(err), err)
		return nil, 3
	}
	return run, 0
}

// refusalReason surfaces the closed-vocabulary token when there is one, so the
// operator sees the same token `dos man wedge` and the recovery table use.
func refusalReason(err error) string {
	var r *armbench.RefusalError
	if errors.As(err, &r) {
		return r.Reason
	}
	return "ERROR"
}

func writeArmbenchJSON(w io.Writer, v any) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func armbenchPonytail(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("armbench ponytail", flag.ContinueOnError)
	fs.SetOutput(stderr)
	checkout := fs.String("checkout", "", "pinned Ponytail checkout at the required revision")
	caveman := fs.String("caveman", "", "Caveman plugin checkout used unchanged by the upstream runner")
	out := fs.String("out", "", "live evidence directory (required with --live)")
	account := fs.String("account", "", "configured Claude account identity (required with --live; secrets are never accepted)")
	python := fs.String("python", "python", "Python executable used by the unchanged upstream runner")
	model := fs.String("model", "haiku", "upstream agent model alias")
	trials := fs.Int("trials", 1, "trial count per task and arm")
	live := fs.Bool("live", false, "execute real provider-backed trials; default is no-spend validation")
	asJSON := fs.Bool("json", false, "emit strict JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	p, err := armbench.Ponytail(armbench.PonytailOptions{Checkout: *checkout, Caveman: *caveman, Out: *out, Account: *account, Python: *python, Model: *model, Trials: *trials, Live: *live})
	if err != nil {
		fmt.Fprintln(stderr, "fak armbench ponytail:", err)
		return 1
	}
	if *asJSON {
		b, _ := json.MarshalIndent(p, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprintf(stdout, "PASS: pinned Ponytail agentic %s; tasks=%d arms=%d trials=%d model=%s\n", p.Mode, len(p.Tasks), len(p.Arms), p.Trials, p.AgentModel)
	}
	return 0
}

func armbenchPonytailManaged(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("armbench ponytail-managed", flag.ContinueOnError)
	fs.SetOutput(stderr)
	upstream := fs.String("upstream-dir", "", "pinned Ponytail checkout")
	tasks := fs.String("task", "todo-null,safe-path,critic-email,rate-limit", "comma-separated task IDs")
	treatments := fs.String("treatments", "baseline,caveman,ponytail", "comma-separated prompt treatments")
	arms := fs.String("managed-arms", "direct,fak_passthrough,shared_prefix_provider_cache_only,tool_result_compression_only,context_shedding_only,compression_shedding_bundle", "comma-separated managed-context arms")
	model := fs.String("model", "haiku", "upstream model alias")
	runs := fs.Int("runs", 1, "runs per cell")
	workers := fs.Int("workers", 1, "parallel upstream cells")
	dry := fs.Bool("dry-run", false, "validate and render without provider calls")
	receipt := fs.String("receipt", "", "secret-free JSON receipt path")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	var managed []armbench.ManagedArm
	for _, v := range splitArmbenchCSV(*arms) {
		managed = append(managed, armbench.ManagedArm(v))
	}
	r, err := armbench.RunManagedMatrix(context.Background(), armbench.ManagedRunOptions{UpstreamDir: *upstream, Tasks: splitArmbenchCSV(*tasks), Treatments: splitArmbenchCSV(*treatments), Arms: managed, Model: *model, Runs: *runs, Workers: *workers, DryRun: *dry, ReceiptPath: *receipt})
	if err != nil {
		fmt.Fprintln(stderr, "fak armbench ponytail-managed:", err)
		return 1
	}
	writeArmbenchJSON(stdout, r)
	return 0
}

func splitArmbenchCSV(s string) []string {
	var out []string
	for _, v := range strings.Split(s, ",") {
		if v = strings.TrimSpace(v); v != "" {
			out = append(out, v)
		}
	}
	return out
}
