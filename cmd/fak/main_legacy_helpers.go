package main

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
	"github.com/anthony-chaudhary/fak/internal/toollint"
)

// fak policy  -  author and validate the deployable capability floor. --dump emits
// the built-in DefaultPolicy as a manifest (the starting point an adopter edits);
// --check validates a manifest against the closed refusal vocabulary and prints
// the floor it admits, so a misconfigured policy is caught BEFORE it gates a run.
//
// --check ALSO accepts a `fak-org-policy/v1` signed envelope (issue #5320): the CLI
// seam onto policy.VerifyEnvelope, so an operator can prove a centrally-issued floor
// offline  -  authentic (signed by the org root key given with --org-key), fresh
// (inside its not_before/expires window), not a rolled-back older version, and
// applicable to this binary (min_version)  -  BEFORE that floor is trusted anywhere.
// The trust anchor is ALWAYS operator-supplied, never read from an enrollment store:
// the verifier is deliberately a pure function over a caller-supplied key (issue
// #5323 pins the key at enroll time and is still open). Routing is by the file's
// SHAPE, never its name, and a verification failure exits non-zero with the closed
// vocabulary refusal named on stderr  -  the envelope arm never fails open.
func cmdPolicy(argv []string) {
	if len(argv) > 0 && argv[0] == "land-rule" {
		fs := flag.NewFlagSet("policy land-rule", flag.ExitOnError)
		candidate := fs.String("candidate", "", "ArgRules-only candidate JSON")
		policyPath := fs.String("policy", "", "policy manifest to merge")
		reloadURL := fs.String("reload-url", "", "POST endpoint for the running gateway policy reload")
		land := fs.Bool("land", false, "write the merged policy and reload (default is dry-run)")
		rollback := fs.Bool("rollback", false, "restore the recorded preimage and reload")
		_ = fs.Parse(argv[1:])
		if err := runPolicyLandRule(*policyPath, *candidate, *reloadURL, *land, *rollback, os.Stdout); err != nil {
			fmt.Fprintln(os.Stderr, "fak policy land-rule:", err)
			os.Exit(1)
		}
		return
	}
	fs := flag.NewFlagSet("policy", flag.ExitOnError)
	verbFlagUsage(fs, "policy")
	dump := fs.Bool("dump", false, "write the built-in DefaultPolicy as a manifest to stdout")
	check := fs.String("check", "", "validate a manifest (or a fak-org-policy/v1 signed envelope) and print the floor it admits")
	orgKey := fs.String("org-key", "", "org root Ed25519 public key, base64 or a file holding it; required to verify a fak-org-policy/v1 envelope")
	orgIssuer := fs.String("org-issuer", "", "require a fak-org-policy/v1 envelope to be issued by this identity")
	orgSeenVersion := fs.Uint64("org-seen-version", 0, "anti-rollback floor: refuse a fak-org-policy/v1 envelope whose version is below this")
	_ = fs.Parse(argv)

	switch {
	case *dump:
		os.Stdout.Write(policy.FromPolicy(adjudicator.DefaultPolicy()).JSON())
	case *check != "":
		report, err := checkPolicyFile(*check, orgCheckOptions{
			Key:            *orgKey,
			ExpectedIssuer: *orgIssuer,
			SeenVersion:    *orgSeenVersion,
			Now:            time.Now(),
			RunningVersion: appversion.Current(),
		})
		if err != nil {
			// This is the command the POLICY_LOAD_FAILED recovery sends an operator
			// to, so it must not dead-end here: the rejection itself carries the
			// knob that produced it and the next step, exactly like the bail that
			// sent them (bail.go).
			writeConfigBail(os.Stderr, configBail{
				Verb:    "fak policy",
				Reason:  bailPolicyLoadFailed,
				Summary: fmt.Sprintf("--check rejected the manifest: %v", err),
				Knobs:   []bailKnob{bailFile(*check, "did not validate").want("a manifest whose every deny cites a closed-vocabulary reason")},
				Check:   "fak policy --dump   # the default manifest, to diff yours against",
				Bind:    []string{"path=" + *check},
			})
			os.Exit(1)
		}
		fmt.Print(report)
	default:
		fmt.Fprintln(os.Stderr, "fak policy: pass --dump (emit the default manifest) or --check FILE (validate one)")
		os.Exit(2)
	}
}

// orgCheckOptions gathers everything the org-envelope arm of `fak policy --check`
// needs from its flags AND from the ambient process, so checkPolicyFile stays a pure
// function of its inputs: it reads neither the clock nor the build identity nor any
// stored key for itself. That mirrors policy.VerifyOptions one field at a time, which
// is the whole point  -  the verifier's determinism is what makes the CLI seam
// table-testable rather than something you can only observe by running a binary.
type orgCheckOptions struct {
	// Key is --org-key verbatim: base64 key material, or the path of a file holding it.
	Key string
	// ExpectedIssuer is --org-issuer; empty accepts whichever issuer the root key signed for.
	ExpectedIssuer string
	// SeenVersion is --org-seen-version, the highest envelope version already accepted.
	SeenVersion uint64
	// Now is the wall clock the not_before/expires window is checked against.
	Now time.Time
	// RunningVersion is the binary version the envelope's min_version is checked against.
	RunningVersion string
}

// checkPolicyFile is the PURE core of `fak policy --check`, factored out (like
// lintExitCode) so both arms are unit-testable without os.Exit. It returns the report
// to print on stdout, or an error to name on stderr with a non-zero exit  -  it never
// returns a floor it did not first validate.
//
// A `fak-org-policy/v1` envelope is fully verified before its wrapped floor is
// rendered; anything else takes the original policy.LoadRuntime path unchanged, so a
// plain manifest  -  including an unreadable or invalid one  -  keeps the exact wording
// this verb has always produced.
//
// The ONE ambient read is the advisory modver rev printed beside a valid manifest's path
// (#4311): it arrives through the policyManifestRevFn seam, is best-effort (a manifest
// outside a git repo, or one that is not a tracked examples/<file>.json module, simply
// resolves to no rev), and is display only  -  it is never consulted to decide whether a
// floor validates, so the accept/reject behavior above is exactly what it was.
func checkPolicyFile(path string, org orgCheckOptions) (string, error) {
	raw, err := os.ReadFile(path)
	if err == nil && isOrgEnvelope(raw) {
		return checkOrgEnvelope(path, raw, org)
	}
	rt, err := policy.LoadRuntime(path)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("OK  %s%s  (manifest valid; every deny cites a closed-vocabulary reason)\n\n%s", path, policyRevTag(policyManifestRevFn(path)), policy.SummaryRuntime(rt)), nil
}

// isOrgEnvelope sniffs whether raw is a signed `fak-org-policy/v1` envelope rather
// than a plain `fak-policy/v1` manifest. It routes on SHAPE, never on the filename: an
// operator may call an envelope anything, and a name is not a trust input.
//
// The two schemas are disjoint at their `version` key. A manifest carries the schema
// TAG there  -  the JSON STRING "fak-policy/v1"  -  while an envelope carries its
// monotonic anti-rollback COUNTER, a JSON number, beside the `alg`/`sig` pair no
// manifest field spells. Requiring the signature pair AND a non-string `version` means
// neither schema can be mistaken for the other.
//
// The sniff only CHOOSES a verifier; it grants nothing. policy.VerifyEnvelope re-decodes
// the same bytes under DisallowUnknownFields, so a near-miss envelope is refused
// MALFORMED rather than quietly admitted, and a file that is neither shape lands on the
// manifest path whose unknown-field rejection reports the real problem loudly.
func isOrgEnvelope(raw []byte) bool {
	var probe struct {
		Version json.RawMessage `json:"version"`
		Alg     string          `json:"alg"`
		Sig     string          `json:"sig"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return false
	}
	if probe.Alg == "" || probe.Sig == "" {
		return false
	}
	v := bytes.TrimSpace(probe.Version)
	return len(v) == 0 || v[0] != '"'
}

// checkOrgEnvelope verifies a `fak-org-policy/v1` envelope against the operator's trust
// anchor and renders its provenance followed by the floor it would install. Every
// failure  -  an absent or malformed --org-key, or any *policy.EnvelopeError from the
// verifier  -  returns an error instead of a report, so the caller exits non-zero and an
// unproven central floor is never printed as though it had been trusted.
func checkOrgEnvelope(path string, raw []byte, org orgCheckOptions) (string, error) {
	key, err := orgRootKey(org.Key)
	if err != nil {
		return "", err
	}
	v, err := policy.VerifyEnvelope(raw, policy.VerifyOptions{
		RootPublicKey:      key,
		ExpectedIssuer:     org.ExpectedIssuer,
		Now:                org.Now,
		HighestSeenVersion: org.SeenVersion,
		RunningVersion:     org.RunningVersion,
	})
	if err != nil {
		// *policy.EnvelopeError already renders its closed-vocabulary Reason plus the
		// Detail token that separates causes sharing one Reason (expired vs rollback).
		return "", err
	}

	e := v.Envelope
	var b strings.Builder
	fmt.Fprintf(&b, "OK  %s  (%s envelope verified: signature, issuer, freshness window, anti-rollback)\n\n", path, policy.OrgEnvelopeVersion)
	fmt.Fprintf(&b, "issuer             : %s\n", e.Issuer)
	fmt.Fprintf(&b, "envelope version   : %d (anti-rollback floor %d)\n", e.Version, org.SeenVersion)
	fmt.Fprintf(&b, "valid              : %s .. %s\n", orgInstant(e.NotBefore), orgInstant(e.Expires))
	fmt.Fprintf(&b, "target             : %s\n", orgFieldOrAny(e.Target))
	fmt.Fprintf(&b, "min fak version    : %s (running %s)\n", orgFieldOrAny(e.MinVersion), org.RunningVersion)
	b.WriteString("\n")
	b.WriteString(policy.SummaryRuntime(v.Runtime))
	return b.String(), nil
}

// orgRootKey resolves --org-key into the pinned org root Ed25519 public key. The value
// is standard base64 key material, or the path of a file holding it  -  a 44-character
// blob is awkward to paste on a command line and operators keep the key in a file.
//
// It fails LOUD on an absent or wrong-sized key rather than skipping the signature
// check: without a trust anchor the envelope cannot be authenticated AT ALL, and the
// one thing this verb must never do is present an unverified central floor as verified.
func orgRootKey(value string) (ed25519.PublicKey, error) {
	v := strings.TrimSpace(value)
	if v == "" {
		return nil, fmt.Errorf("--check names a %s envelope: pass --org-key with the org root public key (base64, or a file holding it)", policy.OrgEnvelopeVersion)
	}
	if b, err := os.ReadFile(v); err == nil {
		v = strings.TrimSpace(string(b))
	}
	key, err := base64.StdEncoding.DecodeString(v)
	if err != nil {
		return nil, fmt.Errorf("--org-key is not base64 key material nor a readable key file: %w", err)
	}
	if len(key) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("--org-key decodes to %d bytes, want a %d-byte Ed25519 public key", len(key), ed25519.PublicKeySize)
	}
	return ed25519.PublicKey(key), nil
}

// orgInstant renders one edge of the envelope's validity window. A zero bound prints
// "unbounded" because that is what the verifier actually did with it (0 means "no bound
// on this side"); printing the 1970 epoch there would misreport the check that ran.
func orgInstant(sec int64) string {
	if sec == 0 {
		return "unbounded"
	}
	return time.Unix(sec, 0).UTC().Format(time.RFC3339)
}

// orgFieldOrAny renders an optional envelope selector, naming the empty case rather
// than leaving a blank the reader has to interpret.
func orgFieldOrAny(s string) string {
	if strings.TrimSpace(s) == "" {
		return "(any)"
	}
	return s
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
	prov := rep.TokenDeltaProvenance
	if prov == "" {
		prov = "SIMULATED"
	}
	fmt.Printf("secondary token delta       : %.2f%% (soft, never gates) [%s]\n", rep.TokenDeltaPct, prov)
	if rep.DollarValuationBasis != "" {
		fmt.Printf("cost per task               : $%.6f [%s, %s]\n", rep.DollarPerTask, rep.DollarValuationBasis, rep.DollarRateTable)
	}
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
	if err == nil {
		return
	}
	// applyPolicy — the launch-time floor install, reached from `fak serve`,
	// `fak chat`, `fak attest`, and the agent verbs — ends here on a floor that
	// would not load. That made the single most common fatal misconfiguration
	// fak has print `fak: policy floor.json: invalid manifest: ...` and stop,
	// with no path named as a knob and nothing to run next.
	//
	// applyPolicy lives in main.go and calls this helper, so recognizing the
	// typed error here upgrades every one of those call sites at once. Every
	// other must() caller is untouched: only a *policy.LoadError takes this path.
	var loadErr *policy.LoadError
	if errors.As(err, &loadErr) {
		observed := "did not validate"
		want := "a manifest whose every deny cites a closed-vocabulary reason"
		if loadErr.Op == policy.LoadOpRead {
			observed = "could not be read"
			want = "a readable path to the capability floor"
		}
		writeConfigBail(os.Stderr, configBail{
			Verb:    "fak",
			Reason:  bailPolicyLoadFailed,
			Summary: fmt.Sprintf("refusing to start on a capability floor that would not load: %v", err),
			Knobs: []bailKnob{
				bailFlag("policy", loadErr.Path),
				bailFile(loadErr.Path, observed).want(want),
			},
			Check: "fak policy --check " + loadErr.Path + "   # the precise rejection, read-only",
			Bind:  []string{"path=" + loadErr.Path},
		})
		os.Exit(1)
	}
	fmt.Fprintln(os.Stderr, "fak:", err)
	os.Exit(1)
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
	if code := runNewModel(os.Stdout, os.Stderr, argv); code != 0 {
		os.Exit(code)
	}
}
