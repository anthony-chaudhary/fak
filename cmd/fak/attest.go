package main

// fak attest — the COMPLIANCE ATTESTATION GENERATOR. It takes the deployable
// capability floor (a --policy manifest) and PROVES it from preflight: it runs
// the SAME adjudication fold `fak preflight` runs, over a probe set, and emits a
// re-checkable attestation document that records, per probe, what the floor was
// expected to do and what it actually did.
//
// The default probe set is DERIVED from the manifest itself — the floor is its
// own spec — so one attestation proves three things at once:
//   - every declared DENY is enforced, with the closed-vocabulary reason the
//     manifest cites for it;
//   - every declared ALLOW (and allow_prefix) is admitted;
//   - the DEFAULT-DENY posture holds for a tool the floor does not name.
//
// An explicit --probes FILE overrides the derived set, so an operator can attest
// arg-value cases (the manifest's arg_rules) or any other shape the name-level
// floor does not capture.
//
// A self-report is not a witness: the recorded verdicts come from the real kernel
// fold, not from the manifest's declaration. Exit 0 if every probe matches its
// expectation (the floor is proven), 1 if any drifts (the floor is NOT what the
// manifest claims) — so `fak attest` can gate a build the way `fak lint` does.
// Exit 2 = usage error.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

func cmdAttest(argv []string) {
	os.Exit(runAttest(os.Stdout, os.Stderr, argv))
}

// runAttest is the testable core: returns the exit code, takes its streams.
func runAttest(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("attest", flag.ContinueOnError)
	fs.SetOutput(stderr)
	policyPath := fs.String("policy", "", "the capability floor manifest to prove (required)")
	probesPath := fs.String("probes", "", "optional JSON file of probes (default: derived from the policy)")
	out := fs.String("out", "", "write the attestation JSON to FILE (default: stdout)")
	asJSON := fs.Bool("json", false, "emit the attestation as JSON on stdout (the machine-readable form)")
	quiet := fs.Bool("quiet", false, "suppress the human summary; still exit by pass/fail")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if *policyPath == "" {
		fmt.Fprintln(stderr, "fak attest: --policy FILE is required (the capability floor to prove)")
		attestUsage(stderr)
		return 2
	}

	raw, err := os.ReadFile(*policyPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak attest: %v\n", err)
		return 2
	}
	sum := sha256.Sum256(raw)

	// Install the floor exactly as `fak preflight --policy` does, so the fold
	// below runs over the SAME adjudicator state an operator's run would.
	applyPolicy(*policyPath)

	probes, err := buildProbes(*probesPath, raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak attest: %v\n", err)
		return 2
	}
	if len(probes) == 0 {
		fmt.Fprintln(stderr, "fak attest: no probes to attest (supply --probes FILE or a policy with rules)")
		return 2
	}

	ctx := ctx()
	results := make([]probeResult, 0, len(probes))
	for _, p := range probes {
		results = append(results, runProbe(ctx, p))
	}

	att := buildAttestation(*policyPath, sum, results)

	// The attestation is always available as JSON (--json, --out, or implicitly
	// when quiet); the human summary is the default interactive surface.
	emitJSON := *asJSON || *out != "" || *quiet
	if *out != "" {
		if err := os.WriteFile(*out, att.JSON(), 0o644); err != nil {
			fmt.Fprintf(stderr, "fak attest: write %s: %v\n", *out, err)
			return 1
		}
	} else if emitJSON {
		fmt.Fprintln(stdout, string(att.JSON()))
	}
	if !*quiet {
		printAttestation(stdout, &att)
	}

	if att.Summary.Failed != 0 {
		return 1
	}
	return 0
}

// probe is one attestation check: a tool call and the verdict the floor owes it.
type probe struct {
	Tool         string `json:"tool"`
	Args         string `json:"args,omitempty"`
	Expect       string `json:"expect"`                  // "allow" | "deny"
	ExpectReason string `json:"expect_reason,omitempty"` // checked when Expect=="deny"; a closed-vocab name
	Origin       string `json:"origin,omitempty"`        // deny | allow | allow_prefix | default_deny | probes-file
}

// probeResult is a probe plus the actual verdict the fold returned.
type probeResult struct {
	probe
	Actual       string `json:"actual"`        // verdict name (ALLOW/DENY/...)
	ActualReason string `json:"actual_reason"` // closed-vocabulary name
	By           string `json:"by"`            // which adjudicator decided
	Pass         bool   `json:"pass"`
}

type policyRef struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type attSummary struct {
	Probes int  `json:"probes"`
	Passed int  `json:"passed"`
	Failed int  `json:"failed"`
	Pass   bool `json:"pass"`
}

// attestation is the re-checkable compliance document.
type attestation struct {
	Schema      string        `json:"schema"`
	FakVersion  string        `json:"fak_version"`
	GeneratedAt time.Time     `json:"generated_at"`
	Policy      policyRef     `json:"policy"`
	Probes      []probeResult `json:"probes"`
	Summary     attSummary    `json:"summary"`
}

// JSON renders the attestation document as indented JSON.
func (a attestation) JSON() []byte {
	b, _ := json.MarshalIndent(a, "", "  ")
	return b
}

// buildProbes loads the probe set: an explicit --probes FILE if given, else the
// set derived from the manifest (the floor is its own spec).
func buildProbes(probesPath string, manifestBytes []byte) ([]probe, error) {
	if probesPath != "" {
		return loadProbesFile(probesPath)
	}
	return deriveProbes(manifestBytes)
}

// loadProbesFile reads an explicit JSON array of probes. It validates each
// expect/expect_reason against the closed vocabulary so a typo is a hard error
// (the same fail-loud discipline `fak policy --check` applies to a manifest),
// rather than a probe that silently can never pass.
func loadProbesFile(path string) ([]probe, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("probes %s: %w", path, err)
	}
	var ps []probe
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&ps); err != nil {
		return nil, fmt.Errorf("probes %s: invalid probe list: %w", path, err)
	}
	for i := range ps {
		ps[i].Origin = "probes-file"
		if ps[i].Args == "" {
			ps[i].Args = "{}"
		}
		if err := validateProbe(ps[i]); err != nil {
			return nil, fmt.Errorf("probes %s: probe %d (%s): %w", path, i, ps[i].Tool, err)
		}
	}
	return ps, nil
}

// deriveProbes turns a manifest into its own proof obligation. The probes are
// sorted for deterministic output; the default-deny probe is appended last as a
// named, always-present witness that the floor is actually closed.
func deriveProbes(manifestBytes []byte) ([]probe, error) {
	m, err := policy.ParseManifest(manifestBytes)
	if err != nil {
		return nil, fmt.Errorf("derive probes: %w", err)
	}
	var probes []probe

	denyTools := sortedKeys(m.Deny)
	for _, tool := range denyTools {
		p := probe{Tool: tool, Args: "{}", Expect: "deny", ExpectReason: m.Deny[tool], Origin: "deny"}
		if err := validateProbe(p); err != nil {
			return nil, fmt.Errorf("deny rule %q: %w", tool, err)
		}
		probes = append(probes, p)
	}

	for _, tool := range sortedStrings(m.Allow) {
		// An allow-listed tool gated by a positive allow_glob arg_rule would be
		// (correctly) DENIED for the missing arg if probed with empty args — so
		// the derived ALLOW probe synthesizes args that satisfy those globs and
		// exercises a genuinely-admissible call instead of a false drift. Other
		// arg-value boundaries (deny_regex/max_bytes) the empty call already
		// satisfies; arbitrary boundaries stay --probes territory (see header).
		probes = append(probes, probe{Tool: tool, Args: synthAllowArgs(tool, m.ArgRules), Expect: "allow", Origin: "allow"})
	}
	for _, prefix := range sortedStrings(m.AllowPrefix) {
		// A prefix is not a concrete tool; synthesize one that matches it so the
		// attestation exercises the prefix rule on a real call.
		probes = append(probes, probe{Tool: prefix + "attest_probe", Args: "{}", Expect: "allow", Origin: "allow_prefix"})
	}

	probes = append(probes, probe{
		Tool: "__fak_attest_unmatched__", Args: "{}",
		Expect: "deny", ExpectReason: "DEFAULT_DENY", Origin: "default_deny",
	})
	return probes, nil
}

// synthAllowArgs builds a minimal args JSON that SATISFIES every positive
// allow_glob arg_rule gating an allow-listed tool, so the derived ALLOW probe
// exercises a genuinely-admissible call rather than the empty-args call an
// allow_glob refuses for a missing arg. Best-effort by design: the real kernel
// verdict still decides pass/fail, so an imperfect synthesis can only leave the
// probe drifting, never manufacture a false ALLOW. Tools with no gating glob keep
// the empty "{}" args (the prior behaviour).
func synthAllowArgs(tool string, rules []policy.ArgRule) string {
	args := map[string]string{}
	for _, r := range rules {
		if r.Tool == tool && r.AllowGlob != "" {
			args[r.Arg] = satisfyGlob(r.AllowGlob)
		}
	}
	if len(args) == 0 {
		return "{}"
	}
	b, err := json.Marshal(args)
	if err != nil {
		return "{}"
	}
	return string(b)
}

// satisfyGlob returns a literal value admitted by an adjudicator allow_glob,
// mirroring adjudicator.pathUnderGlob's two cases: a "**" glob is path-
// containment (a path under the prefix dir); otherwise it is a single
// path.Match (the single-segment wildcards replaced by a literal).
func satisfyGlob(glob string) string {
	if i := strings.Index(glob, "**"); i >= 0 {
		return glob[:i] + "x" // "notes/**" -> "notes/x", "./out/**" -> "./out/x"
	}
	return strings.NewReplacer("*", "x", "?", "x").Replace(glob) // "public.*" -> "public.x"
}

// validateProbe rejects a probe whose expectation the kernel could never satisfy
// (an unknown verdict or a reason outside the closed vocabulary).
func validateProbe(p probe) error {
	switch p.Expect {
	case "allow", "deny":
	default:
		return fmt.Errorf("expect must be \"allow\" or \"deny\", got %q", p.Expect)
	}
	if p.Expect == "deny" && p.ExpectReason != "" {
		if _, ok := abi.ReasonByName(p.ExpectReason); !ok {
			return fmt.Errorf("unknown expect_reason %q (not in the closed refusal vocabulary)", p.ExpectReason)
		}
	}
	return nil
}

// runProbe runs one probe through the real preflight fold and records the result.
func runProbe(ctx context.Context, p probe) probeResult {
	res := abi.ActiveResolver()
	ref, err := res.Put(ctx, []byte(p.Args))
	if err != nil {
		return probeResult{probe: p, Actual: "ERROR", By: "resolver", Pass: false}
	}
	// Attestation is a proof run, not a continuation of the caller's ambient
	// session. Use a non-empty trace id so process-local IFC ledger state from a
	// prior command/test that touched the empty trace cannot contaminate the probe.
	tc := &abi.ToolCall{TraceID: "fak-attest/" + p.Tool, Tool: p.Tool, Args: ref}
	v := kernel.Fold(ctx, abi.AdjudicatorsFor(tc), tc)
	return probeResult{
		probe:        p,
		Actual:       verdictName(v.Kind),
		ActualReason: abi.ReasonName(v.Reason),
		By:           v.By,
		Pass:         probePass(p, v),
	}
}

// probePass is the closed pass criterion: allow must be ALLOW; deny must be DENY,
// and when a reason is asserted it must match (closed-vocabulary name equality).
func probePass(p probe, v abi.Verdict) bool {
	switch p.Expect {
	case "allow":
		return v.Kind == abi.VerdictAllow
	case "deny":
		if v.Kind != abi.VerdictDeny {
			return false
		}
		if p.ExpectReason != "" {
			return abi.ReasonName(v.Reason) == p.ExpectReason
		}
		return true
	}
	return false
}

func buildAttestation(policyPath string, sum [32]byte, results []probeResult) attestation {
	s := attSummary{Probes: len(results)}
	for _, r := range results {
		if r.Pass {
			s.Passed++
		} else {
			s.Failed++
		}
	}
	s.Pass = s.Failed == 0
	return attestation{
		Schema:      "fak-attestation/v1",
		FakVersion:  appversion.Current(),
		GeneratedAt: time.Now().UTC(),
		Policy:      policyRef{Path: policyPath, SHA256: hex.EncodeToString(sum[:])},
		Probes:      results,
		Summary:     s,
	}
}

func printAttestation(w io.Writer, a *attestation) {
	fmt.Fprintf(w, "== fak attest: %s  (sha256 %s) ==\n", a.Policy.Path, a.Policy.SHA256[:12])
	fmt.Fprintf(w, "schema:      %s   fak: %s   generated: %s\n",
		a.Schema, a.FakVersion, a.GeneratedAt.Format(time.RFC3339))
	fmt.Fprintf(w, "%-5s  %-26s  %-10s  %-16s  %-10s  %s\n",
		"PASS", "TOOL", "EXPECT", "EXPECT_REASON", "ACTUAL", "ACTUAL_REASON")
	for _, r := range a.Probes {
		mark := "FAIL"
		if r.Pass {
			mark = "ok"
		}
		tool := r.Tool
		if len(tool) > 26 {
			tool = tool[:23] + "..."
		}
		fmt.Fprintf(w, "%-5s  %-26s  %-10s  %-16s  %-10s  %s\n",
			mark, tool, r.Expect, r.ExpectReason, r.Actual, r.ActualReason)
	}
	verdict := "FAIL"
	if a.Summary.Pass {
		verdict = "PASS"
	}
	fmt.Fprintf(w, "\n%d probe(s): %d passed, %d failed -> %s (the capability floor is %s)\n",
		a.Summary.Probes, a.Summary.Passed, a.Summary.Failed, verdict,
		map[bool]string{true: "PROVEN", false: "NOT proven"}[a.Summary.Pass])
}

func attestUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak attest --policy FILE [--probes FILE] [--out FILE] [--json] [--quiet]

prove the deployable capability floor from preflight: run the real adjudication
fold over a probe set and emit a re-checkable compliance attestation.

  --policy FILE   the capability floor manifest to prove (required)
  --probes FILE   JSON array of probes; default: derived from the policy
  --out FILE      write the attestation JSON to FILE
  --json          emit the attestation as JSON on stdout
  --quiet         no human summary (still exits by pass/fail)

derived probes (no --probes): each deny rule must be DENIED with its cited
reason; each allow / allow_prefix must be ALLOWED; and a tool the floor does not
name must be DENIED DEFAULT_DENY. exit 0 if every probe matches (floor PROVEN),
1 if any drifts (floor NOT proven), 2 on usage error.
`)
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortedStrings(s []string) []string {
	out := append([]string(nil), s...)
	sort.Strings(out)
	return out
}
