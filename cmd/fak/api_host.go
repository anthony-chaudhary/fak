package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/apihostprobe"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func cmdAPIHost(argv []string) { os.Exit(runAPIHost(os.Stdout, os.Stderr, argv)) }

func runAPIHost(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak api-host: missing subcommand (readiness | acceptance)")
		return 2
	}
	switch argv[0] {
	case "readiness":
		return runAPIHostReadiness(stdout, stderr, argv[1:])
	case "acceptance":
		return runAPIHostAcceptance(stdout, stderr, argv[1:])
	case "help", "-h", "--help":
		apiHostUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak api-host: unknown subcommand %q (want readiness | acceptance)\n", argv[0])
		return 2
	}
}

func runAPIHostReadiness(stdout, stderr io.Writer, argv []string) int {
	var targets apiHostRepeatedString
	fs := flag.NewFlagSet("api-host readiness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Var(&targets, "target", "name|base_url[|api_key_env[|model_hint]] (repeatable)")
	fromRoster := fs.String("from-roster", "", "read probe targets from an api_host_roster JSON artifact")
	fromModelAccounts := fs.String("from-model-accounts", "", "read probe targets from a fak model-account roster (fak route --accounts-dump)")
	outPath := fs.String("out", "", "write JSON report here")
	markdownPath := fs.String("markdown", "", "write Markdown report here")
	timeoutS := fs.Float64("timeout-s", 10.0, "HTTP timeout in seconds")
	probeMissingAuth := fs.Bool("probe-missing-auth", false, "send unauthenticated request even when api_key_env is unset")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if err := oneAPIHostSource("readiness", len(targets) > 0, *fromRoster != "", *fromModelAccounts != ""); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}

	var parsed []apihostprobe.ReadinessTarget
	var err error
	switch {
	case *fromRoster != "":
		parsed, err = apihostprobe.LoadReadinessRosterTargets(*fromRoster)
	case *fromModelAccounts != "":
		parsed, err = loadModelAccountReadinessTargets(*fromModelAccounts)
	case len(targets) > 0:
		parsed = make([]apihostprobe.ReadinessTarget, 0, len(targets))
		for _, spec := range targets {
			target, parseErr := apihostprobe.ParseReadinessTarget(spec)
			if parseErr != nil {
				err = parseErr
				break
			}
			parsed = append(parsed, target)
		}
	default:
		parsed = apihostprobe.DefaultReadinessTargets()
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak api-host readiness: %v\n", err)
		return 2
	}

	report := apihostprobe.BuildReadinessReport(context.Background(), parsed, apihostprobe.ReadinessOptions{
		Timeout:          secondsDuration(*timeoutS),
		ProbeMissingAuth: *probeMissingAuth,
	})
	if rc := emitAPIHostReport(stdout, stderr, *outPath, *markdownPath, report, apihostprobe.ReadinessMarkdown(report), "readiness"); rc != 0 {
		return rc
	}
	if report.Summary.ReadinessGate {
		return 0
	}
	return 1
}

func runAPIHostAcceptance(stdout, stderr io.Writer, argv []string) int {
	var targets apiHostRepeatedString
	fs := flag.NewFlagSet("api-host acceptance", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Var(&targets, "target", "name|provider|base_url[|api_key_env[|model_hint]] (repeatable)")
	fromRoster := fs.String("from-roster", "", "read candidate targets from an api_host_roster JSON artifact")
	fromModelAccounts := fs.String("from-model-accounts", "", "read candidate targets from a fak model-account roster (fak route --accounts-dump)")
	outPath := fs.String("out", "", "write JSON report here")
	markdownPath := fs.String("markdown", "", "write Markdown report here")
	timeoutS := fs.Float64("timeout-s", 10.0, "HTTP timeout in seconds")
	probeMissingAuth := fs.Bool("probe-missing-auth", false, "send unauthenticated request even when api_key_env is unset")
	root := fs.String("root", ".", "workspace root used to inspect existing live sweep rows")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if err := oneAPIHostSource("acceptance", len(targets) > 0, *fromRoster != "", *fromModelAccounts != ""); err != nil {
		fmt.Fprintln(stderr, err)
		return 2
	}

	var parsed []apihostprobe.AcceptanceTarget
	var err error
	switch {
	case *fromRoster != "":
		parsed, err = apihostprobe.LoadAcceptanceRosterTargets(*fromRoster)
	case *fromModelAccounts != "":
		parsed, err = loadModelAccountAcceptanceTargets(*fromModelAccounts)
	case len(targets) > 0:
		parsed = make([]apihostprobe.AcceptanceTarget, 0, len(targets))
		for _, spec := range targets {
			target, parseErr := apihostprobe.ParseAcceptanceTarget(spec)
			if parseErr != nil {
				err = parseErr
				break
			}
			parsed = append(parsed, target)
		}
	default:
		parsed = apihostprobe.DefaultAcceptanceTargets()
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak api-host acceptance: %v\n", err)
		return 2
	}

	report := apihostprobe.BuildAcceptanceReport(context.Background(), parsed, apihostprobe.AcceptanceOptions{
		Timeout:          secondsDuration(*timeoutS),
		ProbeMissingAuth: *probeMissingAuth,
		Root:             *root,
	})
	if rc := emitAPIHostReport(stdout, stderr, *outPath, *markdownPath, report, apihostprobe.AcceptanceMarkdown(report), "acceptance"); rc != 0 {
		return rc
	}
	if report.Summary.AcceptanceGate {
		return 0
	}
	return 1
}

func oneAPIHostSource(mode string, target, roster, modelAccounts bool) error {
	n := 0
	for _, enabled := range []bool{target, roster, modelAccounts} {
		if enabled {
			n++
		}
	}
	if n <= 1 {
		return nil
	}
	return fmt.Errorf("fak api-host %s: --target, --from-roster, and --from-model-accounts are mutually exclusive", mode)
}

func loadModelAccountReadinessTargets(path string) ([]apihostprobe.ReadinessTarget, error) {
	r, err := modelroute.LoadRoster(path)
	if err != nil {
		return nil, err
	}
	hints := modelAccountHints(r)
	var out []apihostprobe.ReadinessTarget
	for _, a := range r.Accounts {
		if !modelAccountSupportsModelsProbe(a.Kind) {
			continue
		}
		base := modelAccountBaseURL(a)
		if base == "" {
			continue
		}
		out = append(out, apihostprobe.ReadinessTarget{
			Name:      a.ID,
			BaseURL:   base,
			APIKeyEnv: a.CredEnv,
			ModelHint: firstModelHint(hints[a.ID]),
		})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("model-account roster %s has no OpenAI-compatible/local accounts to probe", path)
	}
	return out, nil
}

func loadModelAccountAcceptanceTargets(path string) ([]apihostprobe.AcceptanceTarget, error) {
	r, err := modelroute.LoadRoster(path)
	if err != nil {
		return nil, err
	}
	hints := modelAccountHints(r)
	var out []apihostprobe.AcceptanceTarget
	for _, a := range r.Accounts {
		base := modelAccountBaseURL(a)
		if base == "" {
			continue
		}
		out = append(out, apihostprobe.AcceptanceTarget{
			Name:      a.ID,
			Provider:  modelAccountAPIHostProvider(a.Kind),
			BaseURL:   base,
			APIKeyEnv: a.CredEnv,
			ModelHint: firstModelHint(hints[a.ID]),
		})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("model-account roster %s has no accounts with a probeable base_url", path)
	}
	return out, nil
}

func modelAccountHints(r modelroute.Roster) map[string][]string {
	out := map[string][]string{}
	for _, b := range r.Bindings {
		model := b.UpstreamModel
		if model == "" {
			model = b.Model
		}
		if model != "" {
			out[b.Account] = append(out[b.Account], model)
		}
	}
	for account := range out {
		sort.Strings(out[account])
	}
	return out
}

func firstModelHint(hints []string) string {
	if len(hints) == 0 {
		return ""
	}
	return hints[0]
}

func modelAccountBaseURL(a modelroute.Account) string {
	if a.BaseURL != "" {
		return a.BaseURL
	}
	return modelroute.KindBaseURL(a.Kind)
}

func modelAccountSupportsModelsProbe(k modelroute.ProviderKind) bool {
	switch k {
	case modelroute.KindOpenAI, modelroute.KindOpenAIResponses, modelroute.KindXAI, modelroute.KindDeepSeek, modelroute.KindLocal:
		return true
	default:
		return false
	}
}

func modelAccountAPIHostProvider(k modelroute.ProviderKind) string {
	switch k {
	case modelroute.KindAnthropic:
		return "anthropic"
	case modelroute.KindGemini:
		return "gemini"
	case modelroute.KindXAI:
		return "xai"
	case modelroute.KindDeepSeek:
		return "deepseek"
	case modelroute.KindLocal, modelroute.KindOpenAIResponses:
		return "openai-compatible"
	default:
		return string(k)
	}
}

func emitAPIHostReport(stdout, stderr io.Writer, outPath, markdownPath string, report any, markdown, label string) int {
	body, err := apihostprobe.MarshalReport(report)
	if err != nil {
		fmt.Fprintf(stderr, "fak api-host %s: %v\n", label, err)
		return 1
	}
	if outPath != "" {
		if err := writeAPIHostText(outPath, body); err != nil {
			fmt.Fprintf(stderr, "fak api-host %s: %v\n", label, err)
			return 1
		}
	} else {
		if _, err := stdout.Write(body); err != nil {
			fmt.Fprintf(stderr, "fak api-host %s: %v\n", label, err)
			return 1
		}
	}
	if markdownPath != "" {
		if err := writeAPIHostText(markdownPath, []byte(markdown)); err != nil {
			fmt.Fprintf(stderr, "fak api-host %s: %v\n", label, err)
			return 1
		}
	}
	return 0
}

func writeAPIHostText(path string, body []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, body, 0o644)
}

func secondsDuration(seconds float64) time.Duration {
	return time.Duration(seconds * float64(time.Second))
}

type apiHostRepeatedString []string

func (r *apiHostRepeatedString) String() string {
	return fmt.Sprint([]string(*r))
}

func (r *apiHostRepeatedString) Set(value string) error {
	*r = append(*r, value)
	return nil
}

func apiHostUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak api-host readiness  [--target SPEC ... | --from-roster FILE | --from-model-accounts FILE] [--out FILE] [--markdown FILE] [--timeout-s N] [--probe-missing-auth]
  fak api-host acceptance [--target SPEC ... | --from-roster FILE | --from-model-accounts FILE] [--out FILE] [--markdown FILE] [--timeout-s N] [--probe-missing-auth] [--root DIR]

readiness target:
  name|base_url[|api_key_env[|model_hint]]

acceptance target:
  name|provider|base_url[|api_key_env[|model_hint]]

model-account roster:
  a fak-route account roster from 'fak route --accounts-dump'; readiness probes OpenAI-compatible/local accounts (openai, openai-responses, xai, deepseek, local) only, while acceptance also classifies native provider accounts as supported-unprobed.
`)
}
