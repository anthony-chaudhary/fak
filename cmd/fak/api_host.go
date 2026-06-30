package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/apihostprobe"
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
	outPath := fs.String("out", "", "write JSON report here")
	markdownPath := fs.String("markdown", "", "write Markdown report here")
	timeoutS := fs.Float64("timeout-s", 10.0, "HTTP timeout in seconds")
	probeMissingAuth := fs.Bool("probe-missing-auth", false, "send unauthenticated request even when api_key_env is unset")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if len(targets) > 0 && *fromRoster != "" {
		fmt.Fprintln(stderr, "fak api-host readiness: --target and --from-roster are mutually exclusive")
		return 2
	}

	var parsed []apihostprobe.ReadinessTarget
	var err error
	switch {
	case *fromRoster != "":
		parsed, err = apihostprobe.LoadReadinessRosterTargets(*fromRoster)
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
	outPath := fs.String("out", "", "write JSON report here")
	markdownPath := fs.String("markdown", "", "write Markdown report here")
	timeoutS := fs.Float64("timeout-s", 10.0, "HTTP timeout in seconds")
	probeMissingAuth := fs.Bool("probe-missing-auth", false, "send unauthenticated request even when api_key_env is unset")
	root := fs.String("root", ".", "workspace root used to inspect existing live sweep rows")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if len(targets) > 0 && *fromRoster != "" {
		fmt.Fprintln(stderr, "fak api-host acceptance: --target and --from-roster are mutually exclusive")
		return 2
	}

	var parsed []apihostprobe.AcceptanceTarget
	var err error
	switch {
	case *fromRoster != "":
		parsed, err = apihostprobe.LoadAcceptanceRosterTargets(*fromRoster)
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
  fak api-host readiness  [--target SPEC ... | --from-roster FILE] [--out FILE] [--markdown FILE] [--timeout-s N] [--probe-missing-auth]
  fak api-host acceptance [--target SPEC ... | --from-roster FILE] [--out FILE] [--markdown FILE] [--timeout-s N] [--probe-missing-auth] [--root DIR]

readiness target:
  name|base_url[|api_key_env[|model_hint]]

acceptance target:
  name|provider|base_url[|api_key_env[|model_hint]]
`)
}
