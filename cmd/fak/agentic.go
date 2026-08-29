package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agentic"
)

func cmdAgentic(args []string) {
	os.Exit(runAgentic(os.Stdout, os.Stderr, args))
}

func runAgentic(stdout, stderr io.Writer, args []string) int {
	objective, jsonOut, err := parseAgenticArgs(args)
	if err != nil {
		fmt.Fprintf(stderr, "fak agentic: %v\n", err)
		agenticUsage(stderr)
		return 2
	}

	plan, err := agentic.Compile(objective)
	if err != nil {
		fmt.Fprintf(stderr, "fak agentic: %v\n", err)
		return 2
	}
	if jsonOut {
		data, err := agentic.Marshal(plan)
		if err != nil {
			fmt.Fprintf(stderr, "fak agentic: encode plan: %v\n", err)
			return 1
		}
		if _, err := stdout.Write(data); err != nil {
			fmt.Fprintf(stderr, "fak agentic: write plan: %v\n", err)
			return 1
		}
		return 0
	}
	if _, err := io.WriteString(stdout, agentic.Render(plan)); err != nil {
		fmt.Fprintf(stderr, "fak agentic: write plan: %v\n", err)
		return 1
	}
	return 0
}

func agenticUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak agentic [--json] --objective OBJECTIVE")
	fmt.Fprintln(w, "       fak agentic [--json] OBJECTIVE...")
	fmt.Fprintln(w, "  compile an objective into a bounded read-only/offline expand, experiment, and contract plan")
}

func parseAgenticArgs(args []string) (string, bool, error) {
	var (
		explicitObjective string
		haveExplicit      bool
		jsonOut           bool
		positional        []string
	)
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "--json":
			if len(positional) > 0 {
				return "", false, fmt.Errorf("unexpected option %q after positional objective text", arg)
			}
			if jsonOut {
				return "", false, fmt.Errorf("--json may be specified only once")
			}
			jsonOut = true
		case arg == "--objective":
			if haveExplicit {
				return "", false, fmt.Errorf("--objective may be specified only once")
			}
			if i+1 >= len(args) || strings.HasPrefix(args[i+1], "-") {
				return "", false, fmt.Errorf("--objective requires a value")
			}
			i++
			explicitObjective = args[i]
			haveExplicit = true
		case strings.HasPrefix(arg, "--objective="):
			if haveExplicit {
				return "", false, fmt.Errorf("--objective may be specified only once")
			}
			explicitObjective = strings.TrimPrefix(arg, "--objective=")
			haveExplicit = true
		case strings.HasPrefix(arg, "-"):
			return "", false, fmt.Errorf("unexpected option %q", arg)
		default:
			positional = append(positional, arg)
		}
	}
	if haveExplicit && len(positional) > 0 {
		return "", false, fmt.Errorf("--objective conflicts with positional objective text")
	}
	if haveExplicit {
		if strings.TrimSpace(explicitObjective) == "" {
			return "", false, fmt.Errorf("objective is required")
		}
		return explicitObjective, jsonOut, nil
	}
	if len(positional) == 0 || strings.TrimSpace(strings.Join(positional, " ")) == "" {
		return "", false, fmt.Errorf("objective is required")
	}
	return strings.Join(positional, " "), jsonOut, nil
}
