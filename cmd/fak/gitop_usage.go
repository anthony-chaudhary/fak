package main

import (
	"strings"
	"time"
)

// runObservedGitOperation closes the os.Exit coverage gap for the three shared-
// trunk Git front doors. The handler returns instead of exiting, so its exact
// terminal code and end-to-end duration can be appended before main exits.
func runObservedGitOperation(start time.Time, operation string, argv []string, handler func() int) int {
	code := handler()
	recordUsage(operation, argv, code, start)
	return code
}

// gitOperationName converts raw commit/sweep/sync argv into a closed, non-secret
// operation vocabulary. Raw args remain only in the salted digest; paths, commit
// messages, remotes, and tokens can never become metric labels.
func gitOperationName(verb string, argv []string) string {
	prefix := ""
	base := verb
	if strings.HasPrefix(base, "dev ") {
		prefix = "dev "
		base = strings.TrimPrefix(base, "dev ")
	}

	var op string
	switch base {
	case "commit":
		op = commitOperationName(argv)
	case "sweep":
		op = sweepOperationName(argv)
	case "sync":
		op = syncOperationName(argv)
	default:
		return verb
	}
	return prefix + op
}

func commitOperationName(argv []string) string {
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		return "commit" // query/queue subcommands are not a ship-latency operation
	}
	flags := parseGitOpFlags(argv, map[string]bool{
		"m": true, "F": true, "path": true, "dir": true, "trunk": true,
		"review-model": true, "review-min-models": true, "review-objective": true,
		"review-endpoint": true, "review-api-key-env": true,
		"core-lock-maintenance-witness": true,
	})
	if flags["preview"] {
		return "commit" // lint-only: useful generic usage, not commit velocity
	}
	if flags["push"] {
		return "commit push"
	}
	return "commit local"
}

func sweepOperationName(argv []string) string {
	flags := parseGitOpFlags(argv, map[string]bool{
		"dir": true, "lane": true, "m": true, "path": true,
	})
	op := "sweep plan"
	if flags["clean-junk"] {
		op = "sweep clean-junk"
	} else if flags["apply"] {
		op = "sweep apply local"
		if flags["push"] {
			op = "sweep apply push"
		}
	}
	return op
}

func syncOperationName(argv []string) string {
	command := "check"
	flagArgv := argv
	if len(argv) > 0 {
		switch argv[0] {
		case "check", "apply", "push", "drain":
			command = argv[0]
			flagArgv = argv[1:]
		}
	}
	flags := parseGitOpFlags(flagArgv, map[string]bool{
		"repo": true, "remote": true, "branch": true, "retries": true,
		"queue-file": true, "budget": true,
	})
	op := "sync " + command
	if (command == "check" || command == "apply") && flags["fetch"] {
		op += " fetch"
	}
	return op
}

// parseGitOpFlags recognizes flag NAMES without mistaking a flag-looking VALUE
// (for example `-m --push` or `--path --push`) for an operation selector. It
// mirrors flag.FlagSet's stop-at-first-positional/-- behavior for the known
// value-taking flags on these three commands.
func parseGitOpFlags(argv []string, valueFlags map[string]bool) map[string]bool {
	out := map[string]bool{}
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if arg == "--" || !strings.HasPrefix(arg, "-") || arg == "-" {
			break
		}
		nameValue := strings.TrimLeft(arg, "-")
		name, _, hasEquals := strings.Cut(nameValue, "=")
		if name == "" {
			continue
		}
		if valueFlags[name] {
			if !hasEquals && i+1 < len(argv) {
				i++
			}
			continue
		}
		out[name] = true
	}
	return out
}
