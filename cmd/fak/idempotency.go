package main

// idempotency.go — `fak idempotency`, the retry-safe executor for non-idempotent
// tool ops (#2093, part of epic #2063). After a tool hang/timeout an agent often
// retries, but a non-idempotent op (create a GitHub issue, push, append to a
// ledger) can DOUBLE-APPLY if the first attempt actually landed before the hang
// signal reached the caller. This verb attaches an idempotency key (derived from
// an op label + a caller-supplied token) and dedupes a repeated key within a
// window against a JSONL ledger, so a post-hang retry is a safe no-op that returns
// the ORIGINAL result.
//
//	fak idempotency run --op issue-create --token $KEY --ledger .idem.jsonl -- gh issue create ...
//	fak idempotency selfcheck            # the issue-create-after-hang scenario, end to end
//
// This shell does only the I/O the pure internal/idempotency leaf must not: parse
// flags, run the wrapped command, and render. The window/ledger logic is the leaf.

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/idempotency"
)

func cmdIdempotency(argv []string) { os.Exit(runIdempotency(os.Stdout, os.Stderr, argv)) }

// runIdempotency dispatches the `fak idempotency` subcommands. Exit codes: 0 ok,
// 1 runtime error, 2 usage error.
func runIdempotency(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		idempotencyUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "run":
		return runIdempotencyRun(stdout, stderr, argv[1:])
	case "selfcheck", "--selfcheck", "-selfcheck":
		return runIdempotencySelfcheck(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		idempotencyUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak idempotency: unknown subcommand %q\n", argv[0])
		idempotencyUsage(stderr)
		return 2
	}
}

func idempotencyUsage(w io.Writer) {
	fmt.Fprint(w, `fak idempotency — retry-safe execution for non-idempotent tool ops (#2093)

  fak idempotency run --op <op> --token <tok> --ledger <path> [--window <dur>] [--json] -- <command>...
      Run <command> once for a fresh key; a repeat of the same key within the
      window REPLAYS the recorded stdout without re-running the command. Derive
      the key from --op + --token, or pass --key explicitly.

  fak idempotency selfcheck [--json]
      Run the issue-create-after-hang scenario end to end and assert a retry
      replays without double-filing while a fresh key proceeds. Exit 0 on PASS.
`)
}

// runIdempotencyRun wraps one mutating command with keyed dedup.
func runIdempotencyRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("idempotency run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "idempotency")
	op := fs.String("op", "", "op label the key is derived from (e.g. issue-create)")
	token := fs.String("token", "", "caller-supplied idempotency token, stable across a retry of the SAME op")
	keyFlag := fs.String("key", "", "explicit idempotency key (overrides --op/--token derivation)")
	ledger := fs.String("ledger", "", "path to the JSONL dedup ledger (required)")
	window := fs.Duration("window", idempotency.DefaultWindow, "dedup window; a key seen within it replays instead of re-running")
	asJSON := fs.Bool("json", false, "emit {replayed,key,op,result} as JSON instead of the raw result")
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}
	cmd := fs.Args()

	key := strings.TrimSpace(*keyFlag)
	if key == "" {
		if strings.TrimSpace(*op) == "" || strings.TrimSpace(*token) == "" {
			fmt.Fprintln(stderr, "fak idempotency run: need --key, or BOTH --op and --token")
			return 2
		}
		key = idempotency.Key(*op, *token)
	}
	if strings.TrimSpace(*ledger) == "" {
		fmt.Fprintln(stderr, "fak idempotency run: --ledger is required")
		return 2
	}
	if len(cmd) == 0 {
		fmt.Fprintln(stderr, "fak idempotency run: a command is required after `--`")
		return 2
	}

	store, err := idempotency.Open(*ledger, *window)
	if err != nil {
		fmt.Fprintf(stderr, "fak idempotency run: %v\n", err)
		return 1
	}
	result, replayed, err := store.Do(key, *op, func() (string, error) {
		return runMutatingCommand(cmd, stderr)
	})
	if err != nil {
		// The op itself failed: nothing is recorded, so it stays retryable.
		fmt.Fprintf(stderr, "fak idempotency run: op failed (not recorded, safe to retry): %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"replayed": replayed,
			"key":      key,
			"op":       *op,
			"result":   result,
		}, "fak idempotency run")
	}
	if replayed {
		fmt.Fprintf(stderr, "fak idempotency: replayed original result for key %s (op not re-run)\n", shortKey(key))
	}
	fmt.Fprint(stdout, result)
	return 0
}

// runMutatingCommand runs cmd, capturing its stdout as the recorded result and
// streaming its stderr through. A non-zero exit is an error, so a failed op is not
// recorded and stays retryable.
func runMutatingCommand(cmd []string, stderr io.Writer) (string, error) {
	c := exec.Command(cmd[0], cmd[1:]...)
	var out bytes.Buffer
	c.Stdout = &out
	c.Stderr = stderr
	if err := c.Run(); err != nil {
		return "", fmt.Errorf("%s: %w", cmd[0], err)
	}
	return out.String(), nil
}

func shortKey(key string) string {
	if len(key) > 12 {
		return key[:12]
	}
	return key
}

// runIdempotencySelfcheck runs the #2093 "Done when" scenario end to end against a
// temp ledger: a retried keyed issue-create after a simulated hang replays the
// original result without double-filing, and a genuinely new op with a fresh key
// proceeds. This is the runnable spine.
func runIdempotencySelfcheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("idempotency selfcheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "idempotency")
	asJSON := fs.Bool("json", false, "emit the selfcheck verdict as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	dir, err := os.MkdirTemp("", "fak-idem-selfcheck-")
	if err != nil {
		fmt.Fprintf(stderr, "fak idempotency selfcheck: %v\n", err)
		return 1
	}
	defer os.RemoveAll(dir)
	ledger := filepath.Join(dir, "idem.jsonl")

	// `filed` is the durable side effect the op would DOUBLE if dedup failed.
	var filed []string
	fileIssue := func(title string) func() (string, error) {
		return func() (string, error) {
			filed = append(filed, title)
			return fmt.Sprintf("created issue #%d: %s", len(filed), title), nil
		}
	}

	fail := func(msg string) int {
		return idempotencyVerdict(stdout, stderr, *asJSON, false, msg, filed)
	}

	// Attempt 1 lands and records; then its signal is "lost" (the hang).
	store, err := idempotency.Open(ledger, idempotency.DefaultWindow)
	if err != nil {
		return idempotencySelfcheckFail(stderr, "", err)
	}
	key := idempotency.Key("issue-create", "epic-2093-file-child")
	res1, replayed1, err := store.Do(key, "issue-create", fileIssue("idempotency keys"))
	if err != nil {
		return idempotencySelfcheckFail(stderr, "attempt 1: ", err)
	}
	fmt.Fprintf(stderr, "  attempt 1: applied=%v result=%q\n", !replayed1, res1)
	if replayed1 {
		return fail("attempt 1 replayed instead of applying")
	}

	// The post-hang retry arrives in a fresh process → reopen the ledger.
	store2, err := idempotency.Open(ledger, idempotency.DefaultWindow)
	if err != nil {
		return idempotencySelfcheckFail(stderr, "", err)
	}
	res2, replayed2, err := store2.Do(key, "issue-create", fileIssue("idempotency keys"))
	if err != nil {
		return idempotencySelfcheckFail(stderr, "retry: ", err)
	}
	fmt.Fprintf(stderr, "  retry after hang: replayed=%v result=%q\n", replayed2, res2)
	if !replayed2 {
		return fail("retry after hang re-applied instead of replaying")
	}
	if res2 != res1 {
		return fail("retry returned a different result than the original")
	}
	if len(filed) != 1 {
		return fail(fmt.Sprintf("op double-applied: %d issues filed, want 1", len(filed)))
	}

	// A genuinely new op with a fresh key proceeds.
	fresh := idempotency.Key("issue-create", "epic-2093-file-sibling")
	res3, replayed3, err := store2.Do(fresh, "issue-create", fileIssue("timeout partial-state child"))
	if err != nil {
		return idempotencySelfcheckFail(stderr, "fresh op: ", err)
	}
	fmt.Fprintf(stderr, "  fresh key: applied=%v result=%q\n", !replayed3, res3)
	if replayed3 {
		return fail("a fresh key replayed instead of proceeding")
	}
	if len(filed) != 2 {
		return fail(fmt.Sprintf("fresh op did not apply: %d issues filed, want 2", len(filed)))
	}

	return idempotencyVerdict(stdout, stderr, *asJSON, true,
		"retry after hang replayed without double-filing; fresh key proceeded", filed)
}

// idempotencySelfcheckFail reports a selfcheck stage that could not RUN -- a ledger that
// would not open, a store call that errored -- which is a different thing from a stage that
// ran and gave the wrong answer (idempotencyVerdict records those). stage is the
// "attempt 1: "-style prefix naming which step died, empty for the ledger opens.
func idempotencySelfcheckFail(stderr io.Writer, stage string, err error) int {
	fmt.Fprintf(stderr, "fak idempotency selfcheck: %s%v\n", stage, err)
	return 1
}

func idempotencyVerdict(stdout, stderr io.Writer, asJSON, pass bool, detail string, filed []string) int {
	if asJSON {
		code := encodeJSONOrFail(stdout, stderr, map[string]any{
			"pass":         pass,
			"detail":       detail,
			"issues_filed": len(filed),
		}, "fak idempotency selfcheck")
		if code != 0 {
			return code
		}
	} else if pass {
		fmt.Fprintf(stdout, "PASS: %s (%d issues filed, no duplicate)\n", detail, len(filed))
	} else {
		fmt.Fprintf(stdout, "FAIL: %s (%d issues filed)\n", detail, len(filed))
	}
	if pass {
		return 0
	}
	return 1
}
