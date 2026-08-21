package main

// idempotency.go — `fak idempotency`, the ambiguity-safe executor for mutating
// tool ops (#2093, #8284, part of epic #2063). A repeated proven result replays;
// an apply error blocks as UNKNOWN_APPLIED until operation-specific read-back or
// an explicit operator resolution proves the effect applied or absent.
//
//	fak idempotency run --op issue-create --token $KEY --ledger .idem.jsonl -- gh issue create ...
//	fak idempotency status --op issue-create --token $KEY --ledger .idem.jsonl
//	fak idempotency resolve --op issue-create --token $KEY --ledger .idem.jsonl --absent
//	fak idempotency selfcheck            # the issue-create-after-hang scenario, end to end
//
// This shell does only the I/O the pure internal/idempotency leaf must not: parse
// flags, run the wrapped command, and render. The window/ledger logic is the leaf.

import (
	"bytes"
	"errors"
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
	case "status":
		return runIdempotencyStatus(stdout, stderr, argv[1:])
	case "resolve":
		return runIdempotencyResolve(stdout, stderr, argv[1:])
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
	fmt.Fprint(w, `fak idempotency — ambiguity-safe execution for non-idempotent tool ops (#2093, #8284)

  fak idempotency run --op <op> --token <tok> --ledger <path> [--window <dur>] [--json] -- <command>...
	  Run <command> once for a fresh or PROVEN_ABSENT key. A recent APPLIED key
	  REPLAYS its recorded stdout. PENDING or UNKNOWN_APPLIED refuses to run.
	  Derive the key from --op + --token, or pass --key explicitly.

  fak idempotency status --ledger <path> (--key <key> | --op <op> --token <tok>) [--json]
	  Read the latest state, including unknown states that never expire.

  fak idempotency resolve --ledger <path> (--key <key> | --op <op> --token <tok>)
	  (--applied-result <stdout> | --absent | --unknown) [--json]
	  APPLIED replays the supplied result; ABSENT permits one new apply; UNKNOWN
	  remains blocked.

  fak idempotency selfcheck [--json]
	  Prove replay, UNKNOWN_APPLIED refusal, and read-back resolution end to end.
`)
}

type idempotencyTargetFlags struct {
	op     *string
	token  *string
	key    *string
	ledger *string
}

func addIdempotencyTargetFlags(fs *flag.FlagSet) idempotencyTargetFlags {
	return idempotencyTargetFlags{
		op:     fs.String("op", "", "op label the key is derived from (e.g. issue-create)"),
		token:  fs.String("token", "", "caller-supplied token stable across retries of the same op"),
		key:    fs.String("key", "", "explicit idempotency key (overrides --op/--token derivation)"),
		ledger: fs.String("ledger", "", "path to the JSONL state ledger (required)"),
	}
}

func (f idempotencyTargetFlags) resolve(command string) (string, bool) {
	key := strings.TrimSpace(*f.key)
	if key == "" {
		if strings.TrimSpace(*f.op) == "" || strings.TrimSpace(*f.token) == "" {
			return command + ": need --key, or BOTH --op and --token", false
		}
		key = idempotency.Key(*f.op, *f.token)
	}
	if strings.TrimSpace(*f.ledger) == "" {
		return command + ": --ledger is required", false
	}
	return key, true
}

// runIdempotencyRun wraps one mutating command with keyed dedup.
func runIdempotencyRun(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("idempotency run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "idempotency")
	target := addIdempotencyTargetFlags(fs)
	window := fs.Duration("window", idempotency.DefaultWindow, "dedup window; a key seen within it replays instead of re-running")
	asJSON := fs.Bool("json", false, "emit {replayed,key,op,result} as JSON instead of the raw result")
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}
	cmd := fs.Args()

	key, ok := target.resolve("fak idempotency run")
	if !ok {
		fmt.Fprintln(stderr, key)
		return 2
	}
	if len(cmd) == 0 {
		fmt.Fprintln(stderr, "fak idempotency run: a command is required after `--`")
		return 2
	}

	store, err := idempotency.Open(*target.ledger, *window)
	if err != nil {
		fmt.Fprintf(stderr, "fak idempotency run: %v\n", err)
		return 1
	}
	result, replayed, err := store.Do(key, *target.op, func() (string, error) {
		return runMutatingCommand(cmd, stderr)
	})
	if err != nil {
		if errors.Is(err, idempotency.ErrUnknownApplied) {
			fmt.Fprintf(stderr, "fak idempotency run: UNKNOWN_APPLIED for key %s; the command may have landed and was not re-run: %v\n", shortKey(key), err)
			fmt.Fprintf(stderr, "  inspect: fak idempotency status --ledger %q --key %s\n", *target.ledger, key)
			fmt.Fprintf(stderr, "  resolve: fak idempotency resolve --ledger %q --key %s (--applied-result <stdout> | --absent | --unknown)\n", *target.ledger, key)
			return 1
		}
		fmt.Fprintf(stderr, "fak idempotency run: command was not started or the state ledger failed before apply: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"replayed": replayed,
			"key":      key,
			"op":       *target.op,
			"result":   result,
		}, "fak idempotency run")
	}
	if replayed {
		fmt.Fprintf(stderr, "fak idempotency: replayed original result for key %s (op not re-run)\n", shortKey(key))
	}
	fmt.Fprint(stdout, result)
	return 0
}

func runIdempotencyStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("idempotency status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "idempotency")
	target := addIdempotencyTargetFlags(fs)
	asJSON := fs.Bool("json", false, "emit the durable state as JSON")
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak idempotency status: unexpected positional arguments")
		return 2
	}
	key, ok := target.resolve("fak idempotency status")
	if !ok {
		fmt.Fprintln(stderr, key)
		return 2
	}
	store, err := idempotency.Open(*target.ledger, idempotency.DefaultWindow)
	if err != nil {
		fmt.Fprintf(stderr, "fak idempotency status: %v\n", err)
		return 1
	}
	rec, found, err := store.Status(key)
	if err != nil {
		fmt.Fprintf(stderr, "fak idempotency status: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"found": found, "key": key, "op": rec.Op, "state": rec.State,
			"result": rec.Result, "applied_at": rec.AppliedAt, "updated_at": rec.UpdatedAt,
		}, "fak idempotency status")
	}
	if !found {
		fmt.Fprintf(stdout, "NOT_FOUND\t%s\n", key)
		return 0
	}
	fmt.Fprintf(stdout, "%s\t%s\t%s\n", rec.State, key, rec.Op)
	return 0
}

func runIdempotencyResolve(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("idempotency resolve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "idempotency")
	target := addIdempotencyTargetFlags(fs)
	appliedResult := fs.String("applied-result", "", "prove applied and record this original stdout for replay")
	absent := fs.Bool("absent", false, "prove the effect absent and permit one fresh apply")
	unknown := fs.Bool("unknown", false, "record that read-back remains inconclusive")
	asJSON := fs.Bool("json", false, "emit the resolved durable state as JSON")
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak idempotency resolve: unexpected positional arguments")
		return 2
	}
	key, ok := target.resolve("fak idempotency resolve")
	if !ok {
		fmt.Fprintln(stderr, key)
		return 2
	}
	appliedSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "applied-result" {
			appliedSet = true
		}
	})
	choices := 0
	if appliedSet {
		choices++
	}
	if *absent {
		choices++
	}
	if *unknown {
		choices++
	}
	if choices != 1 {
		fmt.Fprintln(stderr, "fak idempotency resolve: choose exactly one of --applied-result, --absent, or --unknown")
		return 2
	}

	verdict := idempotency.ResolutionApplied
	result := *appliedResult
	if *absent {
		verdict = idempotency.ResolutionAbsent
	}
	if *unknown {
		verdict = idempotency.ResolutionUnknown
	}
	store, err := idempotency.Open(*target.ledger, idempotency.DefaultWindow)
	if err != nil {
		fmt.Fprintf(stderr, "fak idempotency resolve: %v\n", err)
		return 1
	}
	rec, err := store.Resolve(key, func(idempotency.Record) (idempotency.Resolution, string, error) {
		return verdict, result, nil
	})
	if errors.Is(err, idempotency.ErrUnknownApplied) && verdict == idempotency.ResolutionUnknown {
		rec, _, err = store.Status(key)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak idempotency resolve: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"key": rec.Key, "op": rec.Op, "state": rec.State, "result": rec.Result,
		}, "fak idempotency resolve")
	}
	fmt.Fprintf(stdout, "%s\t%s\t%s\n", rec.State, rec.Key, rec.Op)
	return 0
}

// runMutatingCommand runs cmd, capturing its stdout as the recorded result and
// streaming its stderr through. A non-zero exit is ambiguous because the command
// may have completed its side effect before returning the error.
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

// runIdempotencySelfcheck proves replay and ambiguous-response resolution against
// a real temporary ledger.
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

	// An effect that lands before response loss becomes UNKNOWN_APPLIED. A second
	// call cannot apply until read-back proves the original result.
	ambiguousKey := idempotency.Key("issue-create", "response-loss-child")
	_, _, err = store2.Do(ambiguousKey, "issue-create", func() (string, error) {
		filed = append(filed, "response-loss child")
		return "", errors.New("response lost after create")
	})
	if !errors.Is(err, idempotency.ErrUnknownApplied) {
		return fail(fmt.Sprintf("ambiguous apply returned %v, want UNKNOWN_APPLIED", err))
	}
	_, _, err = store2.Do(ambiguousKey, "issue-create", fileIssue("duplicate child"))
	if !errors.Is(err, idempotency.ErrUnknownApplied) || len(filed) != 3 {
		return fail("UNKNOWN_APPLIED retry was not blocked")
	}
	if _, err := store2.Resolve(ambiguousKey, func(idempotency.Record) (idempotency.Resolution, string, error) {
		return idempotency.ResolutionApplied, "created issue #3: response-loss child", nil
	}); err != nil {
		return idempotencySelfcheckFail(stderr, "resolve: ", err)
	}
	resolved, replayed, err := store2.Do(ambiguousKey, "issue-create", fileIssue("duplicate child"))
	if err != nil || !replayed || resolved != "created issue #3: response-loss child" || len(filed) != 3 {
		return fail("proven-applied resolution did not replay without duplication")
	}

	return idempotencyVerdict(stdout, stderr, *asJSON, true,
		"replay deduped; ambiguous apply blocked; proven-applied read-back replayed", filed)
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
