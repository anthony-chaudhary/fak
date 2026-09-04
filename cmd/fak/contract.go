package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func cmdContract(argv []string) { os.Exit(runContract(os.Stdout, os.Stderr, argv)) }

func runContract(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, contractUsage)
		return 2
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "acquire":
		return runContractAcquire(stdout, stderr, rest)
	case "yield":
		return runContractYield(stdout, stderr, rest)
	case "resume":
		return runContractResume(stdout, stderr, rest)
	case "verify":
		return runContractVerify(stdout, stderr, rest)
	case "close":
		return runContractClose(stdout, stderr, rest)
	case "list":
		return runContractList(stdout, stderr, rest)
	case "reap":
		return runContractReap(stdout, stderr, rest)
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, contractUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak contract: unknown subcommand %q\n%s\n", sub, contractUsage)
		return 2
	}
}

const contractUsage = `fak contract - ticket execution contract leases (over internal/leaseref, #11163)

  fak contract acquire <ticket-id> --budget-tokens <N> --verify-cmd <cmd> [--tier <tier>] [--holder <holder>] [--turn-limit <N>] [--worktree-dir <dir>] [--ttl <sec>] [--dir <dir>] [--json]
      Acquire an execution contract for a ticket. Monotonically versioned under
      refs/fak/locks/contract-<ticket_id>. Initial state is EXECUTING.
      Exit: 0 ok, 3 on collision/held by another live holder, 1 on error, 2 on usage error.

  fak contract yield <ticket-id> [--dir <dir>] [--json]
      Transition contract state to YIELDED_IO.

  fak contract resume <ticket-id> [--dir <dir>] [--json]
      Transition contract state back to EXECUTING.

  fak contract verify <ticket-id> [--dir <dir>] [--json]
      Transition contract state to VERIFYING.

  fak contract close <ticket-id> --status <SUCCEEDED|FAILED> [--tokens-used <N>] [--dir <dir>] [--json]
      Close an execution contract with final status SUCCEEDED or FAILED.

  fak contract list [--live] [--json] [--dir <dir>]
      List contract records under refs/fak/locks/contract-*.
      --live filters to unexpired contracts only.
      --json outputs raw ContractRecord JSON array.

  fak contract reap [--dir <dir>] [--json]
      Delete expired contracts past TTL.

Exit codes: 0 on success, 3 on fence/collision refusal, 1 on error, 2 on usage/flag error.`

func isContractRefusal(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, leaseref.ErrContractHeld) ||
		errors.Is(err, leaseref.ErrCASContended) ||
		errors.Is(err, leaseref.ErrContractExpired) {
		return true
	}
	msg := err.Error()
	return strings.Contains(msg, leaseref.ReasonContractHeld) ||
		strings.Contains(msg, leaseref.ReasonContractContended) ||
		strings.Contains(msg, leaseref.ReasonContractExpired)
}

func contractErrorExit(stderr io.Writer, verb string, err error) int {
	fmt.Fprintf(stderr, "fak contract %s: %v\n", verb, err)
	if isContractRefusal(err) {
		return 3
	}
	return 1
}

func extractTicketIDAndArgs(argv []string) (string, []string) {
	if len(argv) == 0 {
		return "", nil
	}
	if !strings.HasPrefix(argv[0], "-") {
		return argv[0], argv[1:]
	}
	return "", argv
}

func parseContractTicketCmd(fs *flag.FlagSet, argv []string, stderr io.Writer) (string, int, bool) {
	ticketID, flagArgs := extractTicketIDAndArgs(argv)
	if err := fs.Parse(flagArgs); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return "", 0, true
		}
		return "", 2, true
	}
	if ticketID == "" && fs.NArg() > 0 {
		ticketID = fs.Arg(0)
		if fs.NArg() > 1 {
			fmt.Fprintf(stderr, "%s: unexpected argument %q\n", fs.Name(), fs.Arg(1))
			return "", 2, true
		}
	} else if ticketID != "" && fs.NArg() > 0 {
		fmt.Fprintf(stderr, "%s: unexpected argument %q\n", fs.Name(), fs.Arg(0))
		return "", 2, true
	}
	if ticketID == "" {
		fmt.Fprintf(stderr, "%s: <ticket-id> is required\n", fs.Name())
		return "", 2, true
	}
	return ticketID, 0, false
}

func runContractAcquire(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak contract acquire", flag.ContinueOnError)
	fs.SetOutput(stderr)
	budgetTokens := fs.Int64("budget-tokens", 0, "token budget for ticket contract execution (required)")
	verifyCmd := fs.String("verify-cmd", "", "command to verify ticket execution (required)")
	tier := fs.String("tier", leaseref.PaceTierCommodity, "pace tier (frontier, commodity, eval_only)")
	holder := fs.String("holder", "", "holder identity (default: local node ID)")
	session := fs.String("session", "", "owning session ID")
	ttl := fs.Int("ttl", leaseref.DefaultContractTTLSeconds, "contract lease TTL in seconds")
	turnLimit := fs.Int("turn-limit", 0, "maximum turns allowed for ticket execution")
	worktreeDir := fs.String("worktree-dir", "", "path to worktree directory")
	baseSHA := fs.String("base-sha", "", "base git commit SHA")
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	asJSON := fs.Bool("json", false, "emit result as JSON")

	ticketID, code, done := parseContractTicketCmd(fs, argv, stderr)
	if done {
		return code
	}
	if *budgetTokens <= 0 {
		fmt.Fprintln(stderr, "fak contract acquire: --budget-tokens must be positive")
		return 2
	}
	if strings.TrimSpace(*verifyCmd) == "" {
		fmt.Fprintln(stderr, "fak contract acquire: --verify-cmd is required")
		return 2
	}

	*dir = pathutil.ExpandTilde(*dir)
	h := strings.TrimSpace(*holder)
	if h == "" {
		if *session != "" {
			h = leaseref.MintHolder(leaseref.LocalNodeID(*dir), *session)
		} else {
			h = leaseref.LocalNodeID(*dir)
		}
	}

	rec := leaseref.ContractRecord{
		TicketID:    ticketID,
		Holder:      h,
		SessionID:   strings.TrimSpace(*session),
		State:       leaseref.ContractStateExecuting,
		PaceTier:    strings.TrimSpace(*tier),
		TokenBudget: *budgetTokens,
		TurnLimit:   *turnLimit,
		WorktreeDir: strings.TrimSpace(*worktreeDir),
		BaseSHA:     strings.TrimSpace(*baseSHA),
		VerifyCmd:   strings.TrimSpace(*verifyCmd),
		TTLSeconds:  *ttl,
	}
	store := leaseref.NewInDir(*dir)
	acquired, err := store.AcquireContract(context.Background(), rec, time.Now())
	if err != nil {
		return contractErrorExit(stderr, "acquire", err)
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, acquired, "fak contract acquire")
	}
	fmt.Fprintf(stdout, "acquired contract %s (holder=%s, state=%s, generation=%d)\n",
		acquired.TicketID, acquired.Holder, acquired.State, acquired.Generation)
	return 0
}

func runContractYield(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak contract yield", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	asJSON := fs.Bool("json", false, "emit result as JSON")

	ticketID, code, done := parseContractTicketCmd(fs, argv, stderr)
	if done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	updated, err := store.UpdateContractState(context.Background(), ticketID, leaseref.ContractStateYieldedIO, time.Now())
	if err != nil {
		return contractErrorExit(stderr, "yield", err)
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, updated, "fak contract yield")
	}
	fmt.Fprintf(stdout, "contract %s state -> %s\n", updated.TicketID, updated.State)
	return 0
}

func runContractResume(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak contract resume", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	asJSON := fs.Bool("json", false, "emit result as JSON")

	ticketID, code, done := parseContractTicketCmd(fs, argv, stderr)
	if done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	updated, err := store.UpdateContractState(context.Background(), ticketID, leaseref.ContractStateExecuting, time.Now())
	if err != nil {
		return contractErrorExit(stderr, "resume", err)
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, updated, "fak contract resume")
	}
	fmt.Fprintf(stdout, "contract %s state -> %s\n", updated.TicketID, updated.State)
	return 0
}

func runContractVerify(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak contract verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	asJSON := fs.Bool("json", false, "emit result as JSON")

	ticketID, code, done := parseContractTicketCmd(fs, argv, stderr)
	if done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	updated, err := store.UpdateContractState(context.Background(), ticketID, leaseref.ContractStateVerifying, time.Now())
	if err != nil {
		return contractErrorExit(stderr, "verify", err)
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, updated, "fak contract verify")
	}
	fmt.Fprintf(stdout, "contract %s state -> %s\n", updated.TicketID, updated.State)
	return 0
}

func runContractClose(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak contract close", flag.ContinueOnError)
	fs.SetOutput(stderr)
	status := fs.String("status", "", "terminal status: SUCCEEDED or FAILED (required)")
	tokensUsed := fs.Int64("tokens-used", 0, "total tokens used during execution")
	holder := fs.String("holder", "", "holder identity")
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	asJSON := fs.Bool("json", false, "emit result as JSON")

	ticketID, code, done := parseContractTicketCmd(fs, argv, stderr)
	if done {
		return code
	}
	if strings.TrimSpace(*status) == "" {
		fmt.Fprintln(stderr, "fak contract close: --status is required (SUCCEEDED or FAILED)")
		return 2
	}
	st := leaseref.ContractState(strings.ToUpper(strings.TrimSpace(*status)))
	if st != leaseref.ContractStateSucceeded && st != leaseref.ContractStateFailed {
		fmt.Fprintf(stderr, "fak contract close: --status must be SUCCEEDED or FAILED (got %q)\n", *status)
		return 2
	}
	if *tokensUsed < 0 {
		fmt.Fprintln(stderr, "fak contract close: --tokens-used cannot be negative")
		return 2
	}

	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	rec := leaseref.ContractRecord{
		TicketID:   ticketID,
		State:      st,
		TokensUsed: *tokensUsed,
		Holder:     strings.TrimSpace(*holder),
	}
	updated, err := store.UpdateContract(context.Background(), rec, time.Now())
	if err != nil {
		return contractErrorExit(stderr, "close", err)
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, updated, "fak contract close")
	}
	fmt.Fprintf(stdout, "closed contract %s (state=%s, tokens_used=%d)\n",
		updated.TicketID, updated.State, updated.TokensUsed)
	return 0
}

func runContractList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak contract list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	liveOnly := fs.Bool("live", false, "list unexpired contracts only")
	asJSON := fs.Bool("json", false, "emit records as JSON")
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	ctx := context.Background()

	var (
		recs []leaseref.ContractRecord
		err  error
	)
	now := time.Now()
	if *liveOnly {
		recs, _, err = store.LiveContracts(ctx, now)
	} else {
		recs, err = store.ListContracts(ctx)
	}
	if err != nil {
		return contractErrorExit(stderr, "list", err)
	}
	if *asJSON {
		if recs == nil {
			recs = []leaseref.ContractRecord{}
		}
		return encodeJSONOrFail(stdout, stderr, recs, "fak contract list")
	}
	if len(recs) == 0 {
		fmt.Fprintln(stdout, "no contracts under refs/fak/locks/contract-*")
		return 0
	}
	for _, r := range recs {
		status := "LIVE"
		if r.Expired(now) {
			status = "EXPIRED"
		}
		fmt.Fprintf(stdout, "%-20s %-12s %-15s budget=%-8d used=%-8d gen=%-3d %s\n",
			r.TicketID, r.State, r.Holder, r.TokenBudget, r.TokensUsed, r.Generation, status)
	}
	return 0
}

func runContractReap(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak contract reap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	asJSON := fs.Bool("json", false, "emit result as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	reaped, err := store.ReapContracts(context.Background(), time.Now())
	if err != nil {
		return contractErrorExit(stderr, "reap", err)
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"reaped": reaped,
			"count":  len(reaped),
		}, "fak contract reap")
	}
	fmt.Fprintf(stdout, "reaped %d expired contract(s)\n", len(reaped))
	return 0
}
