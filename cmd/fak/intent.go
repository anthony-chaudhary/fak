package main

// fak intent -- the task-claim collision check (#2155), the CLI surface over the
// intent leases in internal/leaseref (refs/fak/locks/intent-<key>).
//
// dos arbitrate + `fak leaseref` stop two agents editing the same FILES; this verb
// stops two agents fixing the same ISSUE/BUG in different files — both would do the
// whole task and one dispatch is wasted. Claim the intent when you PICK the task,
// before any tokens are spent:
//
//   fak intent claim --target "issue #2155" --holder "$ME" [--session S] [--ttl 3600]
//     -> ok: the target is yours (renewing your own claim also lands here)
//     -> exit 3 + INTENT_COLLISION naming the live incumbent (holder, session, age)
//   fak intent release --target "issue #2155"     when shipped or abandoned
//   fak intent list                               every claimed target, live/expired
//
// Cross-machine visibility rides the same fetch/push as the lock leases:
//   git fetch origin '+refs/fak/locks/*:refs/fak/locks/*'   before claiming
//   git push  origin 'refs/fak/locks/intent-*:refs/fak/locks/intent-*'   after
//
// Documented in docs/cli-reference.md.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func cmdIntent(argv []string) { os.Exit(runIntent(os.Stdout, os.Stderr, argv)) }

func runIntent(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, intentUsage)
		return 2
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "claim":
		return runIntentClaim(stdout, stderr, rest)
	case "release":
		return runIntentRelease(stdout, stderr, rest)
	case "list":
		return runIntentList(stdout, stderr, rest)
	case "reap":
		return runIntentReap(stdout, stderr, rest)
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, intentUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak intent: unknown subcommand %q\n%s\n", sub, intentUsage)
		return 2
	}
}

const intentUsage = `fak intent - intent-level collision check at task claim (#2155)

  fak intent claim --target T --holder H [--session S] [--ttl SEC] [--dir DIR]
      Claim a work TARGET (an issue: "#2155" / "issue 2155" / "2155"; or a free-form
      bug signature) BEFORE spending a turn on it. Admitted when the target is free,
      lapsed, or already yours (a renew). Refused INTENT_COLLISION when a LIVE peer
      holds it — the verdict names the incumbent (holder, session, its own target
      words, age) so you can verify and pick different work. TTL defaults to 3600s
      (the fleet peer-fix window); an intent is short-lived by design.

  fak intent release --target T [--dir DIR]
      Release the claim when the target ships or is abandoned. Idempotent.

  fak intent list [--dir DIR]
      Every intent under refs/fak/locks/intent-*, with live/expired status (JSON).

  fak intent reap [--dir DIR]
      Delete the lapsed (reapable) intents. Also folded into 'fak leaseref reap'.

This COMPLEMENTS the file-tree lease ('fak leaseref', dos arbitrate): claim the intent
when you pick the task, take the tree lease when you start editing. Same honest
boundary as the lock leases: visibility rides ordinary git fetch/push of
refs/fak/locks/*; same-host claims are compare-and-swap atomic; a same-fetch-window
cross-machine race is surfaced, not arbitrated.
Exit: 0 ok, 2 usage error, 1 a git/store failure, 3 a structured refusal (INTENT_COLLISION).`

// intentResult is the claim JSON shape: the deny-as-value verdict plus, on admit, the
// written record — the same envelope discipline as leaseref's fencedResult.
type intentResult struct {
	Verdict leaseref.IntentVerdict `json:"verdict"`
	Record  *leaseref.IntentRecord `json:"record,omitempty"`
}

func runIntentClaim(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak intent claim", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	target := fs.String("target", "", "the work target: an issue reference or a bug signature")
	holder := fs.String("holder", "", "claimant identity (machine/session); names you in a peer's refusal")
	session := fs.String("session", "", "owning session id (refs/fak/locks/session-<id>) for liveness classification")
	ttl := fs.Int64("ttl", 0, "claim lifetime in seconds (0 = the 3600s default; intents are short-lived)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *target == "" || *holder == "" {
		fmt.Fprintln(stderr, "fak intent claim: --target and --holder are required")
		return 2
	}
	store := leaseref.NewInDir(*dir)
	rec, v, err := store.ClaimIntent(context.Background(), leaseref.IntentRecord{
		Target:     *target,
		Holder:     *holder,
		SessionID:  *session,
		TTLSeconds: *ttl,
	}, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak intent claim: %v\n", err)
		return 1
	}
	out := intentResult{Verdict: v}
	if v.OK {
		out.Record = &rec
	}
	if code := emitLeaserefJSON(stdout, stderr, out, "intent claim"); code != 0 {
		return code
	}
	if !v.OK {
		return leaserefRefused
	}
	return 0
}

func runIntentRelease(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak intent release", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	target := fs.String("target", "", "the work target to release")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	if *target == "" {
		fmt.Fprintln(stderr, "fak intent release: --target is required")
		return 2
	}
	store := leaseref.NewInDir(*dir)
	if err := store.ReleaseIntent(context.Background(), *target); err != nil {
		fmt.Fprintf(stderr, "fak intent release: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "released intent %s\n", leaseref.IntentKey(*target))
	return 0
}

func runIntentList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak intent list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	recs, err := store.ListIntents(context.Background())
	if err != nil {
		fmt.Fprintf(stderr, "fak intent list: %v\n", err)
		return 1
	}
	now := time.Now()
	rows := make([]map[string]any, 0, len(recs))
	for _, r := range recs {
		rows = append(rows, map[string]any{
			"key":           r.Key,
			"target":        r.Target,
			"holder":        r.Holder,
			"session_id":    r.SessionID,
			"acquired_unix": r.AcquiredAt,
			"renewed_unix":  r.RenewedAt,
			"ttl_seconds":   r.TTLSeconds,
			"live":          !r.Expired(now),
		})
	}
	return emitLeaserefJSON(stdout, stderr, rows, "intent list")
}

func runIntentReap(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak intent reap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo dir (default: git discovery from cwd)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	*dir = pathutil.ExpandTilde(*dir)
	store := leaseref.NewInDir(*dir)
	reaped, err := store.ReapIntents(context.Background(), time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak intent reap: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "reaped %d lapsed intent(s)\n", len(reaped))
	return 0
}
