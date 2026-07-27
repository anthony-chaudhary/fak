package main

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ghexec"
	"github.com/anthony-chaudhary/fak/internal/questionledger"
)

// cmdQuestionLedger is the thin shell over internal/questionledger — the
// deterministic labeling authority for docs/questions/asked.jsonl that the
// /question-loop skill defers to (Go port of the retired tools/question_ledger.py).
func cmdQuestionLedger(argv []string) {
	now := time.Now().UTC().Format("20060102")
	os.Exit(questionledger.Run(os.Stdout, os.Stderr, argv, now, runQuestionLedgerGH))
}

// runQuestionLedgerGH is the gh seam used by `ensure-label --apply`. The call
// is a GitHub network round-trip, so it carries a deadline at this call site:
// ghexec kills a wedged `gh` instead of hanging the invocation forever.
func runQuestionLedgerGH(args []string) (string, string, error) {
	cmd, cancel := ghexec.CommandTimeout(context.Background(), ghexec.DefaultTimeout, args...)
	defer cancel()
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	return out.String(), errb.String(), err
}
