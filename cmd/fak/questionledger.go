package main

import (
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/questionledger"
)

// cmdQuestionLedger is the thin shell over internal/questionledger — the
// deterministic labeling authority for docs/questions/asked.jsonl that the
// /question-loop skill defers to (Go port of the retired tools/question_ledger.py).
func cmdQuestionLedger(argv []string) {
	now := time.Now().UTC().Format("20060102")
	os.Exit(questionledger.Run(os.Stdout, os.Stderr, argv, now, runQuestionLedgerGH))
}

// runQuestionLedgerGH is the gh seam used by `ensure-label --apply`.
func runQuestionLedgerGH(args []string) (string, string, error) {
	cmd := exec.Command("gh", args...)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	return out.String(), errb.String(), err
}
