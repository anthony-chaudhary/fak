package main

// session_receipt.go — `fak session receipt <trace>`, the offline SIGNED TERMINAL
// TURN RECEIPT reader (#2415). A harness's final result message is a self-accounting
// turn receipt (cost, turns, denials) a parent must take on trust. This verb folds the
// guard journal's tamper-evident hash chain into a receipt whose WITNESSED numbers
// (denials-by-reason, admitted results, taint high-water, witness gates) are re-derived
// straight from the journal and VERIFIED against it, while the OBSERVED numbers (relayed
// harness turns / provider tokens the operator may attach) ride only as self-reported
// context. A peer, a dispatcher, or dos-witness-claim can then verify the receipt
// WITHOUT trusting the worker — the receipt becomes dos_verify input, not narrative.
//
//	fak session receipt <trace> [--journal PATH] [--json]
//	                    [--turns N] [--prompt-tokens N] [--completion-tokens N]
//
// It is an OFFLINE verb: it never dials a live gateway, so it reads a recorded run's
// receipt from the durable journal alone (defaulting to the newest repo-local guard
// journal, or $FAK_AUDIT_JOURNAL / --journal). The OBSERVED usage flags default to
// zero because those numbers are not journal-derivable — an operator relaying them
// binds them into (and signs them over) the receipt, but they are never treated as
// kernel proof, which is exactly why they carry the OBSERVED label.

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// sessionReceiptDoc is the one-line JSON/human envelope for `fak session receipt`: the
// folded receipt plus the verification verdict, so a scripted consumer branches on
// Verified without re-running VerifyReceipt itself.
type sessionReceiptDoc struct {
	Receipt     agent.Receipt `json:"receipt"`
	Verified    bool          `json:"verified"`
	VerifyError string        `json:"verify_error,omitempty"`
	Journal     string        `json:"journal"`
}

func runSessionReceipt(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session receipt", flag.ContinueOnError)
	fs.SetOutput(stderr)
	journalPath := fs.String("journal", "", "guard journal path (default: $FAK_AUDIT_JOURNAL, else the newest repo-local guard journal)")
	asJSON := fs.Bool("json", false, "emit the receipt as JSON instead of the per-field labeled human view")
	turns := fs.Int("turns", 0, "OBSERVED: relayed harness turn count to bind into the receipt")
	promptTokens := fs.Int("prompt-tokens", 0, "OBSERVED: relayed provider prompt tokens")
	completionTokens := fs.Int("completion-tokens", 0, "OBSERVED: relayed provider completion tokens")

	// The trace is the single leading positional; flags follow it, mirroring the rest
	// of the `fak session` surface (the id/value come first, then flags).
	if len(argv) == 0 || strings.HasPrefix(argv[0], "-") {
		fmt.Fprintln(stderr, "usage: fak session receipt <trace> [--journal PATH] [--json] [--turns N] [--prompt-tokens N] [--completion-tokens N]")
		return 2
	}
	trace := argv[0]
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak session receipt: unexpected argument %q (the trace comes first, then flags)\n", fs.Arg(0))
		return 2
	}

	path := strings.TrimSpace(*journalPath)
	if path == "" {
		path = defaultGuardJournalPath()
	}
	// Segment-aware (#6488): the receipt totals a whole trace, and a long session is
	// exactly the one that rotates — reading the live segment alone would bill a tail.
	rows, err := journal.ReadAllSegments(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak session receipt: read journal %s: %v\n", path, err)
		return 1
	}
	rows = journal.WithoutCutAnchors(rows)

	obs := agent.ObservedUsage{Turns: *turns, PromptTokens: *promptTokens, CompletionTokens: *completionTokens}
	r := agent.BuildReceipt(trace, rows, obs)
	verifyErr := agent.VerifyReceipt(r, rows)

	if *asJSON {
		doc := sessionReceiptDoc{Receipt: r, Verified: verifyErr == nil, Journal: path}
		if verifyErr != nil {
			doc.VerifyError = verifyErr.Error()
		}
		if code := emitSessionJSON(stdout, stderr, doc); code != 0 {
			return code
		}
	} else {
		fmt.Fprintf(stdout, "journal: %s\n", path)
		agent.RenderReceipt(stdout, r, verifyErr)
	}
	// A failed verification is a non-zero exit so a dispatcher's `fak session receipt`
	// check fails loudly on a tampered or chain-broken journal.
	if verifyErr != nil {
		return 1
	}
	return 0
}
