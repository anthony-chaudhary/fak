package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

const outputSchema = "fak-auditreceipt-verification/v1"

type verificationOutput struct {
	Schema       string                               `json:"schema"`
	Verdict      string                               `json:"verdict"`
	Path         string                               `json:"path"`
	Rows         int                                  `json:"rows,omitempty"`
	UniqueAudits int                                  `json:"unique_audits,omitempty"`
	HeadHash     string                               `json:"head_hash,omitempty"`
	Cursor       *modelroute.AuditReceiptLedgerCursor `json:"cursor,omitempty"`
	Error        string                               `json:"error,omitempty"`
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr *os.File) int {
	if len(args) != 1 {
		fmt.Fprintln(stderr, "usage: auditreceipt LEDGER.jsonl")
		return 2
	}

	path := args[0]
	verification, err := modelroute.VerifyAuditReceiptLedger(path)
	if err != nil {
		verdict := "CORRUPT"
		var integrityErr *modelroute.AuditReceiptLedgerIntegrityError
		if errors.As(err, &integrityErr) && integrityErr.Integrity.TornTail {
			verdict = "TORN"
		}
		emit(stdout, verificationOutput{
			Schema:  outputSchema,
			Verdict: verdict,
			Path:    path,
			Error:   err.Error(),
		})
		return 1
	}

	emit(stdout, verificationOutput{
		Schema:       outputSchema,
		Verdict:      "CLEAN",
		Path:         path,
		Rows:         verification.Rows,
		UniqueAudits: verification.UniqueAudits,
		HeadHash:     verification.HeadHash,
		Cursor:       &verification.Cursor,
	})
	return 0
}

func emit(stdout *os.File, output verificationOutput) {
	if err := json.NewEncoder(stdout).Encode(output); err != nil {
		fmt.Fprintf(os.Stderr, "auditreceipt: encode result: %v\n", err)
	}
}
