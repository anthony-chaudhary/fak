package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

func runIssueReconcile(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("issue reconcile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "JSON scope snapshot file")
	nowRaw := fs.String("now", "", "RFC3339 evaluation time (default: current UTC time)")
	jsonOut := fs.Bool("json", false, "emit machine-readable result")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *file == "" {
		fmt.Fprintln(stderr, "fak-dev issue reconcile: --file is required")
		return 2
	}
	data, err := os.ReadFile(*file)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue reconcile: %v\n", err)
		return 1
	}
	var snapshot issuepolicy.ScopeSnapshot
	if err = json.Unmarshal(data, &snapshot); err != nil {
		fmt.Fprintf(stderr, "fak-dev issue reconcile: decode: %v\n", err)
		return 2
	}
	now := time.Now().UTC()
	if *nowRaw != "" {
		now, err = time.Parse(time.RFC3339, *nowRaw)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue reconcile: --now: %v\n", err)
			return 2
		}
	}
	result := issuepolicy.ReconcileScope(snapshot, now)
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(result)
	} else {
		fmt.Fprintf(stdout, "scope-reconcile: %s key=%s production_credit_current=%t action=%s\n", result.Status, result.Key, result.ProductionCreditCurrent, result.Action)
		for _, change := range result.Changes {
			fmt.Fprintf(stdout, "  change: %s %s — %s\n", change.Dimension, change.Kind, change.Detail)
		}
		for _, gap := range result.Gaps {
			fmt.Fprintf(stdout, "  gap: %s — %s\n", gap.Dimension, gap.Reason)
		}
		for _, unknown := range result.Unknown {
			fmt.Fprintf(stdout, "  unknown: %s\n", unknown)
		}
	}
	if result.ProductionCreditCurrent {
		return 0
	}
	return 3
}
