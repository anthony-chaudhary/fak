package main

import (
	"os"

	"github.com/anthony-chaudhary/fak/internal/godsplitplan"
)

// cmdGodsplitPlan is the thin shell over internal/godsplitplan — the read-only,
// doc-comment-aware boundary + hazard planner for a behavior-preserving Go split
// its --json plan. Exit 0 on success, 1 on a usage error or unreadable file.
func cmdGodsplitPlan(argv []string) {
	os.Exit(godsplitplan.Run(os.Stdout, os.Stderr, argv))
}
