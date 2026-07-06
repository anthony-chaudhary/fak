package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

func printRotationNoCandidate(w io.Writer, prefix string, plan accounts.RotationResult) {
	if len(plan.Pool) == 0 {
		fmt.Fprintf(w, "%s: no eligible accounts in rotation (every seat is reserved, disabled, tombstoned, or has no live credentials)\n", prefix)
		return
	}
	if walled := rotationRuntimeWalledNames(plan); len(walled) > 0 {
		fmt.Fprintf(w, "%s: no runtime-launchable account in rotation; known usage/weekly-capped bucket(s): %s. Wait for reset or move the launch role to an account with room.\n",
			prefix, strings.Join(walled, ", "))
		return
	}
	fmt.Fprintf(w, "%s: only one account bucket in rotation (%s) - nowhere else to rotate; enroll another with `fak accounts add`\n",
		prefix, plan.Pool[0].Name)
}

func rotationRuntimeWalledNames(plan accounts.RotationResult) []string {
	var out []string
	for _, seat := range plan.Pool {
		if seat.Headroom != nil && *seat.Headroom < 0 {
			out = append(out, seat.Name)
		}
	}
	return out
}
