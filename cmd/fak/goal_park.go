package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/goalpark"
)

// goalParkStore is the one place the on-disk park location is named, so the
// writer (guard's long-Retry-After observer), the reader (guard's supervision
// loop) and this CLI can never point at different directories.
func goalParkStore() goalpark.Store {
	return goalpark.Store{Dir: filepath.Join(repoRoot(), ".fak", "goal-park")}
}

// parkGoalIdentity derives the (goal, account) pair a park record is scoped by,
// from exactly the env the dispatcher hands the child. It mirrors — and is the
// intended single source for — guard.go's park TEMPLATE, so the identity a record
// is written under can never drift from the identity the supervision loop matches
// it against; that drift is what left every live record carrying a blank account
// and therefore walling every account on the lane. goal falls back to the lane
// when the dispatcher named no goal; account has NO fallback on purpose, because
// inventing an identity here would resurrect exactly the account-blind park this
// scoping removes — an unattributable run is one that walls nobody.
func parkGoalIdentity() (goal, account string) {
	goal = strings.TrimSpace(os.Getenv("DISPATCH_GOAL"))
	if goal == "" {
		goal = strings.TrimSpace(os.Getenv("DISPATCH_LANE"))
	}
	return goal, strings.TrimSpace(os.Getenv("DISPATCH_ACCOUNT"))
}

func cmdGoalPark(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: fak goal-park status|claim --goal G [--supervisor ID] [--json]")
		return
	}
	verb := args[0]
	fs := flag.NewFlagSet("goal-park "+verb, flag.ExitOnError)
	goal := fs.String("goal", "", "goal identity")
	supervisor := fs.String("supervisor", "fak-supervisor", "claim owner")
	asJSON := fs.Bool("json", false, "JSON output")
	_ = fs.Parse(args[1:])
	store := goalParkStore()
	var out any
	var err error
	switch verb {
	case "status":
		if *goal != "" {
			out, err = store.Load(*goal)
		} else {
			out, err = store.List()
		}
	case "claim":
		if *goal == "" {
			err = fmt.Errorf("--goal is required")
		} else {
			out, err = store.ClaimDue(*goal, *supervisor, time.Now())
		}
	default:
		err = fmt.Errorf("unknown goal-park verb %q", verb)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak goal-park:", err)
		return
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(out)
		return
	}
	b, _ := json.MarshalIndent(out, "", "  ")
	fmt.Println(string(b))
}
