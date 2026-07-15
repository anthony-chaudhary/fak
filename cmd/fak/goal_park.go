package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/goalpark"
)

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
	store := goalpark.Store{Dir: filepath.Join(repoRoot(), ".fak", "goal-park")}
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
