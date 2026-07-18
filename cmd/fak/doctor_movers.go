package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modver"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// runDoctorMovers renders the `fak doctor movers` section (#2472): the fastest-
// moving modules and the dormant-with-open-issues candidates, folded purely from
// the append-only module-versions ledger that `fak version modules --stamp`
// writes. It reads only the ledger (and an optional --issues feed) — no git, no
// live snapshot, no network issue fetch — so it is a read-only operator dogfood
// surface, off the hot path, the modver analogue of the answer-health doctor.
//
// Exit 0 = rendered, 1 = the ledger could not be read, 2 = usage error.
func runDoctorMovers(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("doctor movers", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	ledger := fs.String("ledger", defaultModverLedger, "ledger path (repo-relative unless absolute)")
	issuesPath := fs.String("issues", "", `open-issue feed as JSON [{"number":N,"title":"...","paths":["internal/x/y.go"]}] to cross-reference for dormancy`)
	top := fs.Int("top", modver.DefaultMoversTop, "how many top movers / dormant candidates to show")
	dormantDays := fs.Int("dormant-days", 0, "idle-days window for a dormant module (0 = default 30)")
	asJSON := fs.Bool("json", false, "emit the movers section as JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak doctor movers: unexpected args: %v\n", fs.Args())
		return 2
	}

	path := pathutil.ExpandTilde(*ledger)
	if !filepath.IsAbs(path) {
		root := resolveRoot(pathutil.ExpandTilde(*dir))
		if root == "" {
			fmt.Fprintln(stderr, "fak doctor movers: could not resolve git repo root")
			return 2
		}
		path = filepath.Join(root, path)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Fprintf(stderr, "fak doctor movers: no ledger at %s — run `fak version modules --stamp` first\n", path)
			return 1
		}
		fmt.Fprintf(stderr, "fak doctor movers: %v\n", err)
		return 1
	}

	var issues []modver.OpenIssue
	if *issuesPath != "" {
		ib, ierr := os.ReadFile(pathutil.ExpandTilde(*issuesPath))
		if ierr != nil {
			fmt.Fprintf(stderr, "fak doctor movers: read --issues: %v\n", ierr)
			return 2
		}
		if jerr := json.Unmarshal(ib, &issues); jerr != nil {
			fmt.Fprintf(stderr, "fak doctor movers: parse --issues: %v\n", jerr)
			return 2
		}
	}

	sec := modver.Movers(b, issues, time.Now().UTC(), *dormantDays, *top)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, sec, "fak doctor movers")
	}
	fmt.Fprint(stdout, sec.Render())
	return 0
}
