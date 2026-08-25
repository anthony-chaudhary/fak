package main

// `fak conpty` — the searchable status surface for the pwsh 0xE9 FailFast crash
// class (#3402, parent #2170). It resolves the ConPTY pair on the launch PATH,
// reads each FileVersion, and compares it to the known-good floor. A stale pair
// is the verified root cause of the crash, so `--strict` turns the warning into a
// refusal a fleet launcher can gate on.
//
// Exit 0 = pass or nothing to judge, 1 = defect found (or refused under
// --strict), 2 = usage error.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const conptySchema = "fak-conpty/1"

type conptyPayload struct {
	Schema     string                       `json:"schema"`
	OK         bool                         `json:"ok"`
	Verdict    string                       `json:"verdict"`
	Reason     string                       `json:"reason,omitempty"`
	Floor      string                       `json:"floor"`
	Strict     bool                         `json:"strict"`
	SearchDirs int                          `json:"search_dirs"`
	Components []windowgate.ConPTYComponent `json:"components"`
	NextAction string                       `json:"next_action,omitempty"`
	Issue      string                       `json:"issue"`
}

// conptyBundleDirs collects repeated -bundle flags.
type conptyBundleDirs []string

func (d *conptyBundleDirs) String() string { return strings.Join(*d, string(os.PathListSeparator)) }

func (d *conptyBundleDirs) Set(v string) error {
	for _, p := range filepath.SplitList(v) {
		if p = strings.TrimSpace(p); p != "" {
			*d = append(*d, p)
		}
	}
	return nil
}

// conptySearchDirs puts operator-named bundles ahead of the live launch PATH:
// an explicitly named package directory is the pair the terminal really loads.
// FAK_CONPTY_BUNDLE_DIRS supplies the same list from the environment.
func conptySearchDirs(bundles conptyBundleDirs) []string {
	var dirs []string
	dirs = append(dirs, bundles...)
	dirs = append(dirs, filepath.SplitList(os.Getenv("FAK_CONPTY_BUNDLE_DIRS"))...)
	dirs = append(dirs, windowgate.DefaultConPTYSearchDirs()...)
	return dirs
}

func cmdConPTY(argv []string) { os.Exit(runConPTY(os.Stdout, os.Stderr, argv)) }

func runConPTY(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && (argv[0] == "check" || argv[0] == "report") {
		argv = argv[1:]
	}
	fs := flag.NewFlagSet("conpty", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	strict := fs.Bool("strict", false, "refuse (exit 1) on a stale pair instead of warning")
	floor := fs.String("floor", "", "override the known-good version floor (default "+windowgate.ConPTYVersionFloor+")")
	var bundles conptyBundleDirs
	fs.Var(&bundles, "bundle", "search this terminal package directory first (repeatable); "+
		"needed because %ProgramFiles%\\WindowsApps denies enumeration to an unelevated user")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak conpty: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	rep := windowgate.ConPTYPreflight(windowgate.ConPTYOptions{
		SearchDirs: conptySearchDirs(bundles),
		Floor:      strings.TrimSpace(*floor),
		Strict:     *strict,
	})
	payload := conptyPayload{
		Schema:     conptySchema,
		OK:         rep.OK(),
		Verdict:    rep.Verdict,
		Reason:     rep.Reason,
		Floor:      rep.Floor,
		Strict:     rep.Strict,
		SearchDirs: rep.SearchDirs,
		Components: rep.Components,
		NextAction: rep.NextAction,
		Issue:      "https://github.com/anthony-chaudhary/fak/issues/3402",
	}

	if code := emitJSONOrRender(stdout, stderr, "fak conpty", *asJSON, payload, func(w io.Writer) {
		fmt.Fprintln(w, renderConPTY(payload))
	}); code != 0 {
		return code
	}
	if !payload.OK {
		return 1
	}
	return 0
}

func renderConPTY(p conptyPayload) string {
	var b strings.Builder
	fmt.Fprintf(&b, "conpty: %s", p.Verdict)
	if p.Reason != "" {
		fmt.Fprintf(&b, " (%s)", p.Reason)
	}
	fmt.Fprintf(&b, "\n  floor: >= %s   search dirs: %d\n", p.Floor, p.SearchDirs)
	for _, c := range p.Components {
		switch {
		case !c.Found:
			fmt.Fprintf(&b, "  %-16s not on the launch PATH\n", c.Name)
		case c.Compared == "":
			fmt.Fprintf(&b, "  %-16s %s  [%s: %s]\n", c.Name, c.Path, c.Reason, c.Error)
		default:
			mark := "ok"
			if c.Stale {
				mark = "STALE"
			}
			fmt.Fprintf(&b, "  %-16s %s\n", c.Name, c.Path)
			fmt.Fprintf(&b, "  %-16s   compared %s (%s)  file=%s product=%s  [%s]\n",
				"", c.Compared, c.ComparedSource, orDash(c.FileVersion), orDash(c.ProductVersion), mark)
		}
	}
	if p.NextAction != "" {
		fmt.Fprintf(&b, "  next: %s\n", p.NextAction)
	}
	fmt.Fprintf(&b, "  crash class: pwsh FailFast 0xE9 \"No process is on the other end of the pipe\" (%s)", p.Issue)
	return b.String()
}
