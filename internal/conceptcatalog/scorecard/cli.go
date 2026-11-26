package scorecard

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// ResolvePath is pathlib's Path.resolve(): absolute, links followed where the
// path exists.
func ResolvePath(p string) string {
	abs, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	if real, err := filepath.EvalSymlinks(abs); err == nil {
		return real
	}
	return abs
}

// Collect folds the catalog under dataDir (default <workspace>/DataDirRel)
// against workspace. Both paths are used as given; callers resolve them.
func Collect(workspace, dataDir string) *Payload {
	if dataDir == "" {
		dataDir = filepath.Join(workspace, filepath.FromSlash(DataDirRel))
	}
	data, msg := LoadData(dataDir)
	tree := LoadTree(workspace)
	return BuildPayload(workspace, data, tree, msg)
}

type auditErrorPayload struct {
	Schema     string   `json:"schema"`
	OK         bool     `json:"ok"`
	Verdict    string   `json:"verdict"`
	Finding    string   `json:"finding"`
	Reason     string   `json:"reason"`
	NextAction string   `json:"next_action"`
	Workspace  string   `json:"workspace"`
	Corpus     struct{} `json:"corpus"`
	Kpis       []KPI    `json:"kpis"`
}

// JSON renders the payload exactly as `json.dumps(payload, indent=2)` does.
func (p *Payload) JSON() ([]byte, error) {
	var v any = p
	if p.Corpus == nil {
		v = auditErrorPayload{Schema: p.Schema, OK: p.OK, Verdict: p.Verdict, Finding: p.Finding,
			Reason: p.Reason, NextAction: p.NextAction, Workspace: p.Workspace, Kpis: []KPI{}}
	}
	var compact bytes.Buffer
	enc := json.NewEncoder(&compact)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	var out bytes.Buffer
	out.Grow(compact.Len() * 2)
	if err := json.Indent(&out, bytes.TrimRight(compact.Bytes(), "\n"), "", "  "); err != nil {
		return nil, err
	}
	return asciiJSON(out.Bytes()), nil
}

type cliArgs struct {
	workspace, lookup, compare, markdownDir, data, stamp string
	json, chart, critical, rollup, gaps, pairs, index    bool
	limit                                                int
}

type cliOption struct {
	name   string
	isBool bool
	set    func(*cliArgs, string) error
	help   string
}

var cliOptions = []cliOption{
	{"--workspace", false, func(a *cliArgs, v string) error { a.workspace = v; return nil }, "workspace root (default: repo root)"},
	{"--json", true, func(a *cliArgs, _ string) error { a.json = true; return nil }, "emit machine-readable JSON"},
	{"--chart", true, func(a *cliArgs, _ string) error { a.chart = true; return nil }, "an at-a-glance ASCII chart"},
	{"--critical", true, func(a *cliArgs, _ string) error { a.critical = true; return nil }, "the worst-first clarity backlog"},
	{"--rollup", true, func(a *cliArgs, _ string) error { a.rollup = true; return nil }, "the hierarchy roll-up (namespace at its abstraction heads)"},
	{"--gaps", true, func(a *cliArgs, _ string) error { a.gaps = true; return nil }, "the coverage backlog (unpositioned tree tokens)"},
	{"--pairs", true, func(a *cliArgs, _ string) error { a.pairs = true; return nil }, "the pairwise separation backlog (confusable name-pairs)"},
	{"--index", true, func(a *cliArgs, _ string) error { a.index = true; return nil }, "the name -> concept lookup index"},
	{"--lookup", false, func(a *cliArgs, v string) error { a.lookup = v; return nil }, "resolve ONE name through the index (implies --index)"},
	{"--limit", false, func(a *cliArgs, v string) error {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("argument --limit: invalid int value: %s", pyStrString(v))
		}
		a.limit = n
		return nil
	}, "cap the --index listing (0 = all)"},
	{"--compare", false, func(a *cliArgs, v string) error { a.compare = v; return nil }, "baseline JSON to prove disambiguation-debt dropped"},
	{"--markdown-dir", false, func(a *cliArgs, v string) error { a.markdownDir = v; return nil }, "regenerate the doc folder (e.g. " + GeneratedDocDir + ")"},
	{"--data", false, func(a *cliArgs, v string) error { a.data = v; return nil }, "data directory (default: " + DataDirRel + ")"},
	{"--stamp", false, func(a *cliArgs, v string) error { a.stamp = v; return nil }, "optional stamp embedded in the generated doc"},
}

const cliUsage = "usage: concept_disambiguation_scorecard [-h] [--workspace WORKSPACE] [--json] [--chart] [--critical] [--rollup] [--gaps] [--pairs] [--index] [--lookup LOOKUP] [--limit LIMIT] [--compare COMPARE] [--markdown-dir MARKDOWN_DIR] [--data DATA] [--stamp STAMP]"

const cliDescription = "Concept-disambiguation scorecard (read-only unless --markdown-dir)."

// argparse's default layout for an 80-column terminal: help text starts at
// column 24 and wraps at 78.
const (
	cliHelpColumn = 24
	cliHelpWidth  = 78
)

// cliHelp renders the -h/--help page: usage, description, and one described
// line per option.
func cliHelp() string {
	var b strings.Builder
	b.WriteString(cliUsage + "\n\n" + cliDescription + "\n\noptions:\n")
	writeCLIHelpEntry(&b, "-h, --help", "show this help message and exit")
	for _, o := range cliOptions {
		invocation := o.name
		if !o.isBool {
			invocation += " " + strings.ToUpper(strings.ReplaceAll(strings.TrimPrefix(o.name, "--"), "-", "_"))
		}
		writeCLIHelpEntry(&b, invocation, o.help)
	}
	return b.String()
}

func writeCLIHelpEntry(b *strings.Builder, invocation, help string) {
	pad := strings.Repeat(" ", cliHelpColumn)
	b.WriteString("  " + invocation)
	first := true
	if len(invocation)+2 > cliHelpColumn-2 {
		b.WriteString("\n")
		first = false
	}
	line := ""
	flush := func() {
		if first {
			b.WriteString(strings.Repeat(" ", cliHelpColumn-2-len(invocation)) + line + "\n")
			first = false
		} else {
			b.WriteString(pad + line + "\n")
		}
		line = ""
	}
	for _, word := range strings.Fields(help) {
		if line != "" && cliHelpColumn+len(line)+1+len(word) > cliHelpWidth {
			flush()
		}
		if line != "" {
			line += " "
		}
		line += word
	}
	if line != "" || first {
		flush()
	}
}

// parseCLI accepts the argparse surface: --opt value, --opt=value, and any
// unambiguous prefix of a long option.
func parseCLI(argv []string) (cliArgs, bool, error) {
	var a cliArgs
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if arg == "-h" || arg == "--help" {
			return a, true, nil
		}
		if !strings.HasPrefix(arg, "--") || arg == "--" {
			return a, false, fmt.Errorf("unrecognized arguments: %s", strings.Join(argv[i:], " "))
		}
		name, value, hasValue := strings.Cut(arg, "=")
		var match *cliOption
		var candidates []string
		for k := range cliOptions {
			o := &cliOptions[k]
			if o.name == name {
				match, candidates = o, nil
				break
			}
			if strings.HasPrefix(o.name, name) {
				candidates = append(candidates, o.name)
				match = o
			}
		}
		if len(candidates) > 1 {
			return a, false, fmt.Errorf("ambiguous option: %s could match %s", name, strings.Join(candidates, ", "))
		}
		if match == nil {
			return a, false, fmt.Errorf("unrecognized arguments: %s", arg)
		}
		if match.isBool {
			if hasValue {
				return a, false, fmt.Errorf("argument %s: ignored explicit argument %s", match.name, pyStrString(value))
			}
			_ = match.set(&a, "")
			continue
		}
		if !hasValue {
			if i+1 >= len(argv) || (strings.HasPrefix(argv[i+1], "-") && len(argv[i+1]) > 1 && !isNegativeNumber(argv[i+1])) {
				return a, false, fmt.Errorf("argument %s: expected one argument", match.name)
			}
			i++
			value = argv[i]
		}
		if err := match.set(&a, value); err != nil {
			return a, false, err
		}
	}
	return a, false, nil
}

func isNegativeNumber(s string) bool {
	_, err := strconv.ParseFloat(s, 64)
	return err == nil
}

// Main is the scorecard CLI: the flag surface and outputs of
// tools/concept_disambiguation_scorecard.py. defaultRoot is the workspace used
// when --workspace is not given.
func Main(argv []string, stdout, stderr io.Writer, defaultRoot string) int {
	a, help, err := parseCLI(argv)
	if help {
		fmt.Fprint(stdout, cliHelp())
		return 0
	}
	if err != nil {
		fmt.Fprintln(stderr, cliUsage)
		fmt.Fprintf(stderr, "concept_disambiguation_scorecard: error: %v\n", err)
		return 2
	}
	root := defaultRoot
	if a.workspace != "" {
		root = a.workspace
	}
	root = ResolvePath(root)
	dataPath := ""
	if a.data != "" {
		dataPath = ResolvePath(a.data)
	}
	payload := Collect(root, dataPath)

	if a.compare != "" {
		raw, err := os.ReadFile(a.compare)
		var baseline Value
		if err == nil {
			baseline, err = DecodeValue(raw)
		}
		if _, isObject := baseline.(*Object); err == nil && !isObject {
			err = fmt.Errorf("baseline is not a JSON object")
		}
		if err != nil {
			fmt.Fprintf(stderr, "cannot read baseline %s: %v\n", a.compare, err)
			return 2
		}
		fmt.Fprintln(stdout, RenderCompare(baseline, payload))
		return exitFor(payload)
	}

	if a.markdownDir != "" {
		outDir := ResolvePath(a.markdownDir)
		if err := payload.WriteDocFolder(outDir, a.stamp); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		if !a.json {
			fmt.Fprintf(stdout, "wrote concept-disambiguation doc folder -> %s\n", outDir)
		}
	}

	switch {
	case a.json:
		b, err := payload.JSON()
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		stdout.Write(b)
		fmt.Fprintln(stdout)
	case a.chart:
		fmt.Fprintln(stdout, payload.RenderChart())
	case a.critical:
		fmt.Fprintln(stdout, payload.RenderCritical())
	case a.rollup:
		fmt.Fprintln(stdout, payload.RollupText())
	case a.gaps:
		fmt.Fprintln(stdout, payload.RenderGaps())
	case a.pairs:
		fmt.Fprintln(stdout, payload.PairsText())
	case a.index || a.lookup != "":
		fmt.Fprintln(stdout, payload.IndexText(a.limit, a.lookup))
	case a.markdownDir == "":
		fmt.Fprintln(stdout, payload.RenderText())
	}
	return exitFor(payload)
}

func exitFor(p *Payload) int {
	if p.OK {
		return 0
	}
	return 1
}
