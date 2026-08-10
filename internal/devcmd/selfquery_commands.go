package devcmd

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

func RunFeature(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		writeFeatureUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "query":
		return runFeatureQuery(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		writeFeatureUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak-dev feature: unknown subcommand %q\n", argv[0])
		writeFeatureUsage(stderr)
		return 2
	}
}

func runFeatureQuery(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("feature query", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: search upward for dos.toml)")
	plane := fs.String("plane", "all", "catalog plane: dev, live, or all")
	detail := fs.String("detail", "", "fault detail for one selected card name/detail_ref")
	limit := fs.Int("limit", 0, "cap query cards (0 = bounded default)")
	all := fs.Bool("all", false, "return every matching card")
	missing := fs.String("missing-context", "", "comma-separated missing context keys")
	asJSON := fs.Bool("json", false, "emit stable JSON")
	var args []string
	for rest := argv; ; {
		if err := fs.Parse(rest); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return 0
			}
			return 2
		}
		rest = fs.Args()
		if len(rest) == 0 {
			break
		}
		args = append(args, rest[0])
		rest = rest[1:]
	}
	if len(args) == 0 {
		fmt.Fprintln(stderr, "fak-dev feature query: needs a non-empty query")
		return 2
	}
	rootDir := *root
	if strings.TrimSpace(rootDir) == "" {
		rootDir = devindexRoot()
	}
	dev, err := loadSelfQueryDevCatalog(rootDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev feature query: %v\n", err)
		return 1
	}
	cat, err := selfquery.Load(rootDir, selfquery.Options{Dev: dev, DevLoader: loadSelfQueryDevCatalog, Tools: selfquery.ToolDescriptorsFromMaps(gateway.ToolDescriptorsForResolver())})
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev feature query: %v\n", err)
		return 1
	}
	resp, err := cat.Query(selfquery.Request{Query: joinArgs(args), Plane: selfquery.Plane(*plane), Detail: *detail, Limit: *limit, All: *all, MissingContext: splitCSVDev(*missing)})
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev feature query: %v\n", err)
		return 2
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, resp, "fak-dev feature query")
	}
	if len(resp.Cards) == 0 && resp.Clarifications == nil {
		fmt.Fprintln(stdout, "no matching feature")
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	for _, c := range resp.Cards {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\n", c.Name, c.Kind, c.Effect, c.Source, truncRunes(c.Summary, 96), c.Freshness)
	}
	if err := tw.Flush(); err != nil {
		return 1
	}
	if resp.Clarifications != nil {
		for _, q := range resp.Clarifications.Questions {
			fmt.Fprintf(stdout, "clarify %s: %s\n", q.Key, q.Question)
		}
	}
	return 0
}

func RunCapabilities(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("capabilities", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root")
	limit := fs.Int("limit", 0, "cap cards")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	rootDir := *root
	if strings.TrimSpace(rootDir) == "" {
		rootDir = devindexRoot()
	}
	dev, err := loadSelfQueryDevCatalog(rootDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev capabilities: %v\n", err)
		return 1
	}
	cat, err := selfquery.Load(rootDir, selfquery.Options{Dev: dev, DevLoader: loadSelfQueryDevCatalog, Tools: selfquery.ToolDescriptorsFromMaps(gateway.ToolDescriptorsForResolver())})
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev capabilities: %v\n", err)
		return 1
	}
	resp, err := cat.Capabilities(selfquery.CapabilitiesRequest{Query: joinArgs(fs.Args()), Limit: *limit})
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev capabilities: %v\n", err)
		return 2
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, resp, "fak-dev capabilities")
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tKIND\tEFFECT\tCALL\tSUMMARY")
	for _, c := range resp.Cards {
		call := c.Request.MCPTool
		if call == "" && len(c.Request.Command) > 0 {
			call = c.Request.Command[0]
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n", c.Name, c.Kind, c.Effect, call, truncRunes(c.Summary, 88))
	}
	return flushTab(tw, stderr, "fak-dev capabilities")
}

func devindexRoot() string { return devindex.FindRoot(".") }
func splitCSVDev(s string) []string {
	var out []string
	for _, v := range strings.Split(s, ",") {
		if v = strings.TrimSpace(v); v != "" {
			out = append(out, v)
		}
	}
	return out
}
func writeFeatureUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak-dev feature query <intent> [--json] [--plane dev|live|all] [--root DIR]")
}
