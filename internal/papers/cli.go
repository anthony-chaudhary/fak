package papers

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

const usage = `usage: fak-dev papers <command> [flags]

commands:
  fetch  [--source arxiv|hf|all] [--query Q] [--category C] [--max N] [--date YYYY-MM-DD]
  search "<query>" [--live] [--max N] [--since 30d] [--category C] [--source S] [--json]
  latest [--since 7d] [--category C] [--source arxiv|hf] [--max N] [--json]
  show   <id> [--json]

common: --index PATH (default $FAK_PAPERS_INDEX or <repo>/.fak/papers/index.jsonl)
`

type Env struct {
	Fetcher Fetcher
	Now     func() time.Time
	Getenv  func(string) string
	Cwd     string
}

func Run(argv []string, stdout, stderr io.Writer) int {
	cwd, _ := os.Getwd()
	return RunWith(Env{Fetcher: NewLiveFetcher(), Now: time.Now, Getenv: os.Getenv, Cwd: cwd}, argv, stdout, stderr)
}

func RunWith(env Env, argv []string, stdout, stderr io.Writer) int {
	if env.Now == nil {
		env.Now = time.Now
	}
	if len(argv) == 0 || argv[0] == "-h" || argv[0] == "--help" || argv[0] == "help" {
		fmt.Fprint(stderr, usage)
		if len(argv) == 0 {
			return 2
		}
		return 0
	}
	c := &cli{env: env, stdout: stdout, stderr: stderr}
	switch argv[0] {
	case "fetch":
		return c.fetch(argv[1:])
	case "search":
		return c.search(argv[1:])
	case "latest":
		return c.latest(argv[1:])
	case "show":
		return c.show(argv[1:])
	}
	fmt.Fprintf(stderr, "papers: unknown command %q\n%s", argv[0], usage)
	return 2
}

type cli struct {
	env            Env
	stdout, stderr io.Writer
}

func (c *cli) flagSet(name string) (*flag.FlagSet, *string) {
	fs := flag.NewFlagSet("papers "+name, flag.ContinueOnError)
	fs.SetOutput(c.stderr)
	idx := fs.String("index", "", "index JSONL path")
	return fs, idx
}

// parseInterspersed lets flags follow positionals (`search "q" --live`).
func parseInterspersed(fs *flag.FlagSet, args []string) ([]string, error) {
	var pos []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		args = fs.Args()
		if len(args) == 0 {
			return pos, nil
		}
		pos = append(pos, args[0])
		args = args[1:]
	}
}

func (c *cli) indexPath(flagValue string) string {
	return DefaultIndexPath(flagValue, c.env.Getenv, c.env.Cwd)
}

func (c *cli) fail(err error) int {
	fmt.Fprintf(c.stderr, "papers: %v\n", err)
	return 1
}

func (c *cli) fetch(args []string) int {
	fs, idx := c.flagSet("fetch")
	source := fs.String("source", "all", "arxiv, hf, or all")
	query := fs.String("query", "", "free-text query")
	category := fs.String("category", "", "arXiv category, e.g. cs.LG")
	maxN := fs.Int("max", 25, "max papers per source")
	date := fs.String("date", "", "HF daily papers date YYYY-MM-DD (default latest)")
	asJSON := fs.Bool("json", false, "emit JSON counts")
	pos, err := parseInterspersed(fs, args)
	if err != nil {
		return 2
	}
	if *query == "" && len(pos) > 0 {
		*query = strings.Join(pos, " ")
	}
	switch *source {
	case "arxiv", "hf", "all":
	default:
		fmt.Fprintf(c.stderr, "papers fetch: --source must be arxiv, hf, or all\n")
		return 2
	}
	ctx := context.Background()
	var got []Paper
	var errs []error
	if *source == "arxiv" || *source == "all" {
		ps, err := c.env.Fetcher.FetchArxiv(ctx, ArxivQuery{Text: *query, Category: *category, Max: *maxN})
		switch {
		case errors.Is(err, ErrEmptyArxivQuery) && *source == "all":
			fmt.Fprintln(c.stderr, "papers: arxiv skipped (needs --query or --category)")
		case err != nil:
			errs = append(errs, err)
		}
		got = append(got, ps...)
	}
	if *source == "hf" || *source == "all" {
		var ps []Paper
		var err error
		if *query != "" {
			ps, err = c.env.Fetcher.FetchHFSearch(ctx, *query, *maxN)
		} else {
			ps, err = c.env.Fetcher.FetchHFDaily(ctx, *date, *maxN)
		}
		if err != nil {
			errs = append(errs, err)
		}
		got = append(got, ps...)
	}
	for _, e := range errs {
		fmt.Fprintf(c.stderr, "papers: warning: %v\n", e)
	}
	path := c.indexPath(*idx)
	st, err := c.upsert(path, got)
	if err != nil {
		return c.fail(err)
	}
	if *asJSON {
		writeJSON(c.stdout, st)
	} else {
		fmt.Fprintf(c.stdout, "fetched %d: new %d, updated %d, total %d (%s)\n", len(got), st.New, st.Updated, st.Total, path)
	}
	if len(got) == 0 && len(errs) > 0 {
		return 1
	}
	return 0
}

func (c *cli) upsert(path string, got []Paper) (UpsertStats, error) {
	existing, err := LoadIndex(path)
	if err != nil {
		return UpsertStats{}, err
	}
	merged, st := Upsert(existing, got)
	if st.New == 0 && st.Updated == 0 {
		return st, nil
	}
	return st, SaveIndex(path, merged)
}

func (c *cli) filter(since, category, source string) (Filter, error) {
	t, err := ParseSince(since, c.env.Now())
	if err != nil {
		return Filter{}, err
	}
	return Filter{Since: t, Category: category, Source: source}, nil
}

func (c *cli) search(args []string) int {
	fs, idx := c.flagSet("search")
	live := fs.Bool("live", false, "fetch arXiv + HF for the query and upsert before searching")
	maxN := fs.Int("max", 10, "max results")
	since := fs.String("since", "", "only papers published since (30d, 2w, YYYY-MM-DD)")
	category := fs.String("category", "", "arXiv category filter")
	source := fs.String("source", "", "arxiv or hf filter")
	asJSON := fs.Bool("json", false, "emit JSON")
	pos, err := parseInterspersed(fs, args)
	if err != nil {
		return 2
	}
	q := strings.TrimSpace(strings.Join(pos, " "))
	if q == "" {
		fmt.Fprintln(c.stderr, "papers search: query required")
		return 2
	}
	f, err := c.filter(*since, *category, *source)
	if err != nil {
		fmt.Fprintf(c.stderr, "papers search: %v\n", err)
		return 2
	}
	path := c.indexPath(*idx)
	if *live {
		fetchN := max(*maxN*3, 25)
		ctx := context.Background()
		var got []Paper
		fails := 0
		ps, err := c.env.Fetcher.FetchArxiv(ctx, ArxivQuery{Text: q, Category: *category, Max: fetchN})
		if err != nil {
			fails++
			fmt.Fprintf(c.stderr, "papers: warning: %v\n", err)
		}
		got = append(got, ps...)
		ps, err = c.env.Fetcher.FetchHFSearch(ctx, q, fetchN)
		if err != nil {
			fails++
			fmt.Fprintf(c.stderr, "papers: warning: %v\n", err)
		}
		got = append(got, ps...)
		st, err := c.upsert(path, got)
		if err != nil {
			return c.fail(err)
		}
		fmt.Fprintf(c.stderr, "papers: live fetched %d: new %d, updated %d, total %d\n", len(got), st.New, st.Updated, st.Total)
		if fails == 2 {
			return 1
		}
	}
	all, err := LoadIndex(path)
	if err != nil {
		return c.fail(err)
	}
	hits := Search(all, q, f, *maxN)
	if *asJSON {
		if hits == nil {
			hits = []Hit{}
		}
		writeJSON(c.stdout, hits)
		return 0
	}
	for _, h := range hits {
		fmt.Fprintln(c.stdout, FormatLine(h.Paper))
	}
	if len(hits) == 0 {
		fmt.Fprintf(c.stderr, "papers: no matches in %s (%d indexed; try --live)\n", path, len(all))
	}
	return 0
}

func (c *cli) latest(args []string) int {
	fs, idx := c.flagSet("latest")
	since := fs.String("since", "7d", "only papers published since (7d, 2w, YYYY-MM-DD; empty = all)")
	category := fs.String("category", "", "arXiv category filter")
	source := fs.String("source", "", "arxiv or hf filter")
	maxN := fs.Int("max", 20, "max results")
	asJSON := fs.Bool("json", false, "emit JSON")
	if _, err := parseInterspersed(fs, args); err != nil {
		return 2
	}
	f, err := c.filter(*since, *category, *source)
	if err != nil {
		fmt.Fprintf(c.stderr, "papers latest: %v\n", err)
		return 2
	}
	all, err := LoadIndex(c.indexPath(*idx))
	if err != nil {
		return c.fail(err)
	}
	ps := Latest(all, f, *maxN)
	if *asJSON {
		if ps == nil {
			ps = []Paper{}
		}
		writeJSON(c.stdout, ps)
		return 0
	}
	for _, p := range ps {
		fmt.Fprintln(c.stdout, FormatLine(p))
	}
	return 0
}

type showRecord struct {
	Paper
	AbsURL  string `json:"abs_url"`
	PDF     string `json:"pdf"`
	HFURL   string `json:"hf_url"`
	ReadCmd string `json:"read_cmd"`
}

func (c *cli) show(args []string) int {
	fs, idx := c.flagSet("show")
	asJSON := fs.Bool("json", false, "emit JSON")
	pos, err := parseInterspersed(fs, args)
	if err != nil {
		return 2
	}
	if len(pos) != 1 {
		fmt.Fprintln(c.stderr, "papers show: exactly one id required")
		return 2
	}
	id := CanonicalID(pos[0])
	all, err := LoadIndex(c.indexPath(*idx))
	if err != nil {
		return c.fail(err)
	}
	for _, p := range all {
		if p.ID != id {
			continue
		}
		bare := BareArxivID(id)
		rec := showRecord{Paper: p, AbsURL: AbsURL(id), PDF: PDFURL(id), HFURL: HFURL(id), ReadCmd: "hf papers read " + bare}
		if *asJSON {
			writeJSON(c.stdout, rec)
			return 0
		}
		w := c.stdout
		fmt.Fprintf(w, "id:         %s\n", p.ID)
		fmt.Fprintf(w, "title:      %s\n", p.Title)
		fmt.Fprintf(w, "published:  %s\n", p.Published)
		fmt.Fprintf(w, "source:     %s\n", p.Source)
		if len(p.Authors) > 0 {
			fmt.Fprintf(w, "authors:    %s\n", strings.Join(p.Authors, ", "))
		}
		if len(p.Categories) > 0 {
			fmt.Fprintf(w, "categories: %s\n", strings.Join(p.Categories, ", "))
		}
		if p.Upvotes > 0 {
			fmt.Fprintf(w, "upvotes:    %d\n", p.Upvotes)
		}
		fmt.Fprintf(w, "abs:        %s\npdf:        %s\nhf:         %s\n", rec.AbsURL, rec.PDF, rec.HFURL)
		fmt.Fprintf(w, "\n%s\n\nfull text as markdown: %s\n", p.Abstract, rec.ReadCmd)
		return 0
	}
	fmt.Fprintf(c.stderr, "papers show: %s not in index (fetch it with: fak-dev papers fetch --source arxiv --query %s)\n", id, BareArxivID(id))
	return 1
}

const titleWidth = 96

// FormatLine renders "date  id  [upvotes]  title" for compact listings.
func FormatLine(p Paper) string {
	date := "----------"
	if t := p.PublishedTime(); !t.IsZero() {
		date = t.Format("2006-01-02")
	}
	up := "     "
	if p.Upvotes > 0 {
		up = fmt.Sprintf("[%3d]", p.Upvotes)
	}
	title := []rune(p.Title)
	if len(title) > titleWidth {
		title = append(title[:titleWidth-3], []rune("...")...)
	}
	return fmt.Sprintf("%s  %-18s  %s  %s", date, p.ID, up, string(title))
}

func writeJSON(w io.Writer, v any) {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}
