package papers

import (
	"bytes"
	"context"
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

type fakeFetcher struct {
	arxiv, daily, search []Paper
	calls                []string
}

func (f *fakeFetcher) FetchArxiv(_ context.Context, q ArxivQuery) ([]Paper, error) {
	f.calls = append(f.calls, "arxiv:"+BuildArxivSearchQuery(q.Text, q.Category))
	if BuildArxivSearchQuery(q.Text, q.Category) == "" {
		return nil, ErrEmptyArxivQuery
	}
	return f.arxiv, nil
}

func (f *fakeFetcher) FetchHFDaily(_ context.Context, date string, _ int) ([]Paper, error) {
	f.calls = append(f.calls, "daily:"+date)
	return f.daily, nil
}

func (f *fakeFetcher) FetchHFSearch(_ context.Context, q string, _ int) ([]Paper, error) {
	f.calls = append(f.calls, "search:"+q)
	return f.search, nil
}

func newFake(t *testing.T) *fakeFetcher {
	t.Helper()
	ax, err := ParseArxivAtom(readFixture(t, "arxiv_query.xml"), fixedNow)
	if err != nil {
		t.Fatal(err)
	}
	daily, err := ParseHFPapers(readFixture(t, "hf_daily.json"), fixedNow)
	if err != nil {
		t.Fatal(err)
	}
	search, err := ParseHFPapers(readFixture(t, "hf_search.json"), fixedNow)
	if err != nil {
		t.Fatal(err)
	}
	return &fakeFetcher{arxiv: ax, daily: daily, search: search}
}

func runCLI(t *testing.T, f Fetcher, args ...string) (int, string, string) {
	t.Helper()
	var out, errb bytes.Buffer
	env := Env{Fetcher: f, Now: func() time.Time { return fixedNow }, Getenv: func(string) string { return "" }, Cwd: t.TempDir()}
	code := RunWith(env, args, &out, &errb)
	return code, out.String(), errb.String()
}

// fak-test:runtime fast est=30ms lane=default
func TestCLIDispatch(t *testing.T) {
	f := newFake(t)
	idx := filepath.Join(t.TempDir(), "index.jsonl")

	if code, _, errOut := runCLI(t, f); code != 2 || !strings.Contains(errOut, "usage: fak-dev papers") {
		t.Fatalf("no args: code=%d", code)
	}
	if code, _, _ := runCLI(t, f, "bogus"); code != 2 {
		t.Fatalf("unknown command code=%d", code)
	}
	if code, _, _ := runCLI(t, f, "fetch", "--source", "nope", "--index", idx); code != 2 {
		t.Fatalf("bad source code=%d", code)
	}

	code, out, _ := runCLI(t, f, "fetch", "--source", "all", "--json", "--index", idx)
	if code != 0 {
		t.Fatalf("fetch all code=%d", code)
	}
	var st UpsertStats
	if err := json.Unmarshal([]byte(out), &st); err != nil {
		t.Fatal(err)
	}
	if st != (UpsertStats{New: 3, Total: 3}) {
		t.Fatalf("fetch all (no query: arxiv skipped, hf daily) stats = %+v", st)
	}
	if !reflect.DeepEqual(f.calls, []string{"arxiv:", "daily:"}) {
		t.Fatalf("calls = %v", f.calls)
	}

	f.calls = nil
	code, out, _ = runCLI(t, f, "search", "speculative decoding", "--live", "--max", "2", "--index", idx)
	if code != 0 {
		t.Fatalf("search --live code=%d", code)
	}
	if !reflect.DeepEqual(f.calls, []string{"arxiv:all:speculative AND all:decoding", "search:speculative decoding"}) {
		t.Fatalf("live calls = %v", f.calls)
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 result lines, got %q", out)
	}

	code, out, _ = runCLI(t, f, "search", "--json", "--source", "hf", "speculative", "--index", idx)
	if code != 0 {
		t.Fatalf("search --json code=%d", code)
	}
	var hits []Hit
	if err := json.Unmarshal([]byte(out), &hits); err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if !h.Paper.HasSource(SourceHF) {
			t.Fatalf("source filter leaked %s", h.Paper.ID)
		}
	}
	if len(hits) == 0 || hits[0].Paper.ID != "arxiv:2211.17192" && hits[0].Paper.ID != "arxiv:2311.08981" {
		t.Fatalf("unexpected top hf hit: %+v", hits)
	}

	code, out, _ = runCLI(t, f, "latest", "--source", "hf", "--since", "30d", "--json", "--index", idx)
	if code != 0 {
		t.Fatalf("latest code=%d", code)
	}
	var latest []Paper
	if err := json.Unmarshal([]byte(out), &latest); err != nil {
		t.Fatal(err)
	}
	if got := ids(latest); !reflect.DeepEqual(got, []string{"arxiv:2610.12374", "arxiv:2610.08215", "arxiv:2610.12242"}) {
		t.Fatalf("latest ids = %v", got)
	}

	code, out, _ = runCLI(t, f, "show", "2610.12374v2", "--json", "--index", idx)
	if code != 0 {
		t.Fatalf("show code=%d", code)
	}
	var rec showRecord
	if err := json.Unmarshal([]byte(out), &rec); err != nil {
		t.Fatal(err)
	}
	if rec.ID != "arxiv:2610.12374" || rec.HFURL != "https://huggingface.co/papers/2610.12374" || rec.ReadCmd != "hf papers read 2610.12374" {
		t.Fatalf("show record = %+v", rec)
	}
	if code, _, _ := runCLI(t, f, "show", "9999.99999", "--index", idx); code != 1 {
		t.Fatalf("show missing code=%d", code)
	}
	if code, _, _ := runCLI(t, f, "search", "--index", idx); code != 2 {
		t.Fatalf("search without query code=%d", code)
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestFormatLine(t *testing.T) {
	p := Paper{ID: "arxiv:2610.12374", Title: strings.Repeat("x", 200), Published: "2026-10-08T00:00:00Z", Upvotes: 7}
	got := FormatLine(p)
	if !strings.HasPrefix(got, "2026-10-08  arxiv:2610.12374    [  7]  ") || !strings.HasSuffix(got, "...") {
		t.Fatalf("FormatLine = %q", got)
	}
	if n := len([]rune(got)); n > 140 {
		t.Fatalf("line too long: %d", n)
	}
}
