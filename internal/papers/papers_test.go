package papers

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

var fixedNow = time.Date(2026, 10, 12, 12, 0, 0, 0, time.UTC)

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func ids(ps []Paper) []string {
	out := make([]string, len(ps))
	for i, p := range ps {
		out[i] = p.ID
	}
	return out
}

// fak-test:runtime fast est=5ms lane=default
func TestCanonicalID(t *testing.T) {
	cases := map[string]string{
		"http://arxiv.org/abs/2610.13778v1":        "arxiv:2610.13778",
		"https://arxiv.org/pdf/2610.13778v2":       "arxiv:2610.13778",
		"2211.17192":                               "arxiv:2211.17192",
		"arXiv:2211.17192v3":                       "arxiv:2211.17192",
		"https://huggingface.co/papers/2211.17192": "arxiv:2211.17192",
		"hep-th/9901001v2":                         "arxiv:hep-th/9901001",
		"":                                         "",
	}
	for in, want := range cases {
		if got := CanonicalID(in); got != want {
			t.Errorf("CanonicalID(%q) = %q, want %q", in, got, want)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestParseFixtures(t *testing.T) {
	tests := []struct {
		name       string
		parse      func([]byte, time.Time) ([]Paper, error)
		fixture    string
		wantIDs    []string
		wantSource string
		firstCats  []string
		firstVotes int
		firstPDF   string
	}{
		{
			name: "arxiv atom", parse: ParseArxivAtom, fixture: "arxiv_query.xml",
			wantIDs:    []string{"arxiv:2610.13778", "arxiv:2610.13708", "arxiv:2610.13506"},
			wantSource: SourceArxiv, firstCats: []string{"cs.AI", "cs.LG"},
			firstPDF: "https://arxiv.org/pdf/2610.13778v1",
		},
		{
			name: "hf daily", parse: ParseHFPapers, fixture: "hf_daily.json",
			wantIDs:    []string{"arxiv:2610.12374", "arxiv:2610.08215", "arxiv:2610.12242"},
			wantSource: SourceHF, firstVotes: 259, firstPDF: "https://arxiv.org/pdf/2610.12374",
		},
		{
			name: "hf search", parse: ParseHFPapers, fixture: "hf_search.json",
			wantIDs:    []string{"arxiv:2211.17192", "arxiv:2310.08461", "arxiv:2311.08981"},
			wantSource: SourceHF, firstVotes: 11, firstPDF: "https://arxiv.org/pdf/2211.17192",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ps, err := tc.parse(readFixture(t, tc.fixture), fixedNow)
			if err != nil {
				t.Fatal(err)
			}
			if got := ids(ps); !reflect.DeepEqual(got, tc.wantIDs) {
				t.Fatalf("ids = %v, want %v", got, tc.wantIDs)
			}
			p := ps[0]
			if p.Source != tc.wantSource || p.Title == "" || p.Abstract == "" || len(p.Authors) == 0 {
				t.Fatalf("incomplete record: %+v", p)
			}
			if p.PublishedTime().IsZero() {
				t.Fatalf("published %q did not parse", p.Published)
			}
			if !reflect.DeepEqual(p.Categories, tc.firstCats) {
				t.Errorf("categories = %v, want %v", p.Categories, tc.firstCats)
			}
			if p.Upvotes != tc.firstVotes || p.PDFURL != tc.firstPDF {
				t.Errorf("upvotes/pdf = %d %q, want %d %q", p.Upvotes, p.PDFURL, tc.firstVotes, tc.firstPDF)
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestParseArxivErrorEntry(t *testing.T) {
	body := []byte(`<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors#incorrect_id</id><summary>bad query</summary></entry></feed>`)
	if _, err := ParseArxivAtom(body, fixedNow); err == nil {
		t.Fatal("want error for arXiv error entry")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestBuildArxivSearchQuery(t *testing.T) {
	tests := []struct{ text, cat, want string }{
		{"speculative decoding", "", "all:speculative AND all:decoding"},
		{"kv cache", "cs.LG", "all:kv AND all:cache AND cat:cs.LG"},
		{"", "cs.CL", "cat:cs.CL"},
		{"", "", ""},
	}
	for _, tc := range tests {
		if got := BuildArxivSearchQuery(tc.text, tc.cat); got != tc.want {
			t.Errorf("BuildArxivSearchQuery(%q,%q) = %q, want %q", tc.text, tc.cat, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestMergeAndUpsert(t *testing.T) {
	ax := Paper{ID: "arxiv:1", Source: SourceArxiv, Title: "Arxiv Title", Abstract: "short", Published: "2026-10-01T00:00:00Z",
		URL: AbsURL("1"), Categories: []string{"cs.LG"}, FetchedAt: "2026-10-01T00:00:00Z"}
	hf := Paper{ID: "arxiv:1", Source: SourceHF, Title: "HF Title", Abstract: "a much longer abstract", Published: "2026-10-02T00:00:00Z",
		URL: HFURL("1"), Categories: []string{"cs.AI"}, Upvotes: 42, FetchedAt: "2026-10-03T00:00:00Z"}

	m := Merge(hf, ax)
	want := Paper{ID: "arxiv:1", Source: "arxiv,hf", Title: "Arxiv Title", Abstract: "a much longer abstract",
		Published: "2026-10-01T00:00:00Z", URL: AbsURL("1"), Categories: []string{"cs.AI", "cs.LG"}, Upvotes: 42,
		FetchedAt: "2026-10-03T00:00:00Z"}
	if !reflect.DeepEqual(m, want) {
		t.Fatalf("Merge =\n%+v\nwant\n%+v", m, want)
	}

	idx, st := Upsert(nil, []Paper{ax, {ID: "arxiv:2", Source: SourceHF}})
	if st != (UpsertStats{New: 2, Updated: 0, Total: 2}) {
		t.Fatalf("first upsert stats = %+v", st)
	}
	idx, st = Upsert(idx, []Paper{hf, {ID: "arxiv:2", Source: SourceHF}, {ID: "arxiv:3", Source: SourceArxiv}})
	if st != (UpsertStats{New: 1, Updated: 1, Total: 3}) {
		t.Fatalf("second upsert stats = %+v", st)
	}
	if idx[0].Upvotes != 42 || idx[0].Source != "arxiv,hf" {
		t.Fatalf("merged record = %+v", idx[0])
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestStoreRoundTripAndIndexPath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "index.jsonl")
	ps, err := LoadIndex(path)
	if err != nil || ps != nil {
		t.Fatalf("missing index: %v %v", ps, err)
	}
	in := []Paper{
		{ID: "arxiv:b", Source: SourceHF, Title: "B", Authors: []string{"X"}, Upvotes: 3},
		{ID: "arxiv:a", Source: SourceArxiv, Title: "A <&>", Categories: []string{"cs.LG"}},
	}
	if err := SaveIndex(path, in); err != nil {
		t.Fatal(err)
	}
	out, err := LoadIndex(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out, []Paper{in[1], in[0]}) {
		t.Fatalf("round trip = %+v", out)
	}
	leftovers, _ := filepath.Glob(filepath.Join(dir, "nested", ".index-*"))
	if len(leftovers) != 0 {
		t.Fatalf("temp files left: %v", leftovers)
	}

	env := map[string]string{EnvIndex: "/env/index.jsonl"}
	if got := DefaultIndexPath("/flag.jsonl", func(k string) string { return env[k] }, dir); got != "/flag.jsonl" {
		t.Errorf("flag precedence: %q", got)
	}
	if got := DefaultIndexPath("", func(k string) string { return env[k] }, dir); got != "/env/index.jsonl" {
		t.Errorf("env precedence: %q", got)
	}
	repo := filepath.Join(dir, "repo")
	sub := filepath.Join(repo, "a", "b")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, ".git"), []byte("gitdir: x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(repo, ".fak", "papers", "index.jsonl")
	if got := DefaultIndexPath("", func(string) string { return "" }, sub); got != want {
		t.Errorf("repo default = %q, want %q", got, want)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestSearchRanking(t *testing.T) {
	corpus := []Paper{
		{ID: "arxiv:abs-only", Title: "Serving systems", Abstract: "we study speculative decoding for serving", Published: "2026-10-05T00:00:00Z"},
		{ID: "arxiv:title-hit", Title: "Speculative Decoding at Scale", Abstract: "throughput study", Published: "2026-09-01T00:00:00Z", Categories: []string{"cs.LG"}},
		{ID: "arxiv:unrelated", Title: "Image segmentation", Abstract: "pixels", Published: "2026-10-10T00:00:00Z"},
		{ID: "arxiv:tie-old", Title: "Draft models", Abstract: "speculative", Published: "2026-01-01T00:00:00Z", Source: SourceHF},
		{ID: "arxiv:tie-new", Title: "Draft models", Abstract: "speculative", Published: "2026-08-01T00:00:00Z", Source: SourceArxiv},
		{ID: "arxiv:author", Title: "Untitled", Abstract: "misc", Authors: []string{"Yaniv Leviathan"}, Published: "2026-02-01T00:00:00Z"},
	}
	hitIDs := func(hs []Hit) []string {
		var out []string
		for _, h := range hs {
			out = append(out, h.Paper.ID)
		}
		return out
	}
	tests := []struct {
		name  string
		query string
		f     Filter
		max   int
		want  []string
	}{
		{"title boost beats abstract", "speculative decoding", Filter{}, 2, []string{"arxiv:title-hit", "arxiv:abs-only"}},
		{"ties break newest first", "draft", Filter{}, 0, []string{"arxiv:tie-new", "arxiv:tie-old"}},
		{"authors indexed", "leviathan", Filter{}, 0, []string{"arxiv:author"}},
		{"category filter", "speculative", Filter{Category: "cs.lg"}, 0, []string{"arxiv:title-hit"}},
		{"source filter", "speculative", Filter{Source: SourceHF}, 0, []string{"arxiv:tie-old"}},
		{"since filter", "speculative", Filter{Since: time.Date(2026, 9, 15, 0, 0, 0, 0, time.UTC)}, 0, []string{"arxiv:abs-only"}},
		{"stopwords only", "the of and", Filter{}, 0, nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := hitIDs(Search(corpus, tc.query, tc.f, tc.max)); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("Search(%q) = %v, want %v", tc.query, got, tc.want)
			}
		})
	}

	if got := ids(Latest(corpus, Filter{Since: time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)}, 0)); !reflect.DeepEqual(got,
		[]string{"arxiv:unrelated", "arxiv:abs-only", "arxiv:title-hit"}) {
		t.Fatalf("Latest = %v", got)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestParseSince(t *testing.T) {
	tests := []struct {
		in      string
		want    time.Time
		wantErr bool
	}{
		{"", time.Time{}, false},
		{"30d", fixedNow.AddDate(0, 0, -30), false},
		{"2w", fixedNow.AddDate(0, 0, -14), false},
		{"48h", fixedNow.Add(-48 * time.Hour), false},
		{"2026-10-01", time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC), false},
		{"soon", time.Time{}, true},
	}
	for _, tc := range tests {
		got, err := ParseSince(tc.in, fixedNow)
		if (err != nil) != tc.wantErr || !got.Equal(tc.want) {
			t.Errorf("ParseSince(%q) = %v, %v; want %v, err=%v", tc.in, got, err, tc.want, tc.wantErr)
		}
	}
}
