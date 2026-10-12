package papers

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"reflect"
	"testing"
	"time"
)

type roundTrip func(*http.Request) (*http.Response, error)

func (f roundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func respond(code int, body []byte) *http.Response {
	return &http.Response{StatusCode: code, Status: http.StatusText(code), Body: io.NopCloser(bytes.NewReader(body)), Header: http.Header{}}
}

// fak-test:runtime fast est=1s lane=default
func TestLiveFetcherRetriesAndArxivEtiquette(t *testing.T) {
	search := readFixture(t, "hf_search.json")
	atom := readFixture(t, "arxiv_query.xml")
	var urls, agents []string
	calls := 0
	client := &http.Client{Transport: roundTrip(func(r *http.Request) (*http.Response, error) {
		calls++
		urls = append(urls, r.URL.Host+r.URL.Path)
		agents = append(agents, r.Header.Get("User-Agent"))
		switch {
		case r.URL.Host == "huggingface.co" && calls == 1:
			return respond(http.StatusServiceUnavailable, nil), nil
		case r.URL.Host == "huggingface.co":
			return respond(http.StatusOK, search), nil
		default:
			return respond(http.StatusOK, atom), nil
		}
	})}
	clock := fixedNow
	var slept []time.Duration
	f := &LiveFetcher{HTTPClient: client, Now: func() time.Time { return clock },
		Sleep: func(d time.Duration) { slept = append(slept, d); clock = clock.Add(d) }}

	ps, err := f.FetchHFSearch(context.Background(), "speculative decoding", 2)
	if err != nil {
		t.Fatal(err)
	}
	if got := ids(ps); !reflect.DeepEqual(got, []string{"arxiv:2211.17192", "arxiv:2310.08461"}) {
		t.Fatalf("hf search after retry = %v", got)
	}

	if _, err := f.FetchArxiv(context.Background(), ArxivQuery{Text: "speculative"}); err != nil {
		t.Fatal(err)
	}
	if _, err := f.FetchArxiv(context.Background(), ArxivQuery{Category: "cs.LG"}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(slept, []time.Duration{arxivMinGap}) {
		t.Fatalf("arxiv gap sleeps = %v, want one %v", slept, arxivMinGap)
	}
	for _, a := range agents {
		if a != userAgent {
			t.Fatalf("user agent = %q", a)
		}
	}
	if len(urls) != 4 || urls[2] != "export.arxiv.org/api/query" {
		t.Fatalf("urls = %v", urls)
	}
	if _, err := f.FetchArxiv(context.Background(), ArxivQuery{}); err != ErrEmptyArxivQuery {
		t.Fatalf("empty query err = %v", err)
	}
}
