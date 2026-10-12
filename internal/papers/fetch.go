package papers

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	ArxivAPI       = "https://export.arxiv.org/api/query"
	HFDailyAPI     = "https://huggingface.co/api/daily_papers"
	HFSearchAPI    = "https://huggingface.co/api/papers/search"
	userAgent      = "fak-papers/1.0 (+https://github.com/anthony-chaudhary/fak)"
	arxivPageSize  = 100
	arxivMinGap    = 3 * time.Second
	maxBodyBytes   = 32 << 20
	defaultTimeout = 60 * time.Second
)

var ErrEmptyArxivQuery = errors.New("arxiv query needs free text or a category")

type ArxivQuery struct {
	Text     string
	Category string
	Max      int
	Start    int
}

type Fetcher interface {
	FetchArxiv(ctx context.Context, q ArxivQuery) ([]Paper, error)
	// FetchHFDaily lists HF daily papers for date (YYYY-MM-DD); empty date = latest.
	FetchHFDaily(ctx context.Context, date string, max int) ([]Paper, error)
	FetchHFSearch(ctx context.Context, query string, max int) ([]Paper, error)
}

type LiveFetcher struct {
	HTTPClient *http.Client
	Now        func() time.Time
	Sleep      func(time.Duration)

	mu        sync.Mutex
	lastArxiv time.Time
}

func NewLiveFetcher() *LiveFetcher {
	return &LiveFetcher{HTTPClient: &http.Client{Timeout: defaultTimeout}}
}

// BuildArxivSearchQuery ANDs each free-text term as all:<term> plus an optional cat: filter.
func BuildArxivSearchQuery(text, category string) string {
	var terms []string
	for _, tok := range strings.Fields(text) {
		tok = strings.Trim(tok, `"'()[]{},;:`)
		if tok == "" {
			continue
		}
		terms = append(terms, "all:"+tok)
	}
	if category != "" {
		terms = append(terms, "cat:"+strings.TrimSpace(category))
	}
	return strings.Join(terms, " AND ")
}

func (f *LiveFetcher) FetchArxiv(ctx context.Context, q ArxivQuery) ([]Paper, error) {
	sq := BuildArxivSearchQuery(q.Text, q.Category)
	if sq == "" {
		return nil, ErrEmptyArxivQuery
	}
	want := q.Max
	if want <= 0 {
		want = 20
	}
	var out []Paper
	start := q.Start
	for len(out) < want {
		page := min(want-len(out), arxivPageSize)
		v := url.Values{}
		v.Set("search_query", sq)
		v.Set("sortBy", "submittedDate")
		v.Set("sortOrder", "descending")
		v.Set("start", strconv.Itoa(start))
		v.Set("max_results", strconv.Itoa(page))
		f.arxivGap()
		body, err := f.get(ctx, ArxivAPI+"?"+v.Encode())
		if err != nil {
			return out, fmt.Errorf("arxiv: %w", err)
		}
		got, err := ParseArxivAtom(body, f.now())
		if err != nil {
			return out, err
		}
		out = append(out, got...)
		if len(got) < page {
			break
		}
		start += len(got)
	}
	return out, nil
}

// arxivGap enforces arXiv's API etiquette of >=3s between consecutive requests.
func (f *LiveFetcher) arxivGap() {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.lastArxiv.IsZero() {
		if wait := arxivMinGap - f.now().Sub(f.lastArxiv); wait > 0 {
			sleep := f.Sleep
			if sleep == nil {
				sleep = time.Sleep
			}
			sleep(wait)
		}
	}
	f.lastArxiv = f.now()
}

func (f *LiveFetcher) FetchHFDaily(ctx context.Context, date string, max int) ([]Paper, error) {
	v := url.Values{}
	if date != "" {
		v.Set("date", date)
	}
	if max > 0 {
		v.Set("limit", strconv.Itoa(max))
	}
	u := HFDailyAPI
	if len(v) > 0 {
		u += "?" + v.Encode()
	}
	body, err := f.get(ctx, u)
	if err != nil {
		return nil, fmt.Errorf("hf daily: %w", err)
	}
	ps, err := ParseHFPapers(body, f.now())
	return truncate(ps, max), err
}

func (f *LiveFetcher) FetchHFSearch(ctx context.Context, query string, max int) ([]Paper, error) {
	v := url.Values{}
	v.Set("q", query)
	if max > 0 {
		v.Set("limit", strconv.Itoa(max))
	}
	body, err := f.get(ctx, HFSearchAPI+"?"+v.Encode())
	if err != nil {
		return nil, fmt.Errorf("hf search: %w", err)
	}
	ps, err := ParseHFPapers(body, f.now())
	return truncate(ps, max), err
}

func truncate(ps []Paper, max int) []Paper {
	if max > 0 && len(ps) > max {
		return ps[:max]
	}
	return ps
}

func (f *LiveFetcher) now() time.Time {
	if f.Now != nil {
		return f.Now()
	}
	return time.Now()
}

// get retries transient failures (connection resets, 429, 5xx) twice with backoff.
func (f *LiveFetcher) get(ctx context.Context, u string) ([]byte, error) {
	var lastErr error
	for attempt := range 3 {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(attempt) * time.Second):
			}
		}
		body, retry, err := f.getOnce(ctx, u)
		if err == nil || !retry {
			return body, err
		}
		lastErr = err
	}
	return nil, lastErr
}

func (f *LiveFetcher) getOnce(ctx context.Context, u string) ([]byte, bool, error) {
	client := f.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: defaultTimeout}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, false, err
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := client.Do(req)
	if err != nil {
		return nil, ctx.Err() == nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		retry := resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500
		return nil, retry, fmt.Errorf("status %s", resp.Status)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	return b, err != nil, err
}

type atomFeed struct {
	Entries []atomEntry `xml:"entry"`
}

type atomEntry struct {
	ID        string `xml:"id"`
	Title     string `xml:"title"`
	Summary   string `xml:"summary"`
	Published string `xml:"published"`
	Links     []struct {
		Href  string `xml:"href,attr"`
		Rel   string `xml:"rel,attr"`
		Type  string `xml:"type,attr"`
		Title string `xml:"title,attr"`
	} `xml:"link"`
	Categories []struct {
		Term string `xml:"term,attr"`
	} `xml:"category"`
	Authors []struct {
		Name string `xml:"name"`
	} `xml:"author"`
}

func ParseArxivAtom(body []byte, fetchedAt time.Time) ([]Paper, error) {
	var feed atomFeed
	if err := xml.Unmarshal(body, &feed); err != nil {
		return nil, fmt.Errorf("arxiv atom: %w", err)
	}
	var out []Paper
	for _, e := range feed.Entries {
		if strings.Contains(e.ID, "/api/errors") {
			return nil, fmt.Errorf("arxiv api error: %s", collapseSpace(e.Summary))
		}
		id := CanonicalID(e.ID)
		if id == "" {
			continue
		}
		p := Paper{
			ID:        id,
			Source:    SourceArxiv,
			Title:     collapseSpace(e.Title),
			Abstract:  collapseSpace(e.Summary),
			Published: e.Published,
			URL:       AbsURL(id),
			FetchedAt: fetchedAt.UTC().Format(time.RFC3339),
		}
		for _, l := range e.Links {
			if l.Title == "pdf" || l.Type == "application/pdf" {
				p.PDFURL = l.Href
			}
		}
		if p.PDFURL == "" {
			p.PDFURL = PDFURL(id)
		}
		for _, a := range e.Authors {
			if n := collapseSpace(a.Name); n != "" {
				p.Authors = append(p.Authors, n)
			}
		}
		for _, c := range e.Categories {
			p.Categories = append(p.Categories, c.Term)
		}
		p.Categories = unionSorted(p.Categories, nil)
		out = append(out, p)
	}
	return out, nil
}

// hfEntry covers both /api/daily_papers and /api/papers/search: each item wraps a "paper".
type hfEntry struct {
	Paper struct {
		ID          string `json:"id"`
		Title       string `json:"title"`
		Summary     string `json:"summary"`
		PublishedAt string `json:"publishedAt"`
		Upvotes     int    `json:"upvotes"`
		Authors     []struct {
			Name string `json:"name"`
		} `json:"authors"`
	} `json:"paper"`
	Title       string `json:"title"`
	Summary     string `json:"summary"`
	PublishedAt string `json:"publishedAt"`
}

func ParseHFPapers(body []byte, fetchedAt time.Time) ([]Paper, error) {
	var entries []hfEntry
	if err := json.Unmarshal(body, &entries); err != nil {
		return nil, fmt.Errorf("hf papers json: %w", err)
	}
	out := make([]Paper, 0, len(entries))
	for _, e := range entries {
		id := CanonicalID(e.Paper.ID)
		if id == "" {
			continue
		}
		p := Paper{
			ID:        id,
			Source:    SourceHF,
			Title:     collapseSpace(firstNonEmpty(e.Paper.Title, e.Title)),
			Abstract:  collapseSpace(firstNonEmpty(e.Paper.Summary, e.Summary)),
			Published: firstNonEmpty(e.Paper.PublishedAt, e.PublishedAt),
			URL:       HFURL(id),
			PDFURL:    PDFURL(id),
			Upvotes:   e.Paper.Upvotes,
			FetchedAt: fetchedAt.UTC().Format(time.RFC3339),
		}
		for _, a := range e.Paper.Authors {
			if n := collapseSpace(a.Name); n != "" {
				p.Authors = append(p.Authors, n)
			}
		}
		out = append(out, p)
	}
	return out, nil
}

func firstNonEmpty(xs ...string) string {
	for _, x := range xs {
		if strings.TrimSpace(x) != "" {
			return x
		}
	}
	return ""
}
