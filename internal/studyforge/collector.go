package studyforge

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

type sourceSpec struct{ name, path string }

var sourceSpecs = []sourceSpec{
	{"issues", "issues?state=all&sort=created&direction=asc&per_page=100&page=1"},
	{"pulls", "pulls?state=all&sort=created&direction=asc&per_page=100&page=1"},
	{"discussions", "discussions?direction=asc&per_page=100&page=1"},
	{"releases", "releases?per_page=100&page=1"},
	{"labels", "labels?per_page=100&page=1"},
	{"milestones", "milestones?state=all&sort=due_on&direction=asc&per_page=100&page=1"},
}

// NewCollector returns a REST-first collector with bounded transient retries.
func NewCollector(client *http.Client) *Collector {
	if client == nil {
		client = http.DefaultClient
	}
	return &Collector{Client: client, BaseURL: "https://api.github.com", MaxRetries: 2, Now: time.Now, RetryWait: waitContext}
}

// Capture collects all six top-level sources. On any inaccessible source it returns
// both a persistable partial corpus and an error; the corpus can be supplied as Resume.
func (c *Collector) Capture(ctx context.Context, req CaptureRequest) (Corpus, error) {
	if strings.TrimSpace(req.Owner) == "" || strings.TrimSpace(req.Repository) == "" {
		return Corpus{}, errors.New("owner and repository are required")
	}
	c.defaults()
	if req.Cutoff.IsZero() {
		if req.Resume != nil {
			req.Cutoff, _ = time.Parse(time.RFC3339Nano, req.Resume.Receipt.Cutoff)
		}
		if req.Cutoff.IsZero() {
			req.Cutoff = c.Now().UTC()
		}
	}
	req.Cutoff = req.Cutoff.UTC()
	full := req.Owner + "/" + req.Repository
	corpus := Corpus{Schema: CorpusSchema, Receipt: Receipt{Schema: ReceiptSchema, Repository: full, Cutoff: req.Cutoff.Format(time.RFC3339Nano), StartedAt: c.Now().UTC().Format(time.RFC3339Nano), Status: StatusPartial}}
	if req.Resume != nil {
		corpus = cloneCorpus(*req.Resume)
		if err := validateResume(corpus, full, req.Cutoff); err != nil {
			return Corpus{}, err
		}
		corpus.Receipt.StartedAt = c.Now().UTC().Format(time.RFC3339Nano)
		corpus.Receipt.CompletedAt = ""
	}
	revision, api, err := c.resolveRevision(ctx, req.Owner, req.Repository)
	corpus.Receipt.API = api
	if err != nil {
		corpus.Receipt.Status = StatusFailed
		return corpus, err
	}
	if corpus.Receipt.Revision != "" && corpus.Receipt.Revision != revision {
		return Corpus{}, fmt.Errorf("resume revision changed from %s to %s", corpus.Receipt.Revision, revision)
	}
	corpus.Receipt.Revision = revision

	var failures []error
	for _, spec := range sourceSpecs {
		sr, existing := sourceByName(corpus.Receipt.Sources, spec.name)
		if existing && sr.Status == StatusComplete {
			continue
		}
		if !existing {
			sr = SourceReceipt{Name: spec.name, Endpoint: c.repoURL(req.Owner, req.Repository, spec.path), Status: StatusPartial}
		}
		records := recordsForSource(corpus.Records, spec.name)
		other := recordsExceptSource(corpus.Records, spec.name)
		updated, got, collectErr := c.collectSource(ctx, sr, records, req.Cutoff)
		corpus.Records = append(other, got...)
		upsertSource(&corpus.Receipt.Sources, updated)
		if collectErr != nil {
			failures = append(failures, fmt.Errorf("%s: %w", spec.name, collectErr))
		}
	}
	sortCorpus(&corpus)
	refreshChecksums(&corpus)
	corpus.Receipt.CompletedAt = c.Now().UTC().Format(time.RFC3339Nano)
	if len(failures) == 0 {
		corpus.Receipt.Status = StatusComplete
	} else if completeSourceCount(corpus.Receipt.Sources) == 0 {
		corpus.Receipt.Status = StatusFailed
	} else {
		corpus.Receipt.Status = StatusPartial
	}
	if err := Validate(corpus); err != nil {
		if len(failures) == 0 {
			return corpus, err
		}
	}
	return corpus, errors.Join(failures...)
}

func (c *Collector) defaults() {
	if c.Client == nil {
		c.Client = http.DefaultClient
	}
	if c.BaseURL == "" {
		c.BaseURL = "https://api.github.com"
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	if c.RetryWait == nil {
		c.RetryWait = waitContext
	}
	if c.MaxRetries < 0 {
		c.MaxRetries = 0
	}
}

func (c *Collector) resolveRevision(ctx context.Context, owner, repo string) (string, []APIReceipt, error) {
	metaURL := c.repoURL(owner, repo, "")
	body, hdr, status, err := c.get(ctx, metaURL)
	api := []APIReceipt{apiReceipt("repository", metaURL, body, hdr, status, c.Now())}
	if err != nil {
		return "", api, fmt.Errorf("repository metadata: %w", err)
	}
	var meta struct {
		DefaultBranch string `json:"default_branch"`
	}
	if err := json.Unmarshal(body, &meta); err != nil || meta.DefaultBranch == "" {
		return "", api, errors.New("repository metadata missing default_branch")
	}
	commitURL := c.repoURL(owner, repo, "commits/"+url.PathEscape(meta.DefaultBranch))
	body, hdr, status, err = c.get(ctx, commitURL)
	api = append(api, apiReceipt("revision", commitURL, body, hdr, status, c.Now()))
	if err != nil {
		return "", api, fmt.Errorf("repository revision: %w", err)
	}
	var commit struct {
		SHA string `json:"sha"`
	}
	if err := json.Unmarshal(body, &commit); err != nil || commit.SHA == "" {
		return "", api, errors.New("repository revision missing sha")
	}
	return commit.SHA, api, nil
}

func (c *Collector) collectSource(ctx context.Context, sr SourceReceipt, records []Record, cutoff time.Time) (SourceReceipt, []Record, error) {
	next := sr.Endpoint
	if len(sr.Pages) > 0 {
		next = sr.Pages[len(sr.Pages)-1].Next
	}
	if next == "" && len(sr.Pages) > 0 {
		sr.Status = StatusComplete
		sr.Failure = ""
		return sr, records, nil
	}
	for next != "" {
		body, hdr, status, err := c.get(ctx, next)
		if err != nil {
			sr.Failure = err.Error()
			if len(sr.Pages) == 0 {
				sr.Status = StatusFailed
			} else {
				sr.Status = StatusPartial
			}
			return sr, records, err
		}
		var raws []json.RawMessage
		if err := json.Unmarshal(body, &raws); err != nil {
			sr.Failure = "decode page: " + err.Error()
			sr.Status = StatusPartial
			return sr, records, errors.New(sr.Failure)
		}
		pageNext := linkNext(hdr.Get("Link"))
		page := PageReceipt{Number: len(sr.Pages) + 1, URL: next, ItemCount: len(raws), Checksum: digest(body), Next: pageNext, FetchedAt: c.Now().UTC().Format(time.RFC3339Nano), StatusCode: status}
		fillAPIHeaders(&page.RequestID, &page.ETag, &page.RateLimit, &page.RateRemain, &page.RateReset, hdr)
		pageRecords := make([]Record, 0, len(raws))
		pagePulls, pageExcluded := 0, 0
		for _, raw := range raws {
			record, isPull, excluded, err := normalize(sr.Name, raw, cutoff)
			if err != nil {
				sr.Failure = err.Error()
				sr.Status = StatusPartial
				return sr, records, err
			}
			if isPull {
				pagePulls++
				continue
			}
			if excluded {
				pageExcluded++
				continue
			}
			pageRecords = append(pageRecords, record)
		}
		// Commit the page checkpoint only after every row normalized, so resume
		// can never skip the tail of a malformed page.
		sr.Pages = append(sr.Pages, page)
		sr.FetchedCount += len(raws)
		sr.ClassifiedPullCount += pagePulls
		sr.CutoffExcludedCount += pageExcluded
		records = append(records, pageRecords...)
		next = pageNext
	}
	sr.Status, sr.Failure = StatusComplete, ""
	return sr, records, nil
}

func (c *Collector) get(ctx context.Context, endpoint string) ([]byte, http.Header, int, error) {
	for attempt := 0; ; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			return nil, nil, 0, err
		}
		req.Header.Set("Accept", "application/vnd.github+json")
		req.Header.Set("X-GitHub-Api-Version", "2022-11-28")
		resp, err := c.Client.Do(req)
		if err == nil {
			body, readErr := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
			resp.Body.Close()
			if readErr != nil {
				return body, resp.Header, resp.StatusCode, readErr
			}
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				return body, resp.Header, resp.StatusCode, nil
			}
			err = fmt.Errorf("GET %s: HTTP %d: %s", endpoint, resp.StatusCode, strings.TrimSpace(string(body)))
			if !retryable(resp.StatusCode) || attempt >= c.MaxRetries {
				return body, resp.Header, resp.StatusCode, err
			}
			if waitErr := c.RetryWait(ctx, retryDelay(resp.Header, attempt)); waitErr != nil {
				return body, resp.Header, resp.StatusCode, waitErr
			}
			continue
		}
		if attempt >= c.MaxRetries {
			return nil, nil, 0, err
		}
		if waitErr := c.RetryWait(ctx, retryDelay(nil, attempt)); waitErr != nil {
			return nil, nil, 0, waitErr
		}
	}
}

func (c *Collector) repoURL(owner, repo, suffix string) string {
	base := strings.TrimRight(c.BaseURL, "/") + "/repos/" + url.PathEscape(owner) + "/" + url.PathEscape(repo)
	if suffix != "" {
		base += "/" + suffix
	}
	return base
}

func retryable(code int) bool { return code == 429 || code >= 500 }
func retryDelay(h http.Header, attempt int) time.Duration {
	if h != nil {
		if n, err := strconv.Atoi(h.Get("Retry-After")); err == nil && n >= 0 {
			return time.Duration(n) * time.Second
		}
	}
	return time.Duration(1<<attempt) * 100 * time.Millisecond
}
func waitContext(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
func digest(b []byte) string { sum := sha256.Sum256(b); return "sha256:" + hex.EncodeToString(sum[:]) }
