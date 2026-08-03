package ideascout

// The live fetcher: the only code in the package that touches the network or
// shells out to gh. Everything else in the package works off its bytes.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ghexec"
)

type LiveFetcher struct {
	HTTPClient *http.Client
}

func (f LiveFetcher) FetchArxiv(query string, maxResults int) (string, error) {
	client := f.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	q := url.Values{}
	q.Set("search_query", query)
	q.Set("sortBy", "submittedDate")
	q.Set("sortOrder", "descending")
	q.Set("max_results", strconv.Itoa(maxResults))
	req, err := http.NewRequest("GET", ArxivAPI+"?"+q.Encode(), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "fak-idea-scout/1.0")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("arxiv status %s", resp.Status)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func (f LiveFetcher) FetchGitHub(query string, limit int) ([]GitHubRepo, error) {
	var out []GitHubRepo
	err := ghJSONFn([]string{"search", "repos", query, "--limit", strconv.Itoa(limit), "--sort", "stars", "--json", "fullName,description,url,stargazersCount,pushedAt,updatedAt,createdAt,language"}, 60*time.Second, &out)
	return out, err
}

// FetchGitHubFresh is the recency-first companion to FetchGitHub: the SAME topic
// query (so the neighborhood stays "relative to ours") sorted by most-recently
// updated instead of all-time stars, so newly-created / trending / freshly-pushed
// repos surface where the stars sort would bury them under incumbents.
func (f LiveFetcher) FetchGitHubFresh(query string, limit int) ([]GitHubRepo, error) {
	var out []GitHubRepo
	err := ghJSONFn([]string{"search", "repos", query, "--limit", strconv.Itoa(limit), "--sort", "updated", "--json", "fullName,description,url,stargazersCount,pushedAt,updatedAt,createdAt,language"}, 60*time.Second, &out)
	return out, err
}

func (f LiveFetcher) FetchHackerNews(query string, limit int) (string, error) {
	client := f.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	q := url.Values{}
	q.Set("query", query)
	q.Set("tags", "story")
	q.Set("hitsPerPage", strconv.Itoa(limit))
	req, err := http.NewRequest("GET", HNAlgoliaAPI+"?"+q.Encode(), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "fak-idea-scout/1.0")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("hn status %s", resp.Status)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func (f LiveFetcher) FetchReddit(query string, limit int) (string, error) {
	client := f.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	q := url.Values{}
	q.Set("q", query)
	q.Set("sort", "new")
	q.Set("t", "week")
	q.Set("limit", strconv.Itoa(limit))
	req, err := http.NewRequest("GET", RedditSearchAPI+"?"+q.Encode(), nil)
	if err != nil {
		return "", err
	}
	// Reddit rejects requests without a descriptive, non-default User-Agent.
	req.Header.Set("User-Agent", "fak-idea-scout/1.0 (+https://github.com/anthony-chaudhary/fak)")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("reddit status %s", resp.Status)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// issueListQuery builds the gh argv for one of the two dedup corpora. Both are
// `gh issue list --state all`; `label` is the ONLY difference between them, and
// that one flag pair IS the never-file-twice guarantee:
//
//   - empty label  → the rung 3/4 RECENCY WINDOW: the most recent `limit` issues,
//     whoever opened them, whose coverage shrinks every time the tracker gets busier.
//   - ScoutLabel   → the rung 2 FILING HISTORY: filtered server-side to the exact
//     population being deduped, so its size tracks the scout's own capped output
//     (MaxIssues/day) rather than the tracker's growth rate.
//
// The two are built here, side by side and off one parameter, so the difference is
// structural rather than a coincidence of two literals that can drift apart. A
// regression that hands the scout query an empty label turns the durable rung back
// into a window and silently reopens #5544; TestScoutIndexQueryIsLabelTargeted
// reads this argv back and refuses that.
//
// `--state all` is unconditional and load-bearing on both: a source whose issue was
// triaged and CLOSED is the exact case that must not come back.
func issueListQuery(label string, limit int) []string {
	argv := []string{"issue", "list", "--state", "all"}
	if label != "" {
		argv = append(argv, "--label", label)
	}
	return append(argv, "--limit", strconv.Itoa(limit), "--json", "number,title,body")
}

// ghJSONFn is the shell-out seam. Production always runs ghJSON; the query tests
// swap it to read back the argv a fetcher WOULD send without touching the network,
// which is the only way to pin "the durable rung's query is label-targeted" from a
// test. tools/idea_scout_test.py pins the same property the same way, by patching
// its own gh_json (test_scout_index_query_is_label_targeted_not_windowed).
var ghJSONFn = ghJSON

// FetchExistingIssues is the rung 3/4 corpus: the `limit` most recent issues,
// whoever opened them. A RECENCY WINDOW — it answers "did a human already write
// this up lately", and that is all it is allowed to answer.
func (f LiveFetcher) FetchExistingIssues(limit int) ([]ExistingIssue, error) {
	var out []ExistingIssue
	err := ghJSONFn(issueListQuery("", limit), 60*time.Second, &out)
	return out, err
}

// FetchScoutIssues is the rung 2 corpus: every issue the scout has EVER filed,
// open or closed.
//
// TARGETED, not windowed. `--label idea-scout` is a server-side filter, so the
// result set is the scout's own filing history — it does not thin out because
// unrelated issues were opened this week, which is precisely how the recency
// window in FetchExistingIssues lost the guarantee. Every filed issue carries the
// label (RenderIssue always emits ScoutLabel) and the matching
// `<!-- idea-scout-source: … -->` stamp, so label ⊇ stamped-by-us.
//
// `--state all` is load-bearing: a source whose issue was triaged and CLOSED is
// the exact case that must not come back. The longer deadline is deliberate — a
// years-deep index is a bigger page walk than the 60s window fetch, and a timeout
// here refuses the whole run.
func (f LiveFetcher) FetchScoutIssues(limit int) ([]ExistingIssue, error) {
	var out []ExistingIssue
	err := ghJSONFn(issueListQuery(ScoutLabel, limit), 180*time.Second, &out)
	return out, err
}

func (f LiveFetcher) EnsureLabels() error {
	wanted := []struct {
		name  string
		color string
		desc  string
	}{
		{ScoutLabel, "8a63d2", "Auto-filed by the daily idea-scout; needs human triage"},
		{TriageLabel, "d4c5f9", "Needs human scoping before an agent dispatch can take it"},
		{TriageOnlyLabel, "d4c5f9", "Useful issue, but not a worker-ready dispatch leaf"},
	}
	var errs []string
	for _, w := range wanted {
		if _, _, err := runGH([]string{"label", "create", w.name, "--color", w.color, "--description", w.desc, "--force"}, 30*time.Second); err != nil {
			errs = append(errs, err.Error())
		}
	}
	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

func (f LiveFetcher) CreateIssue(issue IssuePlan, milestone string) (string, error) {
	args := []string{"issue", "create", "--title", issue.Title, "--body", issue.Body}
	for _, lab := range issue.Labels {
		args = append(args, "--label", lab)
	}
	if milestone != "" {
		args = append(args, "--milestone", milestone)
	}
	stdout, _, err := runGH(args, 60*time.Second)
	if err != nil {
		return "", err
	}
	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	if len(lines) == 0 {
		return "", nil
	}
	return strings.TrimSpace(lines[len(lines)-1]), nil
}

func (f LiveFetcher) AddToProject(issueURL, number, owner string) error {
	args := []string{"project", "item-add", number, "--url", issueURL}
	if owner != "" {
		args = append(args, "--owner", owner)
	}
	_, _, err := runGH(args, 60*time.Second)
	return err
}

func ghJSON(args []string, timeout time.Duration, out any) error {
	stdout, _, err := runGH(args, timeout)
	if err != nil {
		return err
	}
	if strings.TrimSpace(stdout) == "" {
		return nil
	}
	return json.Unmarshal([]byte(stdout), out)
}

func runGH(args []string, timeout time.Duration) (string, string, error) {
	cmd, cancel := ghexec.CommandTimeout(context.Background(), timeout, args...)
	defer cancel()
	// WaitDelay is the straggler backstop: if the deadline kill leaves a grandchild
	// holding the output pipe open, cmd.Run could still block past the timeout;
	// WaitDelay forces the pipes closed so the deadline is real (issue #3483).
	cmd.WaitDelay = 10 * time.Second
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err != nil {
		return stdout.String(), stderr.String(), fmt.Errorf("gh %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.String(), stderr.String(), nil
}
