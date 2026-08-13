// Package stalework builds bounded, read-only evidence packets for stale-artifact adjudication.
package stalework

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/docfreshrsi"
)

const Schema = "fak.stale-work.packet.v1"
const maxExcerpt = 240

type Runner func(context.Context, string, ...string) ([]byte, error)
type Options struct {
	Root       string
	Limit      int
	Paths      []string
	OpenDedupe map[string]bool
	Now        time.Time
	Run        Runner
}
type Component struct {
	Name       string `json:"name"`
	Points     int    `json:"points"`
	Provenance string `json:"provenance"`
	Evidence   string `json:"evidence"`
}
type Commit struct {
	SHA        string `json:"sha"`
	Subject    string `json:"subject"`
	Dependency string `json:"dependency"`
}
type Candidate struct {
	Path               string      `json:"path"`
	Batch              string      `json:"batch"`
	Score              int         `json:"score"`
	Status             string      `json:"status"`
	Components         []Component `json:"score_components"`
	LastSemanticCommit string      `json:"last_semantic_commit,omitempty"`
	DependencyCommits  []Commit    `json:"dependency_commits,omitempty"`
	Excerpt            string      `json:"excerpt,omitempty"`
	ExcerptSHA256      string      `json:"excerpt_sha256,omitempty"`
	Exemption          string      `json:"exemption_reason,omitempty"`
	Abstain            string      `json:"abstain_reason,omitempty"`
	DedupeKey          string      `json:"dedupe_key"`
	ProposedDoD        []string    `json:"proposed_dod"`
	VerifyWith         string      `json:"verify_with"`
	Recommendation     string      `json:"recommendation"`
}
type Metrics struct {
	FilesScanned int    `json:"files_scanned"`
	PacketBytes  int    `json:"packet_bytes,omitempty"`
	Alternative  string `json:"alternative"`
}
type Packet struct {
	Schema      string      `json:"schema"`
	Head        string      `json:"head"`
	Candidates  []Candidate `json:"candidates"`
	Exemptions  []Candidate `json:"exemptions"`
	Abstentions []Candidate `json:"abstentions"`
	Metrics     Metrics     `json:"metrics"`
}

var depRE = regexp.MustCompile(`(?:internal|cmd)/[a-zA-Z0-9_.-]+|fak\s+([a-zA-Z0-9_-]+)`)

func GitRunner(ctx context.Context, dir string, args ...string) ([]byte, error) {
	c := exec.CommandContext(ctx, "git", args...)
	configureDispatchHelperCommand(c)
	c.Dir = dir
	b, e := c.CombinedOutput()
	if e != nil {
		return nil, fmt.Errorf("git %s: %s", strings.Join(args, " "), strings.TrimSpace(string(b)))
	}
	return b, nil
}

func Scan(ctx context.Context, o Options) (Packet, error) {
	if o.Limit <= 0 {
		o.Limit = 10
	}
	if o.Now.IsZero() {
		o.Now = time.Now().UTC()
	}
	if o.Run == nil {
		o.Run = GitRunner
	}
	if o.Root == "" {
		o.Root = "."
	}
	head, err := o.Run(ctx, o.Root, "rev-parse", "HEAD")
	if err != nil {
		return Packet{}, err
	}
	var listed []byte
	if len(o.Paths) > 0 {
		listed = []byte(strings.Join(o.Paths, "\n"))
	} else {
		listed, err = o.Run(ctx, o.Root, "ls-files", "*.md", "CLAIMS.md")
		if err != nil {
			return Packet{}, err
		}
	}
	p := Packet{Schema: Schema, Head: strings.TrimSpace(string(head)), Metrics: Metrics{Alternative: "git log + grep + full-file reads (not measured; no gain claimed)"}}
	for _, path := range lines(listed) {
		if path == "" {
			continue
		}
		p.Metrics.FilesScanned++
		c, e := inspect(ctx, o, path)
		if e != nil {
			c = Candidate{Path: path, Batch: path, Status: "candidate", DedupeKey: "stale-work:" + path, ProposedDoD: []string{"adjudicate every packet reason against current source", "update or retain the artifact with a captured witness", "do not delete without dedicated issue approval"}, VerifyWith: "fak stale-work --path " + path + " --json", Recommendation: "adjudicate only; no update/delete recommendation"}
			c.Status = "abstain"
			c.Abstain = e.Error()
			p.Abstentions = append(p.Abstentions, c)
			continue
		}
		switch c.Status {
		case "exempt":
			p.Exemptions = append(p.Exemptions, c)
		case "abstain":
			p.Abstentions = append(p.Abstentions, c)
		default:
			p.Candidates = append(p.Candidates, c)
		}
	}
	sort.Slice(p.Candidates, func(i, j int) bool {
		if p.Candidates[i].Score == p.Candidates[j].Score {
			return p.Candidates[i].Path < p.Candidates[j].Path
		}
		return p.Candidates[i].Score > p.Candidates[j].Score
	})
	if len(p.Candidates) > o.Limit {
		p.Candidates = p.Candidates[:o.Limit]
	}
	sort.Slice(p.Exemptions, func(i, j int) bool { return p.Exemptions[i].Path < p.Exemptions[j].Path })
	sort.Slice(p.Abstentions, func(i, j int) bool { return p.Abstentions[i].Path < p.Abstentions[j].Path })
	return p, nil
}

func inspect(ctx context.Context, o Options, path string) (Candidate, error) {
	c := Candidate{Path: path, Batch: path, Status: "candidate", DedupeKey: "stale-work:" + path, ProposedDoD: []string{"adjudicate every packet reason against current source", "update or retain the artifact with a captured witness", "do not delete without dedicated issue approval"}, VerifyWith: "fak stale-work --path " + path + " --json", Recommendation: "adjudicate only; no update/delete recommendation"}
	raw, err := os.ReadFile(filepath.Join(o.Root, filepath.FromSlash(path)))
	if err != nil {
		return c, err
	}
	text := string(raw)
	low := strings.ToLower(path + "\n" + first(text, 400))
	if strings.Contains(low, "generated") || strings.HasPrefix(path, "vendor/") || strings.HasPrefix(path, "third_party/") {
		c.Status = "exempt"
		c.Exemption = "generated or third-party artifact"
		return c, nil
	}
	if strings.HasPrefix(path, "docs/notes/") && (strings.Contains(low, "historical") || strings.Contains(low, "archive") || strings.Contains(low, "study")) {
		c.Status = "exempt"
		c.Exemption = "historical note"
		return c, nil
	}
	last, err := o.Run(ctx, o.Root, "log", "-1", "--format=%H|%ct", "--", path)
	if err != nil || strings.TrimSpace(string(last)) == "" {
		c.Status = "abstain"
		c.Abstain = "no semantic commit evidence"
		return c, nil
	}
	parts := strings.SplitN(strings.TrimSpace(string(last)), "|", 2)
	c.LastSemanticCommit = parts[0]
	deps := dependencies(text)
	for _, dep := range deps {
		b, e := o.Run(ctx, o.Root, "log", "--format=%H|%s", c.LastSemanticCommit+"..HEAD", "--", dep)
		if e != nil {
			continue
		}
		for _, line := range lines(b) {
			q := strings.SplitN(line, "|", 2)
			if len(q) == 2 {
				c.DependencyCommits = append(c.DependencyCommits, Commit{SHA: q[0], Subject: q[1], Dependency: dep})
			}
		}
	}
	if len(c.DependencyCommits) > 0 {
		add(&c, "dependency_drift", 50, "git", fmt.Sprintf("%d commits to declared dependencies since artifact commit", len(c.DependencyCommits)))
	}
	claims := docfreshrsi.ScanVersionClaims(path, text)
	if len(claims) > 0 {
		add(&c, "outdated_reference", 20, "docfreshrsi", fmt.Sprintf("%d unpointed live version claims", len(claims)))
	}
	if len(parts) == 2 {
		if sec, e := strconv.ParseInt(parts[1], 10, 64); e == nil {
			days := int(o.Now.Sub(time.Unix(sec, 0)).Hours() / 24)
			if days >= 365 {
				add(&c, "age", 5, "git", fmt.Sprintf("%d days; weak feature only", days))
			}
		}
	}
	strong := len(c.DependencyCommits) > 0 || len(claims) > 0
	if !strong {
		c.Status = "abstain"
		c.Abstain = "age alone is insufficient; no semantic drift evidence"
		return c, nil
	}
	c.Excerpt = first(strings.TrimSpace(text), maxExcerpt)
	h := sha256.Sum256([]byte(c.Excerpt))
	c.ExcerptSHA256 = hex.EncodeToString(h[:])
	if len(deps) > 0 {
		c.Batch = deps[0]
	} else {
		c.Batch = path
	}
	c.DedupeKey = "stale-work:" + path
	if o.OpenDedupe[c.DedupeKey] {
		c.Status = "exempt"
		c.Exemption = "already-open candidate"
	}
	return c, nil
}
func add(c *Candidate, n string, p int, prov, e string) {
	c.Components = append(c.Components, Component{n, p, prov, e})
	c.Score += p
}
func dependencies(s string) []string {
	m := depRE.FindAllStringSubmatch(s, -1)
	set := map[string]bool{}
	for _, x := range m {
		v := strings.Fields(x[0])
		d := v[0]
		if d == "fak" && len(x) > 1 {
			d = "cmd/fak"
		}
		set[d] = true
	}
	out := make([]string, 0, len(set))
	for x := range set {
		out = append(out, x)
	}
	sort.Strings(out)
	return out
}
func lines(b []byte) []string {
	return strings.Split(strings.TrimSpace(strings.ReplaceAll(string(b), "\r\n", "\n")), "\n")
}
func first(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
