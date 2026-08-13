// Package stalework builds bounded, read-only evidence packets for stale-artifact adjudication.
package stalework

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
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

const (
	Schema                    = "fak.stale-work.packet.v1"
	CheckpointSchema          = "fak.stale-work.checkpoint.v1"
	DiscoveryComplete         = "COMPLETE"
	DiscoveryPartial          = "PARTIAL"
	ReasonDiscoveryBudget     = "STALE_WORK_DISCOVERY_BUDGET"
	DefaultDiscoveryBudget    = 60 * time.Second
	maxExcerpt                = 240
	discoveryStageHead        = "head"
	discoveryStageEnumeration = "enumeration"
	discoveryStageInspection  = "inspection"
)

type Runner func(context.Context, string, ...string) ([]byte, error)
type Options struct {
	Root       string
	Limit      int
	Paths      []string
	OpenDedupe map[string]bool
	Now        time.Time
	Run        Runner
	Budget     time.Duration
	Resume     *Packet
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
	FilesScanned  int    `json:"files_scanned"`
	FilesTotal    int    `json:"files_total"`
	PacketBytes   int    `json:"packet_bytes,omitempty"`
	BudgetMillis  int64  `json:"budget_millis"`
	ElapsedMillis int64  `json:"elapsed_millis,omitempty"`
	Alternative   string `json:"alternative"`
}
type DiscoveryProgress struct {
	Status       string `json:"status"`
	Reason       string `json:"reason,omitempty"`
	FilesScanned int    `json:"files_scanned"`
	FilesTotal   int    `json:"files_total"`
	NextPath     string `json:"next_path,omitempty"`
}
type Checkpoint struct {
	Schema           string `json:"schema"`
	Stage            string `json:"stage"`
	Head             string `json:"head,omitempty"`
	PathsetSHA256    string `json:"pathset_sha256,omitempty"`
	NextIndex        int    `json:"next_index"`
	NextPath         string `json:"next_path,omitempty"`
	Limit            int    `json:"limit"`
	Now              string `json:"now"`
	OpenDedupeSHA256 string `json:"open_dedupe_sha256"`
}
type Packet struct {
	Schema      string            `json:"schema"`
	Head        string            `json:"head"`
	Discovery   DiscoveryProgress `json:"discovery"`
	Checkpoint  *Checkpoint       `json:"checkpoint,omitempty"`
	Candidates  []Candidate       `json:"candidates"`
	Exemptions  []Candidate       `json:"exemptions"`
	Abstentions []Candidate       `json:"abstentions"`
	Metrics     Metrics           `json:"metrics"`
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
	if o.Budget <= 0 {
		o.Budget = DefaultDiscoveryBudget
	}
	if o.Resume != nil && o.Now.IsZero() && o.Resume.Checkpoint != nil && o.Resume.Checkpoint.Now != "" {
		resumeNow, err := time.Parse(time.RFC3339Nano, o.Resume.Checkpoint.Now)
		if err != nil {
			return Packet{}, fmt.Errorf("resume checkpoint now: %w", err)
		}
		o.Now = resumeNow
	}
	if o.Now.IsZero() {
		o.Now = time.Now().UTC()
	}
	o.Now = o.Now.UTC()
	if err := validateResumeStatic(o); err != nil {
		return Packet{}, err
	}
	if o.Run == nil {
		o.Run = GitRunner
	}
	if o.Root == "" {
		o.Root = "."
	}
	started := time.Now()
	budgeted, cancel := context.WithTimeout(ctx, o.Budget)
	defer cancel()

	p := newPacket(o)
	head, err := o.Run(budgeted, o.Root, "rev-parse", "HEAD")
	if err != nil {
		if ctx.Err() != nil {
			return Packet{}, ctx.Err()
		}
		if discoveryBudgetExceeded(ctx, budgeted) {
			if o.Resume != nil && o.Resume.Checkpoint != nil {
				return pausedResumePacket(p, o, started), nil
			}
			return partialPacket(p, o, started, discoveryStageHead, "", nil, 0), nil
		}
		return Packet{}, err
	}
	headText := strings.TrimSpace(string(head))
	if ctx.Err() != nil {
		return Packet{}, ctx.Err()
	}
	if o.Resume != nil && o.Resume.Checkpoint.Head != "" && o.Resume.Checkpoint.Head != headText {
		return Packet{}, fmt.Errorf("resume checkpoint head %s differs from current head %s", o.Resume.Checkpoint.Head, headText)
	}
	p.Head = headText
	var listed []byte
	if len(o.Paths) > 0 {
		listed = []byte(strings.Join(o.Paths, "\n"))
	} else {
		listed, err = o.Run(budgeted, o.Root, "ls-files", "*.md", "CLAIMS.md")
		if err != nil {
			if ctx.Err() != nil {
				return Packet{}, ctx.Err()
			}
			if discoveryBudgetExceeded(ctx, budgeted) {
				if o.Resume != nil && o.Resume.Checkpoint != nil {
					return pausedResumePacket(p, o, started), nil
				}
				return partialPacket(p, o, started, discoveryStageEnumeration, headText, nil, 0), nil
			}
			return Packet{}, err
		}
	}
	paths := discoveryPaths(listed)
	p.Metrics.FilesTotal = len(paths)

	next, err := resumeAt(o, &p, paths)
	if err != nil {
		return Packet{}, err
	}
	for i := next; i < len(paths); i++ {
		path := paths[i]
		if ctx.Err() != nil {
			return Packet{}, ctx.Err()
		}
		if discoveryBudgetExceeded(ctx, budgeted) {
			return partialPacket(p, o, started, discoveryStageInspection, headText, paths, i), nil
		}
		c, e := inspect(budgeted, o, path)
		if ctx.Err() != nil {
			return Packet{}, ctx.Err()
		}
		if e != nil && discoveryBudgetExceeded(ctx, budgeted) {
			return partialPacket(p, o, started, discoveryStageInspection, headText, paths, i), nil
		}
		p.Metrics.FilesScanned++
		if e != nil {
			c = abstention(path, e)
			p.Abstentions = append(p.Abstentions, c)
			if discoveryBudgetExceeded(ctx, budgeted) {
				return partialPacket(p, o, started, discoveryStageInspection, headText, paths, i+1), nil
			}
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
		if ctx.Err() != nil {
			return Packet{}, ctx.Err()
		}
		if discoveryBudgetExceeded(ctx, budgeted) {
			return partialPacket(p, o, started, discoveryStageInspection, headText, paths, i+1), nil
		}
	}
	p.Checkpoint = nil
	p.Discovery = DiscoveryProgress{
		Status: DiscoveryComplete, FilesScanned: p.Metrics.FilesScanned, FilesTotal: len(paths),
	}
	p.Metrics.FilesTotal = len(paths)
	p.Metrics.ElapsedMillis = 0
	finalizePacket(&p, o.Limit)
	return p, nil
}

func newPacket(o Options) Packet {
	p := Packet{
		Schema: Schema,
		Metrics: Metrics{
			BudgetMillis: durationMillisCeil(o.Budget),
			Alternative:  "git log + grep + full-file reads (not measured; no gain claimed)",
		},
	}
	if o.Resume == nil {
		return p
	}
	p = *o.Resume
	p.Candidates = append([]Candidate(nil), o.Resume.Candidates...)
	p.Exemptions = append([]Candidate(nil), o.Resume.Exemptions...)
	p.Abstentions = append([]Candidate(nil), o.Resume.Abstentions...)
	p.Checkpoint = nil
	p.Metrics.PacketBytes = 0
	p.Metrics.BudgetMillis = durationMillisCeil(o.Budget)
	p.Metrics.ElapsedMillis = 0
	if p.Metrics.Alternative == "" {
		p.Metrics.Alternative = "git log + grep + full-file reads (not measured; no gain claimed)"
	}
	return p
}

func validateResumeStatic(o Options) error {
	if o.Resume == nil {
		return nil
	}
	if o.Resume.Schema != Schema {
		return fmt.Errorf("resume packet schema %q, want %q", o.Resume.Schema, Schema)
	}
	if o.Resume.Discovery.Status != DiscoveryPartial || o.Resume.Checkpoint == nil {
		return errors.New("resume packet has no partial discovery checkpoint")
	}
	cp := o.Resume.Checkpoint
	if cp.Schema != CheckpointSchema {
		return fmt.Errorf("resume checkpoint schema %q, want %q", cp.Schema, CheckpointSchema)
	}
	if cp.Limit != o.Limit {
		return fmt.Errorf("resume checkpoint limit %d differs from requested limit %d", cp.Limit, o.Limit)
	}
	if cp.OpenDedupeSHA256 != openDedupeDigest(o.OpenDedupe) {
		return errors.New("resume checkpoint open-issue dedupe set changed")
	}
	if cp.Now != o.Now.Format(time.RFC3339Nano) {
		return errors.New("resume checkpoint clock changed")
	}
	switch cp.Stage {
	case discoveryStageHead, discoveryStageEnumeration, discoveryStageInspection:
		return nil
	default:
		return fmt.Errorf("resume checkpoint stage %q is invalid", cp.Stage)
	}
}

func resumeAt(o Options, p *Packet, paths []string) (int, error) {
	if o.Resume == nil {
		p.Metrics.FilesScanned = 0
		return 0, nil
	}
	cp := o.Resume.Checkpoint
	switch cp.Stage {
	case discoveryStageHead, discoveryStageEnumeration:
		p.Metrics.FilesScanned = 0
		return 0, nil
	case discoveryStageInspection:
	}
	if cp.PathsetSHA256 != pathsetDigest(paths) {
		return 0, errors.New("resume checkpoint tracked path set changed")
	}
	if cp.NextIndex < 0 || cp.NextIndex > len(paths) {
		return 0, fmt.Errorf("resume checkpoint next index %d outside %d paths", cp.NextIndex, len(paths))
	}
	if p.Metrics.FilesScanned != cp.NextIndex {
		return 0, fmt.Errorf("resume checkpoint scanned %d files but next index is %d", p.Metrics.FilesScanned, cp.NextIndex)
	}
	return cp.NextIndex, nil
}

func partialPacket(p Packet, o Options, started time.Time, stage, head string, paths []string, next int) Packet {
	if p.Schema == "" {
		p.Schema = Schema
	}
	if head != "" {
		p.Head = head
	}
	if next < 0 {
		next = 0
	}
	if next > len(paths) {
		next = len(paths)
	}
	nextPath := ""
	if next < len(paths) {
		nextPath = paths[next]
	}
	p.Metrics.FilesTotal = len(paths)
	p.Metrics.BudgetMillis = durationMillisCeil(o.Budget)
	p.Metrics.ElapsedMillis = durationMillisCeil(time.Since(started))
	p.Discovery = DiscoveryProgress{
		Status: DiscoveryPartial, Reason: ReasonDiscoveryBudget,
		FilesScanned: p.Metrics.FilesScanned, FilesTotal: len(paths), NextPath: nextPath,
	}
	cp := &Checkpoint{
		Schema: CheckpointSchema, Stage: stage, Head: head,
		NextIndex: next, NextPath: nextPath, Limit: o.Limit,
		Now: o.Now.Format(time.RFC3339Nano), OpenDedupeSHA256: openDedupeDigest(o.OpenDedupe),
	}
	if paths != nil {
		cp.PathsetSHA256 = pathsetDigest(paths)
	}
	p.Checkpoint = cp
	finalizePacket(&p, o.Limit)
	return p
}

func pausedResumePacket(p Packet, o Options, started time.Time) Packet {
	cp := *o.Resume.Checkpoint
	p.Checkpoint = &cp
	p.Discovery = o.Resume.Discovery
	p.Metrics.BudgetMillis = durationMillisCeil(o.Budget)
	p.Metrics.ElapsedMillis = durationMillisCeil(time.Since(started))
	finalizePacket(&p, o.Limit)
	return p
}

func finalizePacket(p *Packet, limit int) {
	sort.Slice(p.Candidates, func(i, j int) bool {
		if p.Candidates[i].Score == p.Candidates[j].Score {
			return p.Candidates[i].Path < p.Candidates[j].Path
		}
		return p.Candidates[i].Score > p.Candidates[j].Score
	})
	if len(p.Candidates) > limit {
		p.Candidates = p.Candidates[:limit]
	}
	sort.Slice(p.Exemptions, func(i, j int) bool { return p.Exemptions[i].Path < p.Exemptions[j].Path })
	sort.Slice(p.Abstentions, func(i, j int) bool { return p.Abstentions[i].Path < p.Abstentions[j].Path })
}

func discoveryBudgetExceeded(parent, scan context.Context) bool {
	return parent.Err() == nil && errors.Is(scan.Err(), context.DeadlineExceeded)
}

func durationMillisCeil(d time.Duration) int64 {
	if d <= 0 {
		return 0
	}
	return int64((d + time.Millisecond - 1) / time.Millisecond)
}

func discoveryPaths(listed []byte) []string {
	set := map[string]bool{}
	for _, path := range lines(listed) {
		path = strings.TrimSpace(filepath.ToSlash(path))
		if path != "" {
			set[path] = true
		}
	}
	paths := make([]string, 0, len(set))
	for path := range set {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return paths
}

func pathsetDigest(paths []string) string {
	h := sha256.Sum256([]byte(strings.Join(paths, "\n")))
	return hex.EncodeToString(h[:])
}

func openDedupeDigest(keys map[string]bool) string {
	active := make([]string, 0, len(keys))
	for key, present := range keys {
		if present {
			active = append(active, key)
		}
	}
	sort.Strings(active)
	h := sha256.Sum256([]byte(strings.Join(active, "\n")))
	return hex.EncodeToString(h[:])
}

func abstention(path string, err error) Candidate {
	c := artifactRecord(path)
	c.Status = "abstain"
	c.Abstain = err.Error()
	return c
}

func inspect(ctx context.Context, o Options, path string) (Candidate, error) {
	c := artifactRecord(path)
	if err := ctx.Err(); err != nil {
		return c, err
	}
	raw, err := os.ReadFile(filepath.Join(o.Root, filepath.FromSlash(path)))
	if err != nil {
		return c, err
	}
	if err := ctx.Err(); err != nil {
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
	if ctx.Err() != nil {
		return c, ctx.Err()
	}
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
		if ctx.Err() != nil {
			return c, ctx.Err()
		}
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
	if err := ctx.Err(); err != nil {
		return c, err
	}
	return c, nil
}
func artifactRecord(path string) Candidate {
	return Candidate{Path: path, Batch: path, Status: "candidate", DedupeKey: "stale-work:" + path, ProposedDoD: []string{"adjudicate every packet reason against current source", "update or retain the artifact with a captured witness", "do not delete without dedicated issue approval"}, VerifyWith: "fak stale-work --path " + path + " --json", Recommendation: "adjudicate only; no update/delete recommendation"}
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
