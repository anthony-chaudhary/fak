// Package studylink deterministically joins stable upstream mechanism clusters to
// captured FAK issue state and repository artifacts.
package studylink

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const Schema = "fak.study-join-ledger/1"

type Disposition string

const (
	Landed    Disposition = "landed"
	OpenExact Disposition = "open_exact"
	Partial   Disposition = "partial"
	Conflict  Disposition = "conflict"
	Obsolete  Disposition = "obsolete"
	Uncovered Disposition = "uncovered"
)

var dispositionOrder = []Disposition{Landed, OpenExact, Partial, Conflict, Obsolete, Uncovered}

type Artifact struct {
	Kind         string `json:"kind"`
	ID           string `json:"id"`
	Revision     string `json:"revision,omitempty"`
	Path         string `json:"path,omitempty"`
	State        string `json:"state,omitempty"`
	Title        string `json:"title,omitempty"`
	URL          string `json:"url,omitempty"`
	Exact        bool   `json:"exact,omitempty"`
	RecordDigest string `json:"record_digest,omitempty"`
}

type Evidence struct {
	Query             string   `json:"query"`
	MatchCount        int      `json:"match_count"`
	Matches           []string `json:"matches"`
	FullMatchesDigest string   `json:"full_matches_digest"`
	Digest            string   `json:"digest"`
}

type Join struct {
	ClusterID       string      `json:"cluster_id"`
	Mechanism       string      `json:"mechanism"`
	Signal          string      `json:"signal"`
	Rule            string      `json:"rule"`
	Actionable      bool        `json:"actionable"`
	Disposition     Disposition `json:"disposition"`
	Artifacts       []Artifact  `json:"artifacts,omitempty"`
	Confidence      string      `json:"confidence"`
	Evidence        Evidence    `json:"evidence"`
	ManualReview    bool        `json:"manual_review,omitempty"`
	ManualReason    string      `json:"manual_reason,omitempty"`
	MembersChecksum string      `json:"members_checksum,omitempty"`
}

type CapturedIssue struct {
	Number       int    `json:"number"`
	State        string `json:"state"`
	Title        string `json:"title"`
	URL          string `json:"url"`
	RecordDigest string `json:"record_digest"`
}

type Sources struct {
	IndexPath          string `json:"index_path"`
	IndexSHA256        string `json:"index_sha256"`
	IndexClusterDigest string `json:"index_cluster_digest,omitempty"`
	ForgePath          string `json:"forge_path"`
	ForgeSHA256        string `json:"forge_sha256"`
	ForgeSchema        string `json:"forge_schema"`
	ForgeRevision      string `json:"forge_revision"`
	ForgeCutoff        string `json:"forge_cutoff"`
	ForgeRecordCount   int    `json:"forge_record_count"`
	AdjacencyPath      string `json:"adjacency_path"`
	AdjacencySHA256    string `json:"adjacency_sha256"`
	AdjacencyID        string `json:"adjacency_id"`
	AdjacencyMembers   int    `json:"adjacency_members"`
	RepositoryRoot     string `json:"repository_root"`
	RepositoryRevision string `json:"repository_revision"`
}

type Ledger struct {
	Schema         string          `json:"schema"`
	Cutoff         string          `json:"cutoff"`
	SourceRevision string          `json:"source_revision"`
	Sources        Sources         `json:"sources"`
	CapturedIssues []CapturedIssue `json:"captured_issues,omitempty"`
	Joins          []Join          `json:"joins"`
}

type ManualReviewJoin struct {
	ClusterID         string      `json:"cluster_id"`
	Disposition       Disposition `json:"disposition"`
	Reason            string      `json:"reason"`
	MatchCount        int         `json:"match_count"`
	FullMatchesDigest string      `json:"full_matches_digest"`
	Matches           []string    `json:"matches,omitempty"`
}

type SampleJoin struct {
	ClusterID   string      `json:"cluster_id"`
	Disposition Disposition `json:"disposition"`
	Artifacts   []string    `json:"artifacts,omitempty"`
}

type Summary struct {
	Schema           string              `json:"schema"`
	Total            int                 `json:"total"`
	Actionable       int                 `json:"actionable"`
	Counts           map[Disposition]int `json:"counts"`
	ManualReview     []ManualReviewJoin  `json:"manual_review"`
	StrongSamples    []SampleJoin        `json:"strong_samples"`
	AmbiguousSamples []SampleJoin        `json:"ambiguous_samples"`
	Sources          Sources             `json:"sources"`
}

var ErrInvalid = errors.New("studylink: invalid join ledger")

type CompactIndex struct {
	Schema                  string    `json:"schema"`
	ClustersChecksum        string    `json:"clusters_checksum"`
	CompactClustersChecksum string    `json:"compact_clusters_checksum"`
	Clusters                []Cluster `json:"clusters"`
}

type Cluster struct {
	Key             string `json:"key"`
	Mechanism       string `json:"mechanism"`
	Rule            string `json:"rule"`
	Signal          string `json:"signal"`
	Actionable      bool   `json:"actionable"`
	Confidence      string `json:"confidence"`
	MembersChecksum string `json:"members_checksum"`
}

type ForgeCorpus struct {
	Schema  string        `json:"schema"`
	Receipt ForgeReceipt  `json:"receipt"`
	Records []ForgeRecord `json:"records"`
}

type ForgeReceipt struct {
	Schema     string               `json:"schema"`
	Repository string               `json:"repository"`
	Revision   string               `json:"revision"`
	Cutoff     string               `json:"cutoff"`
	Status     string               `json:"status"`
	Sources    []ForgeReceiptSource `json:"sources"`
}

type ForgeReceiptSource struct {
	Name   string `json:"name"`
	Status string `json:"status"`
}

type ForgeRecord struct {
	Source string   `json:"source"`
	Kind   string   `json:"kind"`
	Number int      `json:"number"`
	Title  string   `json:"title"`
	Body   string   `json:"body"`
	State  string   `json:"state"`
	URL    string   `json:"url"`
	Labels []string `json:"labels"`
}

type AdjacencyManifest struct {
	Schema  string            `json:"schema"`
	ID      string            `json:"id"`
	Members []json.RawMessage `json:"members"`
}

func Summarize(l Ledger) (Summary, error) {
	if err := ValidateStructure(l, nil, ""); err != nil {
		return Summary{}, err
	}
	s := Summary{
		Schema:  "fak.study-link-summary/1",
		Total:   len(l.Joins),
		Counts:  map[Disposition]int{},
		Sources: l.Sources,
	}
	for _, j := range l.Joins {
		s.Counts[j.Disposition]++
		if j.Actionable {
			s.Actionable++
		}
		refs := artifactRefs(j.Artifacts)
		if j.ManualReview || j.Disposition == Partial || j.Disposition == Conflict {
			s.ManualReview = append(s.ManualReview, ManualReviewJoin{
				ClusterID: j.ClusterID, Disposition: j.Disposition, Reason: j.ManualReason,
				MatchCount: j.Evidence.MatchCount, FullMatchesDigest: j.Evidence.FullMatchesDigest,
				Matches: append([]string(nil), j.Evidence.Matches...),
			})
			if len(s.AmbiguousSamples) < 6 {
				s.AmbiguousSamples = append(s.AmbiguousSamples, SampleJoin{j.ClusterID, j.Disposition, refs})
			}
		} else if (j.Disposition == Landed || j.Disposition == OpenExact) && len(s.StrongSamples) < 6 {
			s.StrongSamples = append(s.StrongSamples, SampleJoin{j.ClusterID, j.Disposition, refs})
		}
	}
	sort.Slice(s.ManualReview, func(i, j int) bool { return s.ManualReview[i].ClusterID < s.ManualReview[j].ClusterID })
	return s, nil
}

func artifactRefs(artifacts []Artifact) []string {
	out := make([]string, 0, len(artifacts))
	for _, a := range artifacts {
		switch a.Kind {
		case "issue":
			out = append(out, "#"+a.ID+" ("+a.State+")")
		case "repo_path":
			out = append(out, a.Path)
		}
	}
	return out
}

// ValidateStructure validates ledger-internal invariants and, when index and repoRoot
// are supplied, complete cluster coverage and repository-path existence.
func ValidateStructure(l Ledger, index *CompactIndex, repoRoot string) error {
	if l.Schema != Schema || l.Cutoff == "" || l.SourceRevision == "" || len(l.Joins) == 0 {
		return invalidf("missing schema, cutoff, source revision, or joins")
	}
	validDisposition := map[Disposition]bool{}
	for _, d := range dispositionOrder {
		validDisposition[d] = true
	}
	captured := map[int]CapturedIssue{}
	for _, issue := range l.CapturedIssues {
		if issue.Number <= 0 || issue.State == "" || issue.RecordDigest == "" {
			return invalidf("invalid captured issue %d", issue.Number)
		}
		if _, exists := captured[issue.Number]; exists {
			return invalidf("duplicate captured issue #%d", issue.Number)
		}
		captured[issue.Number] = issue
	}
	seenCluster := map[string]bool{}
	exact := map[string]string{}
	for _, j := range l.Joins {
		if j.ClusterID == "" || seenCluster[j.ClusterID] || !validDisposition[j.Disposition] || j.Evidence.Query == "" || j.Evidence.Digest == "" || j.Evidence.FullMatchesDigest == "" || j.Evidence.MatchCount < len(j.Evidence.Matches) {
			return invalidf("invalid or duplicate cluster %q", j.ClusterID)
		}
		seenCluster[j.ClusterID] = true
		if got := evidenceDigest(j); got != j.Evidence.Digest {
			return invalidf("cluster %s evidence digest mismatch", j.ClusterID)
		}
		if j.Actionable && j.Disposition == Uncovered && len(j.Artifacts) > 0 {
			return invalidf("uncovered cluster %s carries artifacts", j.ClusterID)
		}
		if (j.Disposition == Partial || j.Disposition == Conflict) && !j.ManualReview {
			return invalidf("manual-review disposition %s lacks marker", j.ClusterID)
		}
		for _, a := range j.Artifacts {
			if a.ID == "" || a.Kind == "" {
				return invalidf("cluster %s has incomplete artifact", j.ClusterID)
			}
			switch a.Kind {
			case "issue":
				n, err := strconv.Atoi(a.ID)
				if err != nil || n <= 0 {
					return invalidf("cluster %s has invalid issue id %q", j.ClusterID, a.ID)
				}
				state, ok := captured[n]
				if !ok {
					return invalidf("cluster %s references issue #%d absent from captured issue state", j.ClusterID, n)
				}
				if state.State != a.State || state.RecordDigest != a.RecordDigest {
					return invalidf("cluster %s issue #%d captured state mismatch", j.ClusterID, n)
				}
				if j.Disposition == OpenExact && a.Exact && a.State != "open" {
					return invalidf("cluster %s claims closed issue #%d open", j.ClusterID, n)
				}
				if j.Disposition == Landed && a.Exact && a.State != "closed" {
					return invalidf("cluster %s claims open issue #%d landed", j.ClusterID, n)
				}
				if a.Exact {
					key := "issue:" + a.ID
					if prior := exact[key]; prior != "" && prior != j.ClusterID {
						return invalidf("duplicate exact match %s in %s and %s", key, prior, j.ClusterID)
					}
					exact[key] = j.ClusterID
				}
			case "repo_path":
				if a.Path == "" || filepath.IsAbs(a.Path) || strings.HasPrefix(filepath.Clean(a.Path), "..") {
					return invalidf("cluster %s has unsafe repo path %q", j.ClusterID, a.Path)
				}
				if repoRoot != "" {
					if _, err := os.Stat(filepath.Join(repoRoot, filepath.FromSlash(a.Path))); err != nil {
						return invalidf("cluster %s has broken repo path %s", j.ClusterID, a.Path)
					}
				}
			default:
				return invalidf("cluster %s has unknown artifact kind %q", j.ClusterID, a.Kind)
			}
		}
	}
	if index != nil {
		if len(index.Clusters) != len(l.Joins) {
			return invalidf("missing cluster coverage: ledger=%d index=%d", len(l.Joins), len(index.Clusters))
		}
		for _, c := range index.Clusters {
			if !seenCluster[c.Key] {
				return invalidf("missing cluster coverage for %s", c.Key)
			}
		}
	}
	return nil
}

func evidenceDigest(j Join) string {
	body := struct {
		ClusterID       string     `json:"cluster_id"`
		Query           string     `json:"query"`
		MatchCount      int        `json:"match_count"`
		Matches         []string   `json:"matches"`
		FullMatchDigest string     `json:"full_matches_digest"`
		Artifacts       []Artifact `json:"artifacts,omitempty"`
	}{j.ClusterID, j.Evidence.Query, j.Evidence.MatchCount, j.Evidence.Matches, j.Evidence.FullMatchesDigest, j.Artifacts}
	b, _ := json.Marshal(body)
	return digestBytes(b)
}

func issueRecordDigest(r ForgeRecord) string {
	b, _ := json.Marshal(struct {
		Number int    `json:"number"`
		State  string `json:"state"`
		Title  string `json:"title"`
		Body   string `json:"body"`
		URL    string `json:"url"`
	}{r.Number, r.State, r.Title, r.Body, r.URL})
	return digestBytes(b)
}

func digestBytes(b []byte) string {
	s := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(s[:])
}

func invalidf(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalid, fmt.Sprintf(format, args...))
}
