// Package studyadjacency defines and validates the bounded related-system
// manifest used by runtime studies.
package studyadjacency

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

// Schema is the only manifest schema understood by this package.
const Schema = "fak-study-adjacency/1"

// RequiredCoreRepositories is the issue #9275 minimum adjacency set. Additional
// satellite repositories remain manifest-declared and criterion-governed.
var RequiredCoreRepositories = [...]Repository{
	{Owner: "sgl-project", Repo: "sglang"},
	{Owner: "NVIDIA", Repo: "TensorRT-LLM"},
	{Owner: "ai-dynamo", Repo: "dynamo"},
	{Owner: "ggml-org", Repo: "llama.cpp"},
	{Owner: "flashinfer-ai", Repo: "flashinfer"},
	{Owner: "llm-d", Repo: "llm-d"},
}

// ReceiptStatus describes how much of a source class was captured.
type ReceiptStatus string

const (
	ReceiptComplete    ReceiptStatus = "complete"
	ReceiptPartial     ReceiptStatus = "partial"
	ReceiptMissing     ReceiptStatus = "missing"
	ReceiptUnavailable ReceiptStatus = "unavailable"
)

// Manifest is a finite declaration of repositories adjacent to an anchor
// corpus. DeclaredRepositories is the scope boundary; Members is the proof that
// each declaration was processed.
type Manifest struct {
	Schema               string        `json:"schema"`
	ID                   string        `json:"id"`
	Title                string        `json:"title"`
	Scope                Scope         `json:"scope"`
	Anchor               CorpusWitness `json:"anchor"`
	DeclaredRepositories []Repository  `json:"declared_repositories"`
	Members              []Member      `json:"members"`
}

// Scope records the explicit boundary and the criteria for changing it.
type Scope struct {
	BoundedMeaning    string   `json:"bounded_meaning"`
	InclusionCriteria []string `json:"inclusion_criteria"`
	ExclusionCriteria []string `json:"exclusion_criteria"`
}

// Repository is a canonical GitHub owner/repository identity. Splitting the
// fields prevents a URL, display name, or redirect from masquerading as the
// canonical identity.
type Repository struct {
	Owner string `json:"owner"`
	Repo  string `json:"repo"`
}

// String returns the canonical owner/repository spelling carried by the
// manifest.
func (r Repository) String() string {
	return r.Owner + "/" + r.Repo
}

// Pin fixes evidence to a revision and cutoff while separately dating the
// observation. ObservedAt may be later than Cutoff for endpoint metadata that
// cannot be queried historically.
type Pin struct {
	Revision   string `json:"revision"`
	Cutoff     string `json:"cutoff"`
	ObservedAt string `json:"observed_at"`
}

// CorpusWitness identifies the anchor corpus against which adjacency is
// evaluated.
type CorpusWitness struct {
	Repository        Repository `json:"repository"`
	Pin               Pin        `json:"pin"`
	NormalizedRecords int        `json:"normalized_records"`
	SHA256            string     `json:"sha256"`
	Notes             string     `json:"notes,omitempty"`
}

// Member is one processed repository in the declared adjacency set.
type Member struct {
	Name                string               `json:"name"`
	Repository          Repository           `json:"repository"`
	Pin                 Pin                  `json:"pin"`
	Processed           bool                 `json:"processed"`
	InclusionRationale  string               `json:"inclusion_rationale"`
	DecisionRelation    string               `json:"decision_relation"`
	FreshnessNotes      string               `json:"freshness_notes"`
	PartialNotes        string               `json:"partial_notes"`
	SourceClassReceipts []SourceClassReceipt `json:"source_class_receipts"`
	Candidates          []Candidate          `json:"candidates,omitempty"`
}

// SourceClassReceipt records terminal or explicitly incomplete evidence for
// one source class. A complete status is only valid with a terminal receipt.
type SourceClassReceipt struct {
	Class           string        `json:"class"`
	Status          ReceiptStatus `json:"status"`
	TerminalReceipt string        `json:"terminal_receipt,omitempty"`
	Notes           string        `json:"notes"`
}

// Candidate is one decision-changing mechanism imported from a member. IDs
// are global across all members so records cannot be duplicated under another
// repository.
type Candidate struct {
	ID                       string       `json:"id"`
	Title                    string       `json:"title"`
	Rationale                string       `json:"rationale"`
	RepositoryLinks          []Repository `json:"repository_links"`
	VLLMMechanismLink        string       `json:"vllm_mechanism_link,omitempty"`
	FrontierChangingContrast string       `json:"frontier_changing_contrast,omitempty"`
}

// Load reads, strictly decodes, and validates a manifest file.
func Load(path string) (Manifest, error) {
	f, err := os.Open(path)
	if err != nil {
		return Manifest{}, fmt.Errorf("open study adjacency manifest: %w", err)
	}
	defer f.Close()

	manifest, err := Read(f)
	if err != nil {
		return Manifest{}, fmt.Errorf("decode study adjacency manifest: %w", err)
	}
	return manifest, nil
}

// Read strictly decodes and validates one JSON manifest. Unknown fields and
// trailing JSON values are rejected so schema changes cannot pass silently.
func Read(r io.Reader) (Manifest, error) {
	decoder := json.NewDecoder(r)
	decoder.DisallowUnknownFields()

	var manifest Manifest
	if err := decoder.Decode(&manifest); err != nil {
		return Manifest{}, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return Manifest{}, fmt.Errorf("multiple JSON values")
		}
		return Manifest{}, fmt.Errorf("trailing JSON: %w", err)
	}
	if err := Validate(manifest); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}

// Decode is retained as a descriptive alias for callers that already have a
// JSON decoder-shaped boundary.
func Decode(r io.Reader) (Manifest, error) { return Read(r) }

// WriteCanonicalJSON validates and writes a stable, indented JSON encoding.
// Input collection order does not affect the output.
func WriteCanonicalJSON(w io.Writer, manifest Manifest) error {
	if err := Validate(manifest); err != nil {
		return err
	}
	canonical := canonicalManifest(manifest)
	data, err := json.MarshalIndent(canonical, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal study adjacency manifest: %w", err)
	}
	data = append(data, '\n')
	_, err = w.Write(data)
	return err
}

// CanonicalJSON returns the same bytes as WriteCanonicalJSON.
func CanonicalJSON(manifest Manifest) ([]byte, error) {
	var output bytes.Buffer
	if err := WriteCanonicalJSON(&output, manifest); err != nil {
		return nil, err
	}
	return output.Bytes(), nil
}

func canonicalManifest(manifest Manifest) Manifest {
	manifest.Scope.InclusionCriteria = sortedStrings(manifest.Scope.InclusionCriteria)
	manifest.Scope.ExclusionCriteria = sortedStrings(manifest.Scope.ExclusionCriteria)
	manifest.DeclaredRepositories = append([]Repository(nil), manifest.DeclaredRepositories...)
	sort.Slice(manifest.DeclaredRepositories, func(i, j int) bool {
		return repositoryKey(manifest.DeclaredRepositories[i]) < repositoryKey(manifest.DeclaredRepositories[j])
	})
	manifest.Members = append([]Member(nil), manifest.Members...)
	sort.Slice(manifest.Members, func(i, j int) bool {
		return repositoryKey(manifest.Members[i].Repository) < repositoryKey(manifest.Members[j].Repository)
	})
	for i := range manifest.Members {
		member := &manifest.Members[i]
		member.SourceClassReceipts = append([]SourceClassReceipt(nil), member.SourceClassReceipts...)
		sort.Slice(member.SourceClassReceipts, func(i, j int) bool {
			return strings.ToLower(member.SourceClassReceipts[i].Class) < strings.ToLower(member.SourceClassReceipts[j].Class)
		})
		member.Candidates = append([]Candidate(nil), member.Candidates...)
		sort.Slice(member.Candidates, func(i, j int) bool {
			return strings.ToLower(member.Candidates[i].ID) < strings.ToLower(member.Candidates[j].ID)
		})
		for j := range member.Candidates {
			candidate := &member.Candidates[j]
			candidate.RepositoryLinks = append([]Repository(nil), candidate.RepositoryLinks...)
			sort.Slice(candidate.RepositoryLinks, func(i, j int) bool {
				return repositoryKey(candidate.RepositoryLinks[i]) < repositoryKey(candidate.RepositoryLinks[j])
			})
		}
	}
	return manifest
}
