package gateway

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

// IndexLaneRequest is the MCP argument shape for fak_index_lane. It mirrors
// `fak index lane`: callers may pass one path or a batch.
type IndexLaneRequest struct {
	Root  string   `json:"root,omitempty"`
	Path  string   `json:"path,omitempty"`
	Paths []string `json:"paths,omitempty"`
}

// IndexLaneAnswer is one path's resolved lane plus the commit stamp it implies.
type IndexLaneAnswer struct {
	Path  string `json:"path"`
	Lane  string `json:"lane"`
	Stamp string `json:"stamp,omitempty"`
}

// IndexLaneResponse is the MCP result for fak_index_lane.
type IndexLaneResponse struct {
	Root    string            `json:"root"`
	Results []IndexLaneAnswer `json:"results"`
}

var indexLaneInputSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "root": {"type": "string", "description": "optional repo root; omitted means search upward for dos.toml from the server working directory"},
    "path": {"type": "string", "description": "single path to resolve to a lane and suggested (fak <leaf>) stamp"},
    "paths": {"type": "array", "items": {"type": "string"}, "description": "batch of paths to resolve to lanes and suggested stamps"}
  }
}`)

// IndexSearchRequest is shared by fak_index_leaves/docs/claims/verbs. It mirrors
// the query + --limit surface of `fak index`.
type IndexSearchRequest struct {
	Root  string `json:"root,omitempty"`
	Query string `json:"query,omitempty"`
	Limit int    `json:"limit,omitempty"`
}

var indexSearchInputSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "root": {"type": "string", "description": "optional repo root; omitted means search upward for dos.toml from the server working directory"},
    "query": {"type": "string", "description": "search query; required by fak_index_docs and fak_index_claims, optional for fak_index_leaves and fak_index_verbs"},
    "limit": {"type": "integer", "description": "maximum result count; 0 or omitted means no cap"}
  }
}`)

type IndexLeavesResponse struct {
	Root   string          `json:"root"`
	Leaves []devindex.Leaf `json:"leaves"`
}

type IndexDocsResponse struct {
	Root string         `json:"root"`
	Docs []devindex.Doc `json:"docs"`
}

type IndexClaimsResponse struct {
	Root   string           `json:"root"`
	Claims []devindex.Claim `json:"claims"`
}

type IndexVerbsResponse struct {
	Root  string          `json:"root"`
	Verbs []devindex.Verb `json:"verbs"`
}

func loadDevIndex(root string) (*devindex.Catalog, error) {
	if strings.TrimSpace(root) == "" {
		root = devindex.FindRoot(".")
	}
	return devindex.Load(root)
}

func capResults[T any](xs []T, limit int) []T {
	if limit > 0 && len(xs) > limit {
		return xs[:limit]
	}
	return xs
}

func validateIndexLimit(limit int) error {
	if limit < 0 {
		return errors.New("limit must be non-negative")
	}
	return nil
}

func (s *Server) indexLane(req IndexLaneRequest) (IndexLaneResponse, error) {
	cat, err := loadDevIndex(req.Root)
	if err != nil {
		return IndexLaneResponse{}, err
	}
	paths := append([]string(nil), req.Paths...)
	if req.Path != "" {
		paths = append([]string{req.Path}, paths...)
	}
	if len(paths) == 0 {
		return IndexLaneResponse{}, errors.New("fak_index_lane requires path or paths")
	}
	resp := IndexLaneResponse{Root: cat.Root, Results: make([]IndexLaneAnswer, 0, len(paths))}
	for _, p := range paths {
		lane := cat.LaneForPath(p)
		resp.Results = append(resp.Results, IndexLaneAnswer{
			Path:  p,
			Lane:  lane,
			Stamp: cat.SuggestStamp(p),
		})
	}
	return resp, nil
}

func (s *Server) indexLeaves(req IndexSearchRequest) (IndexLeavesResponse, error) {
	if err := validateIndexLimit(req.Limit); err != nil {
		return IndexLeavesResponse{}, err
	}
	cat, err := loadDevIndex(req.Root)
	if err != nil {
		return IndexLeavesResponse{}, err
	}
	return IndexLeavesResponse{Root: cat.Root, Leaves: capResults(cat.SearchLeaves(req.Query), req.Limit)}, nil
}

func (s *Server) indexDocs(req IndexSearchRequest) (IndexDocsResponse, error) {
	if err := validateIndexLimit(req.Limit); err != nil {
		return IndexDocsResponse{}, err
	}
	if strings.TrimSpace(req.Query) == "" {
		return IndexDocsResponse{}, errors.New("fak_index_docs requires query")
	}
	cat, err := loadDevIndex(req.Root)
	if err != nil {
		return IndexDocsResponse{}, err
	}
	return IndexDocsResponse{Root: cat.Root, Docs: capResults(cat.SearchDocs(req.Query), req.Limit)}, nil
}

func (s *Server) indexClaims(req IndexSearchRequest) (IndexClaimsResponse, error) {
	if err := validateIndexLimit(req.Limit); err != nil {
		return IndexClaimsResponse{}, err
	}
	if strings.TrimSpace(req.Query) == "" {
		return IndexClaimsResponse{}, errors.New("fak_index_claims requires query")
	}
	cat, err := loadDevIndex(req.Root)
	if err != nil {
		return IndexClaimsResponse{}, err
	}
	return IndexClaimsResponse{Root: cat.Root, Claims: capResults(cat.SearchClaims(req.Query), req.Limit)}, nil
}

func (s *Server) indexVerbs(req IndexSearchRequest) (IndexVerbsResponse, error) {
	if err := validateIndexLimit(req.Limit); err != nil {
		return IndexVerbsResponse{}, err
	}
	cat, err := loadDevIndex(req.Root)
	if err != nil {
		return IndexVerbsResponse{}, err
	}
	return IndexVerbsResponse{Root: cat.Root, Verbs: capResults(cat.SearchVerbs(req.Query), req.Limit)}, nil
}

// IndexWorkResponse is the MCP result for fak_index_work: the curated selection
// surface (.github/issue-views.json) — the default view slug, the gh page limit each
// query should pair with, and the named views (each with its gh issue-search query).
type IndexWorkResponse struct {
	Root    string               `json:"root"`
	Default string               `json:"default,omitempty"`
	Limit   int                  `json:"limit,omitempty"`
	Views   []devindex.IssueView `json:"views"`
}

func (s *Server) indexWork(req IndexSearchRequest) (IndexWorkResponse, error) {
	if err := validateIndexLimit(req.Limit); err != nil {
		return IndexWorkResponse{}, err
	}
	cat, err := loadDevIndex(req.Root)
	if err != nil {
		return IndexWorkResponse{}, err
	}
	views, err := cat.IssueViews()
	if err != nil {
		return IndexWorkResponse{}, err
	}
	hits := views.Views
	if strings.TrimSpace(req.Query) != "" {
		hits = views.SearchViews(req.Query)
	}
	return IndexWorkResponse{
		Root:    cat.Root,
		Default: views.Default,
		Limit:   views.PageLimit(),
		Views:   capResults(hits, req.Limit),
	}, nil
}

// IndexFreshnessRequest is the MCP argument shape for fak_index_freshness. It mirrors
// `fak index freshness`: no query — just an optional repo root and a result cap.
type IndexFreshnessRequest struct {
	Root  string `json:"root,omitempty"`
	Limit int    `json:"limit,omitempty"`
}

var indexFreshnessInputSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "root": {"type": "string", "description": "optional repo root; omitted means search upward for dos.toml from the server working directory"},
    "limit": {"type": "integer", "description": "maximum finding count; 0 or omitted means no cap"}
  }
}`)

// IndexFreshnessResponse is the MCP result for fak_index_freshness: every way the dev
// self-index disagrees with its live sources — an undeclared leaf, a dead INDEX.md doc
// link, a CLI verb missing from the catalog, an orphaned dated note, or a dead llms.txt
// link. An empty Drift means the catalog agrees with the tree (the index is fresh).
type IndexFreshnessResponse struct {
	Root  string           `json:"root"`
	Drift []devindex.Drift `json:"drift"`
}

func (s *Server) indexFreshness(req IndexFreshnessRequest) (IndexFreshnessResponse, error) {
	if err := validateIndexLimit(req.Limit); err != nil {
		return IndexFreshnessResponse{}, err
	}
	cat, err := loadDevIndex(req.Root)
	if err != nil {
		return IndexFreshnessResponse{}, err
	}
	return IndexFreshnessResponse{Root: cat.Root, Drift: capResults(cat.CheckFreshness(), req.Limit)}, nil
}

// FeatureQueryRequest is the MCP argument shape for fak_feature_query. It mirrors
// `fak feature query`: a non-empty intent, optional dev/live/all plane, optional
// result limit, and optional detail fault for one selected card.
type FeatureQueryRequest struct {
	Root   string `json:"root,omitempty"`
	Query  string `json:"query,omitempty"`
	Plane  string `json:"plane,omitempty"`
	Limit  int    `json:"limit,omitempty"`
	Detail string `json:"detail,omitempty"`
}

var featureQueryInputSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "root": {"type": "string", "description": "optional repo root; omitted means search upward for dos.toml from the server working directory"},
    "query": {"type": "string", "description": "non-empty intent to match against feature cards"},
    "plane": {"type": "string", "enum": ["dev", "live", "all"], "description": "which catalog plane to query; default all"},
    "limit": {"type": "integer", "description": "maximum result count; 0 or omitted means no cap"},
    "detail": {"type": "string", "description": "optional card name/detail_ref to fault schema, doc snippet, or memory explain plan for"}
  },
  "required": ["query"]
}`)

func (s *Server) featureQuery(req FeatureQueryRequest) (selfquery.Response, error) {
	if strings.TrimSpace(req.Query) == "" {
		return selfquery.Response{}, errors.New("fak_feature_query requires query")
	}
	cat, err := selfquery.Load(req.Root, selfquery.Options{
		Tools: selfquery.ToolDescriptorsFromMaps(toolDescriptors()),
	})
	if err != nil {
		return selfquery.Response{}, err
	}
	return cat.Query(selfquery.Request{
		Query:  req.Query,
		Plane:  selfquery.Plane(req.Plane),
		Limit:  req.Limit,
		Detail: req.Detail,
	})
}

// CapabilitiesRequest is the MCP argument shape for fak_capabilities. Unlike
// FeatureQueryRequest, Query is optional (an empty query lists the whole
// toolbelt in stable order) and there is no plane/detail knob — this surface
// is deliberately narrower than fak_feature_query (#1500, the C2 child of the
// #1494 self-knowledge epic): memory drivers, self-index verbs, and kernel
// shared-path verbs only.
type CapabilitiesRequest struct {
	Root  string `json:"root,omitempty"`
	Query string `json:"query,omitempty"`
	Limit int    `json:"limit,omitempty"`
}

var capabilitiesInputSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "root": {"type": "string", "description": "optional repo root; omitted means search upward for dos.toml from the server working directory"},
    "query": {"type": "string", "description": "optional intent to rank the toolbelt by; omitted lists every card in stable order"},
    "limit": {"type": "integer", "description": "maximum result count; 0 or omitted means no cap"}
  }
}`)

func (s *Server) capabilities(req CapabilitiesRequest) (selfquery.CapabilitiesResponse, error) {
	cat, err := selfquery.Load(req.Root, selfquery.Options{
		Tools: selfquery.ToolDescriptorsFromMaps(toolDescriptors()),
	})
	if err != nil {
		return selfquery.CapabilitiesResponse{}, err
	}
	return cat.Capabilities(selfquery.CapabilitiesRequest{Query: req.Query, Limit: req.Limit})
}
