package gateway

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

// Repository index/lane/docs/work/freshness MCP tools moved to fak-dev with the
// repository-development command surface (#6022). Runtime gateway retains only
// model-serving self-query and capabilities tools below.

// FeatureQueryRequest is the MCP argument shape for fak_feature_query. It mirrors
// `fak feature query`: a non-empty intent, optional dev/live/all plane, optional
// result limit, and optional detail fault for one selected card.
type FeatureQueryRequest struct {
	Root   string `json:"root,omitempty"`
	Query  string `json:"query,omitempty"`
	Plane  string `json:"plane,omitempty"`
	Limit  int    `json:"limit,omitempty"`
	All    bool   `json:"all,omitempty"`
	Detail string `json:"detail,omitempty"`
}

var featureQueryInputSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "root": {"type": "string", "description": "optional repo root; omitted means search upward for dos.toml from the server working directory"},
    "query": {"type": "string", "description": "non-empty intent to match against feature cards"},
    "plane": {"type": "string", "enum": ["dev", "live", "all"], "description": "which catalog plane to query; default all"},
    "limit": {"type": "integer", "description": "maximum result count; 0 or omitted uses the bounded feature-query default"},
    "all": {"type": "boolean", "description": "return every match; explicit opt-out from the default feature-query result cap"},
    "detail": {"type": "string", "description": "optional card name/detail_ref to fault schema, doc snippet, or memory explain plan for"}
  },
  "required": ["query"]
}`)

func (s *Server) featureQuery(req FeatureQueryRequest) (selfquery.Response, error) {
	if strings.TrimSpace(req.Query) == "" {
		return selfquery.Response{}, errors.New("fak_feature_query requires query")
	}
	cat, err := selfquery.Load(req.Root, selfquery.Options{
		Tools: selfquery.ToolDescriptorsFromMaps(s.exposedToolDescriptors()),
	})
	if err != nil {
		return selfquery.Response{}, err
	}
	return cat.Query(selfquery.Request{
		Query:  req.Query,
		Plane:  selfquery.Plane(req.Plane),
		Limit:  req.Limit,
		All:    req.All,
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
		Tools: selfquery.ToolDescriptorsFromMaps(s.exposedToolDescriptors()),
	})
	if err != nil {
		return selfquery.CapabilitiesResponse{}, err
	}
	return cat.Capabilities(selfquery.CapabilitiesRequest{Query: req.Query, Limit: req.Limit})
}
