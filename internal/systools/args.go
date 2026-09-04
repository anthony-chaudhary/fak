package systools

import (
	"bytes"
	"encoding/json"
	"net/url"
	"strings"
	"time"
)

// Tool names advertised to planners and adjudicated by the systools rung.
const (
	ToolGetTime   = "get_time"
	ToolFetchWeb  = "fetch_web"
	ToolWebSearch = "web_search"
)

// GetTimeArgs defines arguments for get_time.
type GetTimeArgs struct {
	Timezone string `json:"timezone,omitempty"`
}

// Validate validates GetTimeArgs.
func (a GetTimeArgs) Validate() *Refusal {
	if a.Timezone != "" {
		if _, err := time.LoadLocation(a.Timezone); err != nil {
			return refuse(CodeMalformed, "get_time: invalid timezone: "+a.Timezone)
		}
	}
	return nil
}

// FetchWebArgs defines arguments for fetch_web.
type FetchWebArgs struct {
	URL      string `json:"url"`
	MaxBytes int    `json:"max_bytes,omitempty"`
}

// Validate validates FetchWebArgs.
func (a FetchWebArgs) Validate() *Refusal {
	if strings.TrimSpace(a.URL) == "" {
		return refuse(CodeMalformed, "fetch_web: missing required field: url")
	}
	u, err := url.Parse(a.URL)
	if err != nil || u.Host == "" {
		return refuse(CodeMalformed, "fetch_web: invalid url: "+a.URL)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return refuse(CodeMalformed, "fetch_web: scheme must be http or https")
	}
	if a.MaxBytes < 0 {
		return refuse(CodeMalformed, "fetch_web: max_bytes must be >= 0")
	}
	return nil
}

// WebSearchArgs defines arguments for web_search.
type WebSearchArgs struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results,omitempty"`
}

// Validate validates WebSearchArgs.
func (a WebSearchArgs) Validate() *Refusal {
	if strings.TrimSpace(a.Query) == "" {
		return refuse(CodeMalformed, "web_search: missing required field: query")
	}
	if a.MaxResults < 0 {
		return refuse(CodeMalformed, "web_search: max_results must be >= 0")
	}
	return nil
}

func decodeArgs(body []byte, v any) *Refusal {
	if len(bytes.TrimSpace(body)) == 0 {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return refuse(CodeMalformed, "cannot decode arguments: "+err.Error())
	}
	return nil
}

// ToolDef represents a tool definition advertised to planners.
type ToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	ReadOnly    bool            `json:"read_only"`
}

// Catalog returns tool definitions for get_time, fetch_web, and web_search.
func Catalog() []ToolDef {
	return []ToolDef{
		{
			Name:        ToolGetTime,
			Description: "Get current system time, timezone, and epoch timestamps.",
			Parameters: json.RawMessage(`{"type":"object","properties":{` +
				`"timezone":{"type":"string","description":"Optional IANA timezone, e.g. UTC, America/New_York"}},` +
				`"additionalProperties":false}`),
			ReadOnly: true,
		},
		{
			Name:        ToolFetchWeb,
			Description: "Fetch web content from an HTTP/HTTPS URL with SSRF protection and byte capping.",
			Parameters: json.RawMessage(`{"type":"object","properties":{` +
				`"url":{"type":"string","description":"HTTP or HTTPS URL to fetch"},` +
				`"max_bytes":{"type":"integer","description":"Optional maximum response bytes to read"}},` +
				`"required":["url"],"additionalProperties":false}`),
			ReadOnly: true,
		},
		{
			Name:        ToolWebSearch,
			Description: "Search the web or documentation for relevant information.",
			Parameters: json.RawMessage(`{"type":"object","properties":{` +
				`"query":{"type":"string","description":"Search query string"},` +
				`"max_results":{"type":"integer","description":"Maximum number of results to return"}},` +
				`"required":["query"],"additionalProperties":false}`),
			ReadOnly: true,
		},
	}
}
