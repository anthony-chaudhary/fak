package session

import (
	"fmt"
	"html"
	"net/url"
	"sort"
	"strings"
	"unicode"
)

const (
	ProjectionManifestSchema = "fak.session-projection-manifest.v1"
	CapabilityCorpusSchema   = "fak.session-capability-corpus.v1"
	ProjectionIdentity       = "Reduced projection"
)

type ProjectionOmission struct {
	Action      string `json:"action"`
	TypedReason string `json:"typed_reason"`
	Handoff     string `json:"handoff"`
}

type ProjectionManifest struct {
	Schema            string               `json:"schema"`
	Name              string               `json:"name"`
	CorpusSchema      string               `json:"corpus_schema"`
	SupportedActions  []string             `json:"supported_actions"`
	Omissions         []ProjectionOmission `json:"omissions"`
	FullClientHandoff string               `json:"full_client_handoff"`
}

type ProjectionCoverage struct {
	Identity          string               `json:"identity"`
	Name              string               `json:"name"`
	CorpusSchema      string               `json:"corpus_schema"`
	SupportedActions  []string             `json:"supported_actions"`
	Omissions         []ProjectionOmission `json:"omissions"`
	FullClientHandoff string               `json:"full_client_handoff"`
}

func ValidateProjectionManifest(manifest ProjectionManifest, corpusSchema string, canonicalActions []string) (ProjectionCoverage, error) {
	if manifest.Schema != ProjectionManifestSchema {
		return ProjectionCoverage{}, fmt.Errorf("projection manifest schema %q is unsupported", manifest.Schema)
	}
	if corpusSchema != CapabilityCorpusSchema || manifest.CorpusSchema != corpusSchema {
		return ProjectionCoverage{}, fmt.Errorf("projection corpus schema %q does not match supported corpus %q", manifest.CorpusSchema, corpusSchema)
	}
	if strings.TrimSpace(manifest.Name) == "" {
		return ProjectionCoverage{}, fmt.Errorf("projection name is required")
	}
	if !validProjectionHandoff(manifest.FullClientHandoff) {
		return ProjectionCoverage{}, fmt.Errorf("full-client handoff must be an absolute URI")
	}

	canonical := make(map[string]bool, len(canonicalActions))
	for _, action := range canonicalActions {
		if action == "" || canonical[action] {
			return ProjectionCoverage{}, fmt.Errorf("canonical action %q is blank or duplicated", action)
		}
		canonical[action] = true
	}
	classified := make(map[string]string, len(canonicalActions))
	for _, action := range manifest.SupportedActions {
		if !canonical[action] {
			return ProjectionCoverage{}, fmt.Errorf("supported action %q is not canonical", action)
		}
		if classified[action] != "" {
			return ProjectionCoverage{}, fmt.Errorf("action %q is classified more than once", action)
		}
		classified[action] = "supported"
	}
	for _, omission := range manifest.Omissions {
		if !canonical[omission.Action] {
			return ProjectionCoverage{}, fmt.Errorf("omitted action %q is not canonical", omission.Action)
		}
		if classified[omission.Action] != "" {
			return ProjectionCoverage{}, fmt.Errorf("action %q is classified more than once", omission.Action)
		}
		if !validProjectionReason(omission.TypedReason) || !validProjectionHandoff(omission.Handoff) {
			return ProjectionCoverage{}, fmt.Errorf("omitted action %q requires an uppercase typed_reason and absolute handoff URI", omission.Action)
		}
		classified[omission.Action] = "omitted"
	}
	for _, action := range canonicalActions {
		if classified[action] == "" {
			return ProjectionCoverage{}, fmt.Errorf("canonical action %q is unclassified", action)
		}
	}

	coverage := ProjectionCoverage{
		Identity: ProjectionIdentity, Name: manifest.Name, CorpusSchema: corpusSchema,
		SupportedActions:  append([]string(nil), manifest.SupportedActions...),
		Omissions:         append([]ProjectionOmission(nil), manifest.Omissions...),
		FullClientHandoff: manifest.FullClientHandoff,
	}
	sort.Strings(coverage.SupportedActions)
	sort.Slice(coverage.Omissions, func(i, j int) bool { return coverage.Omissions[i].Action < coverage.Omissions[j].Action })
	return coverage, nil
}

func validProjectionReason(reason string) bool {
	if reason == "" {
		return false
	}
	for _, r := range reason {
		if r != '_' && !unicode.IsUpper(r) && !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

func validProjectionHandoff(handoff string) bool {
	u, err := url.ParseRequestURI(handoff)
	return err == nil && u.IsAbs()
}
func (c ProjectionCoverage) RenderText() string {
	var out strings.Builder
	fmt.Fprintf(&out, "%s: %s\nFull client: %s\n", ProjectionIdentity, c.Name, c.FullClientHandoff)
	for _, action := range c.SupportedActions {
		fmt.Fprintf(&out, "%s: available\n", action)
	}
	for _, omission := range c.Omissions {
		fmt.Fprintf(&out, "%s: unavailable (%s); handoff=%s\n", omission.Action, omission.TypedReason, omission.Handoff)
	}
	return out.String()
}

func (c ProjectionCoverage) RenderHTML() string {
	var out strings.Builder
	fmt.Fprintf(&out, "<section data-session-client=\"projection\"><h1>%s: %s</h1><p>Full client: <a href=\"%s\">handoff</a></p><ul>", ProjectionIdentity, html.EscapeString(c.Name), html.EscapeString(c.FullClientHandoff))
	for _, action := range c.SupportedActions {
		fmt.Fprintf(&out, "<li data-action=\"%s\">available</li>", html.EscapeString(action))
	}
	for _, omission := range c.Omissions {
		fmt.Fprintf(&out, "<li data-action=\"%s\">unavailable (%s); <a href=\"%s\">handoff</a></li>", html.EscapeString(omission.Action), html.EscapeString(omission.TypedReason), html.EscapeString(omission.Handoff))
	}
	return out.String() + "</ul></section>"
}

// TelemetryLabels returns bounded projection identity only; it never includes session or user content.
func (c ProjectionCoverage) TelemetryLabels() map[string]string {
	return map[string]string{"session_client_kind": "projection", "session_projection": c.Name, "session_projection_corpus": c.CorpusSchema}
}
