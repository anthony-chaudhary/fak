package harnesslint

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// Rule identifiers for harness authoring linting.
const (
	HL001_SECRET_PLAINTEXT   = "SECRET_PLAINTEXT_LEAK"
	HL002_CRLF_LINE_ENDINGS  = "CRLF_LINE_ENDINGS"
	HL003_SINGLE_PLATFORM    = "SINGLE_PLATFORM_TRAP"
	HL004_UNPINNED_MCP_TOOLS = "UNPINNED_MCP_TOOLS"
	HL005_UNKNOWN_FIELDS     = "UNKNOWN_FIELDS"
)

// Severity levels.
const (
	SeverityError = "ERROR"
	SeverityWarn  = "WARN"
)

// Diagnostic identifies a single lint violation or warning.
type Diagnostic struct {
	Rule     string `json:"rule"`
	Severity string `json:"severity"` // "ERROR" or "WARN"
	Message  string `json:"message"`
	Field    string `json:"field"`
}

// LintReport aggregates all diagnostics and counts for a harness lock.
type LintReport struct {
	Valid        bool         `json:"valid"`
	Diagnostics  []Diagnostic `json:"diagnostics"`
	ErrorCount   int          `json:"error_count"`
	WarningCount int          `json:"warning_count"`
}

type lockDocument struct {
	Schema              string                     `json:"schema"`
	ID                  string                     `json:"id,omitempty"`
	Environment         *lockEnvironment           `json:"environment,omitempty"`
	Budget              *lockBudget                `json:"budget,omitempty"`
	Components          []lockComponent            `json:"components,omitempty"`
	Assets              []lockAsset                `json:"assets,omitempty"`
	AssetTrace          json.RawMessage            `json:"asset_trace,omitempty"`
	Decisions           json.RawMessage            `json:"decisions,omitempty"`
	Platforms           []string                   `json:"platforms,omitempty"`
	SinglePlatform      bool                       `json:"single_platform,omitempty"`
	SinglePlatformOptIn bool                       `json:"single_platform_opt_in,omitempty"`
	AllowSinglePlatform bool                       `json:"allow_single_platform,omitempty"`
	Compatibility       *lockCompatibility         `json:"compatibility,omitempty"`
	Tools               []lockTool                 `json:"tools,omitempty"`
	MCPTools            []lockTool                 `json:"mcp_tools,omitempty"`
	Secrets             []lockSecret               `json:"secrets,omitempty"`
	Layers              []lockLayer                `json:"layers,omitempty"`
	TextLayers          map[string]string          `json:"text_layers,omitempty"`
	Metadata            map[string]json.RawMessage `json:"metadata,omitempty"`
	Roots               []string                   `json:"roots,omitempty"`
	Explain             json.RawMessage            `json:"explain,omitempty"`
	Options             map[string]json.RawMessage `json:"options,omitempty"`
	Generator           string                     `json:"generator,omitempty"`
	ContractVersion     string                     `json:"contract_version,omitempty"`
	FAKModule           string                     `json:"fak_module,omitempty"`
	FAKVersion          string                     `json:"fak_version,omitempty"`
	Upgrade             string                     `json:"upgrade,omitempty"`
	Version             string                     `json:"version,omitempty"`
}

type lockEnvironment struct {
	OS        string   `json:"os,omitempty"`
	Arch      string   `json:"arch,omitempty"`
	Contract  string   `json:"contract,omitempty"`
	Platforms []string `json:"platforms,omitempty"`
}

type lockBudget struct {
	ContextTokens int `json:"context_tokens,omitempty"`
	MemoryMiB     int `json:"memory_mib,omitempty"`
	Workers       int `json:"workers,omitempty"`
}

type lockRequirement struct {
	Capability string `json:"capability,omitempty"`
	Range      string `json:"range,omitempty"`
	Optional   bool   `json:"optional,omitempty"`
}

type lockCompatibility struct {
	OS        []string `json:"os,omitempty"`
	Arch      []string `json:"arch,omitempty"`
	Contract  string   `json:"contract,omitempty"`
	Platforms []string `json:"platforms,omitempty"`
}

type lockComponent struct {
	ID            string             `json:"id"`
	Version       string             `json:"version,omitempty"`
	Digest        string             `json:"digest,omitempty"`
	Source        string             `json:"source,omitempty"`
	Reason        string             `json:"reason,omitempty"`
	Provider      string             `json:"provider,omitempty"`
	Provides      []string           `json:"provides,omitempty"`
	Requires      []lockRequirement  `json:"requires,omitempty"`
	Conflicts     []string           `json:"conflicts,omitempty"`
	Compatibility *lockCompatibility `json:"compatibility,omitempty"`
	Cost          *lockBudget        `json:"cost,omitempty"`
	Adapters      []string           `json:"adapters,omitempty"`
}

type lockAsset struct {
	Kind              string   `json:"kind"`
	ID                string   `json:"id,omitempty"`
	Value             string   `json:"value,omitempty"`
	Content           string   `json:"content,omitempty"`
	Text              string   `json:"text,omitempty"`
	Ref               string   `json:"ref,omitempty"`
	Boundary          string   `json:"boundary,omitempty"`
	Grants            []string `json:"grants,omitempty"`
	Denies            []string `json:"denies,omitempty"`
	Source            string   `json:"source,omitempty"`
	Locked            bool     `json:"locked,omitempty"`
	Mandatory         bool     `json:"mandatory,omitempty"`
	Operation         string   `json:"operation,omitempty"`
	Type              string   `json:"type,omitempty"`
	Protocol          string   `json:"protocol,omitempty"`
	SchemaSHA256      string   `json:"schema_sha256,omitempty"`
	SchemaFingerprint string   `json:"schema_fingerprint,omitempty"`
	SchemaDigest      string   `json:"schema_digest,omitempty"`
	SchemaHash        string   `json:"schema_hash,omitempty"`
	Digest            string   `json:"digest,omitempty"`
	Fingerprint       string   `json:"fingerprint,omitempty"`
}

type lockTool struct {
	ID                string `json:"id"`
	Name              string `json:"name,omitempty"`
	Type              string `json:"type,omitempty"`
	Kind              string `json:"kind,omitempty"`
	Protocol          string `json:"protocol,omitempty"`
	Source            string `json:"source,omitempty"`
	Value             string `json:"value,omitempty"`
	SchemaSHA256      string `json:"schema_sha256,omitempty"`
	SchemaFingerprint string `json:"schema_fingerprint,omitempty"`
	SchemaDigest      string `json:"schema_digest,omitempty"`
	SchemaHash        string `json:"schema_hash,omitempty"`
	Digest            string `json:"digest,omitempty"`
	Fingerprint       string `json:"fingerprint,omitempty"`
}

type lockSecret struct {
	ID     string `json:"id,omitempty"`
	Kind   string `json:"kind,omitempty"`
	Value  string `json:"value,omitempty"`
	Ref    string `json:"ref,omitempty"`
	Source string `json:"source,omitempty"`
}

type lockLayer struct {
	ID      string      `json:"id,omitempty"`
	Scope   string      `json:"scope,omitempty"`
	Content string      `json:"content,omitempty"`
	Text    string      `json:"text,omitempty"`
	Assets  []lockAsset `json:"assets,omitempty"`
}

// LintLock parses lock data and evaluates rules HL001-HL005.
func LintLock(data []byte) *LintReport {
	report := &LintReport{
		Valid:       true,
		Diagnostics: make([]Diagnostic, 0),
	}

	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		report.Diagnostics = append(report.Diagnostics, Diagnostic{
			Rule:     HL005_UNKNOWN_FIELDS,
			Severity: SeverityError,
			Message:  "malformed JSON: empty input",
			Field:    "$",
		})
		report.ErrorCount = 1
		report.Valid = false
		return report
	}

	// 1. Check HL005: Unknown fields, unanchored schema, or malformed JSON syntax.
	var strictDoc lockDocument
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	decodeErr := dec.Decode(&strictDoc)
	if decodeErr != nil {
		field := "$"
		msg := decodeErr.Error()
		if strings.Contains(msg, "unknown field") {
			parts := strings.Split(msg, `"`)
			if len(parts) >= 2 {
				field = parts[1]
			}
		}
		report.Diagnostics = append(report.Diagnostics, Diagnostic{
			Rule:     HL005_UNKNOWN_FIELDS,
			Severity: SeverityError,
			Message:  msg,
			Field:    field,
		})
	} else {
		// Strict decode succeeded; ensure schema is anchored and non-empty.
		if strings.TrimSpace(strictDoc.Schema) == "" {
			report.Diagnostics = append(report.Diagnostics, Diagnostic{
				Rule:     HL005_UNKNOWN_FIELDS,
				Severity: SeverityError,
				Message:  "unanchored lock: schema field is missing or empty",
				Field:    "schema",
			})
		}

		// Ensure no trailing tokens.
		var trailing json.RawMessage
		if errTrailer := dec.Decode(&trailing); errTrailer == nil {
			report.Diagnostics = append(report.Diagnostics, Diagnostic{
				Rule:     HL005_UNKNOWN_FIELDS,
				Severity: SeverityError,
				Message:  "malformed JSON: trailing data after top-level object",
				Field:    "$",
			})
		}
	}

	// 2. Decode permissively for rules HL001-HL004 if strict decode had unknown fields.
	var doc lockDocument
	if decodeErr == nil {
		doc = strictDoc
	} else {
		_ = json.Unmarshal(data, &doc)
	}

	// HL001: Plaintext Secrets
	for i, asset := range doc.Assets {
		if strings.EqualFold(asset.Kind, "secret") && asset.Value != "" {
			report.Diagnostics = append(report.Diagnostics, Diagnostic{
				Rule:     HL001_SECRET_PLAINTEXT,
				Severity: SeverityError,
				Message:  fmt.Sprintf("secret asset %q contains plaintext value; secrets must be referenced via ref or environment", asset.ID),
				Field:    fmt.Sprintf("assets[%d].value", i),
			})
		}
	}
	for i, secret := range doc.Secrets {
		if (secret.Kind == "" || strings.EqualFold(secret.Kind, "secret")) && secret.Value != "" {
			report.Diagnostics = append(report.Diagnostics, Diagnostic{
				Rule:     HL001_SECRET_PLAINTEXT,
				Severity: SeverityError,
				Message:  fmt.Sprintf("secret %q contains plaintext value; secrets must be referenced via ref or environment", secret.ID),
				Field:    fmt.Sprintf("secrets[%d].value", i),
			})
		}
	}
	for l, layer := range doc.Layers {
		for a, asset := range layer.Assets {
			if strings.EqualFold(asset.Kind, "secret") && asset.Value != "" {
				report.Diagnostics = append(report.Diagnostics, Diagnostic{
					Rule:     HL001_SECRET_PLAINTEXT,
					Severity: SeverityError,
					Message:  fmt.Sprintf("secret asset %q in layer %q contains plaintext value", asset.ID, layer.ID),
					Field:    fmt.Sprintf("layers[%d].assets[%d].value", l, a),
				})
			}
		}
	}

	// HL002: CRLF Line Endings in hashed text layers
	crlfFound := false
	for i, asset := range doc.Assets {
		if strings.Contains(asset.Value, "\r\n") {
			crlfFound = true
			report.Diagnostics = append(report.Diagnostics, Diagnostic{
				Rule:     HL002_CRLF_LINE_ENDINGS,
				Severity: SeverityError,
				Message:  fmt.Sprintf("CRLF (\\r\\n) line endings detected in asset %q value; must use LF (\\n)", asset.ID),
				Field:    fmt.Sprintf("assets[%d].value", i),
			})
		}
		if strings.Contains(asset.Content, "\r\n") {
			crlfFound = true
			report.Diagnostics = append(report.Diagnostics, Diagnostic{
				Rule:     HL002_CRLF_LINE_ENDINGS,
				Severity: SeverityError,
				Message:  fmt.Sprintf("CRLF (\\r\\n) line endings detected in asset %q content; must use LF (\\n)", asset.ID),
				Field:    fmt.Sprintf("assets[%d].content", i),
			})
		}
		if strings.Contains(asset.Text, "\r\n") {
			crlfFound = true
			report.Diagnostics = append(report.Diagnostics, Diagnostic{
				Rule:     HL002_CRLF_LINE_ENDINGS,
				Severity: SeverityError,
				Message:  fmt.Sprintf("CRLF (\\r\\n) line endings detected in asset %q text; must use LF (\\n)", asset.ID),
				Field:    fmt.Sprintf("assets[%d].text", i),
			})
		}
	}
	for l, layer := range doc.Layers {
		if strings.Contains(layer.Content, "\r\n") {
			crlfFound = true
			report.Diagnostics = append(report.Diagnostics, Diagnostic{
				Rule:     HL002_CRLF_LINE_ENDINGS,
				Severity: SeverityError,
				Message:  fmt.Sprintf("CRLF (\\r\\n) line endings detected in layer %q content; must use LF (\\n)", layer.ID),
				Field:    fmt.Sprintf("layers[%d].content", l),
			})
		}
		if strings.Contains(layer.Text, "\r\n") {
			crlfFound = true
			report.Diagnostics = append(report.Diagnostics, Diagnostic{
				Rule:     HL002_CRLF_LINE_ENDINGS,
				Severity: SeverityError,
				Message:  fmt.Sprintf("CRLF (\\r\\n) line endings detected in layer %q text; must use LF (\\n)", layer.ID),
				Field:    fmt.Sprintf("layers[%d].text", l),
			})
		}
		for a, asset := range layer.Assets {
			if strings.Contains(asset.Value, "\r\n") {
				crlfFound = true
				report.Diagnostics = append(report.Diagnostics, Diagnostic{
					Rule:     HL002_CRLF_LINE_ENDINGS,
					Severity: SeverityError,
					Message:  fmt.Sprintf("CRLF (\\r\\n) line endings detected in layer %q asset %q value", layer.ID, asset.ID),
					Field:    fmt.Sprintf("layers[%d].assets[%d].value", l, a),
				})
			}
		}
	}
	for k, text := range doc.TextLayers {
		if strings.Contains(text, "\r\n") {
			crlfFound = true
			report.Diagnostics = append(report.Diagnostics, Diagnostic{
				Rule:     HL002_CRLF_LINE_ENDINGS,
				Severity: SeverityError,
				Message:  fmt.Sprintf("CRLF (\\r\\n) line endings detected in text_layers[%q]", k),
				Field:    fmt.Sprintf("text_layers[%s]", k),
			})
		}
	}
	if !crlfFound && bytes.Contains(data, []byte("\r\n")) {
		report.Diagnostics = append(report.Diagnostics, Diagnostic{
			Rule:     HL002_CRLF_LINE_ENDINGS,
			Severity: SeverityError,
			Message:  "CRLF (\\r\\n) line endings detected in hashed text layer bytes; must use LF (\\n)",
			Field:    "layers",
		})
	}

	// HL003: Single Platform Trap
	optIn := hasSinglePlatformOptIn(&doc)
	if !optIn {
		if len(doc.Platforms) == 1 {
			report.Diagnostics = append(report.Diagnostics, Diagnostic{
				Rule:     HL003_SINGLE_PLATFORM,
				Severity: SeverityWarn,
				Message:  fmt.Sprintf("platforms contains only a single OS/Arch %q without explicit single-platform opt-in", doc.Platforms[0]),
				Field:    "platforms",
			})
		} else if len(doc.Platforms) == 0 {
			if doc.Compatibility != nil && len(doc.Compatibility.Platforms) == 1 {
				report.Diagnostics = append(report.Diagnostics, Diagnostic{
					Rule:     HL003_SINGLE_PLATFORM,
					Severity: SeverityWarn,
					Message:  fmt.Sprintf("compatibility.platforms contains only a single OS/Arch %q without explicit single-platform opt-in", doc.Compatibility.Platforms[0]),
					Field:    "compatibility.platforms",
				})
			} else if doc.Environment != nil && len(doc.Environment.Platforms) == 1 {
				report.Diagnostics = append(report.Diagnostics, Diagnostic{
					Rule:     HL003_SINGLE_PLATFORM,
					Severity: SeverityWarn,
					Message:  fmt.Sprintf("environment.platforms contains only a single OS/Arch %q without explicit single-platform opt-in", doc.Environment.Platforms[0]),
					Field:    "environment.platforms",
				})
			}
		}
	}

	// HL004: Unpinned MCP Tools
	for i, asset := range doc.Assets {
		if isMCPToolAsset(&asset) {
			if !hasValidAssetFingerprint(&asset) {
				toolID := asset.ID
				if toolID == "" {
					toolID = fmt.Sprintf("asset-%d", i)
				}
				report.Diagnostics = append(report.Diagnostics, Diagnostic{
					Rule:     HL004_UNPINNED_MCP_TOOLS,
					Severity: SeverityWarn,
					Message:  fmt.Sprintf("MCP tool %q lacks tool schema SHA-256 fingerprint", toolID),
					Field:    fmt.Sprintf("assets[%d].schema_sha256", i),
				})
			}
		}
	}
	for i, tool := range doc.Tools {
		if isMCPTool(&tool) {
			if !hasValidToolFingerprint(&tool) {
				toolID := tool.ID
				if toolID == "" {
					toolID = fmt.Sprintf("tool-%d", i)
				}
				report.Diagnostics = append(report.Diagnostics, Diagnostic{
					Rule:     HL004_UNPINNED_MCP_TOOLS,
					Severity: SeverityWarn,
					Message:  fmt.Sprintf("MCP tool %q lacks tool schema SHA-256 fingerprint", toolID),
					Field:    fmt.Sprintf("tools[%d].schema_sha256", i),
				})
			}
		}
	}
	for i, tool := range doc.MCPTools {
		if !hasValidToolFingerprint(&tool) {
			toolID := tool.ID
			if toolID == "" {
				toolID = fmt.Sprintf("mcp-tool-%d", i)
			}
			report.Diagnostics = append(report.Diagnostics, Diagnostic{
				Rule:     HL004_UNPINNED_MCP_TOOLS,
				Severity: SeverityWarn,
				Message:  fmt.Sprintf("MCP tool %q lacks tool schema SHA-256 fingerprint", toolID),
				Field:    fmt.Sprintf("mcp_tools[%d].schema_sha256", i),
			})
		}
	}

	// Compute totals and verdict.
	report.ErrorCount = 0
	report.WarningCount = 0
	for _, d := range report.Diagnostics {
		switch d.Severity {
		case SeverityError:
			report.ErrorCount++
		case SeverityWarn:
			report.WarningCount++
		}
	}
	report.Valid = (report.ErrorCount == 0)

	return report
}

func hasSinglePlatformOptIn(doc *lockDocument) bool {
	if doc.SinglePlatform || doc.SinglePlatformOptIn || doc.AllowSinglePlatform {
		return true
	}
	for _, rawMap := range []map[string]json.RawMessage{doc.Options, doc.Metadata} {
		if rawMap != nil {
			for _, key := range []string{"single_platform", "single_platform_opt_in", "allow_single_platform"} {
				if val, ok := rawMap[key]; ok {
					var b bool
					if json.Unmarshal(val, &b) == nil && b {
						return true
					}
				}
			}
		}
	}
	return false
}

func isMCPToolAsset(asset *lockAsset) bool {
	k := strings.ToLower(asset.Kind)
	if k == "mcp-tool" || k == "mcp_tool" || k == "mcp" {
		return true
	}
	if k == "tool" {
		s := strings.ToLower(asset.Source)
		t := strings.ToLower(asset.Type)
		p := strings.ToLower(asset.Protocol)
		id := strings.ToLower(asset.ID)
		if s == "mcp" || strings.HasPrefix(s, "mcp") ||
			t == "mcp" || p == "mcp" || p == "mcp-tools" ||
			strings.HasPrefix(id, "mcp:") || strings.HasPrefix(id, "mcp_") || strings.HasPrefix(id, "mcp-") {
			return true
		}
	}
	return false
}

func hasValidAssetFingerprint(asset *lockAsset) bool {
	for _, val := range []string{
		asset.SchemaSHA256,
		asset.SchemaFingerprint,
		asset.SchemaDigest,
		asset.SchemaHash,
		asset.Digest,
		asset.Fingerprint,
	} {
		if isSHA256Fingerprint(val) {
			return true
		}
	}
	return false
}

func isMCPTool(tool *lockTool) bool {
	k := strings.ToLower(tool.Kind)
	if k == "mcp-tool" || k == "mcp_tool" || k == "mcp" {
		return true
	}
	s := strings.ToLower(tool.Source)
	t := strings.ToLower(tool.Type)
	p := strings.ToLower(tool.Protocol)
	id := strings.ToLower(tool.ID)
	return s == "mcp" || strings.HasPrefix(s, "mcp") ||
		t == "mcp" || p == "mcp" || p == "mcp-tools" ||
		strings.HasPrefix(id, "mcp:") || strings.HasPrefix(id, "mcp_") || strings.HasPrefix(id, "mcp-")
}

func hasValidToolFingerprint(tool *lockTool) bool {
	for _, val := range []string{
		tool.SchemaSHA256,
		tool.SchemaFingerprint,
		tool.SchemaDigest,
		tool.SchemaHash,
		tool.Digest,
		tool.Fingerprint,
	} {
		if isSHA256Fingerprint(val) {
			return true
		}
	}
	return false
}

func isSHA256Fingerprint(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	if strings.HasPrefix(s, "sha256:") {
		hexPart := strings.TrimPrefix(s, "sha256:")
		return len(hexPart) > 0 && isHexString(hexPart)
	}
	if len(s) == 64 && isHexString(s) {
		return true
	}
	return false
}

func isHexString(s string) bool {
	for _, r := range s {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')) {
			return false
		}
	}
	return true
}
