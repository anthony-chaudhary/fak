package codetools

import (
	"bytes"
	"encoding/json"
	"strings"
)

// args.go — the TYPED operation, argument, and result schemas.
//
// Decoding is STRICT (DisallowUnknownFields). That is a deliberate cost: a model that
// emits `{"filePath": ...}` for a tool whose parameter is `file_path` gets a MALFORMED
// refusal instead of a call that runs with an empty path. The permissive alternative —
// ignore what you do not recognize — is how a typo silently becomes a different
// operation, and a call that quietly reads the wrong file is a wrong answer. UnknownFieldError also names
// the offending key, so the refusal is model-fixable in one turn.
//
// Every tool's arguments decode into a concrete Go struct with a Validate() that owns its
// OWN required-field rules. The rung calls Validate before admitting; the engine calls it
// again on entry. That is not redundancy for its own sake: an engine is reachable through
// abi.RegisterEngine by any kernel, including one whose chain does not carry this
// package's rung, and an engine that trusted its caller to have validated would be a hole
// exactly where the toolset is most exposed.

// Tool names. These are the names a planner emits and the rung matches on; they mirror
// the harness-conventional spellings so a model already fluent in Read/Grep/Glob needs
// no retraining to drive the kernel-mediated versions.
const (
	ToolRead       = "Read"
	ToolGrep       = "Grep"
	ToolGlob       = "Glob"
	ToolWrite      = "Write"
	ToolEdit       = "Edit"
	ToolBash       = "Bash"
)

// ReadArgs names one file and an optional line window. Offset is 1-based (line 1 is the
// first line) to match how every editor and every stack trace numbers lines; Limit is a
// line count. A zero window means "the whole file, up to the byte bound".
type ReadArgs struct {
	FilePath string `json:"file_path"`
	Offset   int    `json:"offset,omitempty"`
	Limit    int    `json:"limit,omitempty"`
}

// Validate enforces ReadArgs' own rules: a path is required, and a negative window is a
// shape error rather than something to clamp silently.
func (a ReadArgs) Validate() *Refusal {
	if strings.TrimSpace(a.FilePath) == "" {
		return refuse(CodeMalformed, "Read: missing required field: file_path")
	}
	if a.Offset < 0 || a.Limit < 0 {
		return refuse(CodeMalformed, "Read: offset and limit must be >= 0")
	}
	return nil
}

// WriteArgs replaces one file with Content. Mode makes creation semantics explicit:
// "create" refuses an existing file, "overwrite" refuses a missing file, and "upsert"
// permits either. No omitted/default mode can silently change a caller's intent.
type WriteArgs struct {
	FilePath        string `json:"file_path"`
	Content         string `json:"content"`
	Mode            string `json:"mode"`
	ExpectedVersion string `json:"expected_version,omitempty"`
}

func (a WriteArgs) Validate() *Refusal {
	if strings.TrimSpace(a.FilePath) == "" {
		return refuse(CodeMalformed, "Write: missing required field: file_path")
	}
	switch a.Mode {
	case "create":
		if a.ExpectedVersion != "" {
			return refuse(CodeMalformed, "Write create must not present expected_version")
		}
		return nil
	case "overwrite":
		if a.ExpectedVersion == "" {
			return refuse(CodeMalformed, "Write overwrite: missing required field: expected_version")
		}
		return nil
	case "upsert":
		return nil
	default:
		return refuse(CodeMalformed, "Write: mode must be create, overwrite, or upsert")
	}
}

// EditArgs performs an exact textual replacement. The default requires exactly one
// match; ReplaceAll requires at least one and replaces every match.
type EditArgs struct {
	FilePath        string `json:"file_path"`
	OldString       string `json:"old_string"`
	NewString       string `json:"new_string"`
	ReplaceAll      bool   `json:"replace_all,omitempty"`
	ExpectedVersion string `json:"expected_version"`
}

func (a EditArgs) Validate() *Refusal {
	if strings.TrimSpace(a.FilePath) == "" {
		return refuse(CodeMalformed, "Edit: missing required field: file_path")
	}
	if a.OldString == "" {
		return refuse(CodeMalformed, "Edit: old_string must not be empty")
	}
	if a.ExpectedVersion == "" {
		return refuse(CodeMalformed, "Edit: missing required field: expected_version")
	}
	return nil
}

// BashArgs runs one command through the platform shell inside a confined workspace cwd.
// TimeoutMS is optional but always capped by the toolset's configured maximum.
type BashArgs struct {
	Command   string `json:"command"`
	Cwd       string `json:"cwd,omitempty"`
	TimeoutMS int    `json:"timeout_ms,omitempty"`
}

func (a BashArgs) Validate() *Refusal {
	if strings.TrimSpace(a.Command) == "" {
		return refuse(CodeMalformed, "Bash: missing required field: command")
	}
	if a.TimeoutMS < 0 {
		return refuse(CodeMalformed, "Bash: timeout_ms must be >= 0")
	}
	return nil
}

// GrepArgs is a regexp content search over a confined subtree. Path defaults to the
// workspace root; Glob filters candidate files by base name before any file is opened.
type GrepArgs struct {
	Pattern    string `json:"pattern"`
	Path       string `json:"path,omitempty"`
	Glob       string `json:"glob,omitempty"`
	MaxMatches int    `json:"max_matches,omitempty"`
}

// Validate enforces GrepArgs' own rules. The pattern itself is compiled by the engine —
// a malformed regexp is a MALFORMED refusal there, not here, because compiling twice to
// validate is wasted work on the hot path.
func (a GrepArgs) Validate() *Refusal {
	if a.Pattern == "" {
		return refuse(CodeMalformed, "Grep: missing required field: pattern")
	}
	if a.MaxMatches < 0 {
		return refuse(CodeMalformed, "Grep: max_matches must be >= 0")
	}
	return nil
}

// GlobArgs is a path-shape search. Pattern is matched against each candidate's
// slash-separated path RELATIVE to the search root, so a pattern is portable across
// checkouts and cannot be written to address a host-absolute location.
type GlobArgs struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path,omitempty"`
}

// Validate enforces GlobArgs' own rules.
func (a GlobArgs) Validate() *Refusal {
	if strings.TrimSpace(a.Pattern) == "" {
		return refuse(CodeMalformed, "Glob: missing required field: pattern")
	}
	return nil
}

// decodeArgs strictly decodes a call's raw argument bytes into v. Empty args decode as
// the zero value so a required-field Validate — not a decode fault — produces the
// refusal, which keeps the message actionable ("missing file_path", not "unexpected end
// of JSON input").
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

// ToolDef is one entry in the catalog a caller advertises to a planner: the tool's name,
// what it does, its JSON-Schema parameters, and whether it is read-only. ReadOnly is part
// of the DEFINITION rather than a caller's annotation because it is the same bit CallMeta
// stamps onto the vDSO scope — one source of truth, so a catalog and a cache key cannot
// disagree about whether a tool mutates.
type ToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	ReadOnly    bool            `json:"read_only"`
}

// Catalog returns the implemented coding-tool definitions in a stable order. It is a pure function of
// the package: a caller binds it into its own planner-facing catalog shape (the owned
// loop's ToolDef, an MCP tools/list, an Anthropic tools[]) without this leaf having to
// know which wire it is being advertised on.
func Catalog() []ToolDef {
	return []ToolDef{
		{
			Name:        ToolRead,
			Description: "Read a file from the workspace. Returns content and an opaque version for guarded mutation, optionally windowed by line offset and limit.",
			Parameters: json.RawMessage(`{"type":"object","properties":{` +
				`"file_path":{"type":"string","description":"workspace-relative or absolute path inside the workspace"},` +
				`"offset":{"type":"integer","description":"1-based first line to return"},` +
				`"limit":{"type":"integer","description":"maximum number of lines to return"}},` +
				`"required":["file_path"],"additionalProperties":false}`),
			ReadOnly: true,
		},
		{
			Name:        ToolGrep,
			Description: "Search workspace file contents with a regular expression. Never invokes a shell.",
			Parameters: json.RawMessage(`{"type":"object","properties":{` +
				`"pattern":{"type":"string","description":"RE2 regular expression"},` +
				`"path":{"type":"string","description":"subtree to search; defaults to the workspace root"},` +
				`"glob":{"type":"string","description":"file base-name filter, e.g. *.go"},` +
				`"max_matches":{"type":"integer"}},` +
				`"required":["pattern"],"additionalProperties":false}`),
			ReadOnly: true,
		},
		{
			Name:        ToolWrite,
			Description: "Atomically create or replace a workspace file. overwrite requires the version returned by Read; upsert creates when absent and requires that version when replacing.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string","enum":["create","overwrite","upsert"]},"expected_version":{"type":"string","description":"opaque version returned by Read; required for overwrite and for upsert when the target exists"}},"required":["file_path","content","mode"],"additionalProperties":false}`),
			ReadOnly:    false,
		},
		{
			Name:        ToolEdit,
			Description: "Atomically replace exact text in the file version returned by Read, or refuse when that observation is stale.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean"},"expected_version":{"type":"string","description":"opaque version returned by Read"}},"required":["file_path","old_string","new_string","expected_version"],"additionalProperties":false}`),
			ReadOnly:    false,
		},
		{
			Name:        ToolBash,
			Description: "Run a bounded command through the platform shell in a confined workspace directory.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"},"cwd":{"type":"string"},"timeout_ms":{"type":"integer","minimum":0}},"required":["command"],"additionalProperties":false}`),
			ReadOnly:    false,
		},
		{
			Name:        ToolGlob,
			Description: "List workspace files whose path matches a glob pattern. Never invokes a shell.",
			Parameters: json.RawMessage(`{"type":"object","properties":{` +
				`"pattern":{"type":"string","description":"glob matched against the path relative to the search root"},` +
				`"path":{"type":"string","description":"subtree to walk; defaults to the workspace root"}},` +
				`"required":["pattern"],"additionalProperties":false}`),
			ReadOnly: true,
		},
		{
			Name:        ToolApplyPatch,
			Description: "Apply a unified diff patch to workspace files atomically with optimistic concurrency verification.",
			Parameters: json.RawMessage(`{"type":"object","properties":{` +
				`"patch":{"type":"string","description":"unified diff patch text"},` +
				`"expected_version":{"type":"string","description":"opaque version returned by Read or SHA-256 hash; required for optimistic concurrency verification"},` +
				`"fuzz_margin":{"type":"integer","description":"maximum line drift tolerance (0-5 lines, default 2) for hunk matching","minimum":0,"maximum":5}},` +
				`"required":["patch"],"additionalProperties":false}`),
			ReadOnly: false,
		},
	}
}
