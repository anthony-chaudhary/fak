// Package codexmemory is a READ-ONLY diagnostic over an OpenAI Codex home
// (default ~/.codex). It reports the operator-visible MEMORY POSTURE of a
// guarded Codex session — whether memories are enabled, whether existing
// memories can be injected into future sessions, whether a new thread can
// become a memory input, whether external-context threads are excluded, and
// what generated state (including Chronicle screen-derived memories) is present
// on disk.
//
// It is the trust-boundary counterpart to fak's own recall/durability gates:
// Codex memory changes what context a FUTURE Codex session may receive, so
// before launching `fak guard -- codex` an operator wants an explicit answer to
// "what is enabled, what can be written, what can be injected later, and where
// the generated state lives."
//
// The package NEVER writes Codex state and NEVER prints raw memory contents — it
// inventories file counts, bytes, and top-level artifact names only. A missing
// or partial codex home is reported honestly ("not configured"), never a crash.
//
// Posture keys (from the Codex config reference, current 2026-06-29):
//   - [features] memories                         — global enable
//   - memories.generate_memories                  — new threads may become inputs
//   - memories.use_memories                       — existing memories injected
//   - memories.disable_on_external_context        — exclude MCP/web/tool threads
//   - memories.min_rate_limit_remaining_percent   — rate-limit floor
//   - memories.extract_model / consolidation_model
//
// See https://developers.openai.com/codex/memories and
// https://developers.openai.com/codex/config-reference.
package codexmemory

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Schema is the stable schema tag carried in the JSON payload so the
// control-pane / TUI can route on a known shape.
const Schema = "fak.codexmemory.doctor.v1"

// triBool is a three-state posture value: a key may be explicitly true,
// explicitly false, or simply absent (unset → Codex default applies). Reporting
// the absent state honestly is the whole point — "not configured" is a real,
// operator-relevant answer, not the same as "false".
type triBool struct {
	Set   bool `json:"set"`
	Value bool `json:"value,omitempty"`
}

// Posture is the read-only memory posture of a Codex home.
type Posture struct {
	Schema    string `json:"schema"`
	CodexHome string `json:"codex_home"`
	// Source records how CodexHome was resolved: "flag", "env" (CODEX_HOME),
	// or "default" (~/.codex). Empty when the home could not be resolved.
	HomeSource string `json:"home_source"`

	ConfigPath   string `json:"config_path"`
	ConfigExists bool   `json:"config_exists"`

	// MemoriesEnabled is [features].memories. Absent ⇒ not configured.
	MemoriesEnabled triBool `json:"memories_enabled"`

	GenerateMemories         triBool `json:"generate_memories"`
	UseMemories              triBool `json:"use_memories"`
	DisableOnExternalContext triBool `json:"disable_on_external_context"`
	RateLimitFloor           *int    `json:"min_rate_limit_remaining_percent,omitempty"`
	RateLimitFloorSet        bool    `json:"min_rate_limit_remaining_percent_set"`
	ExtractModel             string  `json:"extract_model,omitempty"`
	ConsolidationModel       string  `json:"consolidation_model,omitempty"`

	// Memories inventories <home>/memories.
	Memories DirInventory `json:"memories"`
	// Chronicle inventories <home>/memories_extensions/chronicle (screen-derived).
	Chronicle DirInventory `json:"chronicle"`

	// Repo guidance boundary: when RepoRoot is set, report whether AGENTS.md
	// exists so the operator can confirm team invariants are checked in, not
	// stored only in user-local memories.
	RepoRoot     string `json:"repo_root,omitempty"`
	AgentsMD     bool   `json:"agents_md_present"`
	AgentsMDPath string `json:"agents_md_path,omitempty"`

	Findings []Finding `json:"findings"`
	// OK is true when no risky posture was found (no Finding with Risk true).
	OK bool `json:"ok"`
}

// DirInventory is the safe, content-free summary of a memory directory.
type DirInventory struct {
	Path   string `json:"path"`
	Exists bool   `json:"exists"`
	Files  int    `json:"files"`
	Bytes  int64  `json:"bytes"`
	Oldest string `json:"oldest,omitempty"` // RFC3339 mtime
	Newest string `json:"newest,omitempty"`
	// Artifacts lists top-level entry names only (never contents), sorted.
	Artifacts []string `json:"artifacts,omitempty"`
	// LargestBytes is the size of the single largest file (context-noise signal).
	LargestBytes int64 `json:"largest_bytes,omitempty"`
}

// Finding is one operator-visible posture note.
type Finding struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	// Risk true ⇒ this finding should flip the advisory exit code nonzero.
	Risk bool `json:"risk"`
}

// largeMemoryBytes is the per-directory size above which we flag the memory
// store as a context-noise risk (~512 KiB of generated recall is a lot to
// silently inject into a future session).
const largeMemoryBytes int64 = 512 * 1024

// Options configures a doctor run.
type Options struct {
	// CodexHome, when non-empty, is used verbatim (the --codex-home flag).
	CodexHome string
	// RepoRoot, when non-empty, enables the AGENTS.md guidance-boundary check.
	RepoRoot string
	// Env overrides os.Getenv for tests (nil ⇒ the process environment).
	Env func(string) string
	// HomeDir overrides os.UserHomeDir for tests (nil ⇒ os.UserHomeDir).
	HomeDir func() (string, error)
}

// Doctor reads a Codex home and returns its memory posture. It never writes and
// never returns an error for a missing/partial home — an unresolved or absent
// home is reported in the Posture itself.
func Doctor(opts Options) Posture {
	getenv := opts.Env
	if getenv == nil {
		getenv = os.Getenv
	}
	homeDir := opts.HomeDir
	if homeDir == nil {
		homeDir = os.UserHomeDir
	}

	p := Posture{Schema: Schema, RepoRoot: opts.RepoRoot}

	// Resolve the effective Codex home: explicit flag > CODEX_HOME > ~/.codex.
	switch {
	case opts.CodexHome != "":
		p.CodexHome = opts.CodexHome
		p.HomeSource = "flag"
	case strings.TrimSpace(getenv("CODEX_HOME")) != "":
		p.CodexHome = strings.TrimSpace(getenv("CODEX_HOME"))
		p.HomeSource = "env"
	default:
		if h, err := homeDir(); err == nil && h != "" {
			p.CodexHome = filepath.Join(h, ".codex")
			p.HomeSource = "default"
		}
	}

	if p.CodexHome == "" {
		p.Findings = append(p.Findings, Finding{
			Code:    "home-unresolved",
			Message: "could not resolve a Codex home (no --codex-home, no CODEX_HOME, no user home dir)",
			Risk:    false,
		})
		p.OK = true
		return p
	}

	p.ConfigPath = filepath.Join(p.CodexHome, "config.toml")
	if raw, err := os.ReadFile(p.ConfigPath); err == nil {
		p.ConfigExists = true
		applyConfig(&p, parseFlatTOML(string(raw)))
	} else {
		p.Findings = append(p.Findings, Finding{
			Code:    "config-absent",
			Message: "no config.toml under the Codex home — memory posture is the Codex built-in default (not configured here)",
			Risk:    false,
		})
	}

	// Inventory generated state (safe: counts/bytes/names only).
	p.Memories = inventory(filepath.Join(p.CodexHome, "memories"))
	p.Chronicle = inventory(filepath.Join(p.CodexHome, "memories_extensions", "chronicle"))

	// Repo guidance boundary.
	if opts.RepoRoot != "" {
		ap := filepath.Join(opts.RepoRoot, "AGENTS.md")
		if st, err := os.Stat(ap); err == nil && !st.IsDir() {
			p.AgentsMD = true
			p.AgentsMDPath = ap
		}
	}

	deriveFindings(&p)
	p.OK = true
	for _, f := range p.Findings {
		if f.Risk {
			p.OK = false
			break
		}
	}
	return p
}

// applyConfig folds parsed TOML key/values into the posture.
func applyConfig(p *Posture, kv map[string]string) {
	if v, ok := kv["features.memories"]; ok {
		p.MemoriesEnabled = triBool{Set: true, Value: parseBool(v)}
	}
	if v, ok := kv["memories.generate_memories"]; ok {
		p.GenerateMemories = triBool{Set: true, Value: parseBool(v)}
	}
	if v, ok := kv["memories.use_memories"]; ok {
		p.UseMemories = triBool{Set: true, Value: parseBool(v)}
	}
	if v, ok := kv["memories.disable_on_external_context"]; ok {
		p.DisableOnExternalContext = triBool{Set: true, Value: parseBool(v)}
	}
	if v, ok := kv["memories.min_rate_limit_remaining_percent"]; ok {
		if n, ok2 := parseInt(v); ok2 {
			p.RateLimitFloor = &n
			p.RateLimitFloorSet = true
		}
	}
	if v, ok := kv["memories.extract_model"]; ok {
		p.ExtractModel = unquote(v)
	}
	if v, ok := kv["memories.consolidation_model"]; ok {
		p.ConsolidationModel = unquote(v)
	}
}

// deriveFindings turns the read posture into operator-visible findings.
func deriveFindings(p *Posture) {
	enabled := p.MemoriesEnabled.Set && p.MemoriesEnabled.Value

	// External-context inclusion is the headline risk for fak-guarded sessions:
	// a guarded session may touch MCP/web/tool-search/external data, and if
	// disable_on_external_context is false those threads can become memory inputs.
	if enabled || p.GenerateMemories.Value {
		if p.DisableOnExternalContext.Set && !p.DisableOnExternalContext.Value {
			p.Findings = append(p.Findings, Finding{
				Code:    "external-context-included",
				Message: "memories.disable_on_external_context=false — a fak-guarded session that uses MCP/web/external data may become a memory input",
				Risk:    true,
			})
		} else if !p.DisableOnExternalContext.Set {
			p.Findings = append(p.Findings, Finding{
				Code:    "external-context-default",
				Message: "memories.disable_on_external_context is unset (Codex default applies) — confirm external-context threads are excluded for guarded sessions",
				Risk:    false,
			})
		}
	}

	if enabled && !p.RateLimitFloorSet {
		p.Findings = append(p.Findings, Finding{
			Code:    "rate-limit-floor-default",
			Message: "memories.min_rate_limit_remaining_percent is unset — the Codex default rate-limit floor applies",
			Risk:    false,
		})
	}

	if p.Memories.Bytes >= largeMemoryBytes {
		p.Findings = append(p.Findings, Finding{
			Code:    "large-memory-store",
			Message: fmt.Sprintf("memories/ holds %s — large generated recall is a context-noise risk for future sessions", humanBytes(p.Memories.Bytes)),
			Risk:    true,
		})
	}

	// Chronicle is opt-in, screen-derived, stored unencrypted, and raises
	// prompt-injection risk from screen content — flag it separately.
	if p.Chronicle.Exists && p.Chronicle.Files > 0 {
		p.Findings = append(p.Findings, Finding{
			Code:    "chronicle-present",
			Message: fmt.Sprintf("Chronicle screen-derived memories present (%d files/%s) — may derive from screen context; pause Chronicle before sensitive screens and gate read-back", p.Chronicle.Files, humanBytes(p.Chronicle.Bytes)),
			Risk:    true,
		})
	}

	if p.RepoRoot != "" && !p.AgentsMD {
		p.Findings = append(p.Findings, Finding{
			Code:    "agents-md-absent",
			Message: "no AGENTS.md in the repo — team/repo invariants should be checked in, not stored only in user-local Codex memories",
			Risk:    false,
		})
	}
}

// inventory summarizes a directory without reading any file contents. A missing
// directory returns Exists=false, never an error.
func inventory(dir string) DirInventory {
	inv := DirInventory{Path: dir}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return inv // absent or unreadable ⇒ Exists stays false
	}
	inv.Exists = true
	var oldest, newest time.Time
	walk := func(path string, info os.FileInfo) {
		if info.IsDir() {
			return
		}
		inv.Files++
		inv.Bytes += info.Size()
		if info.Size() > inv.LargestBytes {
			inv.LargestBytes = info.Size()
		}
		mt := info.ModTime()
		if oldest.IsZero() || mt.Before(oldest) {
			oldest = mt
		}
		if newest.IsZero() || mt.After(newest) {
			newest = mt
		}
	}
	for _, e := range entries {
		inv.Artifacts = append(inv.Artifacts, e.Name())
		info, ierr := e.Info()
		if ierr != nil {
			continue
		}
		if e.IsDir() {
			// One level of recursion into subdirectories for counts/bytes;
			// artifact NAMES stay top-level only (no content, no deep paths).
			subEntries, serr := os.ReadDir(filepath.Join(dir, e.Name()))
			if serr == nil {
				for _, se := range subEntries {
					sinfo, sierr := se.Info()
					if sierr == nil {
						walk(filepath.Join(dir, e.Name(), se.Name()), sinfo)
					}
				}
			}
			continue
		}
		walk(filepath.Join(dir, e.Name()), info)
	}
	sort.Strings(inv.Artifacts)
	if !oldest.IsZero() {
		inv.Oldest = oldest.UTC().Format(time.RFC3339)
	}
	if !newest.IsZero() {
		inv.Newest = newest.UTC().Format(time.RFC3339)
	}
	return inv
}

// Render produces the compact, payload-free human view.
func Render(p Posture) string {
	var b strings.Builder
	home := p.CodexHome
	if home == "" {
		home = "(unresolved)"
	}
	fmt.Fprintf(&b, "codex-memory doctor: %s\n", home)

	if !p.ConfigExists {
		fmt.Fprintf(&b, "  config: config.toml absent — posture is the Codex built-in default (not configured)\n")
	}
	fmt.Fprintf(&b, "  memories: %s use=%s generate=%s external_context_excluded=%s%s\n",
		triState(p.MemoriesEnabled),
		triState(p.UseMemories),
		triState(p.GenerateMemories),
		triState(p.DisableOnExternalContext),
		externalWarn(p),
	)
	if p.RateLimitFloorSet {
		fmt.Fprintf(&b, "  rate-limit floor: %d%%\n", *p.RateLimitFloor)
	} else {
		fmt.Fprintf(&b, "  rate-limit floor: default (unset)\n")
	}
	if p.ExtractModel != "" || p.ConsolidationModel != "" {
		fmt.Fprintf(&b, "  models: extract=%s consolidation=%s\n", orDash(p.ExtractModel), orDash(p.ConsolidationModel))
	}
	fmt.Fprintf(&b, "  files: memories=%d files/%s chronicle=%d files/%s\n",
		p.Memories.Files, humanBytes(p.Memories.Bytes),
		p.Chronicle.Files, humanBytes(p.Chronicle.Bytes),
	)
	if p.RepoRoot != "" {
		state := "absent WARN"
		if p.AgentsMD {
			state = "present OK"
		}
		fmt.Fprintf(&b, "  repo guidance: AGENTS.md %s\n", state)
	}
	fmt.Fprintf(&b, "  findings: %d\n", len(p.Findings))
	for _, f := range p.Findings {
		tag := "note"
		if f.Risk {
			tag = "WARN"
		}
		fmt.Fprintf(&b, "    [%s] %s: %s\n", tag, f.Code, f.Message)
	}
	return b.String()
}

func externalWarn(p Posture) string {
	if p.DisableOnExternalContext.Set && !p.DisableOnExternalContext.Value &&
		(p.MemoriesEnabled.Value || p.GenerateMemories.Value) {
		return " WARN"
	}
	return ""
}

func triState(t triBool) string {
	if !t.Set {
		return "unset"
	}
	if t.Value {
		return "true"
	}
	return "false"
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%dB", n)
	}
	div, exp := int64(unit), 0
	for x := n / unit; x >= unit; x /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.0f%ciB", float64(n)/float64(div), "KMGTPE"[exp])
}

// --- minimal flat TOML scanner -------------------------------------------------
//
// Codex config.toml uses simple `key = value` lines under `[table]` /
// `[table.sub]` headers. We need only a handful of scalar keys, so rather than
// vendor a full TOML parser we scan flat dotted keys ("features.memories",
// "memories.use_memories", ...). Unsupported constructs (arrays, inline tables,
// multiline strings) are simply skipped — they never carry the posture keys we
// read, and skipping them keeps the diagnostic robust on a partial config.

func parseFlatTOML(s string) map[string]string {
	out := map[string]string{}
	table := ""
	for _, raw := range strings.Split(s, "\n") {
		line := stripComment(raw)
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			// Table header. Ignore array-of-tables ("[[...]]") — not used here.
			name := strings.TrimSpace(line[1 : len(line)-1])
			name = strings.TrimPrefix(name, "[")
			name = strings.TrimSuffix(name, "]")
			table = strings.TrimSpace(name)
			continue
		}
		eq := strings.IndexByte(line, '=')
		if eq < 0 {
			continue
		}
		key := strings.TrimSpace(line[:eq])
		val := strings.TrimSpace(line[eq+1:])
		if key == "" {
			continue
		}
		full := key
		if table != "" {
			full = table + "." + key
		}
		out[full] = val
	}
	return out
}

// stripComment removes a trailing `# comment`, respecting a single quoted
// string so a '#' inside a value is not treated as a comment.
func stripComment(s string) string {
	inStr := false
	var q byte
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inStr {
			if c == q {
				inStr = false
			}
			continue
		}
		switch c {
		case '"', '\'':
			inStr = true
			q = c
		case '#':
			return s[:i]
		}
	}
	return s
}

func parseBool(v string) bool {
	return strings.EqualFold(strings.TrimSpace(v), "true")
}

func parseInt(v string) (int, bool) {
	v = strings.TrimSpace(v)
	n := 0
	if v == "" {
		return 0, false
	}
	for _, r := range v {
		if r < '0' || r > '9' {
			return 0, false
		}
		n = n*10 + int(r-'0')
	}
	return n, true
}

func unquote(v string) string {
	v = strings.TrimSpace(v)
	if len(v) >= 2 {
		if (v[0] == '"' && v[len(v)-1] == '"') || (v[0] == '\'' && v[len(v)-1] == '\'') {
			return v[1 : len(v)-1]
		}
	}
	return v
}
