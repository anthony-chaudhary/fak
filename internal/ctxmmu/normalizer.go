package ctxmmu

import (
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
)

const (
	// DefaultSessionPlaceholder is the canonical placeholder for volatile timestamps.
	DefaultSessionPlaceholder = "$SESSION_START_TIME"
	// DefaultVirtualWorkspace is the virtual root replacing host workspace paths.
	DefaultVirtualWorkspace = "$WORKSPACE"
	// PIDPlaceholder is the canonical placeholder for volatile process IDs.
	PIDPlaceholder = "[PID]"
	// TempFilePlaceholder is the canonical placeholder for ephemeral files in temp directories.
	TempFilePlaceholder = "[TMP]"
	// DurationPlaceholder is the canonical placeholder for volatile execution durations.
	DurationPlaceholder = "[DURATION]"
)

var (
	// ISO8601 / RFC3339 timestamps (e.g. 2026-09-05T12:34:56Z, 2026-09-05 12:34:56.789-07:00).
	reISO8601 = regexp.MustCompile(`\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b`)

	// RFC1123 / RFC822 / UTC date formats (e.g. Sat, 05 Sep 2026 12:34:56 GMT, 05 Sep 2026 12:34:56 UTC).
	reRFC1123 = regexp.MustCompile(`\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}(?:\s+\d{2}:\d{2}:\d{2}(?:\s+(?:UTC|GMT|[+-]\d{4}))?)?\b`)

	// Unix date / ctime / asctime / UTC formats (e.g. Sat Sep  5 12:34:56 2026, Sat Sep 05 12:34:56 UTC 2026).
	reUnixDate = regexp.MustCompile(`\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:\s+\d{2}:\d{2}:\d{2})?(?:\s+(?:UTC|GMT))?\s+\d{4}\b`)

	// Explicit environment banner date (e.g. Today's date: Sat Sep 05 2026).
	reTodaysDate = regexp.MustCompile(`(?i)\b(Today's date:\s*)(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{4}\b`)

	// Labeled Unix epoch timestamps (e.g. timestamp: 1725537600, epoch: 1725537600000).
	reEpochLabeled = regexp.MustCompile(`(?i)\b(timestamp|epoch|unix_time|time|started_at|created_at|ended_at)(\s*[:=]\s*|\s+)(1[6-9]\d{8,12}|2\d{9,12}(?:\.\d+)?)\b`)

	// Bracketed Unix epochs (e.g. [1725537600] or (1725537600)).
	reEpochBracketed = regexp.MustCompile(`(\[|\()\s*(1[6-9]\d{8,12}|2\d{9,12}(?:\.\d+)?)\s*(\]|\))`)

	// Labeled @ epoch (e.g. @1725537600).
	reEpochAt = regexp.MustCompile(`@\s*(1[6-9]\d{8,12}|2\d{9,12})\b`)

	// Ephemeral PIDs (e.g. PID 12345, PID: 12345, pid=12345, Process ID: 12345).
	rePID        = regexp.MustCompile(`(?i)\b(process\s+id|pid)(\s*[:=]\s*|\s+)(\d{1,7})\b`)
	reProcessPID = regexp.MustCompile(`(?i)\b(process)(\s*[:=]\s*|\s+)(\d{2,7})\b`)

	// Temporary directories and files (OpenCode and general temp across Windows and Unix).
	reTempPath = regexp.MustCompile(`(?i)(?:[a-z]:(?:\\\\|[\\/])(?:[^\s\\/"'` + "`" + `]+(?:\\\\|[\\/]))*AppData(?:\\\\|[\\/])Local(?:\\\\|[\\/])Temp|/(?:var/)?tmp)(?:(?:\\\\|[\\/])(opencode\b))?(?:(?:\\\\|[\\/])([^\s"'` + "`" + `;,\)\]}>]+))?`)

	// Generic host home/workspace paths fallback if no explicit workspaceRoot provided.
	reGenericWinPath  = regexp.MustCompile(`(?i)[a-z]:(?:\\\\|[\\/])Users(?:\\\\|[\\/])[^\s\\/"'` + "`" + `]+(?:\\\\|[\\/])(?:OneDrive(?:\\\\|[\\/]))?(?:Desktop(?:\\\\|[\\/]))?(?:work|code|projects|workspace)(?:\\\\|[\\/])(?:fak(?:-private)?|[a-zA-Z0-9_-]+)(?:(\\\\|[\\/])([^\s"'` + "`" + `;,\)\]}>]*))?`)
	reGenericUnixPath = regexp.MustCompile(`/(?:home|Users)/[^\s/"'` + "`" + `]+/(?:work|code|projects|workspace)/(?:fak(?:-private)?|[a-zA-Z0-9_-]+)(?:/([^\s"'` + "`" + `;,\)\]}>]*))?`)

	// Execution durations in tool headers/wrappers (e.g. duration: 12ms, elapsed: 0.45s, took 123ms).
	reDuration = regexp.MustCompile(`(?i)\b(duration|elapsed|took)(\s*[:=]\s*|\s+)(\d+(?:\.\d+)?\s*(?:ms|s|µs|us|ns|sec|seconds?))\b`)
)

// Normalizer canonicalizes ephemeral environment headers, volatile timestamps,
// workspace root paths, and temporary files across subagent turns to guarantee
// KV cache prefix stability and prevent prompt cache invalidation thrashing.
type Normalizer struct {
	SessionStart  string // Canonical timestamp placeholder (default: $SESSION_START_TIME)
	WorkspaceRoot string // Host workspace root (e.g. C:\Users\...\fak or /home/.../fak)
	VirtualRoot   string // Virtual root replacement (default: $WORKSPACE)

	pathRegexes []*regexp.Regexp
}

// DefaultNormalizer is the global default normalizer with standard placeholders.
var DefaultNormalizer = NewNormalizer("", "")

// NewNormalizer creates a new Normalizer with the given session start timestamp/placeholder
// and host workspace root path.
func NewNormalizer(sessionStart, workspaceRoot string) *Normalizer {
	if sessionStart == "" {
		sessionStart = DefaultSessionPlaceholder
	}
	workspaceRoot = strings.TrimRight(workspaceRoot, "/\\")
	n := &Normalizer{
		SessionStart:  sessionStart,
		WorkspaceRoot: workspaceRoot,
		VirtualRoot:   DefaultVirtualWorkspace,
	}
	n.compilePathRegexes()
	return n
}

// WithVirtualRoot sets the virtual root placeholder and returns n for chaining.
func (n *Normalizer) WithVirtualRoot(virtualRoot string) *Normalizer {
	if virtualRoot != "" {
		n.VirtualRoot = virtualRoot
	}
	return n
}

func (n *Normalizer) compilePathRegexes() {
	if n.WorkspaceRoot == "" {
		return
	}
	variants := buildPathVariants(n.WorkspaceRoot)
	sort.Slice(variants, func(i, j int) bool {
		return len(variants[i]) > len(variants[j])
	})

	var regexes []*regexp.Regexp
	for _, v := range variants {
		pattern := `(?i)` + regexp.QuoteMeta(v) + `(?:(\\\\|[\\/])([^\s"'` + "`" + `;,\)\]}>]*))?`
		if re, err := regexp.Compile(pattern); err == nil {
			regexes = append(regexes, re)
		}
	}
	n.pathRegexes = regexes
}

func buildPathVariants(root string) []string {
	if root == "" {
		return nil
	}
	root = filepath.Clean(root)
	root = strings.TrimRight(root, "/\\")

	slashVer := strings.ReplaceAll(root, "\\", "/")
	backslashVer := strings.ReplaceAll(root, "/", "\\")
	jsonVer := strings.ReplaceAll(backslashVer, "\\", "\\\\")

	variants := []string{backslashVer, slashVer, jsonVer}

	if len(root) >= 2 && root[1] == ':' {
		drive := rune(root[0])
		var altDrive rune
		if drive >= 'A' && drive <= 'Z' {
			altDrive = drive + ('a' - 'A')
		} else if drive >= 'a' && drive <= 'z' {
			altDrive = drive - ('a' - 'A')
		}
		if altDrive != 0 {
			altBackslash := string(altDrive) + backslashVer[1:]
			altSlash := string(altDrive) + slashVer[1:]
			altJSON := string(altDrive) + jsonVer[1:]
			variants = append(variants, altBackslash, altSlash, altJSON)
		}
	}

	seen := make(map[string]bool)
	var deduped []string
	for _, v := range variants {
		if !seen[v] {
			seen[v] = true
			deduped = append(deduped, v)
		}
	}
	return deduped
}

func (n *Normalizer) sessionPlaceholder() string {
	if n.SessionStart != "" {
		return n.SessionStart
	}
	return DefaultSessionPlaceholder
}

func (n *Normalizer) virtualRoot() string {
	if n.VirtualRoot != "" {
		return n.VirtualRoot
	}
	return DefaultVirtualWorkspace
}

// NormalizeHeader canonicalizes prompt and shell environment headers.
func (n *Normalizer) NormalizeHeader(header string) string {
	return n.canonicalize(header, true)
}

// NormalizeToolOutput canonicalizes tool execution outputs and wrappers.
func (n *Normalizer) NormalizeToolOutput(toolName string, output string) string {
	_ = toolName
	return n.canonicalize(output, false)
}

func (n *Normalizer) canonicalize(text string, isHeader bool) string {
	if text == "" {
		return ""
	}

	vRoot := n.virtualRoot()
	sPlaceholder := n.sessionPlaceholder()

	// 1. Workspace paths
	if len(n.pathRegexes) > 0 {
		for _, re := range n.pathRegexes {
			text = re.ReplaceAllStringFunc(text, func(match string) string {
				sub := re.FindStringSubmatch(match)
				if len(sub) > 1 && sub[1] != "" {
					subpath := ""
					if len(sub) > 2 && sub[2] != "" {
						subpath = strings.ReplaceAll(sub[2], `\\`, `/`)
						subpath = strings.ReplaceAll(subpath, `\`, `/`)
					}
					if subpath != "" {
						return vRoot + "/" + subpath
					}
					return vRoot + "/"
				}
				return vRoot
			})
		}
	} else {
		// Fallback generic workspace replacement
		text = reGenericWinPath.ReplaceAllStringFunc(text, func(match string) string {
			sub := reGenericWinPath.FindStringSubmatch(match)
			if len(sub) > 1 && sub[1] != "" {
				subpath := ""
				if len(sub) > 2 && sub[2] != "" {
					subpath = strings.ReplaceAll(sub[2], `\\`, `/`)
					subpath = strings.ReplaceAll(subpath, `\`, `/`)
				}
				if subpath != "" {
					return vRoot + "/" + subpath
				}
				return vRoot + "/"
			}
			return vRoot
		})
		text = reGenericUnixPath.ReplaceAllStringFunc(text, func(match string) string {
			sub := reGenericUnixPath.FindStringSubmatch(match)
			if len(sub) > 1 && sub[1] != "" {
				return vRoot + "/" + sub[1]
			}
			return vRoot
		})
	}

	// 2. Ephemeral temporary file paths and directories
	text = reTempPath.ReplaceAllStringFunc(text, func(match string) string {
		sub := reTempPath.FindStringSubmatch(match)
		isOpenCode := false
		if len(sub) > 1 && strings.EqualFold(sub[1], "opencode") {
			isOpenCode = true
		}
		hasSubpath := false
		if len(sub) > 2 && sub[2] != "" {
			hasSubpath = true
		}

		if isOpenCode {
			if hasSubpath {
				return "/tmp/opencode/" + TempFilePlaceholder
			}
			return "/tmp/opencode"
		}
		if hasSubpath {
			return "/tmp/" + TempFilePlaceholder
		}
		return "/tmp"
	})

	// 3. Ephemeral PIDs
	text = rePID.ReplaceAllStringFunc(text, func(m string) string {
		sub := rePID.FindStringSubmatch(m)
		if len(sub) == 4 {
			return sub[1] + sub[2] + PIDPlaceholder
		}
		return m
	})
	text = reProcessPID.ReplaceAllStringFunc(text, func(m string) string {
		sub := reProcessPID.FindStringSubmatch(m)
		if len(sub) == 4 {
			return sub[1] + sub[2] + PIDPlaceholder
		}
		return m
	})

	// 4. Volatile Timestamps
	// Explicit header date: "Today's date: Sat Sep 05 2026"
	text = reTodaysDate.ReplaceAllStringFunc(text, func(m string) string {
		sub := reTodaysDate.FindStringSubmatch(m)
		if len(sub) == 2 {
			return sub[1] + sPlaceholder
		}
		return m
	})

	// ISO8601 / RFC3339
	text = reISO8601.ReplaceAllLiteralString(text, sPlaceholder)

	// RFC1123 / UTC dates
	text = reRFC1123.ReplaceAllLiteralString(text, sPlaceholder)

	// Unix date / ctime
	text = reUnixDate.ReplaceAllLiteralString(text, sPlaceholder)

	// Labeled epochs
	text = reEpochLabeled.ReplaceAllStringFunc(text, func(m string) string {
		sub := reEpochLabeled.FindStringSubmatch(m)
		if len(sub) == 4 {
			return sub[1] + sub[2] + sPlaceholder
		}
		return m
	})

	// Bracketed epochs: [1725537600] -> [$SESSION_START_TIME]
	text = reEpochBracketed.ReplaceAllStringFunc(text, func(m string) string {
		sub := reEpochBracketed.FindStringSubmatch(m)
		if len(sub) == 4 {
			return sub[1] + sPlaceholder + sub[3]
		}
		return m
	})

	// @epoch: @1725537600 -> @$SESSION_START_TIME
	text = reEpochAt.ReplaceAllStringFunc(text, func(m string) string {
		return "@" + sPlaceholder
	})

	// 5. Execution Durations
	text = reDuration.ReplaceAllStringFunc(text, func(m string) string {
		sub := reDuration.FindStringSubmatch(m)
		if len(sub) == 4 {
			return sub[1] + sub[2] + DurationPlaceholder
		}
		return m
	})

	_ = isHeader
	return text
}

// CanonicalizeHeader normalizes volatile timestamps, host workspace paths, PIDs,
// and temporary paths in prompt/header text.
func CanonicalizeHeader(header string, sessionStart string, workspaceRoot string) string {
	n := NewNormalizer(sessionStart, workspaceRoot)
	return n.NormalizeHeader(header)
}

// CanonicalizeToolOutput normalizes volatile timestamps, host workspace paths, PIDs,
// temporary paths, and execution durations in tool execution outputs.
func CanonicalizeToolOutput(toolName string, output string, sessionStart string, workspaceRoot string) string {
	n := NewNormalizer(sessionStart, workspaceRoot)
	return n.NormalizeToolOutput(toolName, output)
}

// ComputeLCP returns the length in bytes of the Longest Common Prefix of s1 and s2.
func ComputeLCP(s1, s2 string) int {
	minLen := len(s1)
	if len(s2) < minLen {
		minLen = len(s2)
	}
	for i := 0; i < minLen; i++ {
		if s1[i] != s2[i] {
			return i
		}
	}
	return minLen
}

// PrefixDivergenceRatio computes the divergence ratio of s1 and s2 based on their
// longest common prefix relative to the shorter string's length:
//
//	divergence = 1.0 - (LCP / min(len(s1), len(s2)))
//
// Returns 0.0 if both strings are empty or if the shorter string is an exact prefix of the longer.
// Returns 1.0 if min(len(s1), len(s2)) > 0 and LCP == 0.
func PrefixDivergenceRatio(s1, s2 string) float64 {
	minLen := len(s1)
	if len(s2) < minLen {
		minLen = len(s2)
	}
	if minLen == 0 {
		if len(s1) == 0 && len(s2) == 0 {
			return 0.0
		}
		return 1.0
	}
	lcp := ComputeLCP(s1, s2)
	return 1.0 - (float64(lcp) / float64(minLen))
}

// LCPPreservationRatio returns the prefix preservation ratio (1.0 - PrefixDivergenceRatio):
//
//	preservation = LCP / min(len(s1), len(s2))
//
// Returns 1.0 if both strings are empty or if the shorter string is an exact prefix of the longer.
func LCPPreservationRatio(s1, s2 string) float64 {
	return 1.0 - PrefixDivergenceRatio(s1, s2)
}

// -----------------------------------------------------------------------------
// MMU Normalizer Integration
// -----------------------------------------------------------------------------

var (
	mmuNormMu      sync.RWMutex
	mmuNormalizers = make(map[*MMU]*Normalizer)
)

// WithNormalizer associates a Normalizer with this MMU for result canonicalization.
func (m *MMU) WithNormalizer(norm *Normalizer) *MMU {
	if m == nil {
		return m
	}
	mmuNormMu.Lock()
	defer mmuNormMu.Unlock()
	mmuNormalizers[m] = norm
	return m
}

// Normalizer returns the Normalizer associated with this MMU, if any.
func (m *MMU) Normalizer() *Normalizer {
	if m == nil {
		return nil
	}
	mmuNormMu.RLock()
	defer mmuNormMu.RUnlock()
	return mmuNormalizers[m]
}
