package vllmcompile

import (
	"fmt"
	"regexp"
	"strings"
)

// CacheTuple identifies the complete hardware and runtime configuration keying compiled artifacts.
type CacheTuple struct {
	Model     string
	Quant     string
	Arch      string
	TP        int
	Ctx       int
	Engine    string
	EngineVer string
	TorchVer  string
}

var archDigits = regexp.MustCompile(`sm[_-]?(\d+)|(\d+)\.(\d+)|(\d+)`)

// NormalizeArch converts compute-capability formats into canonical bare architecture digits.
func NormalizeArch(raw string) string {
	m := archDigits.FindStringSubmatch(strings.TrimSpace(raw))
	if m == nil {
		return "unknown"
	}
	switch {
	case m[1] != "":
		return m[1]
	case m[2] != "":
		return m[2] + m[3]
	default:
		return m[4]
	}
}

var slugUnsafe = regexp.MustCompile(`[^a-z0-9]+`)

// slug sanitizes strings for filesystem-safe path segments.
func slug(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = slugUnsafe.ReplaceAllString(s, "-")
	s = strings.Trim(s, "-")
	if s == "" {
		return "unknown"
	}
	return s
}

// Key builds the deterministic cache directory path segment from the tuple configuration.
func (t CacheTuple) Key() string {
	return fmt.Sprintf(
		"%s/%s/sm%s/tp%d/ctx%d/%s-%s-torch%s",
		slug(t.Model),
		slug(t.Quant),
		NormalizeArch(t.Arch),
		t.TP,
		t.Ctx,
		slug(t.Engine),
		slug(t.EngineVer),
		slug(t.TorchVer),
	)
}

// Readout formats the preflight log line indicating cache hit or rebuild status.
func Readout(key, dir string, populated bool) string {
	state := "rebuild"
	if populated {
		state = "hit"
	}
	return fmt.Sprintf("COMPILE_CACHE %s dir=%s tuple=%s", state, dir, key)
}

// WithCacheTuple populates cache key and enabled status on a compile block from a tuple.
func (b Block) WithCacheTuple(t CacheTuple, populated bool) Block {
	b.CompileCacheKey = t.Key()
	b.CompileCacheEnabled = Bool(populated)
	return b
}
