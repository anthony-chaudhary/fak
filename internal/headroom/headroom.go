package headroom

import (
	"bytes"
	"context"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// Compressor is the generic seam for the context-savings AREA — the one interface
// every plugin "for that area" implements. Headroom (a bridge to the external
// `headroom` service), native (an in-process structural compressor), and noop (the
// identity default) are the shipped plugins; a new one is a single Register call
// from its own init(), so the area is modular by construction: any plugin can be
// added, selected, or left out without touching this seam OR the kernel result
// path that folds it. This is the "implemented as generic functions so any other
// style of plugin for that area can be used or not used modularly" contract — the
// kernel knows only this interface, never a concrete compressor.
//
// A Compressor is handed the model-bound bytes of ONE tool result and returns a
// smaller, MODEL-READABLE rendering plus the stats to prove the saving. For a
// given process it must be a deterministic function of its input (no hidden
// mutable global state) so the ResultAdmitter that folds it stays replay-stable.
type Compressor interface {
	// Name is the stable registry key (also the FAK_COMPRESSOR value).
	Name() string
	// Compress shrinks in.Bytes. It returns the compressed rendering and the
	// measured saving. A plugin that cannot help — incompressible input, an
	// unreachable backend — returns Output.Compressed==false and SHOULD return the
	// input bytes unchanged; the caller then admits the original as-is. An error is
	// advisory: the caller treats any non-Compressed Output as a clean no-op.
	Compress(ctx context.Context, in Input) (Output, error)
}

// Input is one tool result presented to a Compressor.
type Input struct {
	Tool  string      // the producing tool (a router hint; may be "")
	Kind  ContentKind // detected content class (router hint; KindUnknown = sniff it)
	Model string      // target model id, for token-accounting plugins (may be "")
	Bytes []byte      // the model-bound result bytes
}

// Output is a Compressor's verdict on one input.
type Output struct {
	Bytes      []byte // the compressed, model-readable rendering
	Compressed bool   // true iff Bytes is a genuine, smaller rendering of the input
	Codec      string // which transform(s) ran (forensics; e.g. "json-min+line-dedup")
	Retrieval  string // optional handle to retrieve the original (CCR), "" if none
	OrigLen    int    // len(input bytes)
	NewLen     int    // len(output bytes)
}

// SavedRatio is the fraction of bytes removed, in [0,1] (0 when nothing helped).
func (o Output) SavedRatio() float64 {
	if o.OrigLen <= 0 || o.NewLen >= o.OrigLen {
		return 0
	}
	return float64(o.OrigLen-o.NewLen) / float64(o.OrigLen)
}

// ---------------------------------------------------------------------------
// The named-plugin registry — the modular core. Plugins Register() from init();
// exactly one is Selected() at a time (default noop = off). Selection is an
// explicit act separate from registration, so adding a plugin is inert until it
// is asked for by name (Select or FAK_COMPRESSOR).
// ---------------------------------------------------------------------------

var (
	mu       sync.RWMutex
	plugins  = map[string]Compressor{}
	selected = NoopName // identity default: the build compresses NOTHING until asked
	selOnce  sync.Once
)

// Register adds a Compressor to the area registry under c.Name(). It is called
// from a plugin's init(); a duplicate name panics (link-time disjointness, the
// repo idiom — RegisterOp/RegisterEngine behave the same). Registration does NOT
// select the plugin.
func Register(c Compressor) {
	mu.Lock()
	defer mu.Unlock()
	name := c.Name()
	if _, dup := plugins[name]; dup {
		panic("headroom: duplicate compressor " + name)
	}
	plugins[name] = c
}

// Select makes the named plugin the active Compressor and reports whether the
// name was known. An unknown name is REFUSED and leaves the current selection
// unchanged — the fail-closed direction: a typo never silently disables an active
// compressor nor enables an unintended one.
func Select(name string) bool {
	mu.Lock()
	defer mu.Unlock()
	if _, ok := plugins[name]; !ok {
		return false
	}
	selected = name
	return true
}

// Selected returns the active Compressor, resolving FAK_COMPRESSOR exactly once on
// first use (lazily, so it is robust to package init() ordering). It never returns
// nil: with nothing selected and no noop registered it falls back to the literal
// identity, so the result path always has a working, zero-overhead default.
func Selected() Compressor {
	selOnce.Do(func() {
		if name := strings.TrimSpace(os.Getenv("FAK_COMPRESSOR")); name != "" {
			Select(name) // unknown name is ignored -> stays on the noop default
		}
	})
	mu.RLock()
	defer mu.RUnlock()
	if c, ok := plugins[selected]; ok {
		return c
	}
	return noopCompressor{}
}

// Lookup returns the registered Compressor for name (for `--via NAME`).
func Lookup(name string) (Compressor, bool) {
	mu.RLock()
	defer mu.RUnlock()
	c, ok := plugins[name]
	return c, ok
}

// Names lists the registered plugin names, sorted (CLI/forensics).
func Names() []string {
	mu.RLock()
	defer mu.RUnlock()
	out := make([]string, 0, len(plugins))
	for n := range plugins {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// ---------------------------------------------------------------------------
// ContentKind — the generic analogue of Headroom's ContentRouter. A cheap
// structural sniff a plugin uses as a hint to pick its transform.
// ---------------------------------------------------------------------------

// ContentKind is a coarse classification of a result's bytes.
type ContentKind int

const (
	KindUnknown ContentKind = iota
	KindJSON
	KindLog
	KindCode
	KindText
)

// String renders the content kind as its lowercase name ("json", "log", "code",
// "text"), or "unknown" for KindUnknown.
func (k ContentKind) String() string {
	switch k {
	case KindJSON:
		return "json"
	case KindLog:
		return "log"
	case KindCode:
		return "code"
	case KindText:
		return "text"
	default:
		return "unknown"
	}
}

// Detect classifies bytes into a coarse content class with a cheap structural
// sniff (no full parse of the blob). Empty input is KindUnknown.
func Detect(b []byte) ContentKind {
	t := bytes.TrimSpace(b)
	if len(t) == 0 {
		return KindUnknown
	}
	// JSON: brackets at both ends are a strong, cheap tell.
	if (t[0] == '{' || t[0] == '[') && (t[len(t)-1] == '}' || t[len(t)-1] == ']') {
		return KindJSON
	}
	s := string(t)
	// Code: fenced blocks or common source tokens.
	if strings.Contains(s, "```") ||
		strmatch.ContainsAny(s, "func ", "package ", "import ", "def ", "class ", "#include", "=> {", "public ", "private ") {
		return KindCode
	}
	// Log: many lines, a high share carrying a level tag or an ISO/clock stamp.
	lines := strings.Split(s, "\n")
	if len(lines) >= 4 {
		hits := 0
		for _, ln := range lines {
			if looksLogLine(ln) {
				hits++
			}
		}
		if hits*2 >= len(lines) {
			return KindLog
		}
	}
	return KindText
}

func looksLogLine(ln string) bool {
	if strmatch.ContainsAny(ln, "ERROR", "WARN", "INFO", "DEBUG", "TRACE", "FATAL") {
		return true
	}
	// a leading time-ish stamp: "12:34", "2026-06-23", "[12:34:56]".
	for i := 0; i < len(ln) && i < 24; i++ {
		c := ln[i]
		if c == ':' && i >= 2 && isDigit(ln[i-1]) && isDigit(ln[i-2]) {
			return true
		}
	}
	return false
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }
