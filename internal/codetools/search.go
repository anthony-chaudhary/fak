package codetools

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// search.go — Grep and Glob, implemented as Go walks.
//
// NO SHELL. Not "we prefer not to" — this file does not import os/exec, and
// TestOnlyBashEngineImportsExec fails the package if it ever does. Handing a search
// pattern to a shell is how a search tool becomes an execution tool: `grep -r "$pattern"`
// with a pattern of `x"; rm -rf /; "` is a command, not a search, and no amount of
// quoting discipline survives a model that controls the string. RE2 (regexp) and
// filepath.Match have no such reading — a pattern can only ever describe what to match.
//
// BOUNDED IN THREE DIMENSIONS. A walk stops at MaxWalkFiles files visited, a result set
// stops at MaxMatches / MaxEntries rows, and both check ctx between files. The third is
// what makes a cancel or a session terminate actually stop the work rather than merely
// discard it afterwards.
//
// SYMLINKS ARE NOT FOLLOWED during a walk. fs.WalkDir does not descend them, and each
// candidate is re-confined before it is opened, so a link planted inside the tree cannot
// widen a search into the host filesystem.

// grepEngine serves Grep: a bounded regexp content search.
type grepEngine struct{ t *Toolset }

// Caps reports no optional capabilities.
func (grepEngine) Caps() []abi.Capability { return nil }

// WeightBearing declares Grep a deterministic classical tool engine.
func (grepEngine) WeightBearing() bool { return false }

// Complete performs the search.
func (e grepEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	in := bytesOf(ctx, c.Args)
	out, isErr := e.t.grep(ctx, in)
	return result(ctx, c, in, out, isErr, EngineGrep), nil
}

// GrepMatch is one match row: the workspace-relative file, the 1-based line number, and
// the matching line, bounded in length so one pathological minified line cannot flood the
// result. Truncated reports whether this match's line text was capped.
type GrepMatch struct {
	File      string `json:"file"`
	Line      int    `json:"line"`
	Text      string `json:"text"`
	Truncated bool   `json:"truncated"`
}

// maxMatchLineBytes bounds a single reported line. A minified bundle is one 3MB "line";
// reporting it whole would blow the result bound on a single row while telling the caller
// nothing they could act on.
const maxMatchLineBytes = 512

// errFlightAbandoned is what joiners see if a leader's execution dies without
// publishing a result (a panic unwinding through Do).
var errFlightAbandoned = errors.New("codetools: in-flight search abandoned")

// flightGroup coalesces concurrent identical search calls, keyed by query arguments.
// Matching the singleflight pattern in internal/gitbroker/singleflight.go.
type flightGroup[T any] struct {
	mu        sync.Mutex
	m         map[string]*flight[T]
	coalesced atomic.Int64
}

type flight[T any] struct {
	done     chan struct{}
	doneOnce sync.Once
	val      T
	err      error
	waiters  atomic.Int32
	canceled bool
}

func (g *flightGroup[T]) Do(ctx context.Context, key string, fn func() (T, error)) (val T, shared bool, err error) {
	g.mu.Lock()
	if g.m == nil {
		g.m = make(map[string]*flight[T])
	}
	if inflight, ok := g.m[key]; ok {
		if ctx != nil {
			if err := ctx.Err(); err != nil {
				g.mu.Unlock()
				var zero T
				return zero, false, err
			}
		}
		inflight.waiters.Add(1)
		g.mu.Unlock()

		var ctxDone <-chan struct{}
		if ctx != nil {
			ctxDone = ctx.Done()
		}
		select {
		case <-ctxDone:
			inflight.waiters.Add(-1)
			var zero T
			return zero, false, ctx.Err()
		case <-inflight.done:
			inflight.waiters.Add(-1)
			if ctx != nil {
				if err := ctx.Err(); err != nil {
					var zero T
					return zero, false, err
				}
			}
		}

		if inflight.canceled {
			// Leader was canceled, but this joiner's context is still valid.
			// Recover by executing or joining a fresh flight.
			return g.Do(ctx, key, fn)
		}

		g.coalesced.Add(1)
		return inflight.val, true, inflight.err
	}
	f := &flight[T]{
		done: make(chan struct{}),
		err:  errFlightAbandoned,
	}
	g.m[key] = f
	g.mu.Unlock()

	defer func() {
		g.mu.Lock()
		delete(g.m, key)
		g.mu.Unlock()
		f.doneOnce.Do(func() {
			close(f.done)
		})
	}()

	v, ferr := fn()
	f.val, f.err = v, ferr
	if (ctx != nil && ctx.Err() != nil) ||
		errors.Is(ferr, context.Canceled) ||
		errors.Is(ferr, context.DeadlineExceeded) {
		f.canceled = true
	}
	return v, false, ferr
}

// Coalesced reports how many callers joined an in-flight search instead of executing their own.
func (g *flightGroup[T]) Coalesced() int64 { return g.coalesced.Load() }

// Waiters reports how many joiners are currently waiting on key.
func (g *flightGroup[T]) Waiters(key string) int32 {
	g.mu.Lock()
	defer g.mu.Unlock()
	if f, ok := g.m[key]; ok {
		return f.waiters.Load()
	}
	return 0
}

type grepRecord struct {
	pattern          string
	matches          []GrepMatch
	filesScanned     int
	truncated        bool
	truncationReason string
	coalesced        bool
	errJSON          []byte
	isErr            bool
}

func (r *grepRecord) toOutput() ([]byte, bool) {
	if r.isErr {
		return r.errJSON, true
	}
	return okJSON(map[string]any{
		"pattern":           r.pattern,
		"matches":           r.matches,
		"match_count":       len(r.matches),
		"files_scanned":     r.filesScanned,
		"truncated":         r.truncated,
		"truncation_reason": r.truncationReason,
		"coalesced":         r.coalesced,
	}), false
}

func snapToRuneBoundary(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	s = s[:maxBytes]
	for i := 0; i < utf8.UTFMax && len(s) > 0; i++ {
		r, size := utf8.DecodeLastRuneInString(s)
		if r == utf8.RuneError && size == 1 {
			s = s[:len(s)-1]
			continue
		}
		break
	}
	return s
}

func upgradeTruncationReason(current, candidate string) string {
	precedence := func(r string) int {
		switch r {
		case "match_limit":
			return 4
		case "walk_budget":
			return 3
		case "file_size":
			return 2
		case "line_width":
			return 1
		default:
			return 0
		}
	}
	if precedence(candidate) > precedence(current) {
		return candidate
	}
	return current
}

// grep decodes, confines, and executes a Grep.
func (t *Toolset) grep(ctx context.Context, body []byte) ([]byte, bool) {
	RecordSubprocessAvoided()
	var a GrepArgs
	if r := decodeArgs(body, &a); r != nil {
		return r.JSON(), true
	}
	if r := a.Validate(); r != nil {
		return r.JSON(), true
	}
	if r := canceled(ctx); r != nil {
		return r.JSON(), true
	}
	re, err := regexp.Compile(a.Pattern)
	if err != nil {
		return refuse(CodeMalformed, "Grep: bad pattern: "+err.Error()).JSON(), true
	}
	root, r := t.searchRoot(a.Path)
	if r != nil {
		return r.JSON(), true
	}
	limit := t.limits.MaxMatches
	if a.MaxMatches > 0 && a.MaxMatches < limit {
		limit = a.MaxMatches
	}

	key := grepFlightKey(root.Abs, a.Pattern, a.Glob, limit)
	rec, shared, err := t.grepFlight.Do(ctx, key, func() (*grepRecord, error) {
		return t.executeGrep(ctx, a, re, root, limit), nil
	})
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return refuse(CodeCanceled, err.Error()).JSON(), true
		}
		return refuse(CodeIO, "Grep: in-flight search abandoned").JSON(), true
	}
	if rec.isErr {
		return rec.toOutput()
	}
	outRec := *rec
	outRec.coalesced = shared
	return outRec.toOutput()
}

func (t *Toolset) executeGrep(ctx context.Context, a GrepArgs, re *regexp.Regexp, root Resolved, limit int) *grepRecord {
	matches := make([]GrepMatch, 0, 16)
	truncated := false
	truncationReason := ""
	visited := 0
	walkErr := t.walk(ctx, root, func(abs, rel string) error {
		if a.Glob != "" {
			ok, err := filepath.Match(a.Glob, filepath.Base(rel))
			if err != nil {
				return &haltError{refuse(CodeMalformed, "Grep: bad glob: "+err.Error())}
			}
			if !ok {
				return nil
			}
		}
		visited++
		f, err := os.Open(abs)
		if err != nil {
			return nil // an unreadable file is skipped, never fatal to the whole search
		}
		data, readErr := io.ReadAll(io.LimitReader(f, t.limits.MaxReadBytes+1))
		_ = f.Close()
		if readErr != nil {
			return nil
		}
		if int64(len(data)) > t.limits.MaxReadBytes {
			data = data[:t.limits.MaxReadBytes]
			truncated = true
			truncationReason = upgradeTruncationReason(truncationReason, "file_size")
		}

		for i, line := range strings.Split(string(data), "\n") {
			cleanLine := strings.TrimRight(line, "\r")
			if !re.MatchString(cleanLine) {
				continue
			}
			if len(matches) >= limit {
				truncated = true
				truncationReason = upgradeTruncationReason(truncationReason, "match_limit")
				return errStopWalk
			}
			matchTruncated := false
			if len(cleanLine) > maxMatchLineBytes {
				loc := re.FindStringIndex(cleanLine)
				start := 0
				if len(loc) == 2 {
					matchLen := loc[1] - loc[0]
					if matchLen >= maxMatchLineBytes {
						start = loc[0]
					} else {
						mid := loc[0] + matchLen/2
						start = mid - maxMatchLineBytes/2
						if start < 0 {
							start = 0
						}
						if start+maxMatchLineBytes > len(cleanLine) {
							start = len(cleanLine) - maxMatchLineBytes
							if start < 0 {
								start = 0
							}
						}
						for start < loc[0] && !utf8.RuneStart(cleanLine[start]) {
							start++
						}
					}
				}
				cleanLine = snapToRuneBoundary(cleanLine[start:], maxMatchLineBytes)
				matchTruncated = true
				truncated = true
				truncationReason = upgradeTruncationReason(truncationReason, "line_width")
			}
			matches = append(matches, GrepMatch{
				File:      rel,
				Line:      i + 1,
				Text:      cleanLine,
				Truncated: matchTruncated,
			})
		}
		return nil
	})
	if walkErr != nil {
		var halt *haltError
		if errors.As(walkErr, &halt) {
			return &grepRecord{errJSON: halt.r.JSON(), isErr: true}
		}
		if errors.Is(walkErr, context.Canceled) || errors.Is(walkErr, context.DeadlineExceeded) {
			return &grepRecord{errJSON: refuse(CodeCanceled, "operation canceled").JSON(), isErr: true}
		}
		if errors.Is(walkErr, errWalkBudget) {
			truncated = true
			truncationReason = upgradeTruncationReason(truncationReason, "walk_budget")
		} else if !errors.Is(walkErr, errStopWalk) {
			return &grepRecord{errJSON: refuse(CodeIO, "Grep: "+walkErr.Error()).JSON(), isErr: true}
		}
	}
	return &grepRecord{
		pattern:          a.Pattern,
		matches:          matches,
		filesScanned:     visited,
		truncated:        truncated,
		truncationReason: truncationReason,
		isErr:            false,
	}
}

// globEngine serves Glob: a bounded path-shape search.
type globEngine struct{ t *Toolset }

// Caps reports no optional capabilities.
func (globEngine) Caps() []abi.Capability { return nil }

// WeightBearing declares Glob a deterministic classical tool engine.
func (globEngine) WeightBearing() bool { return false }

// Complete performs the listing.
func (e globEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	in := bytesOf(ctx, c.Args)
	out, isErr := e.t.glob(ctx, in)
	return result(ctx, c, in, out, isErr, EngineGlob), nil
}

type globRecord struct {
	pattern          string
	files            []string
	truncated        bool
	truncationReason string
	coalesced        bool
	errJSON          []byte
	isErr            bool
}

func (r *globRecord) toOutput() ([]byte, bool) {
	if r.isErr {
		return r.errJSON, true
	}
	return okJSON(map[string]any{
		"pattern":           r.pattern,
		"files":             r.files,
		"count":             len(r.files),
		"truncated":         r.truncated,
		"truncation_reason": r.truncationReason,
		"coalesced":         r.coalesced,
	}), false
}

// grepFlightKey constructs a length-prefixed singleflight key avoiding delimiter collisions.
func grepFlightKey(absRoot, pattern, glob string, limit int) string {
	return fmt.Sprintf("%d:%s\x00%d:%s\x00%d:%s\x00%d", len(absRoot), absRoot, len(pattern), pattern, len(glob), glob, limit)
}

// globFlightKey constructs a length-prefixed singleflight key avoiding delimiter collisions.
func globFlightKey(absRoot, pattern string) string {
	return fmt.Sprintf("%d:%s\x00%d:%s", len(absRoot), absRoot, len(pattern), pattern)
}

// glob decodes, confines, and executes a Glob. The pattern is matched against the path
// relative to the SEARCH ROOT, supporting recursive `**` wildcards anywhere in the pattern
// (e.g. "**/*.go", "src/**/*.go", or "internal/**/test_*.go").
func (t *Toolset) glob(ctx context.Context, body []byte) ([]byte, bool) {
	RecordSubprocessAvoided()
	var a GlobArgs
	if r := decodeArgs(body, &a); r != nil {
		return r.JSON(), true
	}
	if r := a.Validate(); r != nil {
		return r.JSON(), true
	}
	if r := canceled(ctx); r != nil {
		return r.JSON(), true
	}
	root, r := t.searchRoot(a.Path)
	if r != nil {
		return r.JSON(), true
	}

	key := globFlightKey(root.Abs, a.Pattern)
	rec, shared, err := t.globFlight.Do(ctx, key, func() (*globRecord, error) {
		return t.executeGlob(ctx, a, root), nil
	})
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return refuse(CodeCanceled, err.Error()).JSON(), true
		}
		return refuse(CodeIO, "Glob: in-flight search abandoned").JSON(), true
	}
	if rec.isErr {
		return rec.toOutput()
	}
	outRec := *rec
	outRec.coalesced = shared
	return outRec.toOutput()
}

func (t *Toolset) executeGlob(ctx context.Context, a GlobArgs, root Resolved) *globRecord {
	g, err := compileGlob(a.Pattern)
	if err != nil {
		return &globRecord{errJSON: refuse(CodeMalformed, "Glob: bad pattern: "+err.Error()).JSON(), isErr: true}
	}
	files := make([]string, 0, 16)
	truncated := false
	truncationReason := ""
	walkErr := t.walk(ctx, root, func(abs, rel string) error {
		sub := rel
		if root.Rel != "" && root.Rel != "." {
			sub = strings.TrimPrefix(strings.TrimPrefix(rel, root.Rel), "/")
		}
		if !g.match(sub) {
			return nil
		}
		if len(files) >= t.limits.MaxEntries {
			truncated = true
			truncationReason = "match_limit"
			return errStopWalk
		}
		files = append(files, rel)
		return nil
	})
	if walkErr != nil {
		var halt *haltError
		if errors.As(walkErr, &halt) {
			return &globRecord{errJSON: halt.r.JSON(), isErr: true}
		}
		if errors.Is(walkErr, context.Canceled) || errors.Is(walkErr, context.DeadlineExceeded) {
			return &globRecord{errJSON: refuse(CodeCanceled, "operation canceled").JSON(), isErr: true}
		}
		if errors.Is(walkErr, errWalkBudget) {
			truncated = true
			truncationReason = upgradeTruncationReason(truncationReason, "walk_budget")
		} else if !errors.Is(walkErr, errStopWalk) {
			return &globRecord{errJSON: refuse(CodeIO, "Glob: "+walkErr.Error()).JSON(), isErr: true}
		}
	}
	return &globRecord{
		pattern:          a.Pattern,
		files:            files,
		truncated:        truncated,
		truncationReason: truncationReason,
		isErr:            false,
	}
}

// globPattern holds compiled segments for glob matching with recursive ** wildcards.
type globPattern struct {
	raw      string
	segments []string
}

func compileGlob(pattern string) (*globPattern, error) {
	pattern = filepath.ToSlash(pattern)
	pattern = strings.TrimPrefix(pattern, "./")
	pattern = strings.TrimPrefix(pattern, "/")
	pattern = strings.TrimSuffix(pattern, "/")

	patSegs := splitPathSegments(pattern)
	for _, seg := range patSegs {
		if seg != "**" {
			if _, err := path.Match(seg, ""); err != nil {
				return nil, err
			}
		}
	}
	return &globPattern{raw: pattern, segments: patSegs}, nil
}

func splitPathSegments(p string) []string {
	if p == "" {
		return nil
	}
	raw := strings.Split(p, "/")
	segs := make([]string, 0, len(raw))
	for _, s := range raw {
		if s != "" && s != "." {
			segs = append(segs, s)
		}
	}
	return segs
}

func (g *globPattern) match(pathStr string) bool {
	pathStr = filepath.ToSlash(pathStr)
	pathStr = strings.TrimPrefix(pathStr, "./")
	pathStr = strings.TrimPrefix(pathStr, "/")
	pathSegs := splitPathSegments(pathStr)
	matched, _ := matchSegments(g.segments, pathSegs)
	return matched
}

// matchGlob reports whether pathStr matches the glob pattern. It supports ** recursive
// directory wildcards anywhere in the pattern (e.g. "src/**/*.go", "internal/**/test_*.go",
// or "**/*.go").
func matchGlob(pattern, pathStr string) (bool, error) {
	g, err := compileGlob(pattern)
	if err != nil {
		return false, err
	}
	return g.match(pathStr), nil
}

func matchSegments(patSegs, pathSegs []string) (bool, error) {
	if len(patSegs) == 0 && len(pathSegs) == 0 {
		return true, nil
	}
	if len(patSegs) == 0 {
		return false, nil
	}
	if len(pathSegs) == 0 {
		for _, seg := range patSegs {
			if seg != "**" {
				return false, nil
			}
		}
		return true, nil
	}

	if patSegs[0] == "**" {
		for len(patSegs) > 1 && patSegs[1] == "**" {
			patSegs = patSegs[1:]
		}
		matched, err := matchSegments(patSegs[1:], pathSegs)
		if matched || err != nil {
			return matched, err
		}
		for k := 0; k < len(pathSegs); k++ {
			matched, err := matchSegments(patSegs[1:], pathSegs[k+1:])
			if matched || err != nil {
				return matched, err
			}
		}
		return false, nil
	}

	ok, err := path.Match(patSegs[0], pathSegs[0])
	if err != nil || !ok {
		return false, err
	}
	return matchSegments(patSegs[1:], pathSegs[1:])
}

// searchRoot resolves and confines the subtree a search runs over, defaulting to the
// workspace root. A search root is confined by exactly the same resolve() every file
// operand crosses — a search is a read of many files, and it gets the same boundary.
func (t *Toolset) searchRoot(path string) (Resolved, *Refusal) {
	if strings.TrimSpace(path) == "" {
		return Resolved{Abs: t.root, Rel: "."}, nil
	}
	return t.resolve(path)
}

// errStopWalk ends a walk early because the caller has all the rows it asked for. Not an
// error condition — the callers translate it into truncated=true.
var errStopWalk = errors.New("codetools: walk stopped")

// errWalkBudget ends a walk because it visited MaxWalkFiles entries. Distinct from
// errStopWalk so the result can say the SEARCH was bounded rather than the result set.
var errWalkBudget = errors.New("codetools: walk budget exhausted")

// haltError carries a caller-supplied Refusal out of a walk (a bad pattern discovered
// mid-walk). Distinguished from the two sentinels so a genuine refusal is reported as
// itself rather than folded into "truncated".
type haltError struct{ r *Refusal }

// Error renders the carried refusal.
func (h *haltError) Error() string { return h.r.Error() }

// walk visits every regular file under root, calling fn with its absolute and
// workspace-relative paths. It skips dot-directories (.git, .dos, .cache — noise a coding
// search never wants and the trees the toolset protects), does not follow symlinks, and
// enforces the visit budget and ctx cancellation between entries.
func (t *Toolset) walk(ctx context.Context, root Resolved, fn func(abs, rel string) error) error {
	if t.searchHook != nil {
		t.searchHook()
	}
	visited := 0
	return filepath.WalkDir(root.Abs, func(abs string, d fs.DirEntry, err error) error {
		if err != nil {
			// An unreadable directory is skipped, not fatal: a permission fault deep in
			// a tree must not turn a whole search into a refusal.
			if d != nil && d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		name := d.Name()
		if d.IsDir() {
			if abs != root.Abs && strings.HasPrefix(name, ".") {
				return fs.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			// Symlinks, devices, sockets: never opened. WalkDir does not descend a
			// symlinked directory, and this drops symlinked FILES, so a link is never a
			// door out of the workspace even for a read.
			return nil
		}
		visited++
		if visited > t.limits.MaxWalkFiles {
			return errWalkBudget
		}
		rel, relErr := filepath.Rel(t.root, abs)
		if relErr != nil || hasDotDotPrefix(rel) {
			return nil
		}
		return fn(abs, filepath.ToSlash(rel))
	})
}
