package codetools

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
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

type grepRecord struct {
	pattern          string
	matches          []GrepMatch
	filesScanned     int
	truncated        bool
	truncationReason string
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
	}), false
}

func snapToRuneBoundary(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	s = s[:maxBytes]
	for len(s) > 0 && !utf8.RuneStart(s[len(s)-1]) {
		s = s[:len(s)-1]
	}
	if len(s) > 0 {
		r, size := utf8.DecodeLastRuneInString(s)
		if r == utf8.RuneError && size == 1 {
			s = s[:len(s)-1]
		}
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
	rec := t.executeGrep(ctx, a, re, root, limit)
	return rec.toOutput()
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
		data, err := os.ReadFile(abs)
		if err != nil {
			return nil // an unreadable file is skipped, never fatal to the whole search
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
				cleanLine = snapToRuneBoundary(cleanLine, maxMatchLineBytes)
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

// glob decodes, confines, and executes a Glob. The pattern is matched against the path
// relative to the SEARCH ROOT and, for a leading-`**` pattern, against the base name too,
// so the familiar "**/*.go" spelling works without the caller having to know how deep the
// tree is.
func (t *Toolset) glob(ctx context.Context, body []byte) ([]byte, bool) {
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
	pattern := a.Pattern
	base := strings.TrimPrefix(pattern, "**/")
	files := make([]string, 0, 16)
	truncated := false
	truncationReason := ""
	walkErr := t.walk(ctx, root, func(abs, rel string) error {
		sub := rel
		if root.Rel != "" && root.Rel != "." {
			sub = strings.TrimPrefix(strings.TrimPrefix(rel, root.Rel), "/")
		}
		ok, err := filepath.Match(pattern, sub)
		if err != nil {
			return &haltError{refuse(CodeMalformed, "Glob: bad pattern: "+err.Error())}
		}
		if !ok && base != pattern {
			ok, _ = filepath.Match(base, filepath.Base(sub))
		}
		if !ok {
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
			return halt.r.JSON(), true
		}
		if errors.Is(walkErr, errWalkBudget) {
			truncated = true
			if truncationReason == "" {
				truncationReason = "walk_budget"
			}
		} else if !errors.Is(walkErr, errStopWalk) {
			return refuse(CodeIO, "Glob: "+walkErr.Error()).JSON(), true
		}
	}
	return okJSON(map[string]any{
		"pattern":           a.Pattern,
		"files":             files,
		"count":             len(files),
		"truncated":         truncated,
		"truncation_reason": truncationReason,
	}), false
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
