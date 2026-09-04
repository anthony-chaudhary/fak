// Package wipfence applies and removes the shared-trunk WIP build fence.
//
// Work-in-progress Go that cannot yet compile against committed symbols is
// kept on the trunk behind a `//go:build wip_<slug>` constraint as the file's
// very first line, followed by a blank line, then the package clause — so the
// default `go build` stays green while the WIP lives on disk. This package is
// the pure text engine behind that convention: callers hand it file content
// and a slug and get transformed content back. No git, no filesystem, no I/O.
//
// Invariant: WIP fence transformations are fail-closed and idempotent. Existing
// non-WIP build constraints and mismatched WIP tags are refused rather than clobbered.
// Guard: Any malformed or conflicting constraint state causes operations to fail-closed
// with an explicit error to prevent silent data or build tag loss.
package wipfence

import (
	"fmt"
	"strings"
)

// constraintPrefix marks any Go build constraint line, wip_ or not.
const constraintPrefix = "//go:build"

// tagPrefix is the prefix every WIP fence tag carries.
const tagPrefix = "wip_"

// SlugFromPath derives a valid build-tag slug from a file path's base name:
// strip directories and the trailing ".go", lowercase, replace every run of
// characters outside [a-z0-9] with a single "_", trim leading/trailing "_".
// If the result is empty or does not start with a letter or "_", prefix "f_".
// Example: "internal/Foo/new-thing.go" -> "new_thing"; "123.go" -> "f_123".
func SlugFromPath(path string) string {
	base := path
	if i := strings.LastIndexAny(base, `/\`); i >= 0 {
		base = base[i+1:]
	}
	base = strings.TrimSuffix(base, ".go")
	slug := sanitizeSlug(base)
	if slug == "" {
		return "f_"
	}
	return slug
}

// Fence returns content with a `//go:build wip_<slug>` fence prepended as the
// very first line, followed by a blank line, then the original content.
// The slug is sanitized the same way SlugFromPath sanitizes, and one leading
// "wip_" is stripped first so passing either "foo" or "wip_foo" yields the
// same wip_foo tag; a slug that sanitizes to empty is an error. Idempotent:
// content already fenced with this exact tag returns (content, false, nil).
// A different leading `//go:build` constraint — another wip_ slug or any
// non-wip constraint — is never clobbered and returns an error. changed
// reports whether content was modified.
func Fence(content, slug string) (out string, changed bool, err error) {
	norm := sanitizeSlug(trimWipPrefix(slug))
	if norm == "" {
		return content, false, fmt.Errorf("slug %q sanitizes to an empty build-tag slug", slug)
	}
	tag := tagPrefix + norm
	first, _, _ := strings.Cut(content, "\n")
	if got, ok := parseFenceLine(first); ok {
		if got == tag {
			return content, false, nil
		}
		return content, false, fmt.Errorf("content is already fenced with %q, not %q; unfence it first", got, tag)
	}
	if trimmed := strings.TrimSpace(first); strings.HasPrefix(trimmed, constraintPrefix) {
		return content, false, fmt.Errorf("content already starts with build constraint %q; refusing to replace it", trimmed)
	}
	return constraintPrefix + " " + tag + "\n\n" + content, true, nil
}

// Unfence removes a leading `//go:build wip_<...>` fence line (any wip_ slug,
// so the caller need not remember it) plus the single blank line that follows
// it, restoring the original file. It only removes a wip_ build tag — a
// non-wip `//go:build` constraint is left untouched (changed=false).
// Idempotent: if there is no leading wip_ fence, it returns
// (content, false, nil).
func Unfence(content string) (out string, changed bool, err error) {
	first, rest, hasNL := strings.Cut(content, "\n")
	if _, ok := parseFenceLine(first); !ok {
		return content, false, nil
	}
	if !hasNL {
		return "", true, nil
	}
	if next, after, _ := strings.Cut(rest, "\n"); strings.TrimSpace(next) == "" {
		return after, true, nil
	}
	return rest, true, nil
}

// IsFenced reports whether content's first line is a `//go:build wip_<slug>`
// tag, and returns the full tag body after "//go:build ", e.g. "wip_foo".
// If not fenced, ok is false.
func IsFenced(content string) (tag string, ok bool) {
	first, _, _ := strings.Cut(content, "\n")
	return parseFenceLine(first)
}

// sanitizeSlug normalizes a raw name into a build-tag slug: lowercase, every
// run of characters outside [a-z0-9] collapsed to a single "_", leading and
// trailing "_" trimmed, and an "f_" prefix added when the result would start
// with a digit. It returns "" when nothing survives; callers decide whether
// that is an error (Fence) or falls back to "f_" (SlugFromPath).
func sanitizeSlug(s string) string {
	var b strings.Builder
	pendingSep := false
	for _, r := range strings.ToLower(s) {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			if pendingSep && b.Len() > 0 {
				b.WriteByte('_')
			}
			pendingSep = false
			b.WriteRune(r)
		default:
			pendingSep = true
		}
	}
	out := b.String()
	if out == "" {
		return ""
	}
	if out[0] >= '0' && out[0] <= '9' {
		out = "f_" + out
	}
	return out
}

// trimWipPrefix strips one leading "wip_" (any case) so a caller may pass
// either the bare slug or the full tag without producing a wip_wip_ tag.
func trimWipPrefix(s string) string {
	if len(s) >= len(tagPrefix) && strings.EqualFold(s[:len(tagPrefix)], tagPrefix) {
		return s[len(tagPrefix):]
	}
	return s
}

// parseFenceLine reports whether line's trimmed form is exactly
// "//go:build wip_<ident>" with <ident> matching [A-Za-z_][A-Za-z0-9_]*,
// and returns the full tag body ("wip_<ident>").
func parseFenceLine(line string) (tag string, ok bool) {
	trimmed := strings.TrimSpace(line)
	expr, found := strings.CutPrefix(trimmed, constraintPrefix+" ")
	if !found || !strings.HasPrefix(expr, tagPrefix) {
		return "", false
	}
	if !isTagIdent(strings.TrimPrefix(expr, tagPrefix)) {
		return "", false
	}
	return expr, true
}

// isTagIdent reports whether s matches [A-Za-z_][A-Za-z0-9_]*.
func isTagIdent(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'):
		case c >= '0' && c <= '9':
			if i == 0 {
				return false
			}
		default:
			return false
		}
	}
	return len(s) > 0
}
