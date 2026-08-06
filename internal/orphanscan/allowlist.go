package orphanscan

// allowlist.go — the gate half of orphanscan (#3167). ScanDir answers "what is
// unreferenced"; this file answers "which of those may a green tree still carry".
//
// The scanner alone cannot be a gate. Run over a real repo it reports standing debt as
// well as fresh mistakes, so asserting on a raw finding count either reds the trunk on
// day one or has to be tuned down to a number nobody can defend. A gate needs a
// RATCHET: a checked-in set of known, reasoned exceptions, so the only thing that can
// turn the tree red is an orphan nobody has accounted for — a NEW one.
//
// There are deliberately two suppression tiers. //orphanscan:keep (honored inside
// ScanDir) is the local one and the one to prefer: it lives on the func, travels with
// the code, and is visible to whoever reads the definition. This file is the second
// tier, for what the first cannot express — a func whose package is held by another
// loop's lane lease (so the source cannot be edited at all right now), and the standing
// baseline that lets the ratchet exist. Every entry carries a required reason, because
// an unexplained suppression is how a ratchet quietly becomes a rubber stamp.

import (
	_ "embed"
	"fmt"
	"path"
	"sort"
	"strings"
)

//go:embed keep_allowlist.txt
var keepAllowlistSrc string

// KeepEntry is one checked-in suppression: a func in a package that the gate will not
// fail on, plus the reason it is exempt. Reason is never empty — ParseKeepAllowlist
// refuses an entry without one.
type KeepEntry struct {
	Pkg    string // repo-relative, slash-separated package dir, e.g. "cmd/fak"
	Func   string // the unexported func name, e.g. "cmdRunaway"
	Reason string // why it is exempt; required
}

// Key is the identity a finding is matched on: package dir + func name. It is
// deliberately NOT file:line — a func that moves between files in its own package, or
// down the file as neighbours grow, is the same exemption and must not silently lapse.
func (k KeepEntry) Key() string { return k.Pkg + ":" + k.Func }

// String renders the entry in the on-disk line format, so a gate can print the exact
// line a maintainer would need to add or remove.
func (k KeepEntry) String() string { return fmt.Sprintf("%s %s # %s", k.Pkg, k.Func, k.Reason) }

// KeepAllowlist parses the checked-in keep_allowlist.txt embedded in the binary. An
// error means the file itself is malformed, which is a hard failure: a gate that cannot
// read its own exemption list must not silently fall back to "exempt nothing" (that
// reds the tree for the wrong reason) or to "exempt everything" (that is no gate).
func KeepAllowlist() ([]KeepEntry, error) { return ParseKeepAllowlist(keepAllowlistSrc) }

// ParseKeepAllowlist reads the allowlist line format:
//
//	<package-dir> <func-name> # <reason>
//
// Blank lines and lines whose first non-space rune is '#' are comments. Entries are
// returned sorted by Key. A missing reason, a wrong field count, or a duplicate key is
// an error naming the 1-based line, so a bad edit is reported where it was made.
func ParseKeepAllowlist(src string) ([]KeepEntry, error) {
	var out []KeepEntry
	seen := map[string]int{}
	for i, raw := range strings.Split(src, "\n") {
		lineNo := i + 1
		line := strings.TrimSpace(strings.TrimSuffix(raw, "\r"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		decl, reason, hasReason := strings.Cut(line, "#")
		reason = strings.TrimSpace(reason)
		if !hasReason || reason == "" {
			return nil, fmt.Errorf("keep_allowlist.txt:%d: entry %q has no reason (format: <package-dir> <func-name> # <reason>)", lineNo, line)
		}
		fields := strings.Fields(decl)
		if len(fields) != 2 {
			return nil, fmt.Errorf("keep_allowlist.txt:%d: want exactly <package-dir> <func-name> before '#', got %d field(s) in %q", lineNo, len(fields), strings.TrimSpace(decl))
		}
		e := KeepEntry{Pkg: path.Clean(filepathToSlash(fields[0])), Func: fields[1], Reason: reason}
		if prev, dup := seen[e.Key()]; dup {
			return nil, fmt.Errorf("keep_allowlist.txt:%d: duplicate entry %s (first seen on line %d)", lineNo, e.Key(), prev)
		}
		seen[e.Key()] = lineNo
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key() < out[j].Key() })
	return out, nil
}

// filepathToSlash normalises a hand-typed backslash path in the allowlist to the slash
// form ScanDir's relPrefix uses, so a Windows-authored entry still matches.
func filepathToSlash(p string) string { return strings.ReplaceAll(p, "\\", "/") }

// Gate partitions a scan against the allowlist and is the whole gate decision:
//
//   - unexpected — orphans with no matching allowlist entry. These are the NEW ones; a
//     non-empty slice is what turns a gate red.
//   - unused — allowlist entries for a package that WAS scanned but that no longer
//     correspond to a live orphan, i.e. the exemption has been earned back and the line
//     can be pruned.
//
// scanned is the set of package dirs actually walked. It exists so `unused` stays
// honest under a partial scan: without it, every entry for a package this run did not
// look at would be reported as stale, and a caller scanning one package would be told
// to delete the rest of the file. Entries for an unscanned package are simply not
// judged. Both results are sorted for a stable, diffable report.
func Gate(scanned []string, orphans []Orphan, keep []KeepEntry) (unexpected []Orphan, unused []KeepEntry) {
	allowed := make(map[string]bool, len(keep))
	for _, k := range keep {
		allowed[k.Key()] = true
	}
	live := make(map[string]bool, len(orphans))
	for _, o := range orphans {
		live[orphanKey(o)] = true
	}
	inScan := make(map[string]bool, len(scanned))
	for _, s := range scanned {
		inScan[path.Clean(filepathToSlash(s))] = true
	}

	for _, o := range orphans {
		if !allowed[orphanKey(o)] {
			unexpected = append(unexpected, o)
		}
	}
	for _, k := range keep {
		if inScan[k.Pkg] && !live[k.Key()] {
			unused = append(unused, k)
		}
	}

	sort.Slice(unexpected, func(i, j int) bool { return orphanKey(unexpected[i]) < orphanKey(unexpected[j]) })
	sort.Slice(unused, func(i, j int) bool { return unused[i].Key() < unused[j].Key() })
	return unexpected, unused
}

// orphanKey derives the allowlist identity of a finding: the dir of its File (which
// ScanDir already reports with the package's relPrefix) plus the func name.
func orphanKey(o Orphan) string { return path.Dir(filepathToSlash(o.File)) + ":" + o.Name }

// SuppressionHint renders the exact remedies for an unexpected orphan, in the order a
// maintainer should consider them: wire it (the usual real answer), delete it, or — if
// it truly is reachable in a way a syntactic scan cannot see — suppress it, local tier
// first. A gate that only says "no" costs a reader a lookup every time it fires.
func SuppressionHint(o Orphan) string {
	return fmt.Sprintf("wire it, delete it, add `//orphanscan:keep <reason>` to its doc comment, "+
		"or add `%s %s # <reason>` to internal/orphanscan/keep_allowlist.txt",
		path.Dir(filepathToSlash(o.File)), o.Name)
}
