package wipattr

import "strings"

// ParseHunks turns unified `git diff` output into per-file Hunks, keeping only the
// edit payload (+/- lines) of each '@@' block. It is the pure boundary the cmd shell
// (cmd/fak/wip.go) uses on both the live working-tree diff (`git diff`) and each
// checkpoint's diff (`git diff <obj>^ <obj>`), so attribution never touches git here.
//
// Rules, matching git's unified format:
//   - the current file is taken from a '+++ b/<path>' line (falling back to the
//     'diff --git a/x b/x' header's b-side when a diff has no body, e.g. pure mode
//     changes — those yield no hunks and are simply skipped);
//   - a '@@' line opens a new hunk;
//   - within a hunk, lines starting with '+' or '-' (but NOT the '+++'/'---' headers)
//     are edit payload; context (' '), '\ No newline at end of file', and blank
//     separators are ignored;
//   - a hunk with no payload (all context) is dropped — it names no edit to attribute.
func ParseHunks(diff string) []Hunk {
	var out []Hunk
	var file string
	var cur *Hunk
	flush := func() {
		if cur != nil && len(cur.Edit) > 0 {
			out = append(out, *cur)
		}
		cur = nil
	}
	for _, line := range strings.Split(diff, "\n") {
		switch {
		case strings.HasPrefix(line, "diff --git "):
			flush()
			file = bSideFromGitHeader(line)
		case strings.HasPrefix(line, "+++ "):
			flush()
			file = pathFromFileHeader(line)
		case strings.HasPrefix(line, "--- "):
			// old-file header; ignore (the +++ line carries the authoritative path).
		case strings.HasPrefix(line, "@@"):
			flush()
			cur = &Hunk{File: file}
		case cur != nil && (strings.HasPrefix(line, "+") || strings.HasPrefix(line, "-")):
			cur.Edit = append(cur.Edit, line)
		default:
			// context, '\ No newline...', or preamble between hunks: not payload.
		}
	}
	flush()
	return out
}

// pathFromFileHeader extracts the path from a '+++ b/<path>' line, stripping the
// 'b/' prefix git adds and the '\t' timestamp suffix some diffs carry. A '/dev/null'
// (deletion target) yields "".
func pathFromFileHeader(line string) string {
	rest := strings.TrimSpace(strings.TrimPrefix(line, "+++ "))
	if i := strings.IndexByte(rest, '\t'); i >= 0 {
		rest = rest[:i]
	}
	if rest == "/dev/null" {
		return ""
	}
	return strings.TrimPrefix(rest, "b/")
}

// bSideFromGitHeader extracts the b-side path from a 'diff --git a/x b/x' header,
// used only when a diff body has no '+++' line to key on.
func bSideFromGitHeader(line string) string {
	fields := strings.Fields(line)
	if len(fields) < 4 {
		return ""
	}
	return strings.TrimPrefix(fields[len(fields)-1], "b/")
}
