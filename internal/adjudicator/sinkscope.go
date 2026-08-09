package adjudicator

import "strings"

// sinkscope.go is the per-tool capability/sink declaration table (#5898): the
// split that lets an ACTION rule stop matching an argument that merely
// DESCRIBES a dangerous action, while a DATA rule (secret/DLP shapes) keeps
// seeing every argument. Clean-room after the capability/sink model studied in
// docs/notes/CONCEPT-STUDY-AGNT-2026-08-08.md (INSPIRE, not INTEGRATE).
//
// The problem it closes: commandText joins a tool's "argv"/"args" slice into
// one string for the reversibility families to scan, but a slice element that
// is PAYLOAD — a commit message, an issue body — carries no quote bytes, so
// the quote-stripping mention protection (#2752) cannot see it. A newline or
// `&&` inside the prose manufactures a fresh command segment and its first
// word lands at the head position: a commit message saying "rm -rf build is no
// longer needed" classified as an irreversible filesystem destroy (the
// recorded "commit message mentions rm" over-block class,
// internal/guardaccuracy/complaints_test.go).
//
// The split is the whole point, in both directions:
//
//   - An ACTION rule (the reversibility families) reads only the arguments a
//     tool declares as sink-bearing — the keys that select an operation,
//     destination, executable, query, or path.
//   - A DATA rule never consults this table. The preview secret redaction in
//     this package and every DLP-shaped gate elsewhere (internal/hooks'
//     public-leak scan, the secretgate result rung) keep their full-argument
//     view: a credential is dangerous wherever it appears, so narrowing a data
//     rule to a sink would be a real leak, not a false-positive fix.
//
// FAIL-CLOSED BY CONSTRUCTION: a tool with no declaration keeps the old
// full-argument scan. Narrowing is opt-in per tool, so an undeclared tool can
// never silently loosen, and a wrong declaration is reviewable in one place.

// toolSinkDecl declares one tool's argument surface for action-rule scanning.
type toolSinkDecl struct {
	// capability names, in one word, what the tool can do at all — provenance
	// for the reviewer deciding whether the sink set below is complete.
	capability string
	// sinkKeys are the argument keys whose values reach an execution sink and
	// therefore stay visible to action rules. Keys absent here are payload for
	// THIS tool and are invisible to the reversibility families. An empty set
	// is a valid declaration: it says no argument selects an operation (the
	// operation is fixed by the tool's name), so families that anchor on the
	// tool NAME (toolContains) still classify while argument prose never does.
	sinkKeys map[string]bool
}

// toolSinkDecls is keyed by the LOWERED bare tool name. An MCP-prefixed
// spelling (mcp__git__git_commit) resolves through its last "__" segment, the
// same bare name the toolContains matchers see, so the declaration follows the
// tool across harness prefixes instead of silently lapsing to the full scan.
//
// The first declarations are the tools through which commit messages and
// issue bodies flow — prose that routinely NAMES the operations this package
// polices. Their write is real (git_commit mutates history, create_issue is
// outward-facing via the issue-create-tool family's name match), but no
// argument of theirs is a command an action rule should lex.
var toolSinkDecls = map[string]toolSinkDecl{
	// The commit tool: "message" is prose payload; the repo/path selectors are
	// the only arguments that reach a sink. None is command-bearing, so the
	// reversibility families read no command text from a declared commit call.
	"git_commit": {
		capability: "vcs-write",
		sinkKeys:   map[string]bool{"repo_path": true, "path": true, "cwd": true},
	},
	// The issue-authoring tools: title/body/labels are prose payload and the
	// operation is fixed by the tool name. The empty sink set leaves the
	// issue-create-tool family's toolContains escalation (and its `fak issue
	// create` redirect) fully intact — the call IS outward-facing — while the
	// body's prose can no longer drag in an unrelated family.
	"create_issue": {capability: "tracker-write", sinkKeys: map[string]bool{}},
	"issue_create": {capability: "tracker-write", sinkKeys: map[string]bool{}},
}

// sinkDeclFor resolves a tool name to its declaration: the exact lowered name
// first, then the bare last "__" segment of an MCP-prefixed spelling.
func sinkDeclFor(tool string) (toolSinkDecl, bool) {
	lower := strings.ToLower(tool)
	if d, ok := toolSinkDecls[lower]; ok {
		return d, true
	}
	if i := strings.LastIndex(lower, "__"); i >= 0 {
		if d, ok := toolSinkDecls[lower[i+2:]]; ok {
			return d, true
		}
	}
	return toolSinkDecl{}, false
}

// sinkScopedArgs returns the argument view an action rule may read under decl:
// only sink-declared keys survive. Key comparison is lowered, matching the
// case-insensitive posture of the rest of the argument plumbing.
func sinkScopedArgs(decl toolSinkDecl, args map[string]any) map[string]any {
	if len(args) == 0 {
		return args
	}
	out := make(map[string]any, len(args))
	for k, v := range args {
		if decl.sinkKeys[strings.ToLower(k)] {
			out[k] = v
		}
	}
	return out
}

// actionCommandText is the command text the ACTION rules read. A declared tool
// contributes only its sink-declared arguments; an undeclared tool keeps the
// full commandText scan (the fail-closed default). Data-shaped surfaces — the
// preview rendering with its secret redaction, the confirm-token hash — keep
// calling commandText/argsForToken on the FULL argument set and never route
// through here.
func actionCommandText(tool string, args map[string]any) string {
	if decl, ok := sinkDeclFor(tool); ok {
		return commandText(sinkScopedArgs(decl, args))
	}
	return commandText(args)
}
