package main

import (
	"os"
	"runtime"
	"strings"
)

// guardScratchpadRootsValue expands a FAK_GUARD_SCRATCHPAD_ROOTS list so that every
// declared root is accompanied by the OTHER host spelling of the SAME directory.
//
// WHY. The adjudicator's recursive-delete and out-of-tree-write gates prove
// containment by string comparison against these roots (isUnder, over paths whose
// backslashes canonicalizeArgValue has already folded to '/'). On Windows the same
// directory has two live spellings: the drive-letter form os.TempDir() reports
// (`C:/…/Temp/claude`) and the MSYS form Git Bash — the shell the Bash tool actually
// runs — uses for it (`/c/…/Temp/claude`). Those two strings never compare equal, so
// declaring only one of them left the carve-out dead for whichever shell used the
// other.
//
// That was not theoretical. The recursive-delete rule was the largest remaining
// refusal class in the guard-audit corpus (49 of 103 POLICY_BLOCKs), every one of
// them dated twelve days AFTER the scratch carve-out shipped, and a live A/B probe
// of one throwaway directory inside the session's own harness scratchpad reproduced
// it exactly: the `/c/…` spelling was hard-denied while the byte-equivalent `C:/…`
// spelling was downgraded to the reversibility preview-confirm gate. Same directory,
// same session, two verdicts — decided by nothing but which shell spelled the path.
// Deleting the scratch tree the harness itself designates as throwaway is routine
// work, and refusing it prevented nothing that could be lost.
//
// WHY THIS IS NOT A WIDENING. An alias is a second NAME for a directory that is
// already declared, never an additional directory: scratchpadRootAlias only rewrites
// the root prefix of a path it was handed, so the set of files reachable through the
// carve-out is unchanged. The narrow default (`<temp>/claude`, never the whole OS
// temp directory) therefore stays exactly as narrow as loadGuardCapabilityFloor
// declares it. Operator-declared roots are expanded the same way, because an operator
// declares a DIRECTORY, not a spelling of one.
func guardScratchpadRootsValue(declared string) string {
	var out []string
	seen := map[string]bool{}
	add := func(p string) {
		p = strings.TrimSpace(p)
		if p == "" || seen[strings.ToLower(p)] {
			return
		}
		seen[strings.ToLower(p)] = true
		out = append(out, p)
	}
	for _, root := range strings.Split(declared, string(os.PathListSeparator)) {
		add(root)
		add(scratchpadRootAlias(root))
	}
	return strings.Join(out, string(os.PathListSeparator))
}

// scratchpadRootAlias returns the other host spelling of the same directory, or ""
// when the path has none: a drive-letter root (`C:\t\claude`) aliases to the MSYS
// form (`/c/t/claude`) and an MSYS root aliases back to the drive-letter form.
//
// Windows-only BY CONSTRUCTION, and not merely because the duality is a Windows one.
// FAK_GUARD_SCRATCHPAD_ROOTS is an OS-path-list, so its separator is ':' on POSIX —
// emitting a `C:`-prefixed alias there would split into the two bogus roots "C" and
// "/t/claude", and that second one would hand the carve-out a top-level directory
// nobody declared. Returning "" off Windows makes that unrepresentable rather than
// merely unlikely.
func scratchpadRootAlias(p string) string {
	if runtime.GOOS != "windows" {
		return ""
	}
	p = strings.ReplaceAll(strings.TrimSpace(p), `\`, "/")
	switch {
	case len(p) >= 2 && p[1] == ':' && isASCIIDriveLetter(p[0]):
		// `C:/t/claude` -> `/c/t/claude`
		return "/" + strings.ToLower(p[:1]) + strings.TrimRight(p[2:], "/")
	case len(p) >= 2 && p[0] == '/' && isASCIIDriveLetter(p[1]) && (len(p) == 2 || p[2] == '/'):
		// `/c/t/claude` -> `C:/t/claude`
		return strings.ToUpper(p[1:2]) + ":" + strings.TrimRight(p[2:], "/")
	}
	return ""
}

func isASCIIDriveLetter(b byte) bool {
	return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z')
}
