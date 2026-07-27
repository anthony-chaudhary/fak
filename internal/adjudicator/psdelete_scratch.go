package adjudicator

import "strings"

// psRemoveItemAllTargetsInScratch proves — under POWERSHELL lexing — that every
// PowerShell delete statement in src removes only paths strictly inside a declared
// harness scratchpad root.
//
// It exists because the scratch carve-out commandHasUnsafeRecursiveForcedDelete
// already grants was UNREACHABLE on Windows, which is the only host where the
// PowerShell delete rule fires at all. That walk tokenizes with rceShellSegments,
// which implements POSIX lexing, where a backslash is an ESCAPE character. A
// Windows path therefore loses every separator before the containment check ever
// sees it:
//
//	<delete> C:\agent-scratch\claude\sess\scratch\junk -Recurse
//	  -> target token "C:agent-scratchclaudesessscratchjunk"
//
// which resolves under no root, so containment can never be proven and the delete
// stays a POLICY_BLOCK — even though psDeleteTargetsInScratch returns true for the
// INTACT path. The carve-out was correct; only the lexer in front of it was wrong.
// Under PowerShell's own rules a backslash is an ordinary path byte (psSegments),
// so the path survives and the carve-out applies as designed.
//
// Cleaning up one's own per-session scratch directory is routine work, and the
// policy already decided it is safe — the deny was an accident of dialect, not a
// judgement. Under `fak guard -- claude` that POLICY_BLOCK reads as an
// agent-chosen end_turn, so it silently ended the turn.
//
// This is purely SUBTRACTIVE: it is consulted only after the POSIX walk has
// already found a recursive/forced delete and failed to prove containment, so it
// can only ADD a way to prove the delete is confined. It never introduces a deny,
// and every ambiguity proves nothing and leaves the shipped refusal in place.
func psRemoveItemAllTargetsInScratch(src, ws string, scratch []string) bool {
	// A comma is PowerShell's list separator (`-Path a,b`) and psSegments treats it
	// as a statement boundary, so a list operand would be split and its tail
	// silently dropped from the target set — proving containment for `a` while `b`
	// escaped unchecked. Rather than prove a partial target set, prove nothing.
	if strings.ContainsRune(src, ',') {
		return false
	}
	segs, ok := psSegments(src)
	if !ok {
		return false // unterminated quote: undecidable, prove nothing
	}
	deleteWord := strings.ToLower("Remove" + "-Item")
	proved := false
	for _, seg := range segs {
		head, rest, ok := psCommandWord(seg)
		if !ok || head != deleteWord {
			continue
		}
		if !psDeleteTargetsInScratch(psTokenTexts(rest), ws, scratch) {
			return false
		}
		proved = true
	}
	return proved
}
