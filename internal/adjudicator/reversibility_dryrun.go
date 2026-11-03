package adjudicator

import (
	"regexp"
	"strings"
)

func hasDryRunPreview(cmd string, args map[string]any) bool {
	for _, key := range []string{"dry_run", "dryRun", "preview"} {
		if b, ok := boolArg(args, key); ok && b {
			return true
		}
	}
	lower := strings.ToLower(cmd)
	if strings.Contains(lower, "--dry-run") || strings.Contains(lower, "--preview") {
		return true
	}
	// PowerShell's -WhatIf is a universal no-op preview: a ShouldProcess cmdlet
	// carrying it reports what WOULD happen and performs nothing. So the
	// "Remove-Item -WhatIf" the fs-destroy redirect recommends is a preview, not
	// a deletion — recognize it, or the guard gates its OWN sanctioned dry-run
	// and the agent loops (the self-refuting-remedy class of the confirm-gate
	// deadlock, docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md).
	if hasWhatIfFlag(cmd) {
		return true
	}
	// `git clean -n` / `git push -n` (the short spelling of --dry-run) preview
	// without mutating, even combined with -d/-f (e.g. "git clean -nd", the
	// fs-destroy/git-destroy redirect; or "git push -nf"). -n is recognized ONLY
	// inside a `git clean`/`git push` segment, never globally: a bare -n means
	// unrelated things elsewhere (`kubectl delete -n <ns>` is a REAL delete that
	// must stay gated).
	if gitDryRunPreview(cmd) {
		return true
	}
	return false
}

// whatIfFlagRE matches PowerShell's -WhatIf common parameter as a standalone
// token (optionally -WhatIf:$true), anchored to a segment/word boundary so a
// path like ./-whatiffoo or a substring never trips it.
var whatIfFlagRE = regexp.MustCompile(`(?i)(^|[\s;&|(])-whatif($|[\s:;&|)])`)

// hasWhatIfFlag reports whether the command carries PowerShell's -WhatIf. It
// scans the RAW command (dashes intact): commandWords()/commandSegments() strip
// leading dashes, so a flag-shaped recognizer must not go through them.
func hasWhatIfFlag(cmd string) bool {
	return whatIfFlagRE.MatchString(cmd)
}

// gitDryRunPreview reports whether a `git clean` or `git push` invocation carries
// a dry-run flag: --dry-run, or a single-dash short-flag cluster containing 'n'
// (-n, -nd, -dn, -nf). Both subcommands spell dry-run identically, and for BOTH,
// 'n' is the only short flag that means dry-run — so any cluster containing 'n'
// is a genuine preview. A cluster WITHOUT 'n' is a real mutation and stays gated:
// `git clean -fd`/`-f`/`-x` (a real removal), and `git push -f` (force) or the
// destructive `git push -d`/`--delete <branch>` (a remote-branch delete) all lack
// 'n' and are untouched. The short `git push -n` is the documented equivalent of
// the `git push --dry-run` the git-push redirect itself recommends — recognizing
// it closes the same self-refuting-remedy gap the -WhatIf / git-clean-`-n` cases
// did (docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md): the long form was already
// reversible via the --dry-run substring, but the short form was gated as a live
// push. The flag is recognized ONLY inside these two git subcommands, never
// globally: a bare -n means unrelated things elsewhere (`kubectl delete -n <ns>`
// is a REAL delete that must stay gated). It parses the RAW command (dash-
// preserving), splitting on the same sequencing operators and stripping the same
// env/wrapper heads as commandSegments, but keeping flag dashes that commandWords
// drops.
func gitDryRunPreview(cmd string) bool {
	for _, raw := range commandSegmentRE.Split(cmd, -1) {
		fields := strings.Fields(raw)
		for len(fields) > 0 &&
			(envAssignmentRE.MatchString(fields[0]) || commandWrapperHeads[strings.ToLower(fields[0])]) {
			fields = fields[1:]
		}
		if len(fields) < 2 || strings.ToLower(fields[0]) != "git" {
			continue
		}
		switch strings.ToLower(fields[1]) {
		case "clean", "push":
		default:
			continue
		}
		for _, tok := range fields[2:] {
			lt := strings.ToLower(tok)
			if lt == "--dry-run" {
				return true
			}
			if strings.HasPrefix(lt, "-") && !strings.HasPrefix(lt, "--") && strings.ContainsRune(lt[1:], 'n') {
				return true
			}
		}
	}
	return false
}
