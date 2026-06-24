package adjudicator

import "strings"

// shellWriteVerbs are the shell idioms that WRITE to a path operand (#172 Hole 1).
// Detection is by substring on the command string and is deliberately conservative:
// these are write operations, so a self-modify guard that fires on them cannot
// block a read (`cat`/`grep`/`head` of a guarded file). A redirect (`>`/`>>`) is
// handled separately since it is punctuation, not a word. `shred` joins the
// destructive-overwrite family (`dd`/`truncate`): it overwrites then unlinks the
// named file and has NO read mode, so a substring match cannot block a read.
var shellWriteVerbs = []string{
	"sed -i", "tee ", "dd ", "truncate ", "shred ",
	"git apply", "git checkout", "git restore", "git stash",
	"cp ", "mv ", "install ", "patch ", "chmod ", "chown ", "ln ", "rm ",
	// In-place interpreter edits — the sed -i family across other interpreters
	// (#172 Hole 1, the porous-denylist residual). `perl -i`/`ruby -i` rewrite a
	// file in place exactly as `sed -i` does, but carry none of the leading tokens
	// above, so `perl -i -pe 's/x/y/' internal/abi/kernel.go` would otherwise
	// launder a self-edit past the floor that `sed -i` is caught on.
	"perl -i", "ruby -i",
}

// commandSelfModify reports the guarded glob a shell command would WRITE into, or
// "" if the command is absent / read-only / writes nowhere guarded. It is the
// shell-path dual of the targetPath self-modify check: it gates a Bash/exec call
// whose write target lives in the `command`/`cmd` string rather than a path arg.
//
// It fires only when BOTH hold: (1) the command string contains a write-shaped
// shell verb or an output redirect, AND (2) the command string contains a guarded
// glob fragment. Requiring a write verb is what keeps a read of a guarded file
// (`cat internal/abi/x.go`) allowed — only a write into the tree is refused. This
// is a substring floor, not a shell parser: it is intentionally over-broad (a
// guarded path mentioned anywhere alongside a write verb is refused) because the
// guarded trees are the kernel's own witness machinery, where a false refusal is
// cheap and a false allow is the self-grading-homework failure the floor exists to
// stop.
func commandSelfModify(args map[string]any, globs []string) string {
	cmd, ok := argString(args, "command")
	if !ok || cmd == "" {
		if cmd, ok = argString(args, "cmd"); !ok || cmd == "" {
			return ""
		}
	}
	g := matchGlob(cmd, globs)
	if g == "" {
		return "" // touches no guarded tree — nothing to guard
	}
	if !commandWrites(cmd) {
		return "" // a read of a guarded file is allowed; only writes are refused
	}
	return g
}

// commandWrites reports whether a shell command string contains a write-shaped
// verb or an output redirect (`>`/`>>`). It is substring/byte level, matching the
// conservative floor commandSelfModify documents.
func commandWrites(cmd string) bool {
	lc := strings.ToLower(cmd)
	for _, v := range shellWriteVerbs {
		if strings.Contains(lc, v) {
			return true
		}
	}
	// `find … -delete` / `find … -exec <writer>`: a delete/exec-write idiom whose
	// effect is a WRITE (it removes or rewrites the matched files) but which carries
	// none of shellWriteVerbs' leading tokens — `find internal/abi -delete` would
	// otherwise launder a deletion of a guarded tree past this floor (#172 Hole 1,
	// the porous-denylist `find -delete` gap the audit named). The `-delete` action
	// and an `-exec`/`-execdir` running anything are both treated as writes; a plain
	// read-only `find` (no `-delete`/`-exec`) stays allowed.
	if strings.Contains(lc, "find ") &&
		(strings.Contains(lc, "-delete") ||
			strings.Contains(lc, "-exec")) {
		return true
	}
	// `git -C <dir> <mutating-subcommand>`: the `-C <dir>` flag runs git AS IF in
	// <dir>, so a guarded tree can be the write TARGET via the directory operand
	// even when the only mutating verbs (`reset`, `commit`, `clean`, …) are not in
	// shellWriteVerbs (#172 Hole 1 residual — `git -C internal/abi reset --hard`
	// rewrites the guarded tree but carries none of the `git apply`/`git checkout`
	// tokens caught above). Fire when a `git -C` invocation also names a
	// tree-mutating subcommand; a `git -C <dir> status`/`log`/`diff` read does not.
	// The `-C` flag is CASE-SENSITIVE (lowercase `-c` selects a config override, not
	// a directory), so the flag is matched against the ORIGINAL command; only the
	// subcommand below is matched case-insensitively against lc.
	if strings.Contains(cmd, "git -C ") || strings.Contains(lc, "git --git-dir") {
		for _, sub := range []string{"reset", "commit", "clean", " rm ", " mv ", "checkout", "restore", "stash", "apply"} {
			if strings.Contains(lc, sub) {
				return true
			}
		}
	}
	// Interpreter inline-program write: a general-purpose interpreter invoked with
	// an inline-program flag (`python -c '…'`, `node -e '…'`, …) can open and rewrite
	// a guarded file from inside the program string with NO shellWriteVerb token and
	// NO `>` redirect — `python3 -c "open('internal/abi/x.go','w').write(...)"` would
	// otherwise launder a self-edit past every rung above (#172 Hole 1 residual, the
	// `perl -i`/`ruby -i` gap one interpreter further out — python/node are the ones
	// most likely on a coding agent's PATH). The program string is opaque to a
	// substring floor, so — matching this floor's documented "a guarded tree named
	// alongside a writer is refused; a false refusal is cheap" stance — an inline
	// interpreter program is treated as write-shaped. commandSelfModify only reaches
	// here once the command already names a guarded glob, so a `python -c` that
	// touches nothing guarded is unaffected; only an inline program that mentions a
	// guarded tree is refused. A read-shaped interpreter call WITHOUT an inline-
	// program flag (`python score.py`, `node app.js`) is not matched here.
	for _, ev := range interpreterEvalFlags {
		if strings.Contains(lc, ev) {
			return true
		}
	}
	// In-place / line-editor writes that carry none of the tokens above (#172 Hole 1
	// residual, the sed -i family two tools further out):
	//
	//   - `awk -i inplace` / `gawk -i inplace`: GNU awk's in-place edit flag rewrites
	//     the named file exactly as `sed -i`/`perl -i`/`ruby -i` (already guarded) do,
	//     but `awk` carries no `-i` leading token in shellWriteVerbs. The flag is the
	//     `inplace` extension specifically — a read-only `awk '{print}' file` has no
	//     `-i inplace` and stays allowed. Match the `-i inplace` token sequence.
	//   - `ed` / `ex`: scripted line editors that WRITE their file operand with no
	//     redirect and no caught verb (`ex -s -c wq file`, `ed -s file`). They are
	//     short tokens, so they are matched only in COMMAND POSITION (start of the
	//     command or right after a shell separator), never as a substring of `sed`,
	//     `indexed`, `fixed`, `expr`, etc. commandSelfModify only reaches here once a
	//     guarded glob is already named, so a benign `ed`/`ex` over an unguarded path
	//     is unaffected; only an editor invocation alongside a guarded tree is refused.
	if strings.Contains(lc, "-i inplace") {
		return true
	}
	if hasCommandLeadingToken(lc, "ed") || hasCommandLeadingToken(lc, "ex") {
		return true
	}
	// Archive EXTRACTION / sync INTO a guarded tree (#172 Hole 1 residual): an
	// extractor writes files into a destination directory with none of the tokens
	// above — `tar -xf a.tar -C internal/abi`, `unzip -d internal/abi a.zip`,
	// `rsync src/ internal/abi/` overwrite the guarded tree via the extract/sync
	// DESTINATION. commandSelfModify only reaches here once a guarded glob is named,
	// so the job is to fire on the WRITE (extract/sync) idioms while leaving the
	// READ idioms over the same tree allowed (tar LIST/CREATE, unzip LIST/test).
	if archiveExtractsOrSyncs(lc) {
		return true
	}
	// Output redirect: a bare '>' or '>>' (not the '2>&1'/'>&' fd-dup forms, which
	// are not a file write per se, but conservatively a '>' anywhere is a write).
	return strings.Contains(cmd, ">")
}

// archiveExtractsOrSyncs reports whether lc (the lowercased command) is an archive
// EXTRACTION or a sync that WRITES files into its destination. It deliberately fires
// only on the write idioms, so a read over the same guarded tree stays allowed
// (commandSelfModify has already confirmed a guarded glob is named in the command):
//
//   - `tar` writes only in EXTRACT mode (`-x` / a leading `x` in the bundled-flag
//     form `tar xf`); LIST (`-t`) and CREATE-archive (`-c`, which READS the tree into
//     an archive written elsewhere) are not writes into the tree. So a `tar` command
//     is a write here iff it carries an extract flag.
//   - `unzip` extracts by default; it is a READ only in LIST/test mode (`-l` / `-v` /
//     `-t` / `-z`). So an `unzip` command is a write here unless it carries a
//     list/test flag.
//   - `rsync` writes its DESTINATION operand; once a guarded glob is named anywhere in
//     an rsync command, conservatively treat it as a write into the tree (the floor's
//     documented "a guarded tree named alongside a writer is refused" stance).
func archiveExtractsOrSyncs(lc string) bool {
	if hasCommandLeadingToken(lc, "rsync") {
		return true
	}
	if hasCommandLeadingToken(lc, "tar") && tarExtracts(lc) {
		return true
	}
	if hasCommandLeadingToken(lc, "unzip") && !unzipListsOnly(lc) {
		return true
	}
	return false
}

// tarExtracts reports whether a tar command is in EXTRACT mode (a write), via either
// the `-x` flag or the classic bundled form (`tar xf …`, `tar xzf …`). A tar command
// with no extract flag (LIST `-t`, CREATE `-c`) is not a write into the named tree.
func tarExtracts(lc string) bool {
	if strings.Contains(lc, "-x") || strings.Contains(lc, "--extract") {
		return true
	}
	// Bundled flags: the first word after `tar ` whose run of letters includes 'x'
	// (e.g. `xf`, `xzf`, `xvf`). Restrict to a leading flag cluster so a path or
	// archive name containing 'x' cannot trip it.
	i := strings.Index(lc, "tar ")
	if i < 0 {
		return false
	}
	rest := strings.TrimLeft(lc[i+len("tar "):], " ")
	// Inspect only the first whitespace-delimited token, and only if it is a bare
	// flag cluster (letters, no '/' or '.', i.e. not a path/archive operand).
	end := strings.IndexAny(rest, " \t")
	if end < 0 {
		end = len(rest)
	}
	tok := rest[:end]
	if tok == "" || strings.ContainsAny(tok, "/.-") {
		return false // a path/archive operand or a `-`-prefixed flag handled above
	}
	return strings.Contains(tok, "x")
}

// unzipListsOnly reports whether an unzip command is a LIST/test (a read), not an
// extraction. unzip extracts by default, so only the list/test flags make it a read.
func unzipListsOnly(lc string) bool {
	for _, f := range []string{"-l", "-v", "-t", "-z"} {
		if strings.Contains(lc, f+" ") || strings.HasSuffix(lc, f) {
			return true
		}
	}
	return false
}

// hasCommandLeadingToken reports whether tok appears in lc as a whole word in
// COMMAND POSITION — at the very start of the command (after any leading whitespace),
// or immediately after a shell command separator (`;`, `|`, `&`, `(`, newline, or the
// `&&`/`||` forms, all of which end in one of those bytes) followed by optional
// whitespace — and is itself followed by a space (an argument follows). This pins a
// short, false-positive-prone editor name like `ed`/`ex` to the slot where it names a
// program to RUN, so it never trips on `sed`, `indexed`, `fixed`, `expr`, a path
// fragment, or an ARGUMENT that happens to be the word (`grep ed file` is a read, not
// an `ed` invocation — the `ed` there follows an argument space, not a separator).
// lc is the lowercased command.
func hasCommandLeadingToken(lc, tok string) bool {
	from := 0
	for {
		i := strings.Index(lc[from:], tok+" ")
		if i < 0 {
			return false
		}
		at := from + i
		// Walk back over run-of-whitespace to the boundary byte (or string start).
		j := at
		for j > 0 && (lc[j-1] == ' ' || lc[j-1] == '\t') {
			j--
		}
		if j == 0 || isShellSep(lc[j-1]) {
			return true
		}
		from = at + 1
	}
}

// isShellSep reports whether b is a shell command separator, so a token following it
// (modulo whitespace) is in command position. A plain space is deliberately NOT a
// separator: a bare argument space does not start a new command, so `grep ed file`
// (the word `ed` as a search argument) is not an `ed` invocation. The `&&`/`||`
// chains end in `&`/`|`, which are covered.
func isShellSep(b byte) bool {
	switch b {
	case ';', '|', '&', '(', '\n':
		return true
	}
	return false
}
