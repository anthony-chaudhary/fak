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

// commandArg returns the shell command string carried by a tool-call's args,
// preferring the "command" key and falling back to "cmd". An empty value is treated
// as absent, so ok is false exactly when neither key holds a non-empty command — the
// shared "no command to inspect" early-out for the command-string floors below.
func commandArg(args map[string]any) (string, bool) {
	cmd, ok := argString(args, "command")
	if !ok || cmd == "" {
		if cmd, ok = argString(args, "cmd"); !ok || cmd == "" {
			return "", false
		}
	}
	return cmd, true
}

// guardedCommandTree decodes a tool-call's shell command and reports the guarded glob
// it names. ok is false — the caller should ALLOW — when there is no command or the
// command reaches no guarded tree. It is the front gate for synthToolSelfModify,
// whose envelope check is intentionally about guarded reach by an authored tool.
func guardedCommandTree(args map[string]any, globs []string) (cmd, glob string, ok bool) {
	cmd, has := commandArg(args)
	if !has {
		return "", "", false
	}
	g := matchGlob(cmd, globs)
	if g == "" {
		return "", "", false
	}
	return cmd, g, true
}

// commandSelfModify reports the guarded glob a shell command would WRITE into, or
// "" if the command is absent / read-only / writes nowhere guarded. It is the
// shell-path dual of the targetPath self-modify check: it gates a Bash/exec call
// whose write target lives in the `command`/`cmd` string rather than a path arg.
//
// It fires only when BOTH hold: (1) the command string contains a write-shaped
// shell verb or an output redirect, AND (2) a decoded write TARGET contains a
// guarded glob fragment. Requiring a write target is what keeps a read of a
// guarded file (`cat VERSION > /tmp/v`, `cp VERSION /tmp/v`) allowed — only the
// write into the tree is refused. Opaque eval-shaped writers keep the conservative
// whole-segment target because their real file open happens inside a program string,
// not in shell operands.
func commandSelfModify(args map[string]any, globs []string) string {
	cmd, ok := commandArg(args)
	if !ok {
		return "" // no command to inspect
	}
	if !commandWrites(cmd) {
		return "" // a read of a guarded file is allowed; only writes are refused
	}
	// SSH/SCP identity-file READ (#1086): an `ssh -i <key>` / `scp -i <key>` names a
	// credential glob (id_rsa / id_ed25519) only as the key it READS for auth, not a
	// write target — but the command may carry a write verb elsewhere (scp's own `cp `
	// substring, or a remote build's `cp`/`mv`), so commandWrites is true and the glob
	// matched. That conflates a sanctioned remote build+bench over SSH with a self-edit
	// of a key. Re-check the match with the `-i <keyfile>` operand removed: if no guarded
	// glob remains, the only match WAS the identity read, so allow. A genuine write that
	// still names a guarded glob after the identity arg is stripped (e.g. `tee ~/.ssh/
	// id_rsa`, or a redirect into a key) keeps its deny — the floor is unchanged for it.
	if stripped, did := stripSSHIdentityArg(cmd); did {
		cmd = stripped
	}
	for _, target := range commandWriteTargets(cmd) {
		if g := matchGlob(target, globs); g != "" {
			return g
		}
	}
	return ""
}

// stripSSHIdentityArg removes the `-i <keyfile>` identity operand from an ssh/scp command
// string, returning the remainder and whether a strip happened. It fires only when the
// command is an ssh/scp invocation (leading token, so a tool literally named `ssh-keygen`
// or an arbitrary program with a `-i` flag is unaffected) carrying an `-i <path>` pair.
// The keyfile token is whatever follows `-i` up to the next space — exactly what OpenSSH
// treats as the IdentityFile argument. It is a surgical removal for the self-modify
// re-check ONLY; the original command is what actually runs.
func stripSSHIdentityArg(cmd string) (string, bool) {
	lc := strings.ToLower(cmd)
	// Only ssh / scp / sftp read a `-i <keyfile>` identity; gate on the leading token so
	// the carve-out cannot be reached by an arbitrary program that happens to take `-i`.
	if !hasCommandLeadingToken(lc, "ssh") &&
		!hasCommandLeadingToken(lc, "scp") &&
		!hasCommandLeadingToken(lc, "sftp") {
		return cmd, false
	}
	fields := strings.Fields(cmd)
	out := make([]string, 0, len(fields))
	stripped := false
	for i := 0; i < len(fields); i++ {
		if fields[i] == "-i" && i+1 < len(fields) {
			i++ // skip the keyfile operand too
			stripped = true
			continue
		}
		// The glued `-i<path>` spelling (no space) — OpenSSH accepts it.
		if strings.HasPrefix(fields[i], "-i") && len(fields[i]) > 2 {
			stripped = true
			continue
		}
		out = append(out, fields[i])
	}
	if !stripped {
		return cmd, false
	}
	return strings.Join(out, " "), true
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
	// Interpreter inline-program write: a general-purpose interpreter invoked with an
	// inline-program flag (`python -c '…'`, `node -e '…'`, `ruby -e '…'`, …) can open and
	// rewrite a guarded file from inside the program string with NO shellWriteVerb token and
	// NO `>` redirect — `python3 -c "open('internal/abi/x.go','w').write(...)"` would
	// otherwise launder a self-edit past every rung above (#172 Hole 1 residual, the
	// `perl -i`/`ruby -i` gap one interpreter further out — python/node/ruby are the ones
	// most likely on a coding agent's PATH). The program string is opaque to this floor.
	// Matching is by TOKEN (interpreterEvalMatch), so the idiomatic `ruby -e'…'` (no space),
	// `node --eval=…`, and intervening-flag (`ruby -rjson -e …`) spellings are all caught,
	// not just a `<interp> <flag> ` prefix. commandSelfModify's target extractor treats
	// the opaque inline-program segment as the write target; a read-shaped interpreter call
	// WITHOUT an eval flag (`python score.py`,
	// `node app.js`) names no flag token and is not matched.
	//
	// The eval flag alone is NOT enough — it says "runs an inline program", not "writes a
	// file". SELF_MODIFY is a WRITE floor, so an inline program that only READS (introspection,
	// `os.environ`, `open(p).read()`, `.count('id_rsa')`, `os.path.exists('~/.ssh/id_ed25519')`)
	// must NOT be routed to the whole-segment write-target extractor — where it would match any
	// guarded glob it merely NAMES by substring and deny a harmless read as a self-edit (witnessed
	// live 2026-07-02: read-only `python -c` naming `~/.ssh/id_ed25519` / counting `id_rsa` in a
	// log both denied SELF_MODIFY; the earlier dotfile-boundary fix reached only dot-globs like
	// `.env`, not the non-dot credential globs). Gate the interpreter rung on inlineEvalWriteIntent:
	// a program is write-shaped only when it shows an actual file-MUTATION signal (a write/append/
	// exclusive open, a `.write`/writeFile/File.write, a delete/rename/mkdir/truncate/chmod, or a
	// subprocess that could do any of those). A read that then EXFILTRATES a secret is the domain of
	// the secret/egress rungs, not this one. Direction of safety: the write-intent set is broad —
	// every realistic file write in python/node/ruby carries one of its tokens — so a genuine inline
	// self-edit keeps its deny; a redirect (`os.system('… > .env')`) inside the program is caught
	// independently by the `>` scanner regardless of this gate.
	for _, ev := range interpreterEvalFlags {
		if interpreterEvalMatch(lc, ev) && inlineEvalWriteIntent(lc) {
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
	//     `indexed`, `fixed`, `expr`, etc. commandSelfModify later checks the editor's
	//     file operand against the guarded globs.
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
	// DESTINATION. The target extractor then matches only those extract/sync
	// destinations against guarded globs, while READ idioms over the same tree stay
	// allowed (tar LIST/CREATE, unzip LIST/test).
	if archiveExtractsOrSyncs(lc) {
		return true
	}
	// Output redirect to a REAL file: a `>`/`>>` whose target is an actual path.
	// fd-duplication (`2>&1`, `>&2`, `1>&2`) and a redirect to the null device
	// (`2>/dev/null`, `>/dev/null`) write NO named file, so — matching this floor's
	// "only a WRITE into a guarded tree is refused; a read of a guarded file is
	// allowed" contract — they are NOT writes. The prior `strings.Contains(cmd, ">")`
	// counted a bare `>` anywhere as a write, which refused the ubiquitous
	// `cat internal/kernel/x.go 2>/dev/null` read-with-stderr idiom (and any command
	// naming a guarded glob that merely carried a `2>/dev/null`/`2>&1`) as SELF_MODIFY
	// (#1569). A genuine `> guarded/path` / `>> ~/.ssh/id_rsa` still fires.
	return hasFileWriteRedirect(cmd)
}

// hasFileWriteRedirect reports whether cmd contains an output redirect (`>`/`>>`)
// whose target is a real file, as opposed to an fd-duplication (`2>&1`, `>&2`,
// `1>&2`) or a redirect to the null device (`2>/dev/null`, `>/dev/null`) — neither
// of which writes a named file. It keeps commandWrites true to its documented
// "a read is allowed; only a write is refused" contract: the pervasive trailing
// `… 2>/dev/null` / `… 2>&1` stderr idiom on an otherwise read-only command no
// longer marks it write-shaped, while a genuine `> path` (including
// `>> internal/kernel/x.go` or `> ~/.ssh/id_rsa`) still does. Conservative on the
// rare/odd forms: an unrecognised or empty target (process substitution `>(cmd)`,
// a dangling `>`) is treated as NOT a write only when it names no real path, so the
// floor never loses a redirect into a guarded tree.
func hasFileWriteRedirect(cmd string) bool {
	return len(fileWriteRedirectTargets(cmd)) > 0
}

// fileWriteRedirectTargets returns the real file targets of output redirects in cmd.
// fd-duplication (`2>&1`) and null-device sinks are deliberately absent.
func fileWriteRedirectTargets(cmd string) []string {
	var targets []string
	for i := 0; i < len(cmd); i++ {
		if cmd[i] != '>' {
			continue
		}
		j := i + 1
		if j < len(cmd) && cmd[j] == '>' { // collapse `>>` to its target
			j++
		}
		if j < len(cmd) && cmd[j] == '|' { // `>|` clobber-override still writes a file
			j++
		}
		// fd-duplication: `>&` / `2>&1` / `>&2` duplicates a descriptor, not a file.
		if j < len(cmd) && cmd[j] == '&' {
			continue
		}
		// Skip whitespace between the operator and its target.
		for j < len(cmd) && (cmd[j] == ' ' || cmd[j] == '\t') {
			j++
		}
		// Read the target token up to the next shell boundary.
		k := j
		for k < len(cmd) && !isRedirectTargetBoundary(cmd[k]) {
			k++
		}
		target := cmd[j:k]
		if target == "" || isNullSink(target) {
			continue // no real file written by this redirect
		}
		targets = append(targets, cleanShellOperand(target))
	}
	return targets
}

// isNullSink reports whether a redirect target is the null device — a write that
// reaches no file-system path and so can never touch a guarded tree.
func isNullSink(target string) bool {
	return target == "/dev/null"
}

// isRedirectTargetBoundary reports whether b ends a redirect target token (a shell
// separator, whitespace, another redirect, or a quote).
func isRedirectTargetBoundary(b byte) bool {
	switch b {
	case ' ', '\t', '\n', '\r', ';', '|', '&', '<', '>', '(', ')', '"', '\'', '`':
		return true
	}
	return false
}

// commandWriteTargets returns the shell operands this command writes to. It is still
// deliberately small, not a shell parser: the goal is to keep the SELF_MODIFY floor
// target-scoped for the common write idioms the floor already recognizes.
func commandWriteTargets(cmd string) []string {
	var targets []string
	add := func(target string) {
		target = cleanShellOperand(target)
		if target == "" || isNullSink(target) {
			return
		}
		for _, existing := range targets {
			if existing == target {
				return
			}
		}
		targets = append(targets, target)
	}
	for _, target := range fileWriteRedirectTargets(cmd) {
		add(target)
	}
	for _, segment := range shellSegments(cmd) {
		for _, target := range segmentWriteTargets(segment) {
			add(target)
		}
	}
	return targets
}

func segmentWriteTargets(segment string) []string {
	var targets []string
	add := func(target string) {
		target = cleanShellOperand(target)
		if target != "" && !isNullSink(target) {
			targets = append(targets, target)
		}
	}
	for _, target := range fileWriteRedirectTargets(segment) {
		add(target)
	}
	words := shellWords(segment)
	if len(words) == 0 {
		return targets
	}
	start := commandWordStart(words)
	if start >= len(words) {
		return targets
	}
	head := strings.ToLower(words[start].text)
	args := words[start+1:]
	lc := strings.ToLower(segment)
	switch head {
	case "tee":
		for _, target := range plainOperands(args) {
			add(target)
		}
	case "dd":
		for _, w := range args {
			if target, ok := strings.CutPrefix(w.text, "of="); ok {
				add(target)
			}
		}
	case "truncate", "shred", "rm":
		for _, target := range plainOperands(args) {
			add(target)
		}
	case "cp", "install", "ln":
		if target := lastPlainOperand(args); target != "" {
			add(target)
		}
	case "mv":
		for _, target := range plainOperands(args) {
			add(target)
		}
	case "chmod", "chown":
		for _, target := range operandsAfterFirst(args) {
			add(target)
		}
	case "sed":
		if strings.Contains(lc, "sed -i") {
			if target := lastPlainOperand(args); target != "" {
				add(target)
			}
		}
	case "perl", "ruby":
		if hasInPlaceFlag(args) {
			if target := lastPlainOperand(args); target != "" {
				add(target)
			}
		}
	case "awk", "gawk":
		if strings.Contains(lc, "-i inplace") {
			if target := lastPlainOperand(args); target != "" {
				add(target)
			}
		}
	case "ed", "ex":
		if target := lastPlainOperand(args); target != "" {
			add(target)
		}
	case "git":
		for _, target := range gitWriteTargets(args) {
			add(target)
		}
	case "find":
		for _, target := range findWriteTargets(args) {
			add(target)
		}
	case "tar":
		if tarExtracts(lc) {
			for _, target := range tarExtractTargets(args) {
				add(target)
			}
		}
	case "unzip":
		if !unzipListsOnly(lc) {
			for _, target := range flagValues(args, "-d", "--dir", "--directory") {
				add(target)
			}
		}
	case "rsync":
		if target := lastPlainOperand(args); target != "" {
			add(target)
		}
	case "patch":
		for _, target := range plainOperands(args) {
			add(target)
		}
	}
	for _, ev := range interpreterEvalFlags {
		// Only a write-INTENT inline program contributes its whole opaque segment as a
		// write target (mirrors the commandWrites gate): a read-only one-liner that merely
		// NAMES a guarded path writes nothing, so it must not be handed to matchGlob as a
		// target. Without this, a read segment's guarded name matches by substring even when
		// commandWrites was made true by a DIFFERENT segment's real write verb.
		if interpreterEvalMatch(lc, ev) && inlineEvalWriteIntent(lc) {
			add(segment)
			break
		}
	}
	for _, w := range words {
		if !w.quoted || !looksNestedShell(w.text) {
			continue
		}
		for _, target := range commandWriteTargets(w.text) {
			add(target)
		}
	}
	return targets
}

type shellWord struct {
	text   string
	quoted bool
}

func shellWords(cmd string) []shellWord {
	var out []shellWord
	var b strings.Builder
	var quote byte
	quoted := false
	escaped := false
	flush := func() {
		if b.Len() == 0 && !quoted {
			return
		}
		out = append(out, shellWord{text: b.String(), quoted: quoted})
		b.Reset()
		quoted = false
	}
	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		if escaped {
			b.WriteByte(c)
			escaped = false
			continue
		}
		if c == '\\' && quote != '\'' {
			escaped = true
			continue
		}
		if quote != 0 {
			if c == quote {
				quote = 0
				quoted = true
				continue
			}
			b.WriteByte(c)
			continue
		}
		switch c {
		case '\'', '"', '`':
			quote = c
			quoted = true
		case ' ', '\t', '\n', '\r':
			flush()
		default:
			b.WriteByte(c)
		}
	}
	flush()
	return out
}

func shellSegments(cmd string) []string {
	var segments []string
	var quote byte
	escaped := false
	start := 0
	flush := func(end int) {
		if s := strings.TrimSpace(cmd[start:end]); s != "" {
			segments = append(segments, s)
		}
	}
	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' && quote != '\'' {
			escaped = true
			continue
		}
		if quote != 0 {
			if c == quote {
				quote = 0
			}
			continue
		}
		switch c {
		case '\'', '"', '`':
			quote = c
		case ';', '\n', '|', '&':
			if c == '|' && i > 0 && cmd[i-1] == '>' {
				continue
			}
			if c == '&' && ((i > 0 && cmd[i-1] == '>') || (i+1 < len(cmd) && cmd[i+1] == '>')) {
				continue
			}
			flush(i)
			start = i + 1
		}
	}
	flush(len(cmd))
	return segments
}

func commandWordStart(words []shellWord) int {
	i := 0
	for i < len(words) {
		tok := words[i].text
		if isAssignment(tok) {
			i++
			continue
		}
		if strings.ToLower(tok) != "env" {
			break
		}
		i++
		for i < len(words) {
			tok = words[i].text
			switch {
			case tok == "--":
				i++
				return i
			case tok == "-u" || tok == "--unset":
				i += 2
			case strings.HasPrefix(tok, "-"):
				i++
			case isAssignment(tok):
				i++
			default:
				return i
			}
		}
	}
	return i
}

func plainOperands(words []shellWord) []string {
	var out []string
	skipNext := false
	for _, w := range words {
		tok := cleanShellOperand(w.text)
		if skipNext {
			skipNext = false
			continue
		}
		if tok == "" || tok == "--" {
			continue
		}
		if tok == "<" || tok == ">" || tok == ">>" || tok == ">|" {
			skipNext = tok == "<"
			continue
		}
		if strings.HasPrefix(tok, "-") || strings.HasPrefix(tok, "+") {
			continue
		}
		out = append(out, tok)
	}
	return out
}

func lastPlainOperand(words []shellWord) string {
	ops := plainOperands(words)
	if len(ops) == 0 {
		return ""
	}
	return ops[len(ops)-1]
}

func operandsAfterFirst(words []shellWord) []string {
	ops := plainOperands(words)
	if len(ops) <= 1 {
		return nil
	}
	return ops[1:]
}

func hasInPlaceFlag(words []shellWord) bool {
	for _, w := range words {
		tok := w.text
		if tok == "-i" || strings.HasPrefix(tok, "-i.") || strings.HasPrefix(tok, "-i'") || strings.HasPrefix(tok, "-i\"") {
			return true
		}
	}
	return false
}

func gitWriteTargets(args []shellWord) []string {
	var targets []string
	mutating := false
	for i := 0; i < len(args); i++ {
		tok := strings.ToLower(args[i].text)
		switch tok {
		case "reset", "commit", "clean", "rm", "mv", "checkout", "restore", "stash", "apply":
			mutating = true
		}
	}
	if !mutating {
		return nil
	}
	for i := 0; i < len(args); i++ {
		tok := args[i].text
		if tok == "-C" && i+1 < len(args) {
			targets = append(targets, args[i+1].text)
			i++
			continue
		}
		if target, ok := strings.CutPrefix(tok, "--git-dir="); ok {
			targets = append(targets, target)
		}
	}
	for _, target := range plainOperands(args) {
		targets = append(targets, target)
	}
	return targets
}

func findWriteTargets(args []shellWord) []string {
	writes := false
	var targets []string
	for i := 0; i < len(args); i++ {
		tok := args[i].text
		if tok == "-delete" {
			writes = true
		}
		if tok != "-exec" && tok != "-execdir" {
			continue
		}
		writes = true
		var exec []string
		for j := i + 1; j < len(args); j++ {
			if args[j].text == ";" || args[j].text == "+" {
				break
			}
			exec = append(exec, args[j].text)
		}
		for _, target := range commandWriteTargets(strings.Join(exec, " ")) {
			targets = append(targets, target)
		}
	}
	if !writes {
		return nil
	}
	for _, w := range args {
		tok := cleanShellOperand(w.text)
		if tok == "" || strings.HasPrefix(tok, "-") || tok == "!" || tok == "(" || tok == ")" {
			break
		}
		targets = append(targets, tok)
	}
	if len(targets) == 0 {
		targets = append(targets, ".")
	}
	return targets
}

func tarExtractTargets(args []shellWord) []string {
	targets := flagValues(args, "-C", "--directory")
	for _, w := range args {
		if target, ok := strings.CutPrefix(w.text, "--directory="); ok {
			targets = append(targets, target)
		}
	}
	return targets
}

func flagValues(args []shellWord, names ...string) []string {
	var values []string
	for i := 0; i < len(args); i++ {
		tok := args[i].text
		for _, name := range names {
			if tok == name && i+1 < len(args) {
				values = append(values, args[i+1].text)
				i++
				break
			}
			if strings.HasPrefix(tok, name+"=") {
				values = append(values, strings.TrimPrefix(tok, name+"="))
			}
		}
	}
	return values
}

func cleanShellOperand(s string) string {
	return strings.Trim(strings.TrimSpace(s), "\"'")
}

func looksNestedShell(s string) bool {
	return commandWrites(s) || strings.ContainsAny(s, ";&|>")
}

// archiveExtractsOrSyncs reports whether lc (the lowercased command) is an archive
// EXTRACTION or a sync that WRITES files into its destination. It deliberately fires
// only on the write idioms, so a read over the same guarded tree stays allowed
// (commandSelfModify later matches only the extracted destination against guarded globs):
//
//   - `tar` writes only in EXTRACT mode (`-x` / a leading `x` in the bundled-flag
//     form `tar xf`); LIST (`-t`) and CREATE-archive (`-c`, which READS the tree into
//     an archive written elsewhere) are not writes into the tree. So a `tar` command
//     is a write here iff it carries an extract flag.
//   - `unzip` extracts by default; it is a READ only in LIST/test mode (`-l` / `-v` /
//     `-t` / `-z`). So an `unzip` command is a write here unless it carries a
//     list/test flag.
//   - `rsync` writes its DESTINATION operand.
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

// inlineWriteIntentTokens are the lowercased substrings whose presence in an interpreter
// inline-program string proves it MUTATES the filesystem (as opposed to only reading). The
// set is deliberately BROAD on the write side — missing a write is fail-OPEN (a laundered
// self-edit), while a spurious hit only ever matters when the program ALSO names a guarded
// glob, and even then errs toward the safe (deny) direction. Every realistic file write in
// the three inline interpreters this floor covers (python/node/ruby) carries one of these:
//   - "write": .write / writeFile(Sync) / write_text / write_bytes / File.write / IO.write /
//     os.write / createWriteStream — the single most common write signal;
//   - a file opened for WRITE/APPEND/EXCLUSIVE/UPDATE: the comma-prefixed mode strings below
//     (`open(p, 'w')`, `'a'`, `'x'`, `'w+'`, `'wb'`, …), including the truncate-on-open case
//     that carries no `.write` call. Comma-prefixed so a bare `print('a')` is not a write;
//   - a delete / rename / make / truncate / perms mutation verb;
//   - a subprocess that could do any of the above (`os.system`, `subprocess`, `popen`,
//     `child_process`, `spawn`), which the `>` redirect scanner also backstops.
//
// The residual (a maximally-obfuscated inline writer that avoids every token here) is the
// same class the perl/php/lua interpreters are for interpreterEvalFlags — named, not yet
// closed — and the direct-write and shell-redirect floors still catch its common forms.
var inlineWriteIntentTokens = []string{
	"write",
	"truncate", "unlink", "rmtree", "rmdir", "mkdir", "makedirs", "appendfile",
	".remove(", "os.remove", ".rename(", "os.rename", ".replace(", "shutil",
	"symlink", "chmod", "chown",
	"system(", "popen(", "subprocess", "os.exec", "execsync", "child_process", "spawn(", ".spawn", "execfile", "fileutils",
	// file opened for write/append/exclusive/update — comma-prefixed mode strings, both
	// quotings and with/without a space after the comma (`open(p,'w')`, `open(p, "a+")`).
	",'w", ", 'w", ",\"w", ", \"w",
	",'a", ", 'a", ",\"a", ", \"a",
	",'x", ", 'x", ",\"x", ", \"x",
	",'r+", ", 'r+", ",\"r+", ", \"r+",
}

// inlineEvalWriteIntent reports whether an interpreter inline-program command (already known
// to carry an interp+eval-flag) shows any sign of MUTATING the filesystem. A false result
// means the program only reads, so — the SELF_MODIFY floor being a write floor — its opaque
// segment must not be treated as a write target even when it names a guarded path. See
// inlineWriteIntentTokens for the safety direction and the residual.
func inlineEvalWriteIntent(lc string) bool {
	for _, t := range inlineWriteIntentTokens {
		if strings.Contains(lc, t) {
			return true
		}
	}
	return false
}

// interpreterEvalMatch reports whether lc (the lowercased command) invokes spec.interp
// with one of its inline-eval flags, both matched as TOKENS. The interpreter need only
// appear as a word (namesWord) and a flag as its own argument (namesFlagToken), so the
// flag's delimiter (space, quote, `=`, backtick, end) and any intervening flags between
// the interpreter and the eval flag (`ruby -rjson -e …`) do not let an inline self-edit
// slip — the porous-prefix gap a fixed `<interp> <flag> ` table leaves open.
func interpreterEvalMatch(lc string, spec interpreterEvalSpec) bool {
	if !namesWord(lc, spec.interp) {
		return false
	}
	for _, f := range spec.flags {
		if namesFlagToken(lc, f) {
			return true
		}
	}
	return false
}

// namesWord reports whether tok appears in lc as a whole word: a left boundary of string
// start, whitespace, or a shell separator, and a right boundary of whitespace (an
// interpreter that runs anything always has an argument after it). So `ruby`, `| ruby`,
// and `xargs ruby` match, but `myruby` and a `…/ruby/…` path fragment do not.
func namesWord(lc, tok string) bool {
	for from := 0; ; {
		i := strings.Index(lc[from:], tok)
		if i < 0 {
			return false
		}
		at := from + i
		end := at + len(tok)
		leftOK := at == 0 || lc[at-1] == ' ' || lc[at-1] == '\t' || isShellSep(lc[at-1])
		rightOK := end < len(lc) && (lc[end] == ' ' || lc[end] == '\t')
		if leftOK && rightOK {
			return true
		}
		from = at + 1
	}
}

// namesFlagToken reports whether flag appears in lc as its own argument token: a left
// boundary of whitespace and a right boundary of whitespace, a quote (`'` `"` “ ` “),
// `=`, or end of string. So `-e 'p'`, `-e'p'`, `-e"p"`, `--eval=p`, and a trailing `-e`
// all match, while `-e` inside `--eval` (left boundary `-`) or inside a longer word does
// not. The right-boundary set is what catches the no-space/quoted/`=`-joined spellings a
// fixed `<flag> ` prefix misses.
func namesFlagToken(lc, flag string) bool {
	for from := 0; ; {
		i := strings.Index(lc[from:], flag)
		if i < 0 {
			return false
		}
		at := from + i
		leftOK := at > 0 && (lc[at-1] == ' ' || lc[at-1] == '\t')
		end := at + len(flag)
		rightOK := end >= len(lc)
		if !rightOK {
			switch lc[end] {
			case ' ', '\t', '\'', '"', '`', '=':
				rightOK = true
			}
		}
		if leftOK && rightOK {
			return true
		}
		from = at + 1
	}
}
