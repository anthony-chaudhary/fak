package toolproc

// repeatnormalize.go — the BOUNDARY front door for #4764's classifier: turn a raw
// tool-call observation into a redacted, canonicalized NormalCall whose Identity
// folds equivalent spellings and whose Class is a fail-closed registry verdict.
// This is the half repeatclass.go's ClassifyRepeats fold depends on; keeping it in
// its own file keeps the decision spine readable.
//
// THREE JOBS, in order, so nothing downstream ever sees a secret or a raw body:
//
//  1. REDACT. Secret-bearing argument spans (a `--token ghp_…`, a `sk-…`, an
//     `AKIA…`, a `Bearer …`, a `PASSWORD=…`) are replaced with a placeholder at
//     the token boundary BEFORE any field is retained. The analytics surface holds
//     only redacted canonical text and the tool-result SIZE — never a credential,
//     never an output payload.
//
//  2. CANONICALIZE + FOLD. Quotes are stripped, path separators normalized, and a
//     registered command's flags sorted, so two spellings that mean the same thing
//     collapse to one Identity. `Get-Content -Raw C:\x\SKILL.md`,
//     `Get-Content -LiteralPath C:/x/SKILL.md -Raw`, and `cat C:\x\SKILL.md` all
//     resolve to the immutable identity `read:C:/x/SKILL.md` — the near-duplicate
//     fold the 640+204 skill-read audit line demands.
//
//  3. CLASSIFY. The registered-command matcher assigns a CommandClass by
//     MUTABILITY — the only property that decides whether a repeat is ever safe to
//     serve locally. An unmatched command is CmdUnknown: fail-closed, never reused.
//
// The registry is DATA (registeredSpecs), not scattered ifs, so the closed set of
// commands fak is willing to reason about is auditable in one place and cannot
// drift. Adding a command is adding a row.

import (
	"regexp"
	"sort"
	"strings"
)

// Normalize turns one raw CallRecord into its redacted, canonical NormalCall. It is
// pure and total: every record maps to exactly one NormalCall, and an unrecognized
// command falls through to CmdUnknown rather than erroring. Secrets are redacted
// before any returned field is built.
func Normalize(rec CallRecord, cfg RepeatConfig) NormalCall {
	if cfg.DefaultFreshnessMS == 0 {
		cfg.DefaultFreshnessMS = DefaultFreshnessWindowMS
	}
	toks := redactTokens(tokenize(rec.Raw))
	tool := strings.TrimSpace(rec.Tool)

	prog, sub, rest := splitProgram(toks)
	spec, path := matchSpec(tool, prog, sub, rest)

	switch spec.class {
	case CmdImmutableRead:
		cp := canonPath(path)
		digest := strings.TrimSpace(rec.Digest)
		id := "read:" + cp // program- and spelling-independent
		if digest != "" {
			// Fold the observed content/identity digest into the reuse key: a
			// re-read after a mutation carries a NEW digest → a NEW identity → the
			// stale entry is never served (invalidation after mutation). A read with
			// no observed digest keeps the conservative path-only fold.
			id += "@" + digest
		}
		return NormalCall{
			Tool:      tool,
			Canonical: "read " + cp,
			Path:      cp,
			Digest:    digest,
			Class:     CmdImmutableRead,
			Identity:  id,
		}
	case CmdMutableQuery:
		canon := spec.canonName + joinSorted(rest)
		fresh := spec.freshMS
		if fresh == 0 {
			fresh = cfg.DefaultFreshnessMS
		}
		return NormalCall{
			Tool:      tool,
			Canonical: canon,
			Class:     CmdMutableQuery,
			FreshMS:   fresh,
			Identity:  "query:" + canon,
		}
	case CmdIdempotentWrite:
		// Write args keep their order — a write is never reused, and the semantic
		// order (remote before refspec) makes the canonical form readable.
		canon := strings.TrimRight(spec.canonName+" "+strings.Join(rest, " "), " ")
		return NormalCall{
			Tool:      tool,
			Canonical: canon,
			Class:     CmdIdempotentWrite,
			Identity:  "write:" + canon,
		}
	default:
		canon := strings.TrimSpace(strings.Join(canonTokens(toks), " "))
		return NormalCall{
			Tool:      tool,
			Canonical: canon,
			Class:     CmdUnknown,
			Identity:  "unknown:" + tool + "|" + canon,
		}
	}
}

// cmdSpec is one row of the closed registered-command registry. canonName is the
// stable program label the Identity is built from (a basename, so `./tools/x.py`
// and `x.py` fold); freshMS is a per-command declared freshness window (0 => the
// config default).
type cmdSpec struct {
	class     CommandClass
	canonName string
	freshMS   int64
}

// readerVerbs are the shell programs whose job is to read a file's content — the
// immutable-read front. Detected by basename so a full path folds.
var readerVerbs = map[string]bool{
	"cat": true, "type": true, "bat": true, "get-content": true, "gc": true,
}

// readerTools are fak/host tool names (not shell programs) that denote a file read.
var readerTools = map[string]bool{
	"read": true, "read_file": true, "get-content": true,
}

// registeredSpecs is the CLOSED registry keyed on (program, subcommand). "" as the
// subcommand matches any. First a specific (prog,sub) is tried, then (prog,"").
var registeredSpecs = map[[2]string]cmdSpec{
	{"git", "status"}: {class: CmdMutableQuery, canonName: "git status"},
	{"git", "diff"}:   {class: CmdMutableQuery, canonName: "git diff"},
	{"git", "log"}:    {class: CmdMutableQuery, canonName: "git log"},
	{"git", "push"}:   {class: CmdIdempotentWrite, canonName: "git push"},
	{"git", "commit"}: {class: CmdIdempotentWrite, canonName: "git commit"},
	{"git", "add"}:    {class: CmdIdempotentWrite, canonName: "git add"},
	{"git", "fetch"}:  {class: CmdIdempotentWrite, canonName: "git fetch"},
	{"git", "pull"}:   {class: CmdIdempotentWrite, canonName: "git pull"},
}

// registeredScripts maps a script basename to its mutable-status spec — the
// dispatch-status family the audit shows polled 631+274 times.
var registeredScripts = map[string]cmdSpec{
	"dispatch_status.py": {class: CmdMutableQuery, canonName: "dispatch_status.py"},
}

// matchSpec resolves a normalized invocation to its registry spec and, for a read,
// the file path. It is fail-closed: no match => CmdUnknown.
func matchSpec(tool, prog, sub string, rest []string) (cmdSpec, string) {
	// 1. A read TOOL (host-level Read) or a reader shell verb → immutable read.
	if readerTools[strings.ToLower(tool)] {
		return cmdSpec{class: CmdImmutableRead}, firstPathArg(append([]string{sub}, rest...))
	}
	if readerVerbs[prog] {
		return cmdSpec{class: CmdImmutableRead}, readerPath(sub, rest)
	}
	// 2. A registered (program, subcommand) — specific before wildcard.
	if spec, ok := registeredSpecs[[2]string{prog, sub}]; ok {
		return spec, ""
	}
	if spec, ok := registeredSpecs[[2]string{prog, ""}]; ok {
		return spec, ""
	}
	// 3. A registered script by basename (dispatch_status.py and kin).
	if spec, ok := registeredScripts[baseName(prog)]; ok {
		return spec, ""
	}
	return cmdSpec{class: CmdUnknown}, ""
}

// splitProgram returns the lowercased program basename, its subcommand (lowercased,
// "" if none or if the next token is a flag), and the remaining argument tokens
// (original spelling, minus the program and a consumed git-style subcommand).
func splitProgram(toks []string) (prog, sub string, rest []string) {
	if len(toks) == 0 {
		return "", "", nil
	}
	prog = strings.ToLower(baseName(toks[0]))
	args := toks[1:]
	// A git-style subcommand is the first non-flag token for programs that take one.
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") && takesSubcommand(prog) {
		sub = strings.ToLower(args[0])
		rest = args[1:]
		return prog, sub, rest
	}
	// Otherwise the first arg is exposed as sub for reader path-extraction, but not
	// consumed (rest still includes it via the caller's append for read tools).
	if len(args) > 0 {
		sub = args[0]
	}
	if len(args) > 1 {
		rest = args[1:]
	}
	return prog, sub, rest
}

func takesSubcommand(prog string) bool {
	switch prog {
	case "git", "gh", "cargo", "go", "npm", "docker", "kubectl":
		return true
	}
	return false
}

// readerPath extracts the file a reader command reads: the value of an explicit
// -Path/-LiteralPath flag, else the last non-flag, non-consumed token. sub is the
// first arg (may itself be the path for `cat X`), rest is the tail.
func readerPath(sub string, rest []string) string {
	all := append([]string{sub}, rest...)
	pathFlags := map[string]bool{"-path": true, "-literalpath": true, "-lp": true}
	valueFlags := map[string]bool{"-path": true, "-literalpath": true, "-lp": true, "-encoding": true, "-totalcount": true, "-tail": true, "-head": true, "-first": true}
	var fromFlag, lastNonFlag string
	for i := 0; i < len(all); i++ {
		t := all[i]
		if t == "" {
			continue
		}
		if strings.HasPrefix(t, "-") {
			lf := strings.ToLower(t)
			if eq := strings.IndexByte(lf, '='); eq >= 0 { // -Path=X form
				if pathFlags[lf[:eq]] {
					fromFlag = t[eq+1:]
				}
				continue
			}
			if valueFlags[lf] && i+1 < len(all) {
				if pathFlags[lf] {
					fromFlag = all[i+1]
				}
				i++ // consume the value
			}
			continue
		}
		lastNonFlag = t
	}
	if fromFlag != "" {
		return fromFlag
	}
	return lastNonFlag
}

// firstPathArg returns the first non-flag token — the path for a host read tool
// whose single argument is the file.
func firstPathArg(args []string) string {
	for _, t := range args {
		if t != "" && !strings.HasPrefix(t, "-") {
			return t
		}
	}
	return ""
}

// canonPath normalizes a file path for identity folding: strip surrounding quotes
// (already handled by tokenize), normalize `\` to `/`, and collapse repeated
// slashes. Case is preserved — folding case would falsely merge distinct files on a
// case-sensitive filesystem; only spelling that cannot change identity is folded.
func canonPath(p string) string {
	p = strings.Trim(p, `"'`)
	p = strings.ReplaceAll(p, `\`, `/`)
	for strings.Contains(p, "//") {
		p = strings.ReplaceAll(p, "//", "/")
	}
	return strings.TrimRight(p, "/")
}

// joinSorted returns " a b c" (leading space, sorted) for a registered command's
// flag/arg tail, so flag ORDER never splits an identity: `--short --branch` and
// `--branch --short` fold. Empty tail => "".
func joinSorted(rest []string) string {
	if len(rest) == 0 {
		return ""
	}
	cp := canonTokens(rest)
	sort.Strings(cp)
	return " " + strings.Join(cp, " ")
}

// canonTokens normalizes each token's path separators without reordering — used for
// the unknown canonical form (order-preserving, since we do not understand it).
func canonTokens(toks []string) []string {
	out := make([]string, 0, len(toks))
	for _, t := range toks {
		out = append(out, strings.ReplaceAll(strings.Trim(t, `"'`), `\`, `/`))
	}
	return out
}

func baseName(p string) string {
	p = strings.ReplaceAll(p, `\`, `/`)
	if i := strings.LastIndexByte(p, '/'); i >= 0 {
		return p[i+1:]
	}
	return p
}

// tokenize splits a command line on unquoted whitespace, stripping quotes so
// `"C:\x\SKILL.md"` and `C:\x\SKILL.md` tokenize identically. It is deliberately
// small: backslash-escapes are not interpreted (Windows paths use them literally).
func tokenize(s string) []string {
	var toks []string
	var cur strings.Builder
	inS, inD := false, false
	flush := func() {
		if cur.Len() > 0 {
			toks = append(toks, cur.String())
			cur.Reset()
		}
	}
	for _, r := range s {
		switch {
		case r == '\'' && !inD:
			inS = !inS
		case r == '"' && !inS:
			inD = !inD
		case (r == ' ' || r == '\t' || r == '\n' || r == '\r') && !inS && !inD:
			flush()
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return toks
}

// --- secret redaction -------------------------------------------------------

// selfSecret matches tokens that ARE a credential by their own shape — no
// surrounding flag needed. Anchored so a partial match never leaks a prefix.
var selfSecret = regexp.MustCompile(`^(?:` +
	`gh[posur]_[A-Za-z0-9]{20,}` + // GitHub PAT/OAuth/refresh/server tokens
	`|github_pat_[A-Za-z0-9_]{20,}` +
	`|sk-[A-Za-z0-9_-]{20,}` + // OpenAI-style
	`|xox[baprs]-[A-Za-z0-9-]{10,}` + // Slack
	`|AKIA[0-9A-Z]{16}` + // AWS access key id
	`|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}` + // JWT
	`)$`)

// secretKV matches a KEY=VALUE token whose KEY names a credential.
var secretKV = regexp.MustCompile(`(?i)^([A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|APIKEY|API_KEY|ACCESS_KEY)[A-Za-z0-9_]*)=(.+)$`)

// secretFlags are option flags whose FOLLOWING token is a credential value.
var secretFlags = map[string]bool{
	"--token": true, "--api-key": true, "--apikey": true, "--password": true,
	"--secret": true, "--auth": true, "--pass": true, "bearer": true,
}

const redactedPlaceholder = "<redacted>"

// redactTokens replaces credential spans with a placeholder, token by token, so no
// returned field ever carries a secret. Three shapes: a self-identifying token, a
// KEY=VALUE credential, and the value following a secret-naming flag.
func redactTokens(toks []string) []string {
	out := make([]string, len(toks))
	copy(out, toks)
	for i := 0; i < len(out); i++ {
		t := out[i]
		if selfSecret.MatchString(t) {
			out[i] = redactedPlaceholder
			continue
		}
		if m := secretKV.FindStringSubmatch(t); m != nil {
			out[i] = m[1] + "=" + redactedPlaceholder
			continue
		}
		if secretFlags[strings.ToLower(t)] && i+1 < len(out) {
			out[i+1] = redactedPlaceholder
			i++
		}
	}
	return out
}
