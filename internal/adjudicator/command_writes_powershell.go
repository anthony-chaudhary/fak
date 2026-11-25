package adjudicator

import "strings"

// psWords tokenizes a PowerShell command segment while preserving literal
// backslashes ('\' is a Windows path separator in PowerShell, not an escape
// character; PowerShell escapes with backtick '`').
func psWords(cmd string) []shellWord {
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
		if c == '`' && quote != '\'' {
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
		case '\'', '"':
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

// psFlagValues extracts PowerShell parameter values case-insensitively, handling
// both space-separated (-Path .env) and colon-separated (-Path:.env) forms.
func psFlagValues(args []shellWord, names ...string) []string {
	var values []string
	for i := 0; i < len(args); i++ {
		tok := args[i].text
		lt := strings.ToLower(tok)
		for _, name := range names {
			if lt == name && i+1 < len(args) {
				values = append(values, args[i+1].text)
				i++
				break
			}
			if strings.HasPrefix(lt, name+":") {
				values = append(values, tok[len(name)+1:])
				break
			}
		}
	}
	return values
}

// psPositionalOperands returns positional (non-flag) operands from a PowerShell
// cmdlet invocation, skipping the argument values of common non-path parameters.
func psPositionalOperands(args []shellWord, valueFlags ...string) []string {
	var out []string
	skipNext := false
	for _, w := range args {
		tok := cleanShellOperand(w.text)
		if skipNext {
			skipNext = false
			continue
		}
		if tok == "" || tok == "--" {
			continue
		}
		if strings.HasPrefix(tok, "-") {
			lt := strings.ToLower(tok)
			if !strings.Contains(lt, ":") {
				for _, vf := range append(valueFlags, psCommonValueParams...) {
					if lt == vf {
						skipNext = true
						break
					}
				}
			}
			continue
		}
		out = append(out, tok)
	}
	return out
}

// psCmdletParams lists each write cmdlet's own parameters so an abbreviated
// parameter (`-Va` for -Value) resolves as PowerShell binds it; an unresolved
// abbreviation would otherwise be read as a switch and shift the positional
// path onto its value.
var psCmdletParams = map[string][]string{
	"set-content":   {"path", "literalpath", "value", "passthru", "filter", "include", "exclude", "force", "credential", "nonewline", "encoding", "asbytestream", "stream"},
	"add-content":   {"path", "literalpath", "value", "passthru", "filter", "include", "exclude", "force", "credential", "nonewline", "encoding", "asbytestream", "stream"},
	"out-file":      {"filepath", "literalpath", "encoding", "append", "force", "noclobber", "width", "nonewline", "inputobject"},
	"tee-object":    {"filepath", "literalpath", "append", "encoding", "variable", "inputobject"},
	"clear-content": {"path", "literalpath", "filter", "include", "exclude", "force", "credential", "stream"},
	"remove-item":   {"path", "literalpath", "filter", "include", "exclude", "recurse", "force", "credential", "stream"},
	"copy-item":     {"path", "literalpath", "destination", "container", "force", "filter", "include", "exclude", "recurse", "passthru", "credential", "fromsession", "tosession"},
	"move-item":     {"path", "literalpath", "destination", "force", "filter", "include", "exclude", "passthru", "credential"},
	"rename-item":   {"path", "literalpath", "newname", "force", "passthru", "credential"},
	"new-item":      {"path", "name", "itemtype", "value", "force", "credential"},
}

var psAliasCmdlet = map[string]string{
	"sc": "set-content", "ac": "add-content", "clc": "clear-content", "ni": "new-item",
	"ri": "remove-item", "del": "remove-item", "erase": "remove-item", "rd": "remove-item",
	"cpi": "copy-item", "copy": "copy-item", "mi": "move-item", "move": "move-item",
	"rni": "rename-item", "ren": "rename-item",
}

// psCommonParams are the cmdlet-binding common parameters; the value-taking ones
// are repeated in psCommonValueParams so their value is never taken for a path.
var psCommonParams = []string{
	"verbose", "debug", "whatif", "confirm",
	"erroraction", "warningaction", "informationaction", "progressaction",
	"errorvariable", "warningvariable", "informationvariable", "outvariable",
	"outbuffer", "pipelinevariable",
}

var psCommonValueParams = []string{
	"-erroraction", "-warningaction", "-informationaction", "-progressaction",
	"-errorvariable", "-warningvariable", "-informationvariable", "-outvariable",
	"-outbuffer", "-pipelinevariable",
}

var psParamAliases = map[string]string{
	"ea": "erroraction", "wa": "warningaction", "infa": "informationaction", "proga": "progressaction",
	"ev": "errorvariable", "wv": "warningvariable", "iv": "informationvariable", "ov": "outvariable",
	"ob": "outbuffer", "pv": "pipelinevariable", "vb": "verbose", "db": "debug", "wi": "whatif", "cf": "confirm",
	"pspath": "literalpath", "lp": "literalpath",
}

// psCanonicalParams rewrites each parameter token of a write cmdlet to its full
// lowercase name when it is an alias, exact name, or unique prefix. Ambiguous
// prefixes are left untouched: PowerShell refuses to bind them.
func psCanonicalParams(head string, args []shellWord) []shellWord {
	cmdlet := head
	if c, ok := psAliasCmdlet[head]; ok {
		cmdlet = c
	}
	params := append(append([]string(nil), psCmdletParams[cmdlet]...), psCommonParams...)
	out := make([]shellWord, len(args))
	for i, w := range args {
		out[i] = w
		if w.quoted || len(w.text) < 2 || w.text[0] != '-' || w.text[1] == '-' {
			continue
		}
		name, rest := w.text[1:], ""
		if j := strings.IndexByte(name, ':'); j >= 0 {
			name, rest = name[:j], name[j:]
		}
		if canon := psResolveParam(strings.ToLower(name), params); canon != "" {
			out[i].text = "-" + canon + rest
		}
	}
	return out
}

func psResolveParam(name string, params []string) string {
	if canon, ok := psParamAliases[name]; ok {
		return canon
	}
	match, n := "", 0
	for _, p := range params {
		if p == name {
			return p
		}
		if strings.HasPrefix(p, name) {
			match, n = p, n+1
		}
	}
	if n != 1 {
		return ""
	}
	return match
}
