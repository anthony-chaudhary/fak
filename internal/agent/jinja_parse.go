package agent

// Adapted from ggml-org/llama.cpp common/jinja lexer/parser at
// cd8427d018c07eefab6c110f7f115c31ec4c6157 (MIT). See JINJA-LICENSE.txt.
// This is a bounded chat-template language, not an embedded Python runtime.

import (
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	jinjaMaxSource = 64 << 10
	jinjaMaxTokens = 32768
	jinjaMaxNodes  = 16384
	jinjaMaxDepth  = 64
)

type jinjaError struct {
	phase, reason string
	offset        int
}

func (e *jinjaError) Error() string {
	return fmt.Sprintf("native chat template: %s/%s at byte %d", e.phase, e.reason, e.offset)
}
func jinjaFail(phase, reason string, pos int) error {
	return &jinjaError{phase: phase, reason: reason, offset: pos}
}

type jinjaToken struct {
	kind, text string
	pos        int
}
type jinjaChunk struct {
	kind, text string
	pos        int
}
type jinjaExpr struct {
	kind, text string
	pos        int
	args       []*jinjaExpr
	names      []string
}
type jinjaNode struct {
	kind, text  string
	pos         int
	exprs       []*jinjaExpr
	names       []string
	body, other []*jinjaNode
}
type nativeJinjaProgram struct{ nodes []*jinjaNode }

// The donor's chat defaults normalize newlines, strip one final LF,
// lstrip block indentation, and trim one LF after statement/comment tags.
func jinjaChunks(source string) ([]jinjaChunk, error) {
	if len(source) == 0 || len(source) > jinjaMaxSource || !utf8.ValidString(source) || strings.IndexByte(source, 0) >= 0 {
		return nil, jinjaFail("compile", "source", 0)
	}
	src := strings.ReplaceAll(strings.ReplaceAll(source, "\r\n", "\n"), "\r", "\n")
	src = strings.TrimSuffix(src, "\n")
	var out []jinjaChunk
	pos := 0
	stripNext, trimNext := false, false
	for pos < len(src) {
		at := -1
		for i := pos; i+1 < len(src); i++ {
			if src[i] == '{' && (src[i+1] == '{' || src[i+1] == '%' || src[i+1] == '#') {
				at = i
				break
			}
		}
		if at < 0 {
			at = len(src)
		}
		text := src[pos:at]
		if stripNext {
			text = strings.TrimLeft(text, " \t\r\n")
		} else if trimNext {
			text = strings.TrimPrefix(text, "\n")
		}
		stripNext, trimNext = false, false
		if at < len(src) {
			kind := src[at+1]
			if at+2 < len(src) && src[at+2] == '-' {
				text = strings.TrimRight(text, " \t\r\n")
			} else if kind == '%' || kind == '#' {
				start := strings.LastIndexByte(text, '\n') + 1
				if strings.Trim(text[start:], " \t") == "" {
					text = text[:start]
				}
			}
		}
		if text != "" {
			out = append(out, jinjaChunk{kind: "text", text: text, pos: pos})
		}
		if at == len(src) {
			break
		}
		kind := src[at+1]
		start := at + 2
		if start < len(src) && src[start] == '-' {
			start++
		}
		end := -1
		quote := byte(0)
		escaped := false
		depth := 0
		for i := start; i+1 < len(src); i++ {
			c := src[i]
			if kind == '#' {
				if c == '#' && src[i+1] == '}' {
					end = i
					break
				}
				continue
			}
			if quote != 0 {
				if escaped {
					escaped = false
				} else if c == '\\' {
					escaped = true
				} else if c == quote {
					quote = 0
				}
				continue
			}
			if c == '\'' || c == '"' {
				quote = c
				continue
			}
			if depth == 0 && ((kind == '{' && c == '}' && src[i+1] == '}') || (kind == '%' && c == '%' && src[i+1] == '}')) {
				end = i
				break
			}
			if c == '(' || c == '[' || c == '{' {
				depth++
			}
			if c == ')' || c == ']' || c == '}' {
				depth--
				if depth < 0 {
					return nil, jinjaFail("compile", "delimiter", i)
				}
			}
		}
		if end < 0 {
			return nil, jinjaFail("compile", "unterminated-tag", at)
		}
		inner := src[start:end]
		if strings.HasSuffix(inner, "-") {
			inner = inner[:len(inner)-1]
			stripNext = true
		}
		trimNext = kind == '%' || kind == '#'
		if kind != '#' {
			k := "stmt"
			if kind == '{' {
				k = "expr"
			}
			out = append(out, jinjaChunk{kind: k, text: inner, pos: start})
		}
		if len(out) > jinjaMaxNodes {
			return nil, jinjaFail("compile", "nodes", at)
		}
		pos = end + 2
	}
	return out, nil
}

func jinjaLex(s string, base int) ([]jinjaToken, error) {
	var out []jinjaToken
	add := func(k, t string, p int) error {
		out = append(out, jinjaToken{k, t, base + p})
		if len(out) > jinjaMaxTokens {
			return jinjaFail("compile", "tokens", base+p)
		}
		return nil
	}
	for i := 0; i < len(s); {
		c := s[i]
		if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
			i++
			continue
		}
		start := i
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' {
			i++
			for i < len(s) && ((s[i] >= 'a' && s[i] <= 'z') || (s[i] >= 'A' && s[i] <= 'Z') || (s[i] >= '0' && s[i] <= '9') || s[i] == '_') {
				i++
			}
			if err := add("id", s[start:i], start); err != nil {
				return nil, err
			}
			continue
		}
		if c >= '0' && c <= '9' {
			i++
			for i < len(s) && s[i] >= '0' && s[i] <= '9' {
				i++
			}
			if i+1 < len(s) && s[i] == '.' && s[i+1] >= '0' && s[i+1] <= '9' {
				i++
				for i < len(s) && s[i] >= '0' && s[i] <= '9' {
					i++
				}
			}
			if err := add("number", s[start:i], start); err != nil {
				return nil, err
			}
			continue
		}
		if c == '\'' || c == '"' {
			quote := c
			i++
			var b strings.Builder
			closed := false
			for i < len(s) {
				c = s[i]
				i++
				if c == quote {
					closed = true
					break
				}
				if c != '\\' {
					b.WriteByte(c)
					continue
				}
				if i == len(s) {
					return nil, jinjaFail("compile", "escape", base+i)
				}
				c = s[i]
				i++
				switch c {
				case '\\', '\'', '"':
					b.WriteByte(c)
				case 'n':
					b.WriteByte('\n')
				case 'r':
					b.WriteByte('\r')
				case 't':
					b.WriteByte('\t')
				case 'b':
					b.WriteByte('\b')
				case 'f':
					b.WriteByte('\f')
				case 'x', 'u', 'U':
					n := 2
					if c == 'u' {
						n = 4
					}
					if c == 'U' {
						n = 8
					}
					if i+n > len(s) {
						return nil, jinjaFail("compile", "escape", base+i)
					}
					u, err := strconv.ParseUint(s[i:i+n], 16, 32)
					if err != nil || u > utf8.MaxRune || u >= 0xd800 && u <= 0xdfff {
						return nil, jinjaFail("compile", "escape", base+i)
					}
					b.WriteRune(rune(u))
					i += n
				default:
					return nil, jinjaFail("compile", "escape", base+i-1)
				}
			}
			if !closed || !utf8.ValidString(b.String()) {
				return nil, jinjaFail("compile", "string", base+start)
			}
			if err := add("string", b.String(), start); err != nil {
				return nil, err
			}
			continue
		}
		op := ""
		for _, x := range []string{"==", "!=", ">=", "<=", "//", "**"} {
			if strings.HasPrefix(s[i:], x) {
				op = x
				break
			}
		}
		if op != "" {
			i += len(op)
		} else if strings.ContainsRune("()[]{}.,:|+-*/%~<>=!", rune(c)) {
			op = string(c)
			i++
		} else {
			return nil, jinjaFail("compile", "token", base+i)
		}
		if err := add(op, op, start); err != nil {
			return nil, err
		}
	}
	out = append(out, jinjaToken{"eof", "", base + len(s)})
	return out, nil
}

type jinjaExprParser struct {
	tokens    []jinjaToken
	at, depth int
	count     *int
}

func (p *jinjaExprParser) peek() jinjaToken { return p.tokens[p.at] }
func (p *jinjaExprParser) is(s string) bool {
	t := p.peek()
	return t.kind == s || t.kind == "id" && t.text == s
}
func (p *jinjaExprParser) take(s string) bool {
	if p.is(s) {
		p.at++
		return true
	}
	return false
}
func (p *jinjaExprParser) need(s string) error {
	if !p.take(s) {
		return jinjaFail("compile", "expected-token", p.peek().pos)
	}
	return nil
}
func (p *jinjaExprParser) node(k, t string, pos int, args ...*jinjaExpr) (*jinjaExpr, error) {
	(*p.count)++
	if *p.count > jinjaMaxNodes {
		return nil, jinjaFail("compile", "nodes", pos)
	}
	return &jinjaExpr{kind: k, text: t, pos: pos, args: args}, nil
}
func (p *jinjaExprParser) expression() (*jinjaExpr, error) {
	p.depth++
	defer func() { p.depth-- }()
	if p.depth > jinjaMaxDepth {
		return nil, jinjaFail("compile", "depth", p.peek().pos)
	}
	a, e := p.binary(0)
	if e != nil {
		return nil, e
	}
	if p.take("if") {
		test, e := p.binary(0)
		if e != nil {
			return nil, e
		}
		if p.take("else") {
			other, e := p.expression()
			if e != nil {
				return nil, e
			}
			return p.node("ternary", "", a.pos, a, test, other)
		}
		return p.node("select", "", a.pos, a, test)
	}
	return a, nil
}

var jinjaBinaryLevels = [][]string{{"or"}, {"and"}, {"==", "!=", "<", ">", "<=", ">=", "in", "not in"}, {"+", "-", "~"}, {"*", "/", "//", "%"}}

func (p *jinjaExprParser) binary(level int) (*jinjaExpr, error) {
	if level == len(jinjaBinaryLevels) {
		return p.unary()
	}
	if level == 2 && p.take("not") {
		t := p.tokens[p.at-1]
		p.depth++
		defer func() { p.depth-- }()
		if p.depth > jinjaMaxDepth {
			return nil, jinjaFail("compile", "depth", t.pos)
		}
		v, e := p.binary(2)
		if e != nil {
			return nil, e
		}
		return p.node("unary", "not", t.pos, v)
	}
	left, e := p.binary(level + 1)
	if e != nil {
		return nil, e
	}
	comparisons := 0
	for {
		op := ""
		for _, x := range jinjaBinaryLevels[level] {
			if x == "not in" {
				if p.is("not") && p.at+1 < len(p.tokens) && p.tokens[p.at+1].text == "in" {
					op = x
					p.at += 2
					break
				}
			} else if p.take(x) {
				op = x
				break
			}
		}
		if op == "" {
			return left, nil
		}
		if level == 2 {
			comparisons++
			if comparisons > 1 {
				return nil, jinjaFail("compile", "comparison-chain", left.pos)
			}
		}
		right, e := p.binary(level + 1)
		if e != nil {
			return nil, e
		}
		left, e = p.node("binary", op, left.pos, left, right)
		if e != nil {
			return nil, e
		}
	}
}
func (p *jinjaExprParser) unary() (*jinjaExpr, error) { return p.postfix(true) }
func (p *jinjaExprParser) postfix(filters bool) (*jinjaExpr, error) {
	p.depth++
	defer func() { p.depth-- }()
	if p.depth > jinjaMaxDepth {
		return nil, jinjaFail("compile", "depth", p.peek().pos)
	}
	t := p.peek()
	var a *jinjaExpr
	var e error
	if p.take("+") || p.take("-") {
		v, err := p.postfix(false)
		if err != nil {
			return nil, err
		}
		a, e = p.node("unary", t.text, t.pos, v)
	} else {
		a, e = p.primary()
	}
	if e != nil {
		return nil, e
	}
	for {
		switch {
		case p.take("."):
			t := p.peek()
			if t.kind != "id" {
				return nil, jinjaFail("compile", "attribute", t.pos)
			}
			p.at++
			a, e = p.node("attr", t.text, a.pos, a)
		case p.take("["):
			var start, stop, step *jinjaExpr
			if !p.is(":") && !p.is("]") {
				start, e = p.expression()
				if e != nil {
					return nil, e
				}
			}
			if p.take(":") {
				if !p.is(":") && !p.is("]") {
					stop, e = p.expression()
					if e != nil {
						return nil, e
					}
				}
				if p.take(":") {
					if !p.is("]") {
						step, e = p.expression()
						if e != nil {
							return nil, e
						}
					}
				}
				if e = p.need("]"); e != nil {
					return nil, e
				}
				a, e = p.node("slice", "", a.pos, a, start, stop, step)
			} else {
				if start == nil {
					return nil, jinjaFail("compile", "empty-index", p.peek().pos)
				}
				if e = p.need("]"); e != nil {
					return nil, e
				}
				a, e = p.node("index", "", a.pos, a, start)
			}
		case p.is("("):
			var args []*jinjaExpr
			var names []string
			args, names, e = p.arguments()
			if e != nil {
				return nil, e
			}
			n, err := p.node("call", "", a.pos, append([]*jinjaExpr{a}, args...)...)
			if err != nil {
				return nil, err
			}
			n.names = names
			a = n
		case filters && p.take("|"):
			t := p.peek()
			if t.kind != "id" {
				return nil, jinjaFail("compile", "filter", t.pos)
			}
			p.at++
			var args []*jinjaExpr
			var names []string
			if p.is("(") {
				args, names, e = p.arguments()
				if e != nil {
					return nil, e
				}
			}
			n, err := p.node("filter", t.text, a.pos, append([]*jinjaExpr{a}, args...)...)
			if err != nil {
				return nil, err
			}
			n.names = names
			a = n
		case filters && p.take("is"):
			not := p.take("not")
			t := p.peek()
			if t.kind != "id" {
				return nil, jinjaFail("compile", "test", t.pos)
			}
			p.at++
			var args []*jinjaExpr
			var names []string
			if p.is("(") {
				args, names, e = p.arguments()
				if e != nil {
					return nil, e
				}
			}
			n, err := p.node("test", t.text, a.pos, append([]*jinjaExpr{a}, args...)...)
			if err != nil {
				return nil, err
			}
			n.names = names
			a = n
			if not {
				a, e = p.node("unary", "not", a.pos, a)
			}
		default:
			return a, nil
		}
		if e != nil {
			return nil, e
		}
	}
}
func (p *jinjaExprParser) arguments() ([]*jinjaExpr, []string, error) {
	if e := p.need("("); e != nil {
		return nil, nil, e
	}
	var args []*jinjaExpr
	var names []string
	seen := map[string]bool{}
	keyword := false
	for !p.is(")") {
		name := ""
		if p.peek().kind == "id" && p.at+1 < len(p.tokens) && p.tokens[p.at+1].kind == "=" {
			name = p.peek().text
			p.at += 2
			keyword = true
			if seen[name] {
				return nil, nil, jinjaFail("compile", "duplicate-keyword", p.peek().pos)
			}
			seen[name] = true
		} else if keyword {
			return nil, nil, jinjaFail("compile", "positional-after-keyword", p.peek().pos)
		}
		v, e := p.expression()
		if e != nil {
			return nil, nil, e
		}
		args = append(args, v)
		names = append(names, name)
		if !p.take(",") {
			break
		}
	}
	if e := p.need(")"); e != nil {
		return nil, nil, e
	}
	return args, names, nil
}
func (p *jinjaExprParser) primary() (*jinjaExpr, error) {
	t := p.peek()
	p.at++
	switch t.kind {
	case "id":
		return p.node("id", t.text, t.pos)
	case "number":
		if _, e := strconv.ParseFloat(t.text, 64); e != nil {
			return nil, jinjaFail("compile", "number", t.pos)
		}
		return p.node("number", t.text, t.pos)
	case "string":
		text := t.text
		for p.peek().kind == "string" {
			text += p.peek().text
			p.at++
		}
		return p.node("string", text, t.pos)
	case "(":
		v, e := p.expression()
		if e != nil {
			return nil, e
		}
		if p.take(",") {
			items := []*jinjaExpr{v}
			for !p.is(")") {
				x, e := p.expression()
				if e != nil {
					return nil, e
				}
				items = append(items, x)
				if !p.take(",") {
					break
				}
			}
			if e = p.need(")"); e != nil {
				return nil, e
			}
			return p.node("array", "", t.pos, items...)
		}
		if e = p.need(")"); e != nil {
			return nil, e
		}
		return v, nil
	case "[", "{":
		close := "]"
		kind := "array"
		if t.kind == "{" {
			close = "}"
			kind = "object"
		}
		var items []*jinjaExpr
		for !p.is(close) {
			x, e := p.expression()
			if e != nil {
				return nil, e
			}
			items = append(items, x)
			if kind == "object" {
				if e = p.need(":"); e != nil {
					return nil, e
				}
				v, e := p.expression()
				if e != nil {
					return nil, e
				}
				items = append(items, v)
			}
			if !p.take(",") {
				break
			}
		}
		if e := p.need(close); e != nil {
			return nil, e
		}
		return p.node(kind, "", t.pos, items...)
	default:
		p.at--
		return nil, jinjaFail("compile", "expression", t.pos)
	}
}

type jinjaParser struct {
	chunks                      []jinjaChunk
	at, count, depth, loopDepth int
}

func (p *jinjaParser) expr(c jinjaChunk) (*jinjaExpr, error) {
	t, e := jinjaLex(c.text, c.pos)
	if e != nil {
		return nil, e
	}
	q := jinjaExprParser{tokens: t, count: &p.count}
	v, e := q.expression()
	if e != nil {
		return nil, e
	}
	if !q.is("eof") {
		return nil, jinjaFail("compile", "trailing-expression", q.peek().pos)
	}
	return v, nil
}
func jinjaWord(s string) (string, string) {
	s = strings.TrimSpace(s)
	for i, c := range s {
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			return s[:i], strings.TrimSpace(s[i:])
		}
	}
	return s, ""
}
func (p *jinjaParser) body(stops map[string]bool) ([]*jinjaNode, string, error) {
	p.depth++
	defer func() { p.depth-- }()
	if p.depth > jinjaMaxDepth {
		return nil, "", jinjaFail("compile", "block-depth", 0)
	}
	var out []*jinjaNode
	for p.at < len(p.chunks) {
		c := p.chunks[p.at]
		word, rest := jinjaWord(c.text)
		if c.kind == "stmt" && stops[word] {
			return out, word, nil
		}
		p.at++
		p.count++
		if p.count > jinjaMaxNodes {
			return nil, "", jinjaFail("compile", "nodes", c.pos)
		}
		n := &jinjaNode{kind: c.kind, text: c.text, pos: c.pos}
		if c.kind == "text" {
			out = append(out, n)
			continue
		}
		if c.kind == "expr" {
			v, e := p.expr(c)
			if e != nil {
				return nil, "", e
			}
			n.exprs = []*jinjaExpr{v}
			out = append(out, n)
			continue
		}
		n.kind = word
		n.text = ""
		c.text = rest
		switch word {
		case "if":
			v, e := p.expr(c)
			if e != nil {
				return nil, "", e
			}
			n.exprs = []*jinjaExpr{v}
			body, stop, e := p.body(map[string]bool{"elif": true, "else": true, "endif": true})
			if e != nil {
				return nil, "", e
			}
			n.body = body
			cursor := n
			for stop == "elif" {
				c = p.chunks[p.at]
				p.at++
				_, c.text = jinjaWord(c.text)
				v, e = p.expr(c)
				if e != nil {
					return nil, "", e
				}
				p.count++
				if p.count > jinjaMaxNodes {
					return nil, "", jinjaFail("compile", "nodes", c.pos)
				}
				next := &jinjaNode{kind: "if", pos: c.pos, exprs: []*jinjaExpr{v}}
				cursor.other = []*jinjaNode{next}
				cursor = next
				cursor.body, stop, e = p.body(map[string]bool{"elif": true, "else": true, "endif": true})
				if e != nil {
					return nil, "", e
				}
			}
			if stop == "else" {
				if e = p.end("else"); e != nil {
					return nil, "", e
				}
				cursor.other, stop, e = p.body(map[string]bool{"endif": true})
				if e != nil {
					return nil, "", e
				}
			}
			if stop != "endif" {
				return nil, "", jinjaFail("compile", "unterminated-if", n.pos)
			}
			if e = p.end("endif"); e != nil {
				return nil, "", e
			}
		case "for":
			ts, e := jinjaLex(c.text, c.pos)
			if e != nil {
				return nil, "", e
			}
			q := jinjaExprParser{tokens: ts, count: &p.count}
			for {
				t := q.peek()
				if t.kind != "id" || t.text == "in" {
					return nil, "", jinjaFail("compile", "loop-target", t.pos)
				}
				q.at++
				n.names = append(n.names, t.text)
				if !q.take(",") {
					break
				}
			}
			if e = q.need("in"); e != nil {
				return nil, "", e
			}
			v, e := q.expression()
			if e != nil {
				return nil, "", e
			}
			if !q.is("eof") {
				return nil, "", jinjaFail("compile", "loop-syntax", q.peek().pos)
			}
			n.exprs = []*jinjaExpr{v}
			p.loopDepth++
			n.body, word, e = p.body(map[string]bool{"else": true, "endfor": true})
			p.loopDepth--
			if e != nil {
				return nil, "", e
			}
			if word == "else" {
				if e = p.end("else"); e != nil {
					return nil, "", e
				}
				n.other, word, e = p.body(map[string]bool{"endfor": true})
				if e != nil {
					return nil, "", e
				}
			}
			if word != "endfor" {
				return nil, "", jinjaFail("compile", "unterminated-for", n.pos)
			}
			if e = p.end("endfor"); e != nil {
				return nil, "", e
			}
		case "set":
			ts, e := jinjaLex(c.text, c.pos)
			if e != nil {
				return nil, "", e
			}
			q := jinjaExprParser{tokens: ts, count: &p.count}
			target, e := q.primary()
			if e != nil {
				return nil, "", e
			}
			if target.kind != "id" {
				return nil, "", jinjaFail("compile", "set-target", n.pos)
			}
			if q.take(".") {
				t := q.peek()
				if t.kind != "id" {
					return nil, "", jinjaFail("compile", "set-target", n.pos)
				}
				q.at++
				target, e = q.node("attr", t.text, t.pos, target)
				if e != nil {
					return nil, "", e
				}
			}
			n.exprs = []*jinjaExpr{target}
			if q.take("=") {
				v, e := q.expression()
				if e != nil {
					return nil, "", e
				}
				n.exprs = append(n.exprs, v)
				if !q.is("eof") {
					return nil, "", jinjaFail("compile", "set-syntax", q.peek().pos)
				}
			} else {
				if !q.is("eof") {
					return nil, "", jinjaFail("compile", "set-syntax", q.peek().pos)
				}
				n.body, word, e = p.body(map[string]bool{"endset": true})
				if e != nil {
					return nil, "", e
				}
				if word != "endset" {
					return nil, "", jinjaFail("compile", "unterminated-set", n.pos)
				}
				if e = p.end("endset"); e != nil {
					return nil, "", e
				}
			}
		case "macro":
			ts, e := jinjaLex(c.text, c.pos)
			if e != nil {
				return nil, "", e
			}
			q := jinjaExprParser{tokens: ts, count: &p.count}
			t := q.peek()
			if t.kind != "id" {
				return nil, "", jinjaFail("compile", "macro-name", n.pos)
			}
			q.at++
			n.text = t.text
			if e = q.need("("); e != nil {
				return nil, "", e
			}
			seen := map[string]bool{}
			defaultSeen := false
			for !q.is(")") {
				t = q.peek()
				if t.kind != "id" || seen[t.text] {
					return nil, "", jinjaFail("compile", "macro-parameter", t.pos)
				}
				q.at++
				seen[t.text] = true
				n.names = append(n.names, t.text)
				var v *jinjaExpr
				if q.take("=") {
					defaultSeen = true
					v, e = q.expression()
					if e != nil {
						return nil, "", e
					}
				} else if defaultSeen {
					return nil, "", jinjaFail("compile", "macro-required-after-default", t.pos)
				}
				n.exprs = append(n.exprs, v)
				if !q.take(",") {
					break
				}
			}
			if e = q.need(")"); e != nil {
				return nil, "", e
			}
			if !q.is("eof") {
				return nil, "", jinjaFail("compile", "macro-syntax", q.peek().pos)
			}
			n.body, word, e = p.body(map[string]bool{"endmacro": true})
			if e != nil {
				return nil, "", e
			}
			if word != "endmacro" {
				return nil, "", jinjaFail("compile", "unterminated-macro", n.pos)
			}
			if e = p.end("endmacro"); e != nil {
				return nil, "", e
			}
		case "break", "continue":
			if rest != "" || p.loopDepth == 0 {
				return nil, "", jinjaFail("compile", "loop-control", n.pos)
			}
		case "generation":
			if rest != "" {
				return nil, "", jinjaFail("compile", "generation", n.pos)
			}
			var e error
			n.body, word, e = p.body(map[string]bool{"endgeneration": true})
			if e != nil {
				return nil, "", e
			}
			if word != "endgeneration" {
				return nil, "", jinjaFail("compile", "unterminated-generation", n.pos)
			}
			if e = p.end("endgeneration"); e != nil {
				return nil, "", e
			}
		default:
			return nil, "", jinjaFail("compile", "unsupported-statement", n.pos)
		}
		out = append(out, n)
	}
	return out, "", nil
}
func (p *jinjaParser) end(word string) error {
	if p.at >= len(p.chunks) {
		return jinjaFail("compile", "missing-end", 0)
	}
	c := p.chunks[p.at]
	w, rest := jinjaWord(c.text)
	if c.kind != "stmt" || w != word || rest != "" {
		return jinjaFail("compile", "end-syntax", c.pos)
	}
	p.at++
	return nil
}
func compileNativeJinja(source string) (*nativeJinjaProgram, error) {
	chunks, e := jinjaChunks(source)
	if e != nil {
		return nil, e
	}
	tokens := 0
	for _, c := range chunks {
		if c.kind == "text" {
			continue
		}
		ts, err := jinjaLex(c.text, c.pos)
		if err != nil {
			return nil, err
		}
		tokens += len(ts) - 1
		if tokens > jinjaMaxTokens {
			return nil, jinjaFail("compile", "tokens", c.pos)
		}
	}
	p := jinjaParser{chunks: chunks}
	nodes, _, e := p.body(nil)
	if e != nil {
		return nil, e
	}
	return &nativeJinjaProgram{nodes: nodes}, nil
}
