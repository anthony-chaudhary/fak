package trajquery

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// Parse reads a SELECT statement in the supported subset:
//
//	SELECT <col[,col...] | *> FROM <relation> [WHERE <pred> [AND <pred>]...] [LIMIT <n>]
//
// Predicates are `field <op> literal`; literals are single-quoted strings or bare numbers.
// Operators: = != < > <= >= LIKE. Only AND-conjunction is supported — OR, joins, subqueries
// and functions are rejected, which keeps the scope-rewrite guarantee decidable.
func Parse(sql string) (Query, error) {
	toks, err := tokenize(sql)
	if err != nil {
		return Query{}, err
	}
	p := &parser{toks: toks}
	return p.parseSelect()
}

type token struct {
	text   string
	quoted bool // a single-quoted string literal (so 'FROM' is a value, not a keyword)
}

func tokenize(s string) ([]token, error) {
	var toks []token
	rs := []rune(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(s), ";")))
	i := 0
	for i < len(rs) {
		c := rs[i]
		switch {
		case unicode.IsSpace(c):
			i++
		case c == ',':
			toks = append(toks, token{text: ","})
			i++
		case c == '\'':
			// quoted string literal, '' escapes a quote
			var b strings.Builder
			i++
			for i < len(rs) {
				if rs[i] == '\'' {
					if i+1 < len(rs) && rs[i+1] == '\'' {
						b.WriteRune('\'')
						i += 2
						continue
					}
					i++
					goto done
				}
				b.WriteRune(rs[i])
				i++
			}
			return nil, fmt.Errorf("unterminated string literal")
		done:
			toks = append(toks, token{text: b.String(), quoted: true})
		case c == '=' || c == '<' || c == '>' || c == '!':
			// multi-char operators: <= >= !=
			op := string(c)
			if i+1 < len(rs) && rs[i+1] == '=' {
				op += "="
				i++
			}
			if op == "!" {
				return nil, fmt.Errorf("unexpected '!' (did you mean '!=')")
			}
			toks = append(toks, token{text: op})
			i++
		default:
			// bareword: identifier, keyword, number
			start := i
			for i < len(rs) && !unicode.IsSpace(rs[i]) && rs[i] != ',' && rs[i] != '=' &&
				rs[i] != '<' && rs[i] != '>' && rs[i] != '!' && rs[i] != '\'' {
				i++
			}
			toks = append(toks, token{text: string(rs[start:i])})
		}
	}
	return toks, nil
}

type parser struct {
	toks []token
	pos  int
}

func (p *parser) peek() (token, bool) {
	if p.pos < len(p.toks) {
		return p.toks[p.pos], true
	}
	return token{}, false
}

func (p *parser) next() (token, bool) {
	t, ok := p.peek()
	if ok {
		p.pos++
	}
	return t, ok
}

// kwEq reports whether an UNQUOTED token equals a keyword (case-insensitive).
func kwEq(t token, kw string) bool {
	return !t.quoted && strings.EqualFold(t.text, kw)
}

func (p *parser) parseSelect() (Query, error) {
	var q Query
	head, ok := p.next()
	if !ok || !kwEq(head, "SELECT") {
		return q, fmt.Errorf("expected SELECT")
	}
	cols, err := p.parseColumns()
	if err != nil {
		return q, err
	}
	q.Columns = cols

	from, ok := p.next()
	if !ok || !kwEq(from, "FROM") {
		return q, fmt.Errorf("expected FROM after column list")
	}
	rel, ok := p.next()
	if !ok || rel.quoted || isKeyword(rel.text) {
		return q, fmt.Errorf("expected a relation name after FROM")
	}
	q.From = rel.text

	// optional WHERE
	if t, ok := p.peek(); ok && kwEq(t, "WHERE") {
		p.pos++
		preds, err := p.parseWhere()
		if err != nil {
			return q, err
		}
		q.Where = preds
	}
	// optional LIMIT
	if t, ok := p.peek(); ok && kwEq(t, "LIMIT") {
		p.pos++
		n, ok := p.next()
		if !ok {
			return q, fmt.Errorf("expected a number after LIMIT")
		}
		lim, err := strconv.Atoi(n.text)
		if err != nil || lim < 0 {
			return q, fmt.Errorf("invalid LIMIT %q", n.text)
		}
		q.Limit = lim
	}
	if t, ok := p.peek(); ok {
		return q, fmt.Errorf("unexpected trailing token %q", t.text)
	}
	return q, nil
}

func (p *parser) parseColumns() ([]string, error) {
	var cols []string
	for {
		t, ok := p.next()
		if !ok {
			return nil, fmt.Errorf("expected a column")
		}
		if t.text == "*" && len(cols) == 0 {
			cols = append(cols, "*")
		} else {
			if t.quoted || isKeyword(t.text) || t.text == "," {
				return nil, fmt.Errorf("expected a column name, got %q", t.text)
			}
			cols = append(cols, t.text)
		}
		nt, ok := p.peek()
		if !ok || nt.text != "," {
			break
		}
		p.pos++ // consume comma
		if len(cols) == 1 && cols[0] == "*" {
			return nil, fmt.Errorf("'*' cannot be combined with other columns")
		}
	}
	return cols, nil
}

func (p *parser) parseWhere() ([]Predicate, error) {
	var preds []Predicate
	for {
		field, ok := p.next()
		if !ok || field.quoted || isKeyword(field.text) {
			return nil, fmt.Errorf("expected a field name in WHERE")
		}
		opTok, ok := p.next()
		if !ok {
			return nil, fmt.Errorf("expected an operator after %q", field.text)
		}
		op, err := parseOp(opTok)
		if err != nil {
			return nil, err
		}
		valTok, ok := p.next()
		if !ok {
			return nil, fmt.Errorf("expected a value after operator")
		}
		if !valTok.quoted && isKeyword(valTok.text) {
			return nil, fmt.Errorf("expected a literal value, got keyword %q", valTok.text)
		}
		preds = append(preds, Predicate{Field: field.text, Op: op, Value: valTok.text})

		nt, ok := p.peek()
		if !ok {
			break
		}
		if kwEq(nt, "AND") {
			p.pos++
			continue
		}
		if kwEq(nt, "OR") {
			return nil, fmt.Errorf("OR is not supported (AND-only conjunction keeps scope enforceable)")
		}
		break
	}
	return preds, nil
}

func parseOp(t token) (Op, error) {
	if t.quoted {
		return "", fmt.Errorf("expected an operator, got string %q", t.text)
	}
	switch {
	case t.text == "=":
		return OpEq, nil
	case t.text == "!=":
		return OpNe, nil
	case t.text == "<":
		return OpLt, nil
	case t.text == ">":
		return OpGt, nil
	case t.text == "<=":
		return OpLe, nil
	case t.text == ">=":
		return OpGe, nil
	case strings.EqualFold(t.text, "LIKE"):
		return OpLike, nil
	default:
		return "", fmt.Errorf("unsupported operator %q", t.text)
	}
}

func isKeyword(s string) bool {
	switch strings.ToUpper(s) {
	case "SELECT", "FROM", "WHERE", "AND", "OR", "LIMIT", "LIKE":
		return true
	}
	return false
}
