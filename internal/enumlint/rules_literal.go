package enumlint

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"
)

// rules_literal.go — RULE 2, the composite-literal hole.
//
// The defect: a map or slice literal that IS the vocabulary table for a closed
// enumeration and is missing a member. a map keyed by an enum for help text,
// `[]Verdict` for render order, `AllKinds()` returning the set — every one of
// them is a second copy of the const block, and every one goes wrong by being
// SHORT while nothing notices. A member added to the const block and not to the
// table produces no compile error and no test failure; it produces a runtime
// hole (empty help text, a value that never renders, a Valid() that rejects a
// constant the package itself declares).
//
// Three discriminations keep this from drowning in false positives. Each is a
// threshold, each was set from a MEASURED run over this tree, and the measured
// numbers are recorded beside the constant so a later change can be judged
// against them rather than re-argued.

// LiteralMinMembers is the smallest enumeration whose near-complete literals
// this rule will judge.
//
// The inference "a literal naming all but one constant meant to name them all"
// weakens as the enumeration shrinks. With four constants a three-member subset
// is an utterly ordinary thing to want and the tree is full of them; with twelve
// constants, a literal naming eleven is almost always the table with a hole.
// Measured over C:\work\fak internal/ + cmd/ at the SHA in
// the named SHA in tree_test.go, holding MaxOmitted at 2:
//
//	LiteralMinMembers 2 -> 60 literal findings
//	LiteralMinMembers 4 -> 41 literal findings
//	LiteralMinMembers 6 -> 24 literal findings
//
// Six is where the remaining population stops being "an ordinary subset of a
// small type" and starts being "a table that has drifted". What it costs is
// real and should not be discovered later: a genuinely drifted 4-of-5 table is
// invisible to this rule. Lowering the floor is a follow-on that needs each
// small-enum subset to carry its own //enumlint:exempt reason, which needs the
// lane of every package that owns one.
const LiteralMinMembers = 6

// LiteralMaxOmitted is how far short of its enumeration a literal may fall
// before this rule reads it as a deliberate subset rather than as an exhaustive
// table that drifted.
//
// This threshold is the difference between a landable gate and an unusable one.
// The discriminating fact is HOW DRIFT ARRIVES: constants land one or two at a
// time, so a table that was exhaustive yesterday is short by one or two today. A
// literal naming two of twenty-six never claimed to be the set and never will
// be — it is a test fixture — and judging it produces a hundred non-defects,
// which is how a tree-wide gate gets switched off and the real sites go back to
// being invisible. Measured over this tree at LiteralMinMembers 6:
//
//	LiteralMaxOmitted 1 -> 15 literal findings
//	LiteralMaxOmitted 2 -> 24 literal findings
//	LiteralMaxOmitted 8 -> 61 literal findings
//
// What this deliberately does not catch: a table that has been short by three or
// more since before this gate existed. That is a census, not a drift alarm — the
// census is Scan with both rules enabled, and the alarm is the ratchet.
const LiteralMaxOmitted = 2

// checkLiterals runs rule 2 over one package and returns the findings plus the
// number of enum-keyed literals it read in a judged position.
//
// POSITION is the third discrimination, and the one that matters most. Only a
// literal that is a package-scope declaration's value, or a value a function
// RETURNS, is judged. Those are the two positions in which a literal says "this
// is the set": the vocabulary var, and the AllX() accessor. A literal built
// inside a function body is a fixture — "here are the two kinds this case
// needs" — and holding it to the whole enumeration reports a hundred sites that
// were never claiming anything.
func (p *Package) checkLiterals(exempt func(string) (string, bool), minMembers, maxOmitted int) ([]Finding, int) {
	var (
		out   []Finding
		sites int
	)
	for _, f := range p.files {
		for _, d := range f.Decls {
			switch decl := d.(type) {
			case *ast.GenDecl:
				if decl.Tok != token.VAR && decl.Tok != token.CONST {
					continue
				}
				for _, spec := range decl.Specs {
					vs, ok := spec.(*ast.ValueSpec)
					if !ok || len(vs.Names) == 0 {
						continue
					}
					for _, v := range vs.Values {
						fs, n := p.judgeTree(v, vs.Names[0].Name, exempt, minMembers, maxOmitted)
						out, sites = append(out, fs...), sites+n
					}
				}
			case *ast.FuncDecl:
				if decl.Body == nil {
					continue
				}
				owner := decl.Name.Name
				if decl.Recv != nil && len(decl.Recv.List) == 1 {
					owner = typeName(decl.Recv.List[0].Type) + "." + owner
				}
				ast.Inspect(decl.Body, func(n ast.Node) bool {
					rs, ok := n.(*ast.ReturnStmt)
					if !ok {
						return true
					}
					for _, r := range rs.Results {
						fs, k := p.judgeTree(r, owner, exempt, minMembers, maxOmitted)
						out, sites = append(out, fs...), sites+k
					}
					return false
				})
			}
		}
	}
	return out, sites
}

// judgeTree judges expr and every composite literal nested inside it. Nesting
// matters because a set is often a field of a larger declared value —
// `Ladder{Floor: []Modality{...}}` — and the FIELD, not the outer var, is the
// name a person greps for, so the owner is extended as the walk descends.
func (p *Package) judgeTree(expr ast.Expr, owner string, exempt func(string) (string, bool), minMembers, maxOmitted int) ([]Finding, int) {
	var (
		out   []Finding
		sites int
	)
	var walk func(e ast.Expr, own string)
	walk = func(e ast.Expr, own string) {
		cl, ok := e.(*ast.CompositeLit)
		if !ok {
			return
		}
		if fnd, read, ok := p.judgeLiteral(cl, own, exempt, minMembers, maxOmitted); read {
			sites++
			if ok {
				out = append(out, fnd)
			}
		}
		for _, el := range cl.Elts {
			if kv, isKV := el.(*ast.KeyValueExpr); isKV {
				sub := own
				if k := identName(kv.Key); k != "" {
					sub = own + "." + k
				}
				walk(kv.Value, sub)
				continue
			}
			walk(el, own)
		}
	}
	walk(expr, owner)
	return out, sites
}

// judgeLiteral returns (finding, read, isFinding). `read` says the literal was
// recognised as enumerating a closed set at all — the denominator; `isFinding`
// says it has a hole worth reporting.
func (p *Package) judgeLiteral(cl *ast.CompositeLit, owner string, exempt func(string) (string, bool), minMembers, maxOmitted int) (Finding, bool, bool) {
	var (
		e      *Enum
		named  = map[string]bool{}
		anyBad bool
	)

	switch t := cl.Type.(type) {
	case *ast.MapType:
		got, ok := p.Enums[typeName(t.Key)]
		if !ok {
			return Finding{}, false, false
		}
		e = got
		for _, el := range cl.Elts {
			kv, ok := el.(*ast.KeyValueExpr)
			if !ok {
				anyBad = true
				break
			}
			n := identName(kv.Key)
			if !e.Has(n) {
				anyBad = true
				break
			}
			named[n] = true
		}

	case *ast.ArrayType:
		elt := typeName(t.Elt)
		if got, ok := p.Enums[elt]; ok {
			e = got
			for _, el := range cl.Elts {
				n := identName(el)
				if !e.Has(n) {
					anyBad = true
					break
				}
				named[n] = true
			}
			break
		}
		// The vocabulary-of-structs form: []RefusalReason where RefusalReason
		// has exactly one RefusalCode field. The enumeration is one field deep,
		// and this is the shape fak writes most.
		sf, ok := p.structs[elt]
		if !ok {
			if st, isAnon := t.Elt.(*ast.StructType); isAnon {
				sf, ok = p.enumFieldOf(st)
			}
		}
		if !ok {
			return Finding{}, false, false
		}
		e = p.Enums[sf.typeName]
		if e == nil {
			return Finding{}, false, false
		}
		for _, el := range cl.Elts {
			inner, ok := el.(*ast.CompositeLit)
			if !ok {
				anyBad = true
				break
			}
			n, ok := structFieldConst(inner, sf)
			if !ok || !e.Has(n) {
				anyBad = true
				break
			}
			named[n] = true
		}

	default:
		return Finding{}, false, false
	}

	// A literal this package could not read END TO END is not judged at all. The
	// cost of a skip is a missed finding; the cost of a guess is a false failure
	// in a lane the reporter does not hold, which is how a tree-wide gate gets
	// disabled.
	if anyBad || len(named) == 0 {
		return Finding{}, false, false
	}
	// It IS a recognised consumption site from here on, even if it does not
	// clear the thresholds — that is what makes Sites a real denominator.
	if len(named) == len(e.Members) {
		return Finding{}, true, false
	}
	if len(named) < 2 || len(e.Members) < minMembers || len(e.Members)-len(named) > maxOmitted {
		return Finding{}, true, false
	}

	file, line := p.pos(cl)
	if p.hasExemptDirective(file, line) {
		return Finding{}, true, false
	}
	key := exemptKey(RuleLiteral, p.Dir, owner)
	if exempt != nil {
		if reason, ok := exempt(key); ok && strings.TrimSpace(reason) != "" {
			return Finding{}, true, false
		}
	}

	missing := e.Missing(named)
	name := owner
	if name == "" {
		name = "literal"
	}
	return Finding{
		Rule: RuleLiteral, Pkg: p.Dir, Type: e.Name, Owner: owner,
		File: file, Line: line,
		Covered: len(named), Total: len(e.Members), Missing: missing,
		Msg: literalMsg(p.Dir, name, e, file, line, len(named), missing, key),
	}, true, true
}

// structFieldConst pulls the enum constant out of one element of a
// []StructWithOneEnumField literal, whether the element is keyed or positional.
func structFieldConst(cl *ast.CompositeLit, sf structEnumField) (string, bool) {
	keyed := false
	for _, el := range cl.Elts {
		if _, ok := el.(*ast.KeyValueExpr); ok {
			keyed = true
			break
		}
	}
	if keyed {
		for _, el := range cl.Elts {
			kv, ok := el.(*ast.KeyValueExpr)
			if !ok {
				return "", false
			}
			if identName(kv.Key) == sf.field {
				return identName(kv.Value), true
			}
		}
		return "", false // the field is absent: a zero value, not a constant
	}
	if sf.index >= len(cl.Elts) {
		return "", false
	}
	return identName(cl.Elts[sf.index]), true
}

func literalMsg(pkg, owner string, e *Enum, file string, line int, covered int, missing []Member, key string) string {
	names := make([]string, 0, len(missing))
	for _, m := range missing {
		names = append(names, fmt.Sprintf("%s (%s:%d)", m.Name, m.File, m.Line))
	}
	return fmt.Sprintf("[%s] %s:%d: %s enumerates %d of %s.%s's %d constants (declared at %s:%d) but omits %s "+
		"— add the missing member here, or record %q in internal/enumlint/exempt.go with the reason it is "+
		"deliberately partial",
		RuleLiteral, file, line, owner, covered, pkg, e.Name, len(e.Members), e.File, e.Line,
		strings.Join(names, ", "), key)
}
