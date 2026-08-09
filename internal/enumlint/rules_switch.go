package enumlint

import (
	"fmt"
	"go/ast"
	"strings"
)

// rules_switch.go — RULE 1, the non-exhaustive switch.
//
// The defect: a `switch` whose cases name constants of one closed enumeration,
// with NO `default` clause, that omits at least one member. The omitted member
// falls straight through: it is neither handled nor refused, the function
// returns its zero value or falls to whatever follows, and NOTHING FAILS. Adding
// a constant to the const block leaves every such switch compiling, so the
// expensive half of the edit — finding the consumers — has no mechanical help.
//
// The `default` clause is the discriminator, and it is the right one. A switch
// WITH a default has made a decision about the members it does not name, even if
// that decision is "refuse": internal/turnkind's classifier and
// internal/wipref's Classify both lean on a default that carries a real domain
// value. A switch WITHOUT one has made no decision at all. That is why this rule
// is not "every member needs a case" — that formulation would fire on correct
// code across the tree and would never have landed.

// SwitchMinCovered is the smallest number of enum constants a defaultless
// switch must name before this rule judges it.
//
// A single-arm `switch k { case KindText: ... }` is an `if` somebody wrote as a
// switch. It is not claiming to enumerate anything, so holding it to the whole
// vocabulary would be a false positive with no available fix except an
// exemption. At two arms the site is making a multi-way distinction over the
// type and the claim "these are the cases" is real.
const SwitchMinCovered = 2

// checkSwitches runs rule 1 over one package and returns the findings plus the
// number of switches it successfully resolved to an enumeration (the
// denominator that separates "exhaustive" from "recognised nothing").
func (p *Package) checkSwitches(exempt func(string) (string, bool)) ([]Finding, int) {
	var (
		out   []Finding
		sites int
	)
	owner := p.memberOwners()
	p.eachOwner(func(name string, node ast.Node) {
		ast.Inspect(node, func(n ast.Node) bool {
			sw, ok := n.(*ast.SwitchStmt)
			if !ok {
				return true
			}
			e, covered, ok := p.resolveSwitch(sw, owner)
			if !ok {
				return true
			}
			sites++
			if switchHasDefault(sw) || len(covered) < SwitchMinCovered {
				return true
			}
			missing := e.Missing(covered)
			if len(missing) == 0 {
				return true
			}
			file, line := p.pos(sw)
			if p.hasExemptDirective(file, line) {
				return true
			}
			key := exemptKey(RuleSwitch, p.Dir, name)
			if exempt != nil {
				if reason, ok := exempt(key); ok && strings.TrimSpace(reason) != "" {
					return true
				}
			}
			out = append(out, Finding{
				Rule: RuleSwitch, Pkg: p.Dir, Type: e.Name, Owner: name,
				File: file, Line: line,
				Covered: len(covered), Total: len(e.Members), Missing: missing,
				Msg: switchMsg(p.Dir, name, e, file, line, len(covered), missing, key),
			})
			return true
		})
	})
	return out, sites
}

// resolveSwitch decides whether a switch statement is switching on a closed
// enumeration, using the CASE EXPRESSIONS rather than the tag's type.
//
// This package reads go/ast only — no go/types, no build, no module graph — so
// it cannot ask what type the tag has. It does not need to: a switch every one
// of whose case expressions names a constant of the SAME enumeration is
// switching on that enumeration, because Go would not compile it otherwise.
// Reading the tag's type would additionally require the package to type-check,
// which in this tree means failing whenever a peer's half-written file is on
// disk — the scan would go dark exactly when the fleet is busiest.
//
// The cost is stated plainly: a switch whose cases name constants from ANOTHER
// package (`case wipref.CensusLanded:` read from outside internal/wipref) is not
// resolved, because discovery is per-directory. External TEST packages are
// covered — `package foo_test` sits in the same directory and Config
// IncludeTestFiles reads it — which is where fak's hand-written coverage tables
// actually live. Genuinely cross-package switches are a follow-on, and #5649's
// shared discovery pass is where the tree-wide symbol index for them belongs.
func (p *Package) resolveSwitch(sw *ast.SwitchStmt, owner map[string]*Enum) (*Enum, map[string]bool, bool) {
	if sw.Tag == nil || sw.Body == nil {
		// A tagless `switch { case x == A: }` is a boolean chain, not a
		// multi-way dispatch over a value.
		return nil, nil, false
	}
	var (
		e       *Enum
		covered = map[string]bool{}
	)
	for _, stmt := range sw.Body.List {
		cc, ok := stmt.(*ast.CaseClause)
		if !ok {
			return nil, nil, false
		}
		if cc.List == nil {
			continue // the default clause carries no case expressions
		}
		for _, expr := range cc.List {
			n := identName(expr)
			if n == "" {
				return nil, nil, false // a computed case: not a constant claim
			}
			cand, ok := owner[n]
			if !ok {
				return nil, nil, false // names something that is not an enum constant
			}
			if e == nil {
				e = cand
			} else if e != cand {
				// Two enumerations in one switch cannot both be the subject.
				return nil, nil, false
			}
			covered[n] = true
		}
	}
	if e == nil || len(covered) == 0 {
		return nil, nil, false
	}
	return e, covered, true
}

func switchHasDefault(sw *ast.SwitchStmt) bool {
	for _, stmt := range sw.Body.List {
		if cc, ok := stmt.(*ast.CaseClause); ok && cc.List == nil {
			return true
		}
	}
	return false
}

// memberOwners maps every constant name in the package to the enumeration it
// belongs to. Package-scope constant names are unique in Go, so this mapping is
// unambiguous and one lookup replaces a search over every enum.
func (p *Package) memberOwners() map[string]*Enum {
	out := make(map[string]*Enum)
	for _, name := range p.EnumNames() {
		e := p.Enums[name]
		for _, m := range e.Members {
			out[m.Name] = e
		}
	}
	return out
}

// eachOwner calls fn for every top-level declaration that can contain a
// statement or a composite literal, with the name a person would grep for.
// A method is named "Recv.Method" so two types' same-named methods do not share
// a ratchet key.
func (p *Package) eachOwner(fn func(name string, node ast.Node)) {
	for _, f := range p.files {
		for _, d := range f.Decls {
			switch decl := d.(type) {
			case *ast.FuncDecl:
				if decl.Body == nil {
					continue
				}
				name := decl.Name.Name
				if decl.Recv != nil && len(decl.Recv.List) == 1 {
					name = typeName(decl.Recv.List[0].Type) + "." + name
				}
				fn(name, decl.Body)
			case *ast.GenDecl:
				// A package-scope var can hold a func literal, and that
				// literal's switches are as real as any other's.
				for _, spec := range decl.Specs {
					vs, ok := spec.(*ast.ValueSpec)
					if !ok || len(vs.Names) == 0 {
						continue
					}
					for _, v := range vs.Values {
						fn(vs.Names[0].Name, v)
					}
				}
			}
		}
	}
}

func switchMsg(pkg, owner string, e *Enum, file string, line int, covered int, missing []Member, key string) string {
	names := make([]string, 0, len(missing))
	for _, m := range missing {
		names = append(names, fmt.Sprintf("%s (%s:%d)", m.Name, m.File, m.Line))
	}
	return fmt.Sprintf("[%s] %s:%d: %s switches on %s.%s, covers %d of %d constants and has no default, "+
		"so it silently falls through for %s — add a case (or a default that decides), "+
		"or record %q in internal/enumlint/exempt.go with the reason it is deliberately partial",
		RuleSwitch, file, line, owner, pkg, e.Name, covered, len(e.Members),
		strings.Join(names, ", "), key)
}
