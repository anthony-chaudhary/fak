package enumlint

import (
	"fmt"
	"sort"
	"strings"
)

const (
	RuleSwitch  = "switch"
	RuleLiteral = "literal"
)

func AllRules() []string { return []string{RuleSwitch, RuleLiteral} }

type Config struct {
	Root              string
	IncludeTestFiles  bool
	SkipDirs          []string
	IncludeTopDirs    []string
	Exempt            func(string) (string, bool)
	LiteralMinMembers int
	LiteralMaxOmitted int
}

func (c Config) withDefaults() Config {
	if len(c.SkipDirs) == 0 {
		c.SkipDirs = []string{".git", "vendor", "testdata", "_scratch"}
	}
	if c.Exempt == nil {
		c.Exempt = LookupExemption
	}
	if c.LiteralMinMembers == 0 {
		c.LiteralMinMembers = 2
	}
	if c.LiteralMaxOmitted == 0 {
		c.LiteralMaxOmitted = 64
	}
	return c
}

type ScanError struct {
	Op, Path string
	Err      error
}

func (e *ScanError) Error() string { return fmt.Sprintf("enumlint %s %s: %v", e.Op, e.Path, e.Err) }
func (e *ScanError) Unwrap() error { return e.Err }

type Finding struct {
	Rule, Pkg, Owner, Type, File string
	Line, Covered, Total         int
	Missing                      []Member
	Msg                          string
}

func (f Finding) Key() string                  { return strings.Join([]string{f.Rule, f.Pkg, f.Owner, f.Type}, "\t") }
func exemptKey(rule, pkg, owner string) string { return rule + "|" + pkg + "|" + owner }

type Report struct {
	Packages, Enums, Members, Sites int
	Findings, Exempted              []Finding
	Skipped, Unparsed               []string
}

func Scan(root string, cfg Config) (Report, error) { cfg.Root = root; return Analyze(cfg) }
func Analyze(cfg Config) (Report, error) {
	cfg = cfg.withDefaults()
	if strings.TrimSpace(cfg.Root) == "" {
		return Report{}, fmt.Errorf("root is required")
	}
	pkgs, unparsed, err := Discover(cfg.Root, cfg)
	if err != nil {
		return Report{}, err
	}
	rep := Report{Packages: len(pkgs), Unparsed: unparsed}
	for _, p := range pkgs {
		rep.Enums += len(p.Enums)
		for _, e := range p.Enums {
			rep.Members += len(e.Members)
		}
		sf, se := p.checkSwitches(cfg.Exempt)
		lf, le := p.checkLiterals(cfg.Exempt, cfg.LiteralMinMembers, cfg.LiteralMaxOmitted)
		rep.Sites += se + le
		rep.Findings = append(rep.Findings, sf...)
		rep.Findings = append(rep.Findings, lf...)
		_ = se
		_ = le
	}
	for _, key := range ExemptionKeys() {
		if strings.TrimSpace(ExemptionReason(key)) == "" {
			return Report{}, ErrBlankReason(key)
		}
	}
	sortFindings(rep.Findings)
	sortFindings(rep.Exempted)
	return rep, nil
}
func (r Report) CountByRule() map[string]int {
	m := map[string]int{}
	for _, f := range r.Findings {
		m[f.Rule]++
	}
	return m
}
func (r Report) Census() string {
	return fmt.Sprintf("packages=%d enums=%d members=%d sites=%d", r.Packages, r.Enums, r.Members, r.Sites)
}
func (f Finding) String() string { return f.Msg }

func sortFindings(fs []Finding) {
	sort.Slice(fs, func(i, j int) bool {
		a, b := fs[i], fs[j]
		if a.File != b.File {
			return a.File < b.File
		}
		if a.Line != b.Line {
			return a.Line < b.Line
		}
		if a.Rule != b.Rule {
			return a.Rule < b.Rule
		}
		return a.Owner < b.Owner
	})
}
