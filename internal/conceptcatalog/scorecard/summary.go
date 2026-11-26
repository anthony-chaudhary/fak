package scorecard

import (
	"context"
	"path/filepath"
)

// FamilyCount is one family's discovered/covered pair.
type FamilyCount struct {
	Family     string `json:"family"`
	Discovered int    `json:"discovered"`
	Covered    int    `json:"covered"`
}

// Summary is the slice of a payload the invariant gates consume, in the
// payload's own JSON shape so it decodes into the same consumer structs.
type Summary struct {
	OK     bool   `json:"ok"`
	Reason string `json:"reason"`
	Corpus struct {
		CoverageDebt   int `json:"coverage_debt"`
		ClarityDefects int `json:"clarity_defects"`
		Coverage       struct {
			CoveragePct PyFloat       `json:"coverage_pct"`
			PerFamily   []FamilyCount `json:"per_family"`
		} `json:"coverage"`
	} `json:"corpus"`
}

// Summary extracts the invariant-gate fields; an AUDIT_ERROR payload yields
// zero counters, as decoding its empty corpus did.
func (p *Payload) Summary() Summary {
	var s Summary
	s.OK, s.Reason = p.OK, p.Reason
	s.Corpus.Coverage.PerFamily = []FamilyCount{}
	if p.Corpus == nil {
		return s
	}
	c := p.Corpus
	s.Corpus.CoverageDebt = c.CoverageDebt
	s.Corpus.ClarityDefects = c.ClarityDefects
	s.Corpus.Coverage.CoveragePct = c.Coverage.CoveragePct
	for _, f := range c.Coverage.PerFamily {
		fam, _ := f.Family.(string)
		s.Corpus.Coverage.PerFamily = append(s.Corpus.Coverage.PerFamily, FamilyCount{Family: fam, Discovered: f.Discovered, Covered: f.Covered})
	}
	return s
}

// Generate folds workspace (with an optional data directory) and writes the
// doc folder into outDir, honouring ctx between the expensive phases.
func Generate(ctx context.Context, workspace, dataDir, outDir string) (*Payload, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	workspace = ResolvePath(workspace)
	if dataDir != "" {
		dataDir = ResolvePath(dataDir)
	} else {
		dataDir = filepath.Join(workspace, filepath.FromSlash(DataDirRel))
	}
	data, msg := LoadData(dataDir)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	tree := LoadTree(workspace)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	p := BuildPayload(workspace, data, tree, msg)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if outDir != "" {
		if err := p.WriteDocFolder(outDir, ""); err != nil {
			return nil, err
		}
	}
	return p, nil
}
