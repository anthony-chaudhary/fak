package hooks

import (
	"errors"
	"io/fs"

	"github.com/anthony-chaudhary/fak/internal/hooks/alphafence"
)

// gate_alphafence.go extends the PUBLIC_LEAK family with the engineer's own alpha fence. The
// needle lists name what the maintainers must not publish; the fence lets any user declare
// private workflow files and have the same gates refuse content reproducing them. Findings
// name a position and the fencing entry, never the matched text, so a gate log cannot become
// the leak.

const alphaFenceRel = alphafence.DefaultRel

type alphaFence struct {
	matcher *alphafence.Matcher
	defect  string
}

// parseAlphaFence turns the fence bytes into a scanner. An absent fence scans nothing; a present
// but unusable one is a finding rather than a silent pass.
func parseAlphaFence(raw []byte, present bool) alphaFence {
	if !present {
		return alphaFence{}
	}
	f, err := alphafence.Parse(raw)
	if err != nil {
		return alphaFence{defect: err.Error()}
	}
	m, err := alphafence.NewMatcher(f)
	if err != nil {
		return alphaFence{defect: err.Error()}
	}
	return alphaFence{matcher: m}
}

// alphaFenceAtRoot reads the fence from the working tree only. It is local policy: a copy in
// the staged or committed tree is content under audit and must never choose what is audited.
func alphaFenceAtRoot(root string) alphaFence {
	if root == "" {
		return alphaFence{}
	}
	raw, err := readFileRel(root, alphaFenceRel)
	switch {
	case err == nil:
		return parseAlphaFence(raw, true)
	case errors.Is(err, fs.ErrNotExist):
		return alphaFence{}
	default:
		return alphaFence{defect: "fence exists but cannot be read"}
	}
}

func (a alphaFence) defectFindings() []Finding {
	if a.defect == "" {
		return nil
	}
	return []Finding{{Gate: "PUBLIC_LEAK", File: alphaFenceRel, Detail: "[alpha fence unusable]  " + a.defect}}
}

// alphaFencePublished reports the fence itself entering the repository. Its digests let a
// holder test guessed lines, so it stays on the engineer's machine.
func alphaFencePublished(file string) []Finding {
	if file != alphaFenceRel {
		return nil
	}
	return []Finding{{Gate: "PUBLIC_LEAK", File: file, Detail: "[alpha fence published]  the fence is local state; untrack it and keep it ignored"}}
}

// alphaUnit accumulates one outbound unit (a file, a message, a payload).
type alphaUnit struct {
	file string
	c    *alphafence.Collector
}

func (a alphaFence) unit(file string) alphaUnit {
	return alphaUnit{file: file, c: a.matcher.NewCollector()}
}

func (u alphaUnit) observe(lineNo int, line string) { u.c.Observe(lineNo, line) }

func (u alphaUnit) findings() []Finding {
	var out []Finding
	for _, h := range u.c.Hits() {
		out = append(out, Finding{
			Gate: "PUBLIC_LEAK", File: u.file, Line: h.Line,
			Detail: "[alpha fence: " + h.Label + "]  reproduces a fenced private line (text withheld)",
		})
	}
	return out
}
