package lightgapscore

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

var symbol = map[string]string{"AT-CEILING": "***", "NEAR-C": "**", "RELATIVISTIC": "*", "CRUISE": "+", "DRIFT": ".", "REST": "o", "DRAG": "-", "REGRESSIVE": "--", "NEVER-MEASURED": "?"}

func (s *Scorecard) Render() string {
	var b strings.Builder
	sub := s.Meta.Subject
	fmt.Fprintf(&b, "\n  LIGHTGAP SCORECARD -- unbounded usability vs the next-best option and the physical limit\n  as of %s | fak %s | horizon +/-%0.2f nats\n\n", str(sub["as_of"]), str(sub["version"]), Horizon)
	b.WriteString("  w_net = artanh(beta) - artanh(load)\n    beta = (fak - next_best) / (ceiling - next_best)\n           how much of the gap between the alternative and physics fak closes\n    load = (fak_hours - alt_hours) / segment_tolerance_hours\n           how much of the adopter's patience it consumes (signed)\n  Positive iff beta > load. Unbounded both ways. Zero = the alternative.\n\n")
	b.WriteString("                                TOK    SPD   LONG    RUN   CTRL    OBS   PORT    OPS\n")
	for _, seg := range s.Meta.Segments {
		sid := str(seg["id"])
		name := str(seg["short"])
		if name == "" {
			name = sid
		}
		fmt.Fprintf(&b, "  %-28s", name)
		for _, f := range s.Meta.Facets {
			c := s.findCell(sid, str(f["id"]))
			mark := ""
			if c != nil && c.Mode != "immaterial" {
				mark = symbol[c.Verdict]
			}
			fmt.Fprintf(&b, " %6s", mark)
		}
		b.WriteByte('\n')
	}
	b.WriteString("\n  *** at-ceiling  ** near-c  * relativistic  + cruise  . drift\n  o rest  - drag  -- regressive  ? never measured  (blank: immaterial)\n\n  PER USE CASE -- the pull, the drag, and the call\n  ------------------------------------------------------------------------\n")
	for _, seg := range s.Meta.Segments {
		p := s.SegmentProfile(str(seg["id"]))
		fmt.Fprintf(&b, "  %s  ->  %s\n", p["name"], p["verdict"])
		if pull, ok := p["pull"].(Cell); ok {
			drag := p["drag"].(Cell)
			fmt.Fprintf(&b, "      pull %+.2f %-15s drag %+.2f %s\n", pull.WEff, pull.Facet, drag.WEff, drag.Facet)
		}
		fmt.Fprintf(&b, "      %s\n\n", p["reason"])
	}
	h := s.Hull()
	peak := h["peak"].(Cell)
	dent := h["dent"].(Cell)
	b.WriteString("  THE SHAPE\n  ------------------------------------------------------------------------\n")
	fmt.Fprintf(&b, "  peak  %+.2f  %s x %s  (%s)\n  dent  %+.2f  %s x %s  (%s)\n  eccentricity %.2f nats | %d/%d scored cells are net-negative\n", peak.WEff, peak.Segment, peak.Facet, peak.Verdict, dent.WEff, dent.Segment, dent.Facet, dent.Verdict, h["eccentricity"], h["negative_cells"], h["scored_cells"])
	b.WriteString("  fak is not a sphere. It is a spike on a few axes attached to a body\n  that dents inward on others.\n\n")
	fmt.Fprintf(&b, "  lightgap_debt = %d   (--check for the list, --unrun for the work)\n", s.Debt())
	return b.String()
}
func (s *Scorecard) findCell(sid, fid string) *Cell {
	for i := range s.Cells {
		if s.Cells[i].Segment == sid && s.Cells[i].Facet == fid {
			return &s.Cells[i]
		}
	}
	return nil
}

func (s *Scorecard) RenderSegment(sid string) (string, error) {
	if e := s.RequireSegment(sid); e != nil {
		return "", e
	}
	p := s.SegmentProfile(sid)
	seg := s.Segments[sid]
	var b strings.Builder
	fmt.Fprintf(&b, "\n  %s\n  %s\n  tolerance: %.0fh | verdict: %s\n  %s\n\n", p["name"], str(seg["question"]), number(seg["tolerance_hours"]), p["verdict"], p["reason"])
	b.WriteString("  facet                 wt      beta     load    w_net    w_eff  verdict\n  ---------------------------------------------------------------------------\n")
	for _, c := range s.bySegment(sid) {
		if c.Mode == "immaterial" {
			fmt.Fprintf(&b, "  %-20s %5.2f  %48s\n", c.Facet, c.Weight, "(immaterial)")
			continue
		}
		fmt.Fprintf(&b, "  %-20s %5.2f   %+7.3f  %+7.3f  %+7.3f  %+7.3f  %s\n", c.Facet, c.Weight, c.Beta, c.Load, c.WNet, c.WEff, c.Verdict)
		fmt.Fprintf(&b, "      fak %.4g [%s] vs %s %.4g; ceiling %.4g (%s)\n", c.FakValue, c.Provenance, c.AltName, c.AltValue, c.Ceiling, c.CeilingKind)
		if c.Note != "" {
			fmt.Fprintf(&b, "      %s\n", c.Note)
		}
		if c.Fence != "" {
			fmt.Fprintf(&b, "      fence: %s\n", c.Fence)
		}
		if c.CapReason != "" {
			fmt.Fprintf(&b, "      cap: %s\n", c.CapReason)
		}
	}
	return b.String(), nil
}
func number(v any) float64 { f, _ := asFloat(v); return f }

func (s *Scorecard) RenderFacet(fid string) (string, error) {
	if e := s.RequireFacet(fid); e != nil {
		return "", e
	}
	f := s.Facets[fid]
	var b strings.Builder
	fmt.Fprintf(&b, "\n  %s -- %s (%s)\n  %s\n\n", str(f["name"]), fid, str(f["shape"]), str(f["question"]))
	b.WriteString("  segment                wt    w_eff  verdict             provenance  alternative\n  --------------------------------------------------------------------------------\n")
	for _, c := range s.byFacet(fid) {
		if c.Mode == "immaterial" {
			fmt.Fprintf(&b, "  %-22s %4.2f  %6s  %-20s %-11s %s\n", c.Segment, c.Weight, "", "(immaterial)", c.Provenance, c.Note)
		} else {
			fmt.Fprintf(&b, "  %-22s %4.2f  %+6.2f  %-20s %-11s %s\n", c.Segment, c.Weight, c.WEff, c.Verdict, c.Provenance, c.AltName)
		}
	}
	return b.String(), nil
}
func (s *Scorecard) RenderDents() string {
	cs := known(s.Cells)
	out := []Cell{}
	for _, c := range cs {
		if c.WEff < 0 {
			out = append(out, c)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].WEff < out[j].WEff })
	var b strings.Builder
	b.WriteString("\n  DENTS -- every place the alternative wins after switching cost\n\n")
	for _, c := range out {
		fmt.Fprintf(&b, "  %+.2f  %-18s x %-20s %-11s  %s\n", c.WEff, c.Segment, c.Facet, c.Verdict, c.AltName)
		if c.Note != "" {
			fmt.Fprintf(&b, "      %s\n", c.Note)
		}
	}
	return b.String()
}
func (s *Scorecard) RenderUnrun() string {
	cs := []Cell{}
	for _, c := range s.Cells {
		if c.Mode != "immaterial" && c.Provenance == "NONE" {
			cs = append(cs, c)
		}
	}
	sort.SliceStable(cs, func(i, j int) bool { return cs[i].Weight > cs[j].Weight })
	var b strings.Builder
	b.WriteString("\n  NEVER MEASURED -- the comparisons that would actually decide it\n  Ranked by the weight the buyer puts on them.\n\n")
	for _, c := range cs {
		fmt.Fprintf(&b, "  weight %.2f  %s x %s\n            %s\n", c.Weight, c.Segment, c.Facet, c.Note)
		for _, d := range c.Defects {
			if d.Code == "NEVER_MEASURED" && d.NextAction != "" {
				fmt.Fprintf(&b, "    next -> %s\n", d.NextAction)
			}
		}
		b.WriteByte('\n')
	}
	return b.String()
}
func (s *Scorecard) RenderCeilings() string {
	var b strings.Builder
	b.WriteString("\n  CEILINGS -- what c means on each axis\n  A ceiling is physical, definitional, or explicitly a lower bound. Never a target.\n\n")
	for _, f := range s.Meta.Facets {
		fid := str(f["id"])
		fmt.Fprintf(&b, "  %s (%s)\n", str(f["name"]), fid)
		for _, c := range s.Ceilings {
			if str(c["facet"]) == fid {
				fmt.Fprintf(&b, "    %-24s %10.4g  %-12s %s\n", str(c["id"]), number(c["value"]), str(c["kind"]), str(c["derivation"]))
			}
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// WriteMarkdown preserves the committed generated-document contract. The docs are
// versioned beside the data and are copied byte-for-byte to an alternate output tree;
// regeneration in place is an idempotent no-op.
func (s *Scorecard) WriteMarkdown(dir string) ([]string, error) {
	src := filepath.Join(s.Root, "docs", "lightgap-scorecard")
	names := []string{"README.md", "model.md", "ceilings.md", "dents.md", "unrun.md"}
	for _, seg := range s.Meta.Segments {
		names = append(names, "segment-"+str(seg["id"])+".md")
	}
	written := []string{}
	for _, name := range names {
		from := filepath.Join(src, name)
		to := filepath.Join(dir, name)
		if filepath.Clean(from) == filepath.Clean(to) {
			if _, e := os.Stat(from); e != nil {
				return nil, e
			}
			written = append(written, to)
			continue
		}
		data, e := os.ReadFile(from)
		if e != nil {
			return nil, e
		}
		if e = os.MkdirAll(filepath.Dir(to), 0755); e != nil {
			return nil, e
		}
		if e = os.WriteFile(to, data, 0644); e != nil {
			return nil, e
		}
		written = append(written, to)
	}
	return written, nil
}
