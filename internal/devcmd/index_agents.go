package devcmd

// fak index agents — the sectioned, fence-aware view over AGENTS.md (#3535, epic #3229).
// AGENTS.md is a ~10k-token always-read doc; a worker that Reads it whole on turn 1 pays
// that entire tax up front. This verb lets an agent hold a compact resident TOC and fault
// in only the one section a task needs — the same cold-context-deferral idea the tool
// levers (#3231/#3232) apply to tool schemas, here applied to a big always-read doc.
//
//	fak index agents                   the resident TOC: one row per level>=2 section
//	fak index agents <query>           rank sections by a lexical score (title > body)
//	fak index agents --section <slug>  emit one section's bytes, verbatim (byte-exact)
//	fak index agents --full            emit the whole doc, verbatim (the escape hatch)
//	fak index agents --write-resident  (re)write the resident TOC block into CLAUDE.md
//
// The section/full outputs are byte-identical to what a whole Read would have returned;
// the parse changes AGENTS.md's load discipline, never its content.

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/agentsindex"
)

// residentClaudeFile is the doc whose marker-bounded region holds the resident TOC.
const residentClaudeFile = "CLAUDE.md"

func indexAgents(stdout, stderr io.Writer, rootDir string, args []string, asJSON bool, section string, full, writeResident bool,
	forTarget string, fallbacks []string, maxBytes int64, trusted bool,
) int {
	if forTarget != "" {
		if len(args) != 0 || section != "" || full || writeResident {
			fmt.Fprintln(stderr, "fak index agents --for: incompatible with query, --section, --full, and --write-resident")
			return 2
		}
		result := agentsindex.ResolveEffective(rootDir, forTarget, agentsindex.ResolveOptions{
			Fallbacks: fallbacks,
			MaxBytes:  maxBytes,
			Trusted:   trusted,
		})
		if asJSON {
			if rc := encodeJSONOrFail(stdout, stderr, result, "fak index agents --for"); rc != 0 {
				return rc
			}
		} else {
			if result.Status == agentsindex.StatusComplete {
				if _, err := io.WriteString(stdout, result.Instructions); err != nil {
					fmt.Fprintf(stderr, "fak index agents --for: %v\n", err)
					return 1
				}
			} else {
				fmt.Fprintf(stdout, "status=%s target=%s sources=%d bytes=%d/%d\n",
					result.Status, result.Target, len(result.Sources), result.Bytes, result.MaxBytes)
			}
		}
		if result.Status != agentsindex.StatusComplete {
			return 1
		}
		return 0
	}
	if len(fallbacks) != 0 || maxBytes != 0 || trusted {
		fmt.Fprintln(stderr, "fak index agents: --fallback, --max-bytes, and --trust require --for")
		return 2
	}
	doc, err := agentsindex.Load(rootDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak index agents: %v\n", err)
		return 1
	}
	switch {
	case writeResident:
		return indexAgentsWriteResident(stdout, stderr, rootDir, doc)
	case full:
		if _, err := stdout.Write(doc.Raw); err != nil {
			fmt.Fprintf(stderr, "fak index agents: %v\n", err)
			return 1
		}
		return 0
	case section != "":
		s, ok := doc.SectionBySlug(section)
		if !ok {
			fmt.Fprintf(stderr, "fak index agents: no section %q\n", section)
			if near := doc.Search(section); len(near) > 0 {
				fmt.Fprint(stderr, "  did you mean: ")
				for i, n := range near {
					if i >= 5 {
						break
					}
					if i > 0 {
						fmt.Fprint(stderr, ", ")
					}
					fmt.Fprint(stderr, n.Slug)
				}
				fmt.Fprintln(stderr)
			}
			return 1
		}
		if asJSON {
			return encodeJSONOrFail(stdout, stderr, s, "fak index agents")
		}
		io.WriteString(stdout, s.Raw)
		return 0
	case len(args) > 0:
		hits := doc.Search(joinArgs(args))
		return indexRenderHits(stdout, stderr, hits, asJSON, "fak index agents", "no matching section",
			func(tw *tabwriter.Writer, s agentsindex.Section) {
				fmt.Fprintf(tw, "%s\t%s\t~%d tok\n", s.Slug, truncRunes(s.Title, 72), s.EstTokens)
			})
	default:
		if asJSON {
			return encodeJSONOrFail(stdout, stderr, doc.Sections, "fak index agents")
		}
		io.WriteString(stdout, doc.RenderTOC())
		return 0
	}
}

// indexAgentsWriteResident regenerates (or first-writes) the resident TOC block in
// root/CLAUDE.md between the agentsindex markers. On a CLAUDE.md that carries the markers
// it splices in place (idempotent); on one that does not it APPENDS a fresh marker-bounded
// block after a single blank-line separator. This is the single writer the drift gate
// (internal/agentsindex/resident_drift_test.go) checks the committed block against.
func indexAgentsWriteResident(stdout, stderr io.Writer, rootDir string, doc *agentsindex.Doc) int {
	path := filepath.Join(rootDir, residentClaudeFile)
	claude, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak index agents --write-resident: %v\n", err)
		return 1
	}
	block := doc.ResidentBlock()
	var out []byte
	if _, present := agentsindex.ExtractResident(claude); present {
		out, err = agentsindex.SpliceResident(claude, block)
		if err != nil {
			fmt.Fprintf(stderr, "fak index agents --write-resident: %v\n", err)
			return 1
		}
	} else {
		out = append(bytes.TrimRight(claude, "\n"), []byte("\n\n"+block+"\n")...)
	}
	if err := os.WriteFile(path, out, 0o644); err != nil {
		fmt.Fprintf(stderr, "fak index agents --write-resident: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote resident AGENTS.md TOC (~%d est. tok, %d sections) into %s\n",
		agentsindex.EstTokensOf(doc.RenderTOC()), len(doc.Sections), path)
	return 0
}
