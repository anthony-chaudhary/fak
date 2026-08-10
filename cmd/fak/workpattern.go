package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/codelint"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
	"github.com/anthony-chaudhary/fak/internal/worktype"
)

const workpatternReportSchema = "fak.workpattern-report/1"

type workpatternInput struct {
	Kind   string `json:"kind"`
	Path   string `json:"path"`
	Digest string `json:"digest,omitempty"`
}
type workpatternReport struct {
	Schema           string                   `json:"schema"`
	CatalogVersion   string                   `json:"catalog_version"`
	DetectorVersions map[string]string        `json:"detector_versions"`
	Inputs           []workpatternInput       `json:"inputs"`
	Catalog          *worktype.PatternCatalog `json:"catalog,omitempty"`
	Source           *codelint.MotifReport    `json:"source,omitempty"`
	Trajectory       *trajectory.Report       `json:"trajectory,omitempty"`
	Findings         int                      `json:"findings"`
	Abstentions      int                      `json:"abstentions"`
	Errors           []string                 `json:"errors,omitempty"`
}

func cmdWorkpattern(args []string) error {
	fs := flag.NewFlagSet("workpattern", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	asJSON := fs.Bool("json", false, "emit JSON")
	source := fs.String("source", "", "repository root")
	corpus := fs.String("trajectory", "", "trajectory JSONL")
	chat := fs.String("chat", "", "scrubbed chat JSON")
	excerpts := fs.Bool("include-excerpts", false, "opt in to scrubbed excerpts")
	mode := "list"
	parseArgs := args
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		mode = args[0]
		parseArgs = args[1:]
	}
	if err := fs.Parse(parseArgs); err != nil {
		return err
	}
	if mode != "list" && mode != "source" && mode != "trajectory" && mode != "report" {
		return fmt.Errorf("unknown mode %q", mode)
	}
	rep := workpatternReport{Schema: workpatternReportSchema, CatalogVersion: "1.0.0", DetectorVersions: map[string]string{"source": codelint.MotifMinerVersion, "trajectory": trajectory.MineSchemaVersion}}
	cat := worktype.SeedPatternCatalog()
	if err := cat.Validate(); err != nil {
		return err
	}
	if mode == "list" || mode == "report" {
		rep.Catalog = &cat
	}
	if mode == "source" || mode == "report" {
		if *source == "" {
			return fmt.Errorf("--source is required")
		}
		r, err := codelint.MineMotifs(*source)
		if err != nil {
			return err
		}
		rep.Source = &r
		rep.Findings += len(r.Findings)
		rep.Abstentions += len(r.Skipped)
		rep.Inputs = append(rep.Inputs, inputDigestRoot("source", *source))
	}
	if mode == "trajectory" || mode == "report" {
		if (*corpus == "") == (*chat == "") {
			return fmt.Errorf("exactly one of --trajectory or --chat is required")
		}
		path := *corpus
		kind := "trajectory"
		var r *trajectory.Recorder
		var err error
		if *chat != "" {
			path = *chat
			kind = "scrubbed-chat"
			f, e := os.Open(path)
			if e != nil {
				return e
			}
			r, _, err = trajectory.ImportScrubbedChat(f)
			f.Close()
		} else {
			f, e := os.Open(path)
			if e != nil {
				return e
			}
			r, _, err = trajectory.ImportFrom(f)
			f.Close()
		}
		if err != nil {
			return err
		}
		tr := r.Mine(trajectory.MineOptions{Excerpts: *excerpts})
		rep.Trajectory = &tr
		rep.Findings += len(tr.Segments)
		rep.Abstentions += tr.OverlapDropped
		rep.Inputs = append(rep.Inputs, inputDigest(kind, path))
	}
	sort.Slice(rep.Inputs, func(i, j int) bool { return rep.Inputs[i].Kind < rep.Inputs[j].Kind })
	if *asJSON {
		b, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(b))
		return nil
	}
	return renderWorkpattern(os.Stdout, rep)
}
func inputDigestRoot(kind, path string) workpatternInput {
	if kind == "source" {
		h := sha256.Sum256([]byte(filepath.ToSlash(filepath.Clean(path))))
		return workpatternInput{Kind: kind, Path: filepath.ToSlash(path), Digest: hex.EncodeToString(h[:])}
	}
	return inputDigest(kind, path)
}

func inputDigest(kind, path string) workpatternInput {
	b, err := os.ReadFile(path)
	if err != nil {
		return workpatternInput{Kind: kind, Path: filepath.ToSlash(path)}
	}
	h := sha256.Sum256(b)
	return workpatternInput{Kind: kind, Path: filepath.ToSlash(path), Digest: hex.EncodeToString(h[:])}
}
func renderWorkpattern(w io.Writer, r workpatternReport) error {
	fmt.Fprintf(w, "WORKPATTERNS catalog=%s findings=%d abstentions=%d\n", r.CatalogVersion, r.Findings, r.Abstentions)
	if r.Catalog != nil {
		fmt.Fprintf(w, "catalog: %d patterns, %d reusable subpatterns\n", len(r.Catalog.Patterns), len(r.Catalog.Subpatterns))
		for _, p := range r.Catalog.Patterns {
			fmt.Fprintf(w, "  %s  %s\n", p.ID, p.Name)
		}
	}
	if r.Source != nil {
		fmt.Fprintln(w, "source findings:")
		for _, f := range r.Source.Findings {
			fmt.Fprintf(w, "  %s  %s:%d  %s\n", f.CatalogID, f.Path, f.StartLine, f.Reason)
		}
	}
	if r.Trajectory != nil {
		fmt.Fprintln(w, "trajectory findings:")
		for _, s := range r.Trajectory.Segments {
			fmt.Fprintf(w, "  %s  %s[%d..%d]  %s\n", s.Subpattern, s.TraceID, s.StartSeq, s.EndSeq, s.Reason)
		}
	}
	return nil
}
