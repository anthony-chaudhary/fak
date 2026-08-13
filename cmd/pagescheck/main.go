package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pagespublish"
)

type stringList []string

func (s *stringList) String() string     { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error { *s = append(*s, v); return nil }

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var report pagespublish.Report
	var err error
	switch os.Args[1] {
	case "source":
		fs := flag.NewFlagSet("pagescheck source", flag.ExitOnError)
		root := fs.String("root", "docs", "Pages source directory")
		_ = fs.Parse(os.Args[2:])
		report, err = pagespublish.AuditSource(*root)
	case "artifact":
		fs := flag.NewFlagSet("pagescheck artifact", flag.ExitOnError)
		root := fs.String("root", "_site", "generated Pages artifact")
		base := fs.String("base-url", "", "canonical site base URL required in sitemap")
		minimum := fs.Int("minimum-pages", 1, "minimum generated HTML page count")
		manifest := fs.Bool("write-manifest", false, "write an exact deploy manifest into the artifact")
		var required stringList
		fs.Var(&required, "require", "required artifact-relative page (repeatable)")
		_ = fs.Parse(os.Args[2:])
		report, err = pagespublish.AuditArtifact(*root, *base, *minimum, required, *manifest)
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "pagescheck:", err)
		os.Exit(1)
	}
	if err := pagespublish.WriteJSON(report); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func usage() { fmt.Fprintln(os.Stderr, "usage: pagescheck source|artifact [flags]"); os.Exit(2) }
