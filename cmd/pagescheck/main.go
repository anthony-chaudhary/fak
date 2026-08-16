package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pagespublish"
	"github.com/anthony-chaudhary/fak/internal/seoaeoscore"
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
	case "freshness":
		fs := flag.NewFlagSet("pagescheck freshness", flag.ExitOnError)
		root := fs.String("root", ".", "repository root")
		targetsPath := fs.String("targets", ".github/pages-freshness-targets.json", "freshness target manifest")
		_ = fs.Parse(os.Args[2:])
		targets, loadErr := pagespublish.LoadFreshnessTargets(filepath.Join(*root, filepath.FromSlash(*targetsPath)))
		if loadErr != nil {
			fmt.Fprintln(os.Stderr, "pagescheck:", loadErr)
			os.Exit(1)
		}
		freshness, auditErr := pagespublish.AuditFreshness(*root, targets, time.Now())
		if writeErr := pagespublish.WriteFreshnessJSON(freshness); writeErr != nil {
			fmt.Fprintln(os.Stderr, writeErr)
			os.Exit(1)
		}
		if auditErr != nil {
			fmt.Fprintln(os.Stderr, "pagescheck:", auditErr)
			os.Exit(1)
		}
		return
	case "source":
		fs := flag.NewFlagSet("pagescheck source", flag.ExitOnError)
		root := fs.String("root", "docs", "Pages source directory")
		_ = fs.Parse(os.Args[2:])
		report, err = pagespublish.AuditSource(*root)
	case "seo":
		fs := flag.NewFlagSet("pagescheck seo", flag.ExitOnError)
		root := fs.String("root", ".", "repository root")
		scoreFloor := fs.Float64("minimum-score", 0, "minimum published-corpus score")
		debtCeiling := fs.Int("maximum-debt", -1, "maximum published-corpus SEO debt (-1 disables)")
		orphanCeiling := fs.Int("maximum-orphans", -1, "maximum discovery orphans (-1 disables)")
		_ = fs.Parse(os.Args[2:])
		payload := seoaeoscore.Build(*root, "published")
		if err = validateSEO(payload.Corpus, *scoreFloor, *debtCeiling, *orphanCeiling); err == nil {
			enc := json.NewEncoder(os.Stdout)
			enc.SetEscapeHTML(false)
			err = enc.Encode(payload)
			if err == nil {
				return
			}
		}
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
func usage() {
	fmt.Fprintln(os.Stderr, "usage: pagescheck source|freshness|seo|artifact [flags]")
	os.Exit(2)
}

func validateSEO(c seoaeoscore.Corpus, scoreFloor float64, debtCeiling, orphanCeiling int) error {
	if c.OverallScore < scoreFloor {
		return fmt.Errorf("published SEO score %.1f is below %.1f", c.OverallScore, scoreFloor)
	}
	if debtCeiling >= 0 && c.SEODebt > debtCeiling {
		return fmt.Errorf("published SEO debt %d exceeds %d", c.SEODebt, debtCeiling)
	}
	if orphanCeiling >= 0 && c.DiscoveryOrphans > orphanCeiling {
		return fmt.Errorf("discovery orphans %d exceeds %d", c.DiscoveryOrphans, orphanCeiling)
	}
	return nil
}
