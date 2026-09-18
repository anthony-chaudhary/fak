package devcmd

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// issueContractSectionsSchema is the JSON schema key emitted by
// `fak-dev issue contract-sections --json`.
const issueContractSectionsSchema = issuepolicy.RequiredSectionsSchema

// IssueRequiredSections returns the canonical required-section contract. It is
// exported so other packages can consume the same manifest the CLI emits
// without shelling out.
func IssueRequiredSections() issuepolicy.RequiredSectionsContract {
	return issuepolicy.RequiredSections()
}

// IssueRequiredSectionsJSON returns the contract encoded as indented JSON.
func IssueRequiredSectionsJSON() ([]byte, error) {
	return issuepolicy.RequiredSectionsJSON()
}

// runIssueContractSections prints the canonical machine-readable required-section
// contract. It is read-only, takes no file argument, and performs no network I/O.
// Exit 0 on success, 2 on bad flags.
func runIssueContractSections(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("issue contract-sections", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the required-section contract as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak-dev issue contract-sections: unexpected positional arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}

	contract := issuepolicy.RequiredSections()
	if *asJSON {
		b, err := issuepolicy.RequiredSectionsJSON()
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev issue contract-sections: encode json: %v\n", err)
			return 1
		}
		if _, err := stdout.Write(append(b, '\n')); err != nil {
			fmt.Fprintf(stderr, "fak-dev issue contract-sections: write json: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprint(stdout, renderIssueContractSections(contract))
	return 0
}

func renderIssueContractSections(c issuepolicy.RequiredSectionsContract) string {
	var b strings.Builder
	fmt.Fprintf(&b, "issue-required-sections: schema=%s required=%d project_work=%d production=%d\n",
		c.Schema, len(c.Sections), len(c.ProjectWorkSections), len(c.ProductionSections))
	b.WriteString("required sections:\n")
	for _, s := range c.Sections {
		fmt.Fprintf(&b, "  - %s [%s] headings: %s\n", s.Field, s.Source, strings.Join(s.Headings, ", "))
	}
	fmt.Fprintf(&b, "problem frame: ## %s  %s  checks=%s  verdicts=%s\n",
		c.ProblemFrame.Section, c.ProblemFrame.CentralityLine,
		strings.Join(c.ProblemFrame.CheckLabels, ","), strings.Join(c.ProblemFrame.Verdicts, "|"))
	b.WriteString("project-work sections:\n")
	for _, s := range c.ProjectWorkSections {
		fmt.Fprintf(&b, "  - %s [%s] headings: %s\n", s.Field, s.Source, strings.Join(s.Headings, ", "))
	}
	b.WriteString("production sections:\n")
	for _, s := range c.ProductionSections {
		fmt.Fprintf(&b, "  - %s [%s] headings: %s\n", s.Field, s.Source, strings.Join(s.Headings, ", "))
	}
	return b.String()
}
