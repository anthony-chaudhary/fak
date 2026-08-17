package disambiguation

import "testing"

func TestIndexCLISourceAddsTermAndDetectsRemovedVerb(t *testing.T) {
	before := `usage:
  fak serve --listen ADDR
  fak retired --json
`
	prior := IndexCLISource(before, nil).Terms
	after := `usage:
  fak serve --listen ADDR
  fak inspect cache --json
`
	got := IndexCLISource(after, prior)

	if !hasCLITerm(got.Terms, CLITerm{Term: "inspect", Kind: CLITermCommand, Invocation: "fak inspect"}) {
		t.Fatalf("new command not derived: %+v", got.Terms)
	}
	if !hasCLITerm(got.Terms, CLITerm{Term: "cache", Kind: CLITermSubcommand, Invocation: "fak inspect cache"}) {
		t.Fatalf("new subcommand not derived: %+v", got.Terms)
	}
	if !hasCLITerm(got.Terms, CLITerm{Term: "--json", Kind: CLITermFlag, Invocation: "fak inspect cache"}) {
		t.Fatalf("flag not derived: %+v", got.Terms)
	}
	if !hasCLITerm(got.Stale, CLITerm{Term: "retired", Kind: CLITermCommand, Invocation: "fak retired"}) {
		t.Fatalf("removed verb not stale: %+v", got.Stale)
	}
}

func TestIndexCLISourceIsDeterministicAndIgnoresHelpProse(t *testing.T) {
	help := "fak prose is not an indented synopsis\n  fak serve --listen ADDR\n  fak serve --listen ADDR\n"
	got := IndexCLISource(help, nil)
	if len(got.Terms) != 2 {
		t.Fatalf("terms = %+v", got.Terms)
	}
	if got.Stale == nil {
		t.Fatal("stale must encode as an array")
	}
}

func hasCLITerm(terms []CLITerm, want CLITerm) bool {
	for _, term := range terms {
		if term == want {
			return true
		}
	}
	return false
}
