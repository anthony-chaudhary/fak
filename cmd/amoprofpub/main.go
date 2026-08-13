// Command amoprofpub deterministically turns an AMOProf directory or .tgz into
// Confluence storage-XHTML pages and a complete attachment manifest.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/amoprofpub"
)

func main() {
	input := flag.String("input", "", "AMOProf directory or tgz")
	out := flag.String("out", "", "generated publication directory")
	title := flag.String("title", "AMOProf report", "parent page title")
	space := flag.String("space", "MPL", "Confluence space")
	parent := flag.String("parent-id", "", "optional existing parent page")
	flag.Parse()
	m, err := amoprofpub.Generate(amoprofpub.Options{Input: *input, Out: *out, Title: *title, Space: *space, ParentID: *parent})
	if err != nil {
		fmt.Fprintln(os.Stderr, "amoprofpub:", err)
		os.Exit(1)
	}
	if err = json.NewEncoder(os.Stdout).Encode(m); err != nil {
		fmt.Fprintln(os.Stderr, "amoprofpub:", err)
		os.Exit(1)
	}
}
