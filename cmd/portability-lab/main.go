package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/portabilitylab"
)

func main() {
	out := flag.String("out", "", "write authoritative JSON report")
	work := flag.String("work", "", "clean-room directory (default: temporary)")
	flag.Parse()
	root := *work
	cleanup := func() {}
	if root == "" {
		var err error
		root, err = os.MkdirTemp("", "fak-portability-lab-")
		if err != nil {
			fatal(err)
		}
		cleanup = func() { os.RemoveAll(root) }
	}
	defer cleanup()
	r, err := portabilitylab.Run(root)
	if *out != "" {
		if e := os.MkdirAll(filepath.Dir(*out), 0755); e != nil {
			fatal(e)
		}
		if e := portabilitylab.WriteReport(*out, r); e != nil {
			fatal(e)
		}
	}
	b, _ := json.MarshalIndent(r, "", "  ")
	fmt.Println(string(b))
	if err != nil {
		fatal(err)
	}
}
func fatal(err error) { fmt.Fprintln(os.Stderr, "portability-lab:", err); os.Exit(1) }
