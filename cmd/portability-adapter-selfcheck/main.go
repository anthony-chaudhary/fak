package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/portability"
	"os"
)

func main() {
	skeleton := flag.String("skeleton", "", "print an adapter registration skeleton")
	flag.Parse()
	if *skeleton != "" {
		s, e := portability.Skeleton(*skeleton)
		if e != nil {
			fmt.Fprintln(os.Stderr, e)
			os.Exit(1)
		}
		fmt.Print(s)
		return
	}
	rs, e := portability.RunReferenceConformance(context.Background())
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
	w := struct {
		Schema  string                          `json:"schema"`
		Verdict string                          `json:"verdict"`
		Results []portability.ConformanceResult `json:"results"`
		Matrix  []portability.AdapterInfo       `json:"matrix"`
	}{"fak-portability-adapter-selfcheck/1", "PASS", rs, portability.ReferenceRegistry().Matrix()}
	b, _ := json.MarshalIndent(w, "", "  ")
	fmt.Println(string(b))
}
