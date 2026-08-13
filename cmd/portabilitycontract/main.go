package main

import (
	"encoding/json"
	"flag"
	"fmt"
	pc "github.com/anthony-chaudhary/fak/internal/portabilitycontract"
	"os"
)

func main() {
	explain := flag.Bool("explain", false, "render the contract for people")
	check := flag.Bool("check", false, "validate package and round-trip it")
	write := flag.Bool("write-identities", false, "derive content and package identities before writing")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: portabilitycontract (--check|--explain|--write-identities) PACKAGE.json")
		os.Exit(2)
	}
	b, e := os.ReadFile(flag.Arg(0))
	fatal(e)
	var p pc.Package
	fatal(json.Unmarshal(b, &p))
	if *write {
		for ci := range p.Collections {
			for oi := range p.Collections[ci].Objects {
				id, e := p.Collections[ci].Objects[oi].CanonicalContentID()
				fatal(e)
				p.Collections[ci].Objects[oi].ContentID = id
			}
		}
		p.PackageID = ""
		id, e := pc.PackageIdentity(p)
		fatal(e)
		p.PackageID = id
		out, e := json.MarshalIndent(p, "", "  ")
		fatal(e)
		out = append(out, '\n')
		fatal(os.WriteFile(flag.Arg(0), out, 0644))
		fmt.Println(id)
		return
	}
	fatal(p.Validate())
	rt, e := pc.RoundTrip(b)
	fatal(e)
	if !pc.EqualJSON(b, rt) {
		fatal(fmt.Errorf("round-trip changed semantics"))
	}
	if *explain {
		fmt.Print(pc.Explain(p))
		return
	}
	if *check {
		fmt.Printf("VALID %s objects=%d\n", p.PackageID, count(p))
		return
	}
	fatal(fmt.Errorf("choose an action"))
}
func count(p pc.Package) int {
	n := 0
	for _, c := range p.Collections {
		n += len(c.Objects)
	}
	return n
}
func fatal(e error) {
	if e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
