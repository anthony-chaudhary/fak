package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	pc "github.com/anthony-chaudhary/fak/internal/portabilitycontract"
)

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("portabilitycontract", flag.ContinueOnError)
	fs.SetOutput(stderr)
	explain := fs.Bool("explain", false, "render the contract for people")
	check := fs.Bool("check", false, "validate package and round-trip it")
	write := fs.Bool("write-identities", false, "derive content and package identities before writing")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: portabilitycontract (--check|--explain|--write-identities) PACKAGE.json")
		return 2
	}
	if err := act(fs.Arg(0), *explain, *check, *write, stdout); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}

func act(path string, explain, check, write bool, stdout io.Writer) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	var p pc.Package
	if e := json.Unmarshal(b, &p); e != nil {
		return e
	}
	if write {
		return writeIdentities(path, p, stdout)
	}
	if e := p.Validate(); e != nil {
		return e
	}
	rt, e := pc.RoundTrip(b)
	if e != nil {
		return e
	}
	if !pc.EqualJSON(b, rt) {
		return fmt.Errorf("round-trip changed semantics")
	}
	if explain {
		fmt.Fprint(stdout, pc.Explain(p))
		return nil
	}
	if check {
		fmt.Fprintf(stdout, "VALID %s objects=%d\n", p.PackageID, count(p))
		return nil
	}
	return fmt.Errorf("choose an action")
}

func writeIdentities(path string, p pc.Package, stdout io.Writer) error {
	for ci := range p.Collections {
		for oi := range p.Collections[ci].Objects {
			id, e := p.Collections[ci].Objects[oi].CanonicalContentID()
			if e != nil {
				return e
			}
			p.Collections[ci].Objects[oi].ContentID = id
		}
	}
	p.PackageID = ""
	id, e := pc.PackageIdentity(p)
	if e != nil {
		return e
	}
	p.PackageID = id
	out, e := json.MarshalIndent(p, "", "  ")
	if e != nil {
		return e
	}
	out = append(out, '\n')
	if e := os.WriteFile(path, out, 0644); e != nil {
		return e
	}
	fmt.Fprintln(stdout, id)
	return nil
}

func count(p pc.Package) int {
	n := 0
	for _, c := range p.Collections {
		n += len(c.Objects)
	}
	return n
}
