package main

import (
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"os"
	"strings"
)

type obsFlags map[string]string

func (f *obsFlags) String() string { return "ARM=FILE" }
func (f *obsFlags) Set(v string) error {
	a, p, ok := strings.Cut(v, "=")
	if !ok {
		return fmt.Errorf("want ARM=FILE")
	}
	if *f == nil {
		*f = map[string]string{}
	}
	(*f)[a] = p
	return nil
}
func main() {
	m := flag.String("manifest", "", "run manifest")
	out := flag.String("out", "", "report output")
	var obs obsFlags
	flag.Var(&obs, "observations", "ARM=FILE")
	flag.Parse()
	if *m == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "usage: crossauditcalibrate --manifest M --observations ARM=F --out R")
		os.Exit(2)
	}
	mb, err := os.ReadFile(*m)
	if err != nil {
		panic(err)
	}
	rows := map[string][]byte{}
	for a, p := range obs {
		b, e := os.ReadFile(p)
		if e != nil {
			panic(e)
		}
		rows[a] = b
	}
	b, err := modelroute.CalibrationReportFromJSON(mb, rows)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, b, 0644); err != nil {
		panic(err)
	}
}
