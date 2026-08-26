package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/study"
)

func defaultStudyStore() (string, error) {
	d, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(d, "fak", "study"), nil
}

func runStudy(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak study <add|get|search> [args]")
		return 2
	}
	switch args[0] {
	case "add":
		return runStudyAdd(stdout, stderr, args[1:])
	case "get":
		return runStudyGet(stdout, stderr, args[1:])
	case "search":
		return runStudySearch(stdout, stderr, args[1:])
	default:
		fmt.Fprintf(stderr, "study: unknown subcommand %q\n", args[0])
		return 2
	}
}
func studyStoreFlag(fs *flag.FlagSet) (*string, error) {
	d, err := defaultStudyStore()
	if err != nil {
		return nil, err
	}
	return fs.String("store", d, "receipt store"), nil
}
func runStudyAdd(out, errw io.Writer, args []string) int {
	fs := flag.NewFlagSet("study add", flag.ContinueOnError)
	fs.SetOutput(errw)
	file := fs.String("file", "", "record JSON")
	store, e := studyStoreFlag(fs)
	if e != nil {
		return 1
	}
	if fs.Parse(args) != nil || *file == "" || fs.NArg() != 0 {
		fmt.Fprintln(errw, "usage: fak study add --file RECORD.json [--store PATH]")
		return 2
	}
	b, e := os.ReadFile(*file)
	if e != nil {
		fmt.Fprintf(errw, "study add: %v\n", e)
		return 1
	}
	var r study.Record
	if e = json.Unmarshal(b, &r); e != nil {
		fmt.Fprintf(errw, "study add: %v\n", e)
		return 1
	}
	receipt, e := study.Add(*store, r)
	if e != nil {
		fmt.Fprintf(errw, "study add: durable receipt unavailable: %v\n", e)
		return 1
	}
	json.NewEncoder(out).Encode(receipt)
	return 0
}
func runStudyGet(out, errw io.Writer, args []string) int {
	fs := flag.NewFlagSet("study get", flag.ContinueOnError)
	fs.SetOutput(errw)
	store, e := studyStoreFlag(fs)
	if e != nil {
		return 1
	}
	if fs.Parse(args) != nil || fs.NArg() != 1 {
		fmt.Fprintln(errw, "usage: fak study get ID [--store PATH]")
		return 2
	}
	r, e := study.Get(*store, fs.Arg(0))
	if e != nil {
		fmt.Fprintf(errw, "study get: %v\n", e)
		return 1
	}
	json.NewEncoder(out).Encode(r)
	return 0
}
func runStudySearch(out, errw io.Writer, args []string) int {
	fs := flag.NewFlagSet("study search", flag.ContinueOnError)
	fs.SetOutput(errw)
	store, e := studyStoreFlag(fs)
	if e != nil {
		return 1
	}
	limit := fs.Int("limit", 20, "maximum results")
	if fs.Parse(args) != nil || fs.NArg() != 1 {
		fmt.Fprintln(errw, "usage: fak study search QUERY [--limit N] [--store PATH]")
		return 2
	}
	matches, e := study.Search(*store, fs.Arg(0), *limit)
	if e != nil {
		fmt.Fprintf(errw, "study search: durable receipt unavailable: %v\n", e)
		return 1
	}
	json.NewEncoder(out).Encode(matches)
	return 0
}
