package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/categorybaseline"
)

var maturityBaselineRunWitness = func(root, name string, args []string, stdout, stderr io.Writer) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir, cmd.Stdout, cmd.Stderr = root, stdout, stderr
	return cmd.Run()
}

func runMaturityBaseline(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		return maturityBaselineUsage(stderr)
	}
	switch argv[0] {
	case "list":
		fs := flag.NewFlagSet("fak maturity baseline list", flag.ContinueOnError)
		fs.SetOutput(stderr)
		workspace := fs.String("workspace", "", "workspace root (default: repo root)")
		asJSON := fs.Bool("json", false, "emit registry JSON")
		if !parseFlags(fs, argv[1:]) || fs.NArg() != 0 {
			return 2
		}
		root := *workspace
		if root == "" {
			root = repoRoot()
		}
		r := categorybaseline.Load(root)
		if *asJSON {
			if writeIndentedJSON(stdout, r) != nil {
				return 1
			}
			return 0
		}
		if len(r.Categories) == 0 {
			fmt.Fprintln(stdout, "no enforced category baselines")
			return 0
		}
		for _, c := range r.Categories {
			fmt.Fprintf(stdout, "%s: %s complete -> %s next (witness: %s)\n", c.Name, c.CompletedLayer, c.NextLayer, c.Witness)
		}
		return 0
	case "promote":
		return runMaturityBaselinePromote(stdout, stderr, argv[1:])
	case "remove":
		fs := flag.NewFlagSet("fak maturity baseline remove", flag.ContinueOnError)
		fs.SetOutput(stderr)
		workspace := fs.String("workspace", "", "workspace root")
		category := fs.String("category", "", "category to stop enforcing")
		if !parseFlags(fs, argv[1:]) || strings.TrimSpace(*category) == "" || fs.NArg() != 0 {
			return 2
		}
		root := *workspace
		if root == "" {
			root = repoRoot()
		}
		if err := categorybaseline.Save(root, categorybaseline.Remove(categorybaseline.Load(root), *category)); err != nil {
			fmt.Fprintf(stderr, "fak maturity baseline remove: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "removed category baseline %s\n", *category)
		return 0
	default:
		return maturityBaselineUsage(stderr)
	}
}

func runMaturityBaselinePromote(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak maturity baseline promote", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root")
	category := fs.String("category", "", "category name")
	layers := fs.String("layers", "", "ordered comma-separated layers")
	completed := fs.String("completed", "", "witnessed completed layer")
	next := fs.String("next", "", "next layer that should receive capacity")
	witness := fs.String("witness", "", "witness executable (run directly, without a shell)")
	var witnessArgs stringList
	fs.Var(&witnessArgs, "witness-arg", "witness argument; repeatable")
	if !parseFlags(fs, argv) || fs.NArg() != 0 {
		return 2
	}
	if strings.TrimSpace(*category) == "" || strings.TrimSpace(*layers) == "" || strings.TrimSpace(*completed) == "" || strings.TrimSpace(*next) == "" || strings.TrimSpace(*witness) == "" {
		fmt.Fprintln(stderr, "fak maturity baseline promote: --category, --layers, --completed, --next, and --witness are required")
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	args := []string(witnessArgs)
	if err := maturityBaselineRunWitness(root, *witness, args, stdout, stderr); err != nil {
		fmt.Fprintf(stderr, "fak maturity baseline promote: witness failed; registry unchanged: %v\n", err)
		return 1
	}
	witnessText := strings.TrimSpace(strings.Join(append([]string{*witness}, args...), " "))
	r, ok := categorybaseline.Upsert(categorybaseline.Load(root), categorybaseline.Category{Name: *category, Layers: strings.Split(*layers, ","), CompletedLayer: *completed, NextLayer: *next, Witness: witnessText})
	if !ok {
		fmt.Fprintln(stderr, "fak maturity baseline promote: invalid layer order or next/completed declaration")
		return 2
	}
	if err := categorybaseline.Save(root, r); err != nil {
		fmt.Fprintf(stderr, "fak maturity baseline promote: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "promoted %s: %s complete -> %s next; dispatch enforcement is now on\n", *category, *completed, *next)
	return 0
}

func maturityBaselineUsage(w io.Writer) int {
	fmt.Fprintln(w, "usage: fak maturity baseline list [--json] | promote --category C --layers A,B --completed A --next B --witness EXE [--witness-arg ARG] | remove --category C")
	fmt.Fprintln(w, "rollback: use remove --category C, or promote the category again with a corrected witnessed boundary")
	return 2
}
