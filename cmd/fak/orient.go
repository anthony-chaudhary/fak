package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func cmdOrient(argv []string) { os.Exit(runOrient(os.Stdout, os.Stderr, argv)) }

type orientPathFlags []string

func (p *orientPathFlags) String() string {
	return strings.Join(*p, ",")
}

func (p *orientPathFlags) Set(v string) error {
	for _, part := range strings.Split(v, ",") {
		if s := strings.TrimSpace(part); s != "" {
			*p = append(*p, s)
		}
	}
	return nil
}

func runOrient(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak orient", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: search upward for dos.toml)")
	asJSON := fs.Bool("json", false, "emit JSON")
	readLeases := fs.Bool("leases", true, "read live refs/fak/locks leases")
	var paths orientPathFlags
	fs.Var(&paths, "paths", "path or glob to orient around; repeat or comma-separate")
	fs.Var(&paths, "path", "alias for --paths")
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	for _, arg := range fs.Args() {
		_ = paths.Set(arg)
	}
	if len(paths) == 0 {
		fmt.Fprintln(stderr, "fak orient: needs --paths <path-or-glob>")
		return 2
	}

	rootDir := pathutil.ExpandTilde(*root)
	if rootDir == "" {
		rootDir = devindex.FindRoot(".")
	}
	cat, err := devindex.Load(rootDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak orient: %v\n", err)
		return 1
	}
	var leases []devindex.OrientationLease
	if *readLeases {
		var lerr error
		leases, lerr = orientLiveLeases(rootDir)
		if lerr != nil {
			fmt.Fprintf(stderr, "fak orient: live leases unavailable: %v\n", lerr)
		}
	}
	rows := cat.Orient(paths, leases)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rows, "fak orient")
	}
	return renderOrient(stdout, stderr, rows)
}

func orientLiveLeases(root string) ([]devindex.OrientationLease, error) {
	store := leaseref.NewInDir(root)
	recs, _, err := store.Live(context.Background(), time.Now())
	if err != nil {
		return nil, err
	}
	out := make([]devindex.OrientationLease, 0, len(recs))
	for _, rec := range recs {
		out = append(out, devindex.OrientationLease{
			ID:         rec.ID,
			Holder:     rec.Holder,
			Tree:       append([]string(nil), rec.TreeGlobs...),
			TTLSeconds: rec.TTLSeconds,
		})
	}
	return out, nil
}

func renderOrient(stdout, stderr io.Writer, rows []devindex.Orientation) int {
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	for _, row := range rows {
		fmt.Fprintf(tw, "%s\tlane=%s\tstamp=%s\ttest=%s\ttier=%s\tlease=%s\n",
			row.Path, orientCell(row.Lane, "unknown"), orientCell(row.Stamp, "unknown"),
			orientCell(row.TestTarget, "n/a"), orientTierCell(row), orientLeaseCell(row.LiveLeases))
		if len(row.LaneTree) > 0 {
			fmt.Fprintf(tw, "\t\tlane_tree=%s\n", strings.Join(row.LaneTree, ","))
		}
	}
	if err := tw.Flush(); err != nil {
		fmt.Fprintf(stderr, "fak orient: %v\n", err)
		return 1
	}
	return 0
}

func orientCell(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func orientTierCell(row devindex.Orientation) string {
	if row.Tier == nil {
		return "n/a"
	}
	if row.TierName == "" {
		return fmt.Sprintf("%d", *row.Tier)
	}
	return fmt.Sprintf("%d/%s", *row.Tier, row.TierName)
}

func orientLeaseCell(leases []devindex.OrientationLease) string {
	if len(leases) == 0 {
		return "none"
	}
	var b bytes.Buffer
	for i, lease := range leases {
		if i > 0 {
			b.WriteString(";")
		}
		b.WriteString(lease.ID)
		if lease.Holder != "" {
			b.WriteString("@")
			b.WriteString(lease.Holder)
		}
		if len(lease.Tree) > 0 {
			b.WriteString("[")
			b.WriteString(strings.Join(lease.Tree, ","))
			b.WriteString("]")
		}
	}
	return b.String()
}
