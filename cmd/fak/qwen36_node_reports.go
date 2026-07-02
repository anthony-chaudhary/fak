package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/qwen36nodereports"
)

func cmdQwen36NodeReports(argv []string) {
	os.Exit(runQwen36NodeReports(os.Stdout, os.Stderr, argv))
}

func runQwen36NodeReports(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("qwen36-node-reports", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := resolveRoot("")
	if root == "" {
		root = "."
	}
	base := root
	if st, err := os.Stat(filepath.Join(root, "fak", "experiments")); err == nil && st.IsDir() {
		base = filepath.Join(root, "fak")
	}
	inbox := fs.String("inbox", filepath.Join(root, "tools", "_registry", "qwen36-report-inbox"), "Taildrop inbox")
	outDir := fs.String("out-dir", filepath.Join(base, "experiments", "qwen36", "node-reports"), "extracted report output directory")
	archive := fs.String("archive", "", "specific qwen36-node-reports-*.zip to import")
	wait := fs.Bool("wait", false, "wait for one Taildrop file before importing")
	skipTaildrop := fs.Bool("skip-taildrop", false, "do not run tailscale file get first")
	replace := fs.Bool("replace", false, "replace an existing extracted report directory")
	logTailLines := fs.Int("log-tail-lines", 60, "server log tail lines")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	summary := qwen36nodereports.ImportReportBundle(qwen36nodereports.ImportArgs{
		Inbox: *inbox, OutDir: *outDir, Archive: *archive, Wait: *wait,
		SkipTaildrop: *skipTaildrop, Replace: *replace, LogTailLines: *logTailLines,
	})
	data, err := qwen36nodereports.MarshalJSON(summary)
	if err != nil {
		fmt.Fprintf(stderr, "qwen36-node-reports: %v\n", err)
		return 2
	}
	fmt.Fprintln(stdout, string(data))
	if summary["imported"] == true {
		return 0
	}
	return 1
}
