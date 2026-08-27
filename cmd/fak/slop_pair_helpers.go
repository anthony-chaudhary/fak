package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func readJSONFileInto(path string, dst any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, dst)
}

func readJSONLCorpus[T any](path, label string) []T {
	if path == "" {
		return nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", label, err)
		os.Exit(1)
	}
	var rows []T
	for i, raw := range bytes.Split(b, []byte("\n")) {
		line := bytes.TrimSpace(raw)
		if len(line) == 0 {
			continue
		}
		var row T
		if err := json.Unmarshal(line, &row); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %s line %d: %v\n", label, path, i+1, err)
			os.Exit(1)
		}
		rows = append(rows, row)
	}
	return rows
}

func gardenWalkPolicy(budget, skipFresh int, skipActive bool) gardenbundle.WalkPolicy {
	return gardenbundle.WalkPolicy{Budget: budget, SkipFreshDays: skipFresh, SkipInProgress: skipActive, DryRun: true}
}

func scoreboardClient(token, apiBase string) (*scoreboard.Client, error) {
	var opts []scoreboard.Option
	if apiBase != "" {
		opts = append(opts, scoreboard.WithAPIBase(apiBase))
	}
	return scoreboard.NewClient(token, opts...)
}

func projectRows[A, B any](rows []A, project func(A) B) []B {
	out := make([]B, 0, len(rows))
	for _, row := range rows {
		out = append(out, project(row))
	}
	return out
}

func writeUsageLines(w io.Writer, lines ...string) {
	for _, line := range lines {
		fmt.Fprintln(w, line)
	}
}

func exitf(code int, format string, args ...any) {
	fmt.Fprintf(os.Stderr, format, args...)
	os.Exit(code)
}

func renderJSONOrHuman[T any](stdout io.Writer, asJSON bool, value T, human func(io.Writer, T), failed bool) int {
	if asJSON {
		b, _ := json.MarshalIndent(value, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		human(stdout, value)
	}
	if failed {
		return 1
	}
	return 0
}

func standardScorecardDoc(title, description, heading, autoGen, law, debtKey, extra string) scorecard.MarkdownDoc {
	return scorecard.MarkdownDoc{
		Title: title, Description: description, Heading: heading, AutoGen: autoGen,
		Law: law, DebtKey: debtKey, HeaderExtra: extra,
	}
}

func emitLeaserefResult[T any](stdout, stderr io.Writer, value T, err error, prefix, operation string, ok func(T) bool) int {
	if err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", prefix, err)
		return 1
	}
	return emitLeaserefOutcome(stdout, stderr, value, ok(value), operation)
}

func validateAndSaveAccounts(stderr io.Writer, path string, reg accounts.Registry, invalidFormat string) bool {
	if err := reg.Validate(); err != nil {
		fmt.Fprintf(stderr, invalidFormat, err)
		return false
	}
	return saveAccountsRegistry(stderr, path, reg)
}

func firstTrimmedOr(fallback string, values ...string) string {
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			return value
		}
	}
	return fallback
}

func sessionDescriptorID(d session.Descriptor) string {
	return firstTrimmedOr("", d.ID, d.Trace)
}

func parsedRFC3339UTC(raw string, fallback time.Time) (time.Time, error) {
	if strings.TrimSpace(raw) == "" {
		return fallback.UTC(), nil
	}
	parsed, err := time.Parse(time.RFC3339, strings.TrimSpace(raw))
	if err != nil {
		return time.Time{}, err
	}
	return parsed.UTC(), nil
}

func profilesWereExplicit(fs *flag.FlagSet) bool {
	explicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "output-profile" || f.Name == "work-profile" {
			explicit = true
		}
	})
	return explicit
}

func backgroundCommand(name string, args ...string) *exec.Cmd {
	cmd := exec.Command(name, args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd
}

func loadJSONFileOrExit[T any](path, prefix string) T {
	var value T
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", prefix, err)
		os.Exit(1)
	}
	if err := json.Unmarshal(b, &value); err != nil {
		fmt.Fprintf(os.Stderr, "%s: parsing %s: %v\n", prefix, path, err)
		os.Exit(1)
	}
	return value
}

func emitPrefixedJSON(stdout, stderr io.Writer, value any, prefix string) int {
	return encodeJSONOrFailPrefixed(stdout, stderr, value, prefix)
}

func resolveCronTimeAndSlot(stderr io.Writer, command, rawAt, rawSlot string, interval time.Duration) (time.Time, string, bool) {
	now, err := parsedRFC3339UTC(rawAt, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "%s: --at %q is not RFC3339: %v\n", command, strings.TrimSpace(rawAt), err)
		return time.Time{}, "", false
	}
	slot := strings.TrimSpace(rawSlot)
	if slot == "" {
		slot = cronFireSlot(now, interval)
	}
	return now, slot, true
}
