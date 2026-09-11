package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"text/tabwriter"
	"time"
)

// fak cache is the thin gateway cache-admin umbrella: it folds the live
// /v1/fak/cacheprt/* health endpoints served by `fak serve` into three
// read-only operator views. It never fabricates data - if the gateway is
// unreachable it says so and exits 1.
//
//	fak cache status      [--addr URL] [--json] [--compact]
//	fak cache namespaces  [--addr URL] [--json] [--compact]
//	fak cache coldcliff   [--addr URL] [--json] [--compact]
func cmdCache(argv []string) {
	os.Exit(runCache(os.Stdout, os.Stderr, argv))
}

func runCache(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		cacheUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "status":
		return runCacheStatus(stdout, stderr, argv[1:])
	case "namespaces":
		return runCacheNamespaces(stdout, stderr, argv[1:])
	case "coldcliff":
		return runCacheColdcliff(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		cacheUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak cache: unknown subcommand %q\n", argv[0])
		cacheUsage(stderr)
		return 2
	}
}

func cacheUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak cache status      [--addr URL] [--json] [--compact]
  fak cache namespaces  [--addr URL] [--json] [--compact]
  fak cache coldcliff   [--addr URL] [--json] [--compact]

Each subcommand GETs the matching /v1/fak/cacheprt/* health endpoint on the
running fak serve gateway. --addr overrides the target (default $FAK_ADDR or
http://127.0.0.1:8080). --json emits the endpoint's JSON verbatim; --compact
prints one summary line with the key fields. Read-only: no cache state is
mutated. Connection-refused exits 1 with "gateway unreachable"; no data is
ever fabricated.

`)
}

// cacheAdminDefaultAddr mirrors the other connection-flag families (`fak ps`,
// `fak session`): FAK_ADDR env override, then the `fak serve` default 8080.
func cacheAdminDefaultAddr() string {
	addr := strings.TrimSpace(os.Getenv("FAK_ADDR"))
	if addr == "" {
		addr = "http://127.0.0.1:8080"
	}
	if !strings.Contains(addr, "://") {
		addr = "http://" + addr
	}
	return strings.TrimRight(addr, "/")
}

// cacheAdminGet fetches one cacheprt endpoint as raw JSON, refusing to
// interpret non-JSON bodies or non-200s as data.
func cacheAdminGet(addr, path string) (json.RawMessage, error) {
	url := addr + path
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("gateway unreachable at %s (is `fak serve` running?)", addr)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", url, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gateway at %s returned HTTP %d for %s", addr, resp.StatusCode, path)
	}
	out := json.RawMessage(strings.TrimSpace(string(body)))
	if !json.Valid(out) {
		return nil, fmt.Errorf("gateway at %s returned non-JSON body for %s", addr, path)
	}
	return out, nil
}

func runCacheStatus(stdout, stderr io.Writer, argv []string) int {
	return runCacheAdminGet(stdout, stderr, argv, "fak cache status", "/v1/fak/cacheprt/summary", renderCacheStatus)
}

func runCacheNamespaces(stdout, stderr io.Writer, argv []string) int {
	return runCacheAdminGet(stdout, stderr, argv, "fak cache namespaces", "/v1/fak/cacheprt/namespaces", renderCacheNamespaces)
}

func runCacheColdcliff(stdout, stderr io.Writer, argv []string) int {
	return runCacheAdminGet(stdout, stderr, argv, "fak cache coldcliff", "/v1/fak/cacheprt/coldcliff", renderCacheColdcliff)
}

// runCacheAdminGet is the shared plumbing for the three views: parse flags,
// GET the endpoint, then render verbatim JSON (--json), compact (--compact),
// or the default pretty view.
func runCacheAdminGet(stdout, stderr io.Writer, argv []string, label, path string, render func(io.Writer, json.RawMessage, bool)) int {
	fs := flag.NewFlagSet(label, flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", cacheAdminDefaultAddr(), "gateway base URL (default $FAK_ADDR or http://127.0.0.1:8080)")
	asJSON := fs.Bool("json", false, "emit the endpoint JSON verbatim")
	compact := fs.Bool("compact", false, "print one summary line with the key fields")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "%s: unexpected argument %q\n", label, fs.Arg(0))
		return 2
	}
	body, err := cacheAdminGet(*addr, path)
	if err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", label, err)
		return 1
	}
	if *asJSON {
		fmt.Fprintf(stdout, "%s\n", body)
		return 0
	}
	render(stdout, body, *compact)
	return 0
}

// cacheAdminJSONString unwraps a JSON string value's outer quoting when the
// raw fragment is a string; other fragments (numbers, objects) are trimmed
// and returned as-is.
func cacheAdminJSONString(v json.RawMessage) string {
	s := strings.TrimSpace(string(v))
	if strings.HasPrefix(s, "\"") {
		var out string
		if err := json.Unmarshal(v, &out); err == nil {
			return out
		}
	}
	return s
}

// cacheAdminFieldLines renders the endpoint's top-level JSON object as sorted
// "field=value" fragments. Unknown shapes print the verbatim JSON instead of
// inventing fields.
func cacheAdminFieldLines(w io.Writer, body json.RawMessage) {
	var obj map[string]json.RawMessage
	fields := make([]string, 0, 8)
	if err := json.Unmarshal(body, &obj); err == nil && len(obj) > 0 {
		keys := make([]string, 0, len(obj))
		for k := range obj {
			keys = append(keys, k)
		}
		sortStrings(keys)
		for _, k := range keys {
			val := cacheAdminJSONString(obj[k])
			if len(val) > 48 {
				val = val[:45] + "..."
			}
			fields = append(fields, fmt.Sprintf("%s=%s", k, val))
		}
	}
	if len(fields) == 0 {
		fmt.Fprintf(w, "%s\n", body)
		return
	}
	fmt.Fprintf(w, "%s\n", strings.Join(fields, " "))
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// renderCacheStatus prints the summary endpoint: verbatim JSON (--json), one
// line (--compact), else a small table. The table reinterprets the summary as
// a key/value map so endpoint schema drift stays visible instead of being
// papered over with hardcoded fields.
func renderCacheStatus(w io.Writer, body json.RawMessage, compact bool) {
	if compact {
		cacheAdminFieldLines(w, body)
		return
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil || len(obj) == 0 {
		fmt.Fprintf(w, "cache status:\n%s\n", body)
		return
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "field\tvalue\t")
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sortStrings(keys)
	for _, k := range keys {
		fmt.Fprintf(tw, "%s\t%s\t\n", k, cacheAdminJSONString(obj[k]))
	}
	tw.Flush()
}

// renderCacheNamespaces prints one row per namespace, tolerating both wire
// shapes: a bare row array or the gateway's wrapped record
// ({"namespaces":[...],"degraded":true,"degraded_reason":...}). Rows only
// carry values the endpoint actually reported; "degraded" is surfaced rather
// than smoothed over, and empty rows print "namespaces: none". --compact
// collapses the whole table to one line.
func renderCacheNamespaces(w io.Writer, body json.RawMessage, compact bool) {
	var wrapper struct {
		Namespaces     []map[string]json.RawMessage `json:"namespaces"`
		Degraded       *bool                        `json:"degraded"`
		DegradedReason json.RawMessage              `json:"degraded_reason"`
	}
	var bareRows []map[string]json.RawMessage
	rows := bareRows
	degraded := false
	degradedReason := ""
	if err := json.Unmarshal(body, &wrapper); err == nil && wrapper.Namespaces != nil {
		rows = wrapper.Namespaces
		if wrapper.Degraded != nil {
			degraded = *wrapper.Degraded
		}
		degradedReason = strings.TrimSpace(string(wrapper.DegradedReason))
	} else if err2 := json.Unmarshal(body, &bareRows); err2 == nil {
		rows = bareRows
	} else {
		fmt.Fprintf(w, "cache namespaces:\n%s\n", body)
		return
	}
	if compact {
		stats := make([]string, 0, len(rows)+1)
		for _, row := range rows {
			stats = append(stats, fmt.Sprintf("%s=%s", cacheAdminRowName(row), cacheAdminRowValue(row)))
		}
		if degraded {
			stats = append(stats, "degraded=true")
		}
		if len(stats) == 0 {
			fmt.Fprintln(w, "namespaces: none")
			return
		}
		fmt.Fprintf(w, "namespaces(%d): %s\n", len(rows), strings.Join(stats, " "))
		return
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "namespace	fields	")
	for _, row := range rows {
		name := cacheAdminRowName(row)
		keys := make([]string, 0, len(row)-1)
		for k := range row {
			if k == "name" || k == "namespace" || k == "id" {
				continue
			}
			keys = append(keys, k)
		}
		sortStrings(keys)
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			parts = append(parts, fmt.Sprintf("%s=%s", k, cacheAdminJSONString(row[k])))
		}
		fmt.Fprintf(tw, "%s	%s	\n", name, strings.Join(parts, " "))
	}
	tw.Flush()
	if degraded {
		if degradedReason != "" {
			fmt.Fprintf(w, "degraded: true (%s)\n", cacheAdminJSONString(wrapper.DegradedReason))
		} else {
			fmt.Fprintln(w, "degraded: true")
		}
	}
}

// renderCacheColdcliff prints the cold-cliff read: whether the frozen-
// trajectory cold cliff fired, with the verdict evidence. --compact prints
// the fields on one line.
func renderCacheColdcliff(w io.Writer, body json.RawMessage, compact bool) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil || len(obj) == 0 {
		fmt.Fprintf(w, "cache coldcliff:\n%s\n", body)
		return
	}
	if compact {
		cacheAdminFieldLines(w, body)
		return
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "field	value	")
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sortStrings(keys)
	for _, k := range keys {
		fmt.Fprintf(tw, "%s	%s	\n", k, cacheAdminJSONString(obj[k]))
	}
	tw.Flush()
}

func cacheAdminRowName(row map[string]json.RawMessage) string {
	for _, k := range []string{"name", "namespace", "id"} {
		if v, ok := row[k]; ok {
			return cacheAdminJSONString(v)
		}
	}
	return "?"
}

func cacheAdminRowValue(row map[string]json.RawMessage) string {
	for _, k := range []string{"ratio", "hit_rate", "value", "tokens", "nodes", "bytes", "size"} {
		if v, ok := row[k]; ok {
			s := cacheAdminJSONString(v)
			if len(s) > 32 {
				s = s[:29] + "..."
			}
			return s
		}
	}
	if len(row) > 0 {
		keys := make([]string, 0, len(row))
		for k := range row {
			keys = append(keys, k)
		}
		sortStrings(keys)
		for _, k := range keys {
			switch row[k][0] {
			case '{', '[':
				continue
			default:
				return fmt.Sprintf("%s=%s", k, cacheAdminJSONString(row[k]))
			}
		}
	}
	return "-"
}
