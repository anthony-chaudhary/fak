package devcmd

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// TicketInfo represents metadata and content of an architectural ticket.
type TicketInfo struct {
	ID            string   `json:"id"`
	Title         string   `json:"title"`
	Tranche       string   `json:"tranche"`
	Path          string   `json:"path"`
	Key           string   `json:"key,omitempty"`
	Lane          string   `json:"lane,omitempty"`
	GithubIssue   string   `json:"github_issue,omitempty"`
	ExpectedSteps int      `json:"expected_steps,omitempty"`
	TargetPaths   []string `json:"target_paths,omitempty"`
	Body          string   `json:"body,omitempty"`
}

// TrancheInfo represents a group of tickets under a tranche directory in docs/tickets/.
type TrancheInfo struct {
	Tranche     string       `json:"tranche"`
	Path        string       `json:"path"`
	TicketCount int          `json:"ticket_count"`
	Tickets     []TicketInfo `json:"tickets"`
}

// RunCompanionTicket implements the 'fak-dev ticket' command suite.
// It discovers, inspects, and reads migration ticket specifications residing in companion fak-private/docs/tickets/.
func RunCompanionTicket(stdout, stderr io.Writer, argv []string) int {
	_, privRoot := ResolveCompanionRoots()

	isHelp := len(argv) == 0
	for _, a := range argv {
		if a == "--help" || a == "-h" || a == "help" {
			isHelp = true
			break
		}
	}

	if privRoot == "" {
		if isHelp {
			fmt.Fprintln(stdout, "fak-dev ticket: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.")
			writeTicketHelp(stdout)
			return 0
		}
		fmt.Fprintln(stderr, "fak-dev ticket: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.")
		return 1
	}

	if len(argv) == 0 {
		writeTicketHelp(stdout)
		return 0
	}

	subcmd := argv[0]
	subArgs := argv[1:]

	if subcmd == "help" || subcmd == "-h" || subcmd == "--help" {
		writeTicketHelp(stdout)
		return 0
	}

	switch subcmd {
	case "list", "ls":
		return runTicketList(stdout, stderr, privRoot, subArgs)
	case "show", "view":
		return runTicketShow(stdout, stderr, privRoot, subArgs)
	case "next":
		return runTicketNext(stdout, stderr, privRoot, subArgs)
	default:
		fmt.Fprintf(stderr, "fak-dev ticket: unknown subcommand %q\n", subcmd)
		writeTicketHelp(stderr)
		return 2
	}
}

func writeTicketHelp(w io.Writer) {
	fmt.Fprintln(w, "Usage: fak-dev ticket <subcommand> [flags] [args]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Discover, inspect, and read migration ticket specifications from companion fak-private.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Subcommands:")
	fmt.Fprintln(w, "  list [--json] [--tranche <name>] [--lane <name>]")
	fmt.Fprintln(w, "      Enumerate ticket tranches and specifications")
	fmt.Fprintln(w, "  show <ticket-id> [--json] (alias: view)")
	fmt.Fprintln(w, "      Display ticket content and metadata")
	fmt.Fprintln(w, "  next [args...]")
	fmt.Fprintln(w, "      Proxy to companion 'fak-sync issue next'")
	fmt.Fprintln(w, "  help")
	fmt.Fprintln(w, "      Display this help message")
}

func writeTicketListHelp(w io.Writer) {
	fmt.Fprintln(w, "Usage: fak-dev ticket list [flags]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Enumerate ticket tranches and specifications under docs/tickets/.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Flags:")
	fmt.Fprintln(w, "  --json            Output structured JSON")
	fmt.Fprintln(w, "  --tranche <name>  Filter by tranche directory name")
	fmt.Fprintln(w, "  --lane <name>     Filter by ticket lane")
}

func writeTicketShowHelp(w io.Writer) {
	fmt.Fprintln(w, "Usage: fak-dev ticket show <ticket-id> [--json]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Display ticket content or metadata. Aliased as 'fak-dev ticket view'.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Flags:")
	fmt.Fprintln(w, "  --json            Output structured metadata and body as JSON")
}

func isTicketFile(filename string) bool {
	lower := strings.ToLower(filename)
	if !strings.HasSuffix(lower, ".md") {
		return false
	}
	base := strings.TrimSuffix(lower, ".md")
	if base == "index" || base == "audit" || base == "readme" {
		return false
	}
	return true
}

func parseTicketFile(filePath, trancheName, privRoot string, includeBody bool) (*TicketInfo, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	base := filepath.Base(filePath)
	id := strings.TrimSuffix(base, filepath.Ext(base))

	relPath, err := filepath.Rel(privRoot, filePath)
	if err != nil {
		relPath = filePath
	}
	relPath = filepath.ToSlash(relPath)

	info := &TicketInfo{
		ID:      id,
		Tranche: trancheName,
		Path:    relPath,
	}

	lines := strings.Split(string(data), "\n")
	inRouting := false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Parse Title (first H1 heading)
		if info.Title == "" && strings.HasPrefix(trimmed, "# ") {
			info.Title = strings.TrimSpace(strings.TrimPrefix(trimmed, "# "))
		}

		// Parse Key from HTML comment e.g. <!-- fak-dev-key: companion-ticket-reader -->
		if info.Key == "" && strings.Contains(trimmed, "-key:") {
			if start := strings.Index(trimmed, "-key:"); start >= 0 {
				rest := trimmed[start+len("-key:"):]
				if end := strings.Index(rest, "-->"); end >= 0 {
					info.Key = strings.TrimSpace(rest[:end])
				}
			}
		}

		// Parse Issue from comment e.g. <!-- fak-private-issue: anthony-chaudhary/fak-private#714 -->
		if info.GithubIssue == "" && strings.Contains(trimmed, "-issue:") {
			if start := strings.Index(trimmed, "-issue:"); start >= 0 {
				rest := trimmed[start+len("-issue:"):]
				if end := strings.Index(rest, "-->"); end >= 0 {
					info.GithubIssue = strings.TrimSpace(rest[:end])
				}
			}
		}

		// Parse routing block
		if strings.HasPrefix(trimmed, "```routing") {
			inRouting = true
			continue
		}
		if inRouting {
			if strings.HasPrefix(trimmed, "```") {
				inRouting = false
				continue
			}
			if strings.HasPrefix(trimmed, "lane:") {
				info.Lane = strings.TrimSpace(strings.TrimPrefix(trimmed, "lane:"))
			} else if strings.HasPrefix(trimmed, "expected_steps:") {
				val := strings.TrimSpace(strings.TrimPrefix(trimmed, "expected_steps:"))
				if n, err := strconv.Atoi(val); err == nil {
					info.ExpectedSteps = n
				}
			} else if strings.HasPrefix(trimmed, "github_issue:") {
				if info.GithubIssue == "" {
					info.GithubIssue = strings.TrimSpace(strings.TrimPrefix(trimmed, "github_issue:"))
				}
			} else if strings.HasPrefix(trimmed, "paths:") {
				raw := strings.TrimSpace(strings.TrimPrefix(trimmed, "paths:"))
				raw = strings.TrimPrefix(raw, "[")
				raw = strings.TrimSuffix(raw, "]")
				parts := strings.Split(raw, ",")
				for _, p := range parts {
					cleanP := strings.Trim(strings.TrimSpace(p), `"'`)
					if cleanP != "" {
						info.TargetPaths = append(info.TargetPaths, cleanP)
					}
				}
			}
		}
	}

	if info.Title == "" {
		info.Title = info.ID
	}

	if includeBody {
		info.Body = string(data)
	}

	return info, nil
}

func enumerateTranches(ticketsDir, privRoot string) ([]TrancheInfo, error) {
	entries, err := os.ReadDir(ticketsDir)
	if err != nil {
		return nil, err
	}

	var tranches []TrancheInfo
	var directTickets []TicketInfo

	for _, entry := range entries {
		if entry.IsDir() {
			trancheName := entry.Name()
			tranchePath := filepath.Join(ticketsDir, trancheName)
			files, err := os.ReadDir(tranchePath)
			if err != nil {
				continue
			}

			var tickets []TicketInfo
			for _, f := range files {
				if f.IsDir() || !isTicketFile(f.Name()) {
					continue
				}
				filePath := filepath.Join(tranchePath, f.Name())
				info, err := parseTicketFile(filePath, trancheName, privRoot, false)
				if err != nil {
					continue
				}
				tickets = append(tickets, *info)
			}

			if len(tickets) > 0 {
				sort.Slice(tickets, func(i, j int) bool {
					return tickets[i].ID < tickets[j].ID
				})
				relTranche, _ := filepath.Rel(privRoot, tranchePath)
				tranches = append(tranches, TrancheInfo{
					Tranche:     trancheName,
					Path:        filepath.ToSlash(relTranche),
					TicketCount: len(tickets),
					Tickets:     tickets,
				})
			}
		} else if isTicketFile(entry.Name()) {
			filePath := filepath.Join(ticketsDir, entry.Name())
			info, err := parseTicketFile(filePath, "tickets", privRoot, false)
			if err == nil {
				directTickets = append(directTickets, *info)
			}
		}
	}

	if len(directTickets) > 0 {
		sort.Slice(directTickets, func(i, j int) bool {
			return directTickets[i].ID < directTickets[j].ID
		})
		relPath, _ := filepath.Rel(privRoot, ticketsDir)
		tranches = append(tranches, TrancheInfo{
			Tranche:     "tickets",
			Path:        filepath.ToSlash(relPath),
			TicketCount: len(directTickets),
			Tickets:     directTickets,
		})
	}

	sort.Slice(tranches, func(i, j int) bool {
		return tranches[i].Tranche < tranches[j].Tranche
	})

	return tranches, nil
}

func runTicketList(stdout, stderr io.Writer, privRoot string, args []string) int {
	var jsonOut bool
	var filterTranche string
	var filterLane string

	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "--json":
			jsonOut = true
		case arg == "--tranche":
			if i+1 < len(args) {
				i++
				filterTranche = args[i]
			}
		case strings.HasPrefix(arg, "--tranche="):
			filterTranche = strings.TrimPrefix(arg, "--tranche=")
		case arg == "--lane":
			if i+1 < len(args) {
				i++
				filterLane = args[i]
			}
		case strings.HasPrefix(arg, "--lane="):
			filterLane = strings.TrimPrefix(arg, "--lane=")
		case arg == "--help" || arg == "-h" || arg == "help":
			writeTicketListHelp(stdout)
			return 0
		default:
			fmt.Fprintf(stderr, "fak-dev ticket list: unknown flag %q\n", arg)
			writeTicketListHelp(stderr)
			return 2
		}
	}

	ticketsDir := filepath.Join(privRoot, "docs", "tickets")
	tranches, err := enumerateTranches(ticketsDir, privRoot)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev ticket list: failed to read tickets directory %s: %v\n", ticketsDir, err)
		return 1
	}

	if filterTranche != "" {
		var filtered []TrancheInfo
		for _, tr := range tranches {
			if strings.EqualFold(tr.Tranche, filterTranche) {
				filtered = append(filtered, tr)
			}
		}
		tranches = filtered
	}

	if filterLane != "" {
		var filtered []TrancheInfo
		for _, tr := range tranches {
			var matchingTickets []TicketInfo
			for _, tk := range tr.Tickets {
				if strings.EqualFold(tk.Lane, filterLane) {
					matchingTickets = append(matchingTickets, tk)
				}
			}
			if len(matchingTickets) > 0 {
				tr.Tickets = matchingTickets
				tr.TicketCount = len(matchingTickets)
				filtered = append(filtered, tr)
			}
		}
		tranches = filtered
	}

	if jsonOut {
		if tranches == nil {
			tranches = []TrancheInfo{}
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(tranches); err != nil {
			fmt.Fprintf(stderr, "fak-dev ticket list: JSON encoding failed: %v\n", err)
			return 1
		}
		return 0
	}

	if len(tranches) == 0 {
		fmt.Fprintln(stdout, "No tickets found.")
		return 0
	}

	for i, tr := range tranches {
		if i > 0 {
			fmt.Fprintln(stdout)
		}
		fmt.Fprintf(stdout, "Tranche: %s (%d tickets)\n", tr.Tranche, tr.TicketCount)
		for _, tk := range tr.Tickets {
			fmt.Fprintf(stdout, "  %s: %s\n", tk.ID, tk.Title)
		}
	}

	return 0
}

func findTicket(ticketsDir, privRoot, target string) (*TicketInfo, error) {
	cleanTarget := strings.TrimSpace(target)
	cleanTarget = strings.TrimSuffix(cleanTarget, ".md")

	tranches, err := enumerateTranches(ticketsDir, privRoot)
	if err != nil {
		return nil, fmt.Errorf("failed to read tickets directory: %w", err)
	}

	var allTickets []TicketInfo
	for _, tr := range tranches {
		allTickets = append(allTickets, tr.Tickets...)
	}

	// 1. Direct path check (e.g. tranche/TICKET-01-foo or docs/tickets/tranche/TICKET-01-foo.md)
	directCandidate := filepath.Join(privRoot, cleanTarget+".md")
	if fi, err := os.Stat(directCandidate); err == nil && !fi.IsDir() {
		rel, _ := filepath.Rel(ticketsDir, directCandidate)
		tranche := filepath.Dir(rel)
		if tranche == "." {
			tranche = "tickets"
		}
		return parseTicketFile(directCandidate, tranche, privRoot, true)
	}

	directCandidate = filepath.Join(ticketsDir, cleanTarget+".md")
	if fi, err := os.Stat(directCandidate); err == nil && !fi.IsDir() {
		rel, _ := filepath.Rel(ticketsDir, directCandidate)
		tranche := filepath.Dir(rel)
		if tranche == "." {
			tranche = "tickets"
		}
		return parseTicketFile(directCandidate, tranche, privRoot, true)
	}

	// 2. Exact match candidates: ID, Path, Key, GithubIssue
	var exactMatches []TicketInfo
	for _, tk := range allTickets {
		if strings.EqualFold(tk.ID, cleanTarget) {
			exactMatches = append(exactMatches, tk)
		} else if strings.EqualFold(tk.Path, cleanTarget) || strings.EqualFold(tk.Path, cleanTarget+".md") {
			exactMatches = append(exactMatches, tk)
		} else if tk.Key != "" && strings.EqualFold(tk.Key, cleanTarget) {
			exactMatches = append(exactMatches, tk)
		} else if tk.GithubIssue != "" && (strings.EqualFold(tk.GithubIssue, cleanTarget) || strings.HasSuffix(tk.GithubIssue, "#"+strings.TrimPrefix(cleanTarget, "#"))) {
			exactMatches = append(exactMatches, tk)
		}
	}

	if len(exactMatches) == 1 {
		fullPath := filepath.Join(privRoot, exactMatches[0].Path)
		return parseTicketFile(fullPath, exactMatches[0].Tranche, privRoot, true)
	}
	if len(exactMatches) > 1 {
		lines := make([]string, len(exactMatches))
		for i, m := range exactMatches {
			lines[i] = fmt.Sprintf("  %s (%s)", m.ID, m.Path)
		}
		return nil, fmt.Errorf("ambiguous ticket ID %q, matched multiple tickets:\n%s", cleanTarget, strings.Join(lines, "\n"))
	}

	// 3. Prefix match on ID (e.g. "TICKET-01" matches "TICKET-01-companion-aware-precommit-hooks")
	var prefixMatches []TicketInfo
	targetLower := strings.ToLower(cleanTarget)
	for _, tk := range allTickets {
		idLower := strings.ToLower(tk.ID)
		if strings.HasPrefix(idLower, targetLower+"-") || strings.HasPrefix(idLower, targetLower+"_") {
			prefixMatches = append(prefixMatches, tk)
		}
	}

	if len(prefixMatches) == 1 {
		fullPath := filepath.Join(privRoot, prefixMatches[0].Path)
		return parseTicketFile(fullPath, prefixMatches[0].Tranche, privRoot, true)
	}
	if len(prefixMatches) > 1 {
		lines := make([]string, len(prefixMatches))
		for i, m := range prefixMatches {
			lines[i] = fmt.Sprintf("  %s (%s)", m.ID, m.Path)
		}
		return nil, fmt.Errorf("ambiguous ticket ID %q, matched multiple tickets:\n%s", cleanTarget, strings.Join(lines, "\n"))
	}

	// 4. Substring match on ID
	var subMatches []TicketInfo
	for _, tk := range allTickets {
		idLower := strings.ToLower(tk.ID)
		if strings.Contains(idLower, targetLower) {
			subMatches = append(subMatches, tk)
		}
	}

	if len(subMatches) == 1 {
		fullPath := filepath.Join(privRoot, subMatches[0].Path)
		return parseTicketFile(fullPath, subMatches[0].Tranche, privRoot, true)
	}
	if len(subMatches) > 1 {
		lines := make([]string, len(subMatches))
		for i, m := range subMatches {
			lines[i] = fmt.Sprintf("  %s (%s)", m.ID, m.Path)
		}
		return nil, fmt.Errorf("ambiguous ticket ID %q, matched multiple tickets:\n%s", cleanTarget, strings.Join(lines, "\n"))
	}

	return nil, fmt.Errorf("ticket %q not found under docs/tickets", cleanTarget)
}

func runTicketShow(stdout, stderr io.Writer, privRoot string, args []string) int {
	var jsonOut bool
	var ticketTarget string

	for _, arg := range args {
		switch {
		case arg == "--json":
			jsonOut = true
		case arg == "--help" || arg == "-h" || arg == "help":
			writeTicketShowHelp(stdout)
			return 0
		case strings.HasPrefix(arg, "-"):
			fmt.Fprintf(stderr, "fak-dev ticket show: unknown flag %q\n", arg)
			writeTicketShowHelp(stderr)
			return 2
		default:
			if ticketTarget == "" {
				ticketTarget = arg
			} else {
				fmt.Fprintf(stderr, "fak-dev ticket show: unexpected argument %q\n", arg)
				writeTicketShowHelp(stderr)
				return 2
			}
		}
	}

	if ticketTarget == "" {
		fmt.Fprintln(stderr, "fak-dev ticket show: ticket ID required")
		writeTicketShowHelp(stderr)
		return 2
	}

	ticketsDir := filepath.Join(privRoot, "docs", "tickets")
	ticket, err := findTicket(ticketsDir, privRoot, ticketTarget)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev ticket show: %v\n", err)
		return 1
	}

	if jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(ticket); err != nil {
			fmt.Fprintf(stderr, "fak-dev ticket show: JSON encoding failed: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprint(stdout, ticket.Body)
	if !strings.HasSuffix(ticket.Body, "\n") {
		fmt.Fprintln(stdout)
	}
	return 0
}

func runTicketNext(stdout, stderr io.Writer, privRoot string, args []string) int {
	subArgs := []string{"-C", privRoot, "run", "./cmd/fak-sync", "issue", "next"}
	subArgs = append(subArgs, args...)

	cmd := exec.Command("go", subArgs...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.Stdin = os.Stdin

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode()
		}
		fmt.Fprintf(stderr, "fak-dev ticket next: failed to execute companion issue next: %v\n", err)
		return 1
	}
	return 0
}
