package main

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// accounts_report.go — the human-readable rendering half of `fak accounts`: the roster
// table (`list`), the login-status table (`status`), and the summary/collapse lines both
// hang off. Split out of accounts.go so the verb implementations own the decisions and
// this file owns only how they READ. Nothing here decides anything — the closed statuses,
// can_serve, warnings, and next actions are computed in internal/accounts and are only
// formatted here.

func printAccountsTable(w io.Writer, reg accounts.Registry, showAll bool) {
	// One provenance line above the table: WHICH fak build rendered this and the registry
	// schema it speaks. It is the cheap visibility half of `fak accounts version` — an operator
	// reading a roster sees the tool version inline, so a stale binary is obvious at a glance.
	fmt.Fprintf(w, "# fak %s · registry %s\n", appversion.Current(), accounts.RegistryVersion)
	// Reconcile groups the seats by the account each truly resolves to, so the table can
	// flag a seat that is really a duplicate of another (one rate-limit bucket presented
	// as several) and a seat whose setup token belongs to a different login than its own.
	rec := reg.Reconcile()
	report := reg.LoginReport()
	if !showAll {
		report = report.WithoutTombstoned()
	}
	obsByName := loginObservationsByName(report)
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tSTATUS\tLOGIN\tIDENTITY\tCREDS\tREHOME\tFLAG")
	dupes, twins := 0, 0
	accountSet := map[string]bool{}
	for _, h := range reg.Homes {
		// Tombstoned seats are retired bookkeeping, not a mainline roster row: with
		// dozens of them they bury the live seats an operator actually reads. Collapse
		// them into the one-line count below the table unless --all is asked for; the
		// login summary still carries the tombstoned=N tally, and --json stays complete.
		if !h.Active() && !showAll {
			continue
		}
		name := h.Name
		if h.Default {
			name += " *"
		}
		status := string(h.Status)
		if status == "" {
			status = "active"
		}
		login := string(h.LoginStatus())
		ident := h.Identity.Email
		exists := h.Identity.Exists
		hasCreds := h.Identity.HasCreds
		if obs, ok := obsByName[h.Name]; ok {
			login = string(obs.Status)
			ident = obs.Email
			exists = obs.Exists
			hasCreds = obs.HasCreds
		}
		if ident == "" {
			ident = "-"
		}
		creds := "-"
		if exists {
			if hasCreds {
				creds = "yes"
			} else {
				creds = "NO"
			}
		}
		rehome := ""
		if h.RehomeTo != "" {
			rehome = "-> " + h.RehomeTo
		}
		// Flags accumulate (a seat can be both a name-lie AND a duplicate): tombstone,
		// name<>identity, dup->canonical, and the token-twin split-identity warning.
		var flags []string
		if !h.Active() {
			flags = append(flags, "TOMBSTONED")
		}
		if h.NameLie() {
			flags = append(flags, "WARN name<>identity")
		}
		if si, ok := rec[h.Name]; ok {
			if si.Account != "" {
				accountSet[si.Account] = true
			}
			if si.Role == accounts.RoleDuplicate {
				dupes++
				flags = append(flags, "dup -> "+si.Canonical)
			}
			if len(si.TokenTwin) > 0 {
				twins++
				flags = append(flags, "token-twin -> "+strings.Join(si.TokenTwin, ","))
			}
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", name, status, login, ident, creds, rehome, strings.Join(flags, "; "))
	}
	tw.Flush()
	printLoginSummary(w, report, "login")
	// A one-line reconcile summary when there is anything to collapse or warn about, so
	// the operator sees "N seats are really M accounts" instead of inferring it per row.
	if dupes > 0 || twins > 0 {
		fmt.Fprintf(w, "reconcile: %d active seat(s) resolve to %d distinct account(s)",
			len(rec), len(accountSet))
		if dupes > 0 {
			fmt.Fprintf(w, "; %d duplicate seat(s) collapse onto their canonical", dupes)
		}
		if twins > 0 {
			fmt.Fprintf(w, "; %d seat(s) carry another login's setup token (token-twin)", twins)
		}
		fmt.Fprintln(w)
	}
}

func printAccountsStatus(w io.Writer, report accounts.LoginReport, showAll bool) {
	if !showAll {
		report = report.WithoutTombstoned()
	}
	fmt.Fprintf(w, "# %s\n", report.Schema)
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tLOGIN\tCAN_SERVE\tACCOUNT\tIDENTITY\tROLES\tNEXT_ACTION\tWARNING")
	for _, obs := range report.Seats {
		// Same collapse as the list table: a retired seat is not a mainline status row.
		// Hide it unless --all; the summary line below still tallies tombstoned=N.
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			obs.Name,
			obs.Status,
			yesNo(obs.CanServe),
			dash(obs.Account),
			dash(obs.Email),
			dash(strings.Join(obs.Roles, ",")),
			dash(obs.NextAction),
			dash(loginWarningsText(obs.Warnings)),
		)
	}
	tw.Flush()
	printLoginSummary(w, report, "summary")
}

func loginObservationsByName(report accounts.LoginReport) map[string]accounts.LoginObservation {
	out := make(map[string]accounts.LoginObservation, len(report.Seats))
	for _, obs := range report.Seats {
		out[obs.Name] = obs
	}
	return out
}

func printLoginSummary(w io.Writer, report accounts.LoginReport, prefix string) {
	// Denominator is the active-style roster (Total minus tombstoned/disabled/
	// missing_dir), not every seat ever registered: reporting "5/36" when 22 seats
	// are tombstoned understates the servable pool. The terminal-class counts still
	// appear in the per-status breakdown below for context.
	fmt.Fprintf(w, "%s: %d/%d active seat(s) can serve; %d distinct account(s)",
		prefix, report.Summary.CanServe, report.Summary.ActiveStyleSeats, report.Summary.DistinctAccounts)
	for _, part := range sortedLoginStatusParts(report.Summary.ByStatus) {
		fmt.Fprintf(w, "; %s", part)
	}
	if report.Summary.WarningSeats > 0 {
		fmt.Fprintf(w, "; %d warning seat(s)", report.Summary.WarningSeats)
	}
	fmt.Fprintln(w)
}

func sortedLoginStatusParts(by map[string]int) []string {
	keys := make([]string, 0, len(by))
	for k := range by {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, by[k]))
	}
	return parts
}

func loginWarningsText(ws []accounts.LoginWarning) string {
	if len(ws) == 0 {
		return ""
	}
	out := make([]string, len(ws))
	for i, w := range ws {
		out[i] = string(w)
	}
	sort.Strings(out)
	return strings.Join(out, ",")
}
