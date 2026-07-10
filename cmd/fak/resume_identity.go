package main

// resume_identity.go — the read-side of the A-identity-join cluster (#4115). The producer
// (guard SessionStart, #4113) records uuid<->trace rows into the durable, GC-immune
// resume_identity.jsonl store; the watchdog folds drive-state but had no way to answer "which
// trace is this transcript UUID?" (or the reverse). This verb RESOLVES either direction from
// that store so an operator holding a guard trace can find the transcript UUID `claude --resume`
// needs, and vice-versa. It resolves the identity ONLY — it does not resume (that is #1206).

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// runResumeIdentity resolves a UUID or trace against resume_identity.jsonl and prints the paired
// id — plus the recorded handle/account/via provenance when present. Exit 0 on a match, 4 on no
// join (a first-class "not found", not a crash), 2 on usage. Streams are explicit so a test
// drives it without a process.
func runResumeIdentity(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume identity", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "resume")
	regDirFlag := fs.String("reg-dir", "", "registry dir holding resume_identity.jsonl (default: the same regDir the watchdog resolves — $FLEET_REG_DIR, else the host Fleet registry, else <repo>/tools/_registry)")
	asJSON := fs.Bool("json", false, "emit the resolved join as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	query := strings.TrimSpace(fs.Arg(0))
	if query == "" {
		fmt.Fprintln(stderr, "fak resume identity: want a UUID or trace to resolve, e.g. `fak resume identity <uuid>`")
		return 2
	}
	regDir := resolveSweepRegDir(*regDirFlag)
	m := resume.ResolveIdentity(resume.LoadIdentityRows(regDir), query)
	if !m.OK {
		if *asJSON {
			data, _ := json.Marshal(map[string]any{"query": query, "matched": false})
			fmt.Fprintln(stdout, string(data))
		} else {
			fmt.Fprintf(stderr, "no identity join for %q in %s\n", query, resume.IdentityLedgerPath(regDir))
		}
		return 4
	}
	if *asJSON {
		data, _ := json.Marshal(map[string]any{
			"query":     m.Query,
			"paired":    m.Paired,
			"direction": m.Direction,
			"handle":    m.Row.Handle,
			"account":   m.Row.Account,
			"via":       m.Row.Via,
			"matched":   true,
		})
		fmt.Fprintln(stdout, string(data))
		return 0
	}
	// Human: name both ends of the join in the direction it resolved.
	if m.Direction == "uuid->trace" {
		fmt.Fprintf(stdout, "uuid %s -> trace %s%s\n", m.Query, m.Paired, identityProvenance(m.Row))
	} else {
		fmt.Fprintf(stdout, "trace %s -> uuid %s%s\n", m.Query, m.Paired, identityProvenance(m.Row))
	}
	return 0
}

// identityProvenance renders the optional handle/account/via suffix for the human line — an
// empty string when the row carried none, so the join reads clean when no provenance was recorded.
func identityProvenance(r resume.IdentityRow) string {
	var parts []string
	if h := strings.TrimSpace(r.Handle); h != "" {
		parts = append(parts, "handle="+h)
	}
	if a := strings.TrimSpace(r.Account); a != "" {
		parts = append(parts, "account="+a)
	}
	if v := strings.TrimSpace(r.Via); v != "" {
		parts = append(parts, "via="+v)
	}
	if len(parts) == 0 {
		return ""
	}
	return " (" + strings.Join(parts, ", ") + ")"
}
