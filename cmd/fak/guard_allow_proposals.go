package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// guard_allow_proposals.go implements the propose-only self-widening seam for the
// operator allow overlay (#5182, epic #5170 "Policy Amendment Classes").
//
// The invariant guard_allow.go establishes is that the always-allow overlay is
// operator-authored, out-of-band from the agent. This file gives the wrapped agent a
// sanctioned way to ASK without any way to GRANT: it may record allow/allow_prefix
// entries in a separate PROPOSALS file (`fak guard allow --propose ...`), and only a
// human operator, reviewing them with `fak guard allow --from-proposals`, can merge
// them into the real overlay — and only by adding the explicit `--apply` confirm flag.
// The default `--from-proposals` is strictly list-only; nothing is ever auto-applied.
//
// The proposals file lives beside the overlay in the same guard config dir (default
// .fak/guard/allow-proposals.json, or next to whatever $FAK_GUARD_ALLOW_OVERLAY names)
// so the two travel together and are diffable side by side. Applying goes through the
// EXISTING saveGuardAllowOverlay path — same normalization, same atomic write — and
// then clears the applied proposals so a stale request can never be re-applied later.
const guardAllowProposalsVersion = "fak-guard-allow-proposals/v1"

// guardAllowProposalsFile is the file name, placed in the same directory as the
// operator allow overlay (see guardAllowProposalsPath).
const guardAllowProposalsFile = "allow-proposals.json"

// guardAllowProposalEntry is one pending amendment request: exact tool names and/or
// tool-name prefixes, plus an optional free-text reason the proposer attaches so the
// reviewing operator can judge intent without replaying the session.
type guardAllowProposalEntry struct {
	Allow       []string `json:"allow,omitempty"`
	AllowPrefix []string `json:"allow_prefix,omitempty"`
	Reason      string   `json:"reason,omitempty"`
}

// guardAllowProposals is the on-disk schema of the proposals file. Like the overlay it
// is a strict widen-only shape — no deny, no arg-rules — because a proposal that could
// tighten or reshape the danger floor is not an amendment class this seam admits.
type guardAllowProposals struct {
	Version   string                    `json:"version"`
	Proposals []guardAllowProposalEntry `json:"proposals,omitempty"`
}

// guardAllowProposalsPath mirrors guardAllowOverlayPath's location logic by reusing
// its directory: the proposals file sits beside the (default write-target) overlay,
// whether that is repo-local .fak/guard/ or an $FAK_GUARD_ALLOW_OVERLAY override.
func guardAllowProposalsPath() string {
	return filepath.Join(filepath.Dir(guardAllowOverlayPath()), guardAllowProposalsFile)
}

// loadGuardAllowProposals reads and validates the proposals file with the same
// discipline loadGuardAllowOverlay uses: a MISSING file is the common "nothing
// proposed" case and yields an empty set; a present-but-malformed file fails loud so
// a reviewing operator never silently approves less (or more) than was written.
func loadGuardAllowProposals(path string) (guardAllowProposals, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return guardAllowProposals{Version: guardAllowProposalsVersion}, nil
		}
		return guardAllowProposals{}, fmt.Errorf("guard allow proposals %s: %w", path, err)
	}
	var props guardAllowProposals
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&props); err != nil {
		return guardAllowProposals{}, fmt.Errorf("guard allow proposals %s: invalid: %w", path, err)
	}
	if v := strings.TrimSpace(props.Version); v != "" && v != guardAllowProposalsVersion {
		return guardAllowProposals{}, fmt.Errorf("guard allow proposals %s: unsupported version %q (want %s)", path, props.Version, guardAllowProposalsVersion)
	}
	props.Version = guardAllowProposalsVersion
	kept := props.Proposals[:0]
	for _, p := range props.Proposals {
		p.Allow = guardAllowNormalize(p.Allow)
		p.AllowPrefix = guardAllowNormalize(p.AllowPrefix)
		p.Reason = strings.TrimSpace(p.Reason)
		if len(p.Allow) == 0 && len(p.AllowPrefix) == 0 {
			continue // an entry proposing nothing is noise, drop it
		}
		kept = append(kept, p)
	}
	props.Proposals = kept
	return props, nil
}

// saveGuardAllowProposals writes the proposals as pretty, newline-terminated JSON via
// the same atomic-replace writer the overlay uses.
func saveGuardAllowProposals(path string, props guardAllowProposals) error {
	props.Version = guardAllowProposalsVersion
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("guard allow proposals: mkdir %s: %w", dir, err)
		}
	}
	b, err := json.MarshalIndent(props, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	return writeGuardAllowOverlayAtomic(path, b)
}

// guardAllowProposalsRoute peels the proposals modes off the `fak guard allow` argv
// BEFORE cmdGuardAllow's flag parse (which would reject the unknown flags as errors).
// When a proposals flag is present it runs the mode and EXITS the process (the same
// os.Exit discipline cmdGuardAllow's own mode switch uses); otherwise it returns and
// the caller falls through to cmdGuardAllow unchanged. Kept as a separate router so
// guard_allow.go's operator surface needs no structural change — the seam is additive.
func guardAllowProposalsRoute(argv []string) {
	for _, a := range argv {
		switch a {
		case "--from-proposals", "-from-proposals", "--propose", "-propose":
			os.Exit(cmdGuardAllowProposals(os.Stdout, os.Stderr, argv))
		case "--":
			return // flags after a literal -- are positional, never modes
		}
	}
}

// cmdGuardAllowProposals parses and runs the two proposals modes:
//
//	fak guard allow --propose [--prefix] [--reason <why>] <tool>...   record a proposal (never widens)
//	fak guard allow --from-proposals                                  list pending proposals (review, list-only)
//	fak guard allow --from-proposals --apply [--user]                 operator confirm: merge into the real overlay, clear applied
func cmdGuardAllowProposals(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("guard allow proposals", flag.ExitOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprintln(stderr, guardAllowProposalsUsage()) }
	propose := fs.Bool("propose", false, "record the positional tool name(s) as a PROPOSAL in the proposals file; never touches the real overlay")
	fromProposals := fs.Bool("from-proposals", false, "list the pending allow proposals for operator review (list-only by default)")
	apply := fs.Bool("apply", false, "with --from-proposals: operator confirm — merge the pending proposals into the real allow overlay, then clear them")
	prefix := fs.Bool("prefix", false, "with --propose: treat the positional args as allow_prefix entries rather than exact names")
	reason := fs.String("reason", "", "with --propose: an optional reason attached to the proposal for the reviewing operator")
	user := fs.Bool("user", false, "with --from-proposals --apply: merge into the per-user home overlay instead of the repo-local overlay")
	_ = fs.Parse(argv)

	proposalsPath := guardAllowProposalsPath()
	switch {
	case *propose && *fromProposals:
		fmt.Fprintln(stderr, "fak guard allow: --propose and --from-proposals are mutually exclusive")
		return 2
	case *propose:
		if *apply {
			fmt.Fprintln(stderr, "fak guard allow: --apply is a review flag; it cannot be combined with --propose")
			return 2
		}
		return runGuardAllowPropose(stdout, stderr, proposalsPath, fs.Args(), *prefix, *reason)
	case *fromProposals:
		return runGuardAllowFromProposals(stdout, stderr, proposalsPath, *apply, *user)
	default:
		fmt.Fprintln(stderr, guardAllowProposalsUsage())
		return 2
	}
}

// runGuardAllowPropose appends one proposal entry. This is the only verb a wrapped
// agent is meant to reach: it records the request and nothing else — the real overlay
// is never read, never written.
func runGuardAllowPropose(stdout, stderr io.Writer, proposalsPath string, args []string, prefix bool, reason string) int {
	names := guardAllowNormalize(args)
	if len(names) == 0 {
		fmt.Fprintln(stderr, guardAllowProposalsUsage())
		return 2
	}
	props, err := loadGuardAllowProposals(proposalsPath)
	if err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 1
	}
	entry := guardAllowProposalEntry{Reason: strings.TrimSpace(reason)}
	if prefix {
		entry.AllowPrefix = names
	} else {
		entry.Allow = names
	}
	props.Proposals = append(props.Proposals, entry)
	if err := saveGuardAllowProposals(proposalsPath, props); err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 1
	}
	fmt.Fprintf(stdout, "fak guard allow: recorded a PROPOSAL for %s — the allow overlay is unchanged.\n", strings.Join(names, ", "))
	fmt.Fprintf(stdout, "  proposals file: %s (%d pending)\n", proposalsPath, len(props.Proposals))
	fmt.Fprintln(stdout, "  An operator reviews with: fak guard allow --from-proposals   (and applies with --apply)")
	return 0
}

// runGuardAllowFromProposals lists the pending proposals; with apply it merges them
// into the real overlay via saveGuardAllowOverlay and clears the applied entries.
// The no-confirm default NEVER writes the overlay — propose-only is the whole point.
func runGuardAllowFromProposals(stdout, stderr io.Writer, proposalsPath string, apply, user bool) int {
	props, err := loadGuardAllowProposals(proposalsPath)
	if err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 1
	}
	if len(props.Proposals) == 0 {
		fmt.Fprintf(stdout, "fak guard allow: no pending proposals in %s — nothing to review.\n", proposalsPath)
		return 0
	}
	fmt.Fprintf(stdout, "Pending allow proposals in %s:\n", proposalsPath)
	var allow, allowPrefix []string
	for i, p := range props.Proposals {
		fmt.Fprintf(stdout, "  [%d]", i+1)
		if len(p.Allow) > 0 {
			fmt.Fprintf(stdout, " allow: %s", strings.Join(p.Allow, ", "))
			allow = append(allow, p.Allow...)
		}
		if len(p.AllowPrefix) > 0 {
			fmt.Fprintf(stdout, " allow_prefix: %s", strings.Join(p.AllowPrefix, ", "))
			allowPrefix = append(allowPrefix, p.AllowPrefix...)
		}
		if p.Reason != "" {
			fmt.Fprintf(stdout, "   reason: %s", p.Reason)
		}
		fmt.Fprintln(stdout)
	}
	if !apply {
		fmt.Fprintln(stdout, "\nList-only: the allow overlay is UNCHANGED. To approve and merge these entries")
		fmt.Fprintln(stdout, "into the real overlay (operator, out-of-band from the agent):")
		fmt.Fprintln(stdout, "  fak guard allow --from-proposals --apply")
		return 0
	}

	overlayPath, err := guardAllowWritePath(user)
	if err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 1
	}
	ov, err := loadGuardAllowOverlay(overlayPath)
	if err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 1
	}
	allow = guardAllowNormalize(allow)
	allowPrefix = guardAllowNormalize(allowPrefix)
	ov.Allow = append(ov.Allow, allow...)
	ov.AllowPrefix = append(ov.AllowPrefix, allowPrefix...)
	if err := saveGuardAllowOverlay(overlayPath, ov); err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 1
	}
	// Clear the applied proposals AFTER the overlay write succeeded, so a failed
	// apply never loses the pending requests. An empty (version-only) file is left
	// behind rather than deleting, keeping the path stable for the next proposer.
	if err := saveGuardAllowProposals(proposalsPath, guardAllowProposals{Version: guardAllowProposalsVersion}); err != nil {
		fmt.Fprintln(stderr, "fak guard allow:", err)
		return 1
	}
	fmt.Fprintf(stdout, "\nApplied %d proposal(s) to the operator allow overlay and cleared them.\n", len(props.Proposals))
	printGuardAllowShellAttachments(stdout, allow)
	fmt.Fprintln(stdout, "  Takes effect on the next `fak guard` launch (or POST /v1/fak/policy/reload on a running gateway).")
	printGuardAllowOverlay(stdout, overlayPath, ov)
	return 0
}

// guardAllowProposalsUsage is the one-screen help for the proposals modes.
func guardAllowProposalsUsage() string {
	return strings.Join([]string{
		"fak guard allow — propose-only self-widening (#5182): proposals are requests, never grants.",
		"",
		"usage:",
		"  fak guard allow --propose [--prefix] [--reason <why>] <tool>...",
		"      record the name(s) in the PROPOSALS file (" + guardAllowProposalsFile + " beside the overlay).",
		"      The real allow overlay is never touched — a wrapped agent can ask, not grant.",
		"  fak guard allow --from-proposals",
		"      list the pending proposals for operator review. List-only; applies nothing.",
		"  fak guard allow --from-proposals --apply [--user]",
		"      operator confirm: merge the approved entries into the real allow overlay",
		"      (same normalized, atomic write path as `fak guard allow <tool>`), then clear them.",
		"",
		"Put flags before positional names (Go flag parsing stops at the first non-flag argument).",
	}, "\n")
}
