package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/portability"
)

func runContinuity(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] == "help" || argv[0] == "-h" || argv[0] == "--help" {
		continuityHelp(stdout)
		return 0
	}
	sub, args := argv[0], argv[1:]
	if sub == "ux-selfcheck" {
		jsonOut := len(args) > 0 && args[0] == "--json"
		return runContinuityUXSelfcheck(stdout, jsonOut)
	}
	if sub == "explain" {
		return runContinuityExplain(stdout, args)
	}
	sub, impliedChannel := continuityCanonicalTask(sub)
	if sub == "registry-selfcheck" {
		jsonOut := len(args) > 0 && args[0] == "--json"
		return runContinuityRegistrySelfcheck(stdout, stderr, jsonOut)
	}
	if sub == "org-selfcheck" {
		jsonOut := len(args) > 0 && args[0] == "--json"
		return runContinuityOrgSelfcheck(stdout, stderr, jsonOut)
	}
	if sub == "sync-plan" {
		return runContinuitySyncPlan(stdout, stderr, args)
	}
	if sub == "sync-apply" {
		return runContinuitySyncApply(stdout, stderr, args)
	}
	fs := flag.NewFlagSet("fak profile continuity "+sub, flag.ContinueOnError)
	fs.SetOutput(stderr)
	home := fs.String("home", "", "isolated fak home (required)")
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON")
	commit := fs.Bool("commit", false, "perform mutation (mutation previews by default)")
	var selectors stringListFlag
	fs.Var(&selectors, "select", "object selector kind or kind:name (repeatable)")
	pkg := fs.String("package", "", "portable package path")
	out := fs.String("out", "", "output package path")
	receipt := fs.String("receipt", "", "switch receipt ID")
	interrupt := fs.Int("interrupt-after", 0, "test-only interruption after N staged objects")
	channel := fs.String("channel", "", "egress channel: public, organization, private, machine-local")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *channel == "" && impliedChannel != "" {
		*channel = impliedChannel
	}
	if sub == "selfcheck" {
		return runContinuitySelfcheck(stdout, stderr, *jsonOut)
	}
	if *home == "" {
		fmt.Fprintln(stderr, "--home is required (use an isolated directory)")
		return 2
	}
	s := portability.New(*home)
	var v any
	var err error
	switch sub {
	case "preview":
		if *channel == "" {
			v, err = s.Discover(selectors)
		} else {
			v, err = continuityEgressPreview(s, selectors, portability.Channel(*channel))
		}
	case "export":
		if *out == "" {
			*out = filepath.Join(*home, "exports", "continuity.fakpkg.json")
		}
		var p portability.Package
		var r portability.Receipt
		if *channel == "" {
			p, r, err = s.Export(*out, selectors, *commit)
			v = map[string]any{"package": p, "receipt": r, "path": *out}
		} else {
			var previews []portability.EgressPreview
			p, r, previews, err = s.ExportEgress(*out, selectors, portability.Channel(*channel), *commit)
			v = map[string]any{"package": p, "receipt": r, "path": *out, "egress_previews": previews}
		}
	case "apply":
		if *pkg == "" {
			fmt.Fprintln(stderr, "--package is required")
			return 2
		}
		v, err = s.Apply(*pkg, *commit, *interrupt)
	case "switch":
		id := strings.TrimSpace(*pkg)
		if id == "" {
			fmt.Fprintln(stderr, "--package must name the applied package ID")
			return 2
		}
		v, err = s.Switch(id, *commit)
	case "receipts":
		v, err = continuityReceipts(*home)
	case "status":
		id, e := s.Active()
		err = e
		if e == nil {
			var behavior map[string]string
			if id != "" {
				behavior, err = s.Readback()
			}
			v = map[string]any{"active": id, "behavior": behavior}
		}
	case "rollback":
		if *receipt == "" {
			fmt.Fprintln(stderr, "--receipt is required")
			return 2
		}
		v, err = s.Rollback(*receipt, *commit)
	default:
		fmt.Fprintf(stderr, "unknown continuity task %q\n", sub)
		continuityHelp(stderr)
		return 2
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak profile continuity %s: %v\n", sub, err)
		return 1
	}
	return emitContinuity(stdout, v, *jsonOut)
}

type stringListFlag []string

func (s *stringListFlag) String() string     { return strings.Join(*s, ",") }
func (s *stringListFlag) Set(v string) error { *s = append(*s, v); return nil }
func emitContinuity(w io.Writer, v any, j bool) int {
	b, _ := json.MarshalIndent(v, "", "  ")
	if j {
		fmt.Fprintln(w, string(b))
		return 0
	}
	switch x := v.(type) {
	case portability.Preview:
		fmt.Fprintf(w, "Preview: %d portable object(s), %d rejected\n", len(x.Objects), len(x.Rejected))
		for _, o := range x.Objects {
			state := "active-compatible"
			if !o.Active {
				state = "inspect-only"
			}
			fmt.Fprintf(w, "  %s  %s\n", o.ID, state)
		}
		for _, r := range x.Rejected {
			fmt.Fprintf(w, "  REJECT %s\n", r)
		}
	default:
		fmt.Fprintln(w, string(b))
	}
	return 0
}
func continuityHelp(w io.Writer) {
	fmt.Fprint(w, `fak profile continuity — move a safe managed context between homes

Backup, restore, switch, share, and publish managed work

Start with one task (mutations preview unless --commit is present):
  backup    create a private Package from the current Context
  restore   preview/apply a Package without activating it
  switch    activate an applied Package as the current Context
  share     preview an organization Channel export
  publish   preview a public Channel export

Inspect and recover:
  status    what Context is active and why
  preview   Objects included/rejected; use --json for a scriptable diff
  explain   --reason CODE gives meaning and at most two Next actions
  receipts  Transaction evidence is retained under HOME/receipts
  recover   alias for rollback; the prior Context stays available

Mental model:
`)
	for _, line := range continuityTaskLines(80) {
		fmt.Fprintf(w, "  %s\n", line)
	}
	fmt.Fprint(w, `
Reason codes: READY, CONFLICT, INCOMPATIBLE, EGRESS_DENIED, INTERRUPTED, ROLLED_BACK.

Foundation/expert tasks:
  export    Package creation with repeatable --select kind[:name]
  apply     inactive Package transaction
  sync-plan / sync-apply  explicit three-way policy and precedence
  rollback  receipt-selected reversal
  selfcheck / ux-selfcheck / registry-selfcheck  captured witnesses

Expert controls remain auditable: --json, repeatable --select, --channel, --package,
--receipt, --commit, and sync policy/precedence flags. No mutation occurs by default.
`)
}
func continuityEgressPreview(s portability.Store, selectors []string, channel portability.Channel) (any, error) {
	preview, err := s.DiscoverForEgress(selectors)
	if err != nil {
		return nil, err
	}
	previews := make([]portability.EgressPreview, 0, len(preview.Objects))
	for _, object := range preview.Objects {
		plan, err := portability.PreviewEgress(channel, object.Payload)
		if err != nil {
			return nil, err
		}
		previews = append(previews, plan)
	}
	return map[string]any{"channel": channel, "previews": previews, "source_rejected_count": len(preview.Rejected)}, nil
}

func runContinuitySyncPlan(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("continuity sync-plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	base := fs.String("base", "", "common-base package")
	local := fs.String("local", "", "local package")
	remote := fs.String("remote", "", "remote package")
	out := fs.String("out", "", "replayable plan path")
	channel := fs.String("channel", string(portability.ChannelPrivate), "egress channel")
	jsonOut := fs.Bool("json", false, "JSON output")
	if err := fs.Parse(args); err != nil || *base == "" || *local == "" || *remote == "" {
		fmt.Fprintln(stderr, "usage: fak continuity sync-plan --base P --local P --remote P [--out PLAN] [--channel private] [--json]")
		return 2
	}
	bp, err := portability.ReadPackage(*base)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	lp, err := portability.ReadPackage(*local)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	rp, err := portability.ReadPackage(*remote)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	plan, err := portability.PreviewMerge(&bp, lp, rp, portability.Channel(*channel))
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *out != "" {
		if err := portability.WriteMergePlan(*out, plan); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	if *jsonOut {
		emitContinuity(stdout, plan, true)
		if len(plan.Conflicts) > 0 {
			return 3
		}
		return 0
	}
	fmt.Fprintf(stdout, "SYNC PLAN %s: %d step(s), %d conflict(s), channel=%s\n", plan.ID, len(plan.Steps), len(plan.Conflicts), plan.Channel)
	for _, s := range plan.Steps {
		fmt.Fprintf(stdout, "  %-18s %-9s %s", s.ObjectID, s.Action, s.Source)
		if len(s.Paths) > 0 {
			fmt.Fprintf(stdout, " (%s)", strings.Join(s.Paths, ", "))
		}
		fmt.Fprintln(stdout)
	}
	for _, c := range plan.Conflicts {
		fmt.Fprintf(stdout, "  CONFLICT %-20s %s %s: %s\n", c.Kind, c.ObjectID, c.Path, c.Explanation)
	}
	if len(plan.Conflicts) > 0 {
		fmt.Fprintln(stdout, "BLOCKED: resolve typed conflicts; no bytes were written and no last-writer-wins was used")
		return 3
	}
	fmt.Fprintln(stdout, "READY: replay with fak continuity sync-apply --plan PLAN --out PACKAGE --home HOME --commit")
	return 0
}

func runContinuitySyncApply(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("continuity sync-apply", flag.ContinueOnError)
	fs.SetOutput(stderr)
	home := fs.String("home", "", "continuity home")
	planPath := fs.String("plan", "", "merge plan")
	out := fs.String("out", "", "merged package export")
	commit := fs.Bool("commit", false, "atomically commit")
	jsonOut := fs.Bool("json", false, "JSON output")
	interrupt := fs.Int("interrupt-after", 0, "test-only interruption before commit")
	if err := fs.Parse(args); err != nil || *home == "" || *planPath == "" || *out == "" {
		fmt.Fprintln(stderr, "usage: fak continuity sync-apply --home DIR --plan PLAN --out PACKAGE [--commit] [--json]")
		return 2
	}
	p, err := portability.ReadMergePlan(*planPath)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	r, err := portability.New(*home).CommitMerge(p, *out, *commit, *interrupt)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *jsonOut {
		return emitContinuity(stdout, r, true)
	}
	fmt.Fprintf(stdout, "%s %s receipt=%s package=%s\n", r.Operation, r.Status, r.ID, r.PackageID)
	return 0
}
