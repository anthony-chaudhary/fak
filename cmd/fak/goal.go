package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/goalregistry"
)

func cmdGoal(args []string) { os.Exit(runGoal(os.Stdout, os.Stderr, args)) }

func runGoal(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak goal create|show|list|update|bind|unbind ...")
		return 2
	}
	fs := flag.NewFlagSet("goal "+args[0], flag.ContinueOnError)
	fs.SetOutput(stderr)
	registry := fs.String("registry", goalregistry.DefaultPath(), "canonical goal registry JSON path")
	id := fs.String("id", "", "opaque fak goal ID")
	title := fs.String("title", "", "user-safe goal title")
	summary := fs.String("summary", "", "scrubbed goal summary (never a raw prompt)")
	lifecycle := fs.String("lifecycle", string(goalregistry.Active), "active|achieved|abandoned|superseded|blocked|paused")
	namespace := fs.String("namespace", "", "typed binding namespace, for example fak:trajctl or github:issue")
	externalID := fs.String("external-id", "", "opaque external binding ID")
	revision := fs.String("revision", "", "optional external binding revision")
	actor := fs.String("actor", "", "provenance actor")
	authority := fs.String("authority", "", "provenance authority")
	witness := fs.String("witness", "", "optional provenance witness reference")
	parent := fs.String("parent-goal", "", "goal this intent decomposes from")
	derived := fs.String("derived-from", "", "goal this intent derives from")
	supersedes := fs.String("supersedes", "", "goal this intent supersedes")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	s := goalregistry.Store{Path: *registry}
	p := goalregistry.Provenance{Actor: *actor, Authority: *authority, Witness: *witness}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	fail := func(err error) int { fmt.Fprintf(stderr, "fak goal: %v\n", err); return 1 }
	requireID := func() error {
		if strings.TrimSpace(*id) == "" {
			return fmt.Errorf("--id is required")
		}
		return nil
	}
	switch args[0] {
	case "create":
		var rels []goalregistry.Relation
		for _, x := range []struct{ kind, id string }{{"parent_goal", *parent}, {"derived_from", *derived}, {"supersedes", *supersedes}} {
			if strings.TrimSpace(x.id) != "" {
				rels = append(rels, goalregistry.Relation{Kind: x.kind, GoalID: x.id})
			}
		}
		g, err := s.Create(*title, *summary, p, rels)
		if err != nil {
			return fail(err)
		}
		_ = enc.Encode(g)
	case "show":
		if err := requireID(); err != nil {
			return fail(err)
		}
		g, bindings, err := s.Show(*id)
		if err != nil {
			return fail(err)
		}
		_ = enc.Encode(struct {
			Goal     goalregistry.Goal      `json:"goal"`
			Bindings []goalregistry.Binding `json:"bindings"`
		}{g, bindings})
	case "list":
		goals, err := s.List()
		if err != nil {
			return fail(err)
		}
		_ = enc.Encode(goals)
	case "update":
		if err := requireID(); err != nil {
			return fail(err)
		}
		g, err := s.Update(*id, *title, *summary, goalregistry.Lifecycle(*lifecycle))
		if err != nil {
			return fail(err)
		}
		_ = enc.Encode(g)
	case "bind":
		if err := requireID(); err != nil {
			return fail(err)
		}
		b, err := s.Bind(*id, *namespace, *externalID, *revision, p)
		if err != nil {
			return fail(err)
		}
		_ = enc.Encode(b)
	case "unbind":
		if err := requireID(); err != nil {
			return fail(err)
		}
		if err := s.Unbind(*id, *namespace, *externalID, *revision); err != nil {
			return fail(err)
		}
		_ = enc.Encode(map[string]any{"ok": true, "goal_id": *id})
	default:
		fmt.Fprintf(stderr, "fak goal: unknown subcommand %q\n", args[0])
		return 2
	}
	return 0
}
