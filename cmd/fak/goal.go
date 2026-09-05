package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/goalregistry"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

func cmdGoal(args []string) { os.Exit(runGoal(os.Stdout, os.Stderr, args)) }

func runGoal(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak goal create|show|list|update|transition|reopen|bind|resolve|topology|backfill-root|unbind|sync ...")
		return 2
	}
	if args[0] == "sync" {
		return runGoalSync(stdout, stderr, args[1:])
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
	sessionRegistry := fs.String("session-registry", sessionregistry.DefaultPath(), "session registry or journal path")
	rootRegistrationID := fs.String("root-registration-id", "", "execution root to bind")
	apply := fs.Bool("apply", false, "append the witnessed binding (default is dry-run)")
	evidenceClass := fs.String("evidence-class", "", "harness_assertion|agent_assertion|operator_declaration|independent_witness")
	evidenceAuthor := fs.String("evidence-author", "", "outcome evidence author")
	evidenceRef := fs.String("evidence-ref", "", "durable outcome evidence reference")
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
		return writeGoalMutation(enc, fail, func() (goalregistry.Goal, error) {
			return s.Create(*title, *summary, p, rels)
		})
	case "show":
		if err := requireID(); err != nil {
			return fail(err)
		}
		g, bindings, err := s.Show(*id)
		if err != nil {
			return fail(err)
		}
		evidence, err := s.OutcomeEvidence(*id)
		if err != nil {
			return fail(err)
		}
		_ = enc.Encode(struct {
			Goal            goalregistry.Goal              `json:"goal"`
			Bindings        []goalregistry.Binding         `json:"bindings"`
			OutcomeEvidence []goalregistry.OutcomeEvidence `json:"outcome_evidence"`
		}{g, bindings, evidence})
	case "list":
		goals, err := s.List()
		if err != nil {
			return fail(err)
		}
		_ = enc.Encode(goals)
	case "update":
		return writeGoalMutationRequiringID(enc, fail, requireID, func() (goalregistry.Goal, error) {
			return s.Update(*id, *title, *summary, goalregistry.Lifecycle(*lifecycle))
		})
	case "transition":
		return writeGoalMutationRequiringID(enc, fail, requireID, func() (goalregistry.Goal, error) {
			return s.Transition(*id, goalregistry.Lifecycle(*lifecycle), goalregistry.OutcomeEvidence{Class: goalregistry.EvidenceClass(*evidenceClass), Author: *evidenceAuthor, Reference: *evidenceRef})
		})
	case "reopen":
		return writeGoalMutationRequiringID(enc, fail, requireID, func() (goalregistry.Goal, error) {
			return s.Reopen(*id, *evidenceAuthor, *evidenceRef)
		})
	case "bind":
		if err := requireID(); err != nil {
			return fail(err)
		}
		b, err := s.Bind(*id, *namespace, *externalID, *revision, p)
		if err != nil {
			return fail(err)
		}
		_ = enc.Encode(b)
	case "resolve":
		g, b, err := s.Resolve(*namespace, *externalID, *revision)
		if err != nil {
			return fail(err)
		}
		_ = enc.Encode(struct {
			Schema  string               `json:"schema"`
			GoalID  string               `json:"goal_id"`
			Goal    goalregistry.Goal    `json:"goal"`
			Binding goalregistry.Binding `json:"binding"`
			Env     map[string]string    `json:"env"`
		}{"fak-goal-resolution/1", g.GoalID, g, b, map[string]string{"FAK_GOAL_ID": g.GoalID}})
	case "topology":
		if err := requireID(); err != nil {
			return fail(err)
		}
		if _, err := s.RequireGoal(*id); err != nil {
			return fail(err)
		}
		groups, err := (sessionregistry.Store{Path: *sessionRegistry}).GoalTopology(*id)
		if err != nil {
			return fail(err)
		}
		type rootView struct {
			RootRegistrationID string                   `json:"root_registration_id"`
			Registrations      []sessionregistry.Record `json:"registrations"`
		}
		roots := make([]rootView, 0, len(groups))
		for _, group := range groups {
			if len(group) > 0 {
				roots = append(roots, rootView{group[0].RootRegistrationID, group})
			}
		}
		_ = enc.Encode(struct {
			Schema         string     `json:"schema"`
			GoalID         string     `json:"goal_id"`
			ExecutionRoots []rootView `json:"execution_roots"`
		}{"fak-goal-topology/1", *id, roots})
	case "backfill-root":
		if err := requireID(); err != nil {
			return fail(err)
		}
		if strings.TrimSpace(*witness) == "" {
			return fail(fmt.Errorf("--witness is required; canonical identity is never inferred"))
		}
		if _, err := s.RequireGoal(*id); err != nil {
			return fail(err)
		}
		rootStore := sessionregistry.Store{Path: *sessionRegistry}
		rows, err := rootStore.BindGoalRoot(*rootRegistrationID, *id, false)
		if err != nil {
			return fail(err)
		}
		if *apply {
			if _, err := s.Bind(*id, "fak:session-root", *rootRegistrationID, "", p); err != nil {
				return fail(err)
			}
			rows, err = rootStore.BindGoalRoot(*rootRegistrationID, *id, true)
			if err != nil {
				return fail(err)
			}
		}
		ids := make([]string, 0, len(rows))
		for _, row := range rows {
			ids = append(ids, row.RegistrationID)
		}
		_ = enc.Encode(struct {
			Schema             string   `json:"schema"`
			Applied            bool     `json:"applied"`
			GoalID             string   `json:"goal_id"`
			RootRegistrationID string   `json:"root_registration_id"`
			Witness            string   `json:"witness"`
			RegistrationIDs    []string `json:"registration_ids"`
		}{"fak-goal-root-backfill/1", *apply, *id, *rootRegistrationID, strings.TrimSpace(*witness), ids})
	case "unbind":
		if err := requireID(); err != nil {
			return fail(err)
		}
		if err := s.Unbind(*id, *namespace, *externalID, *revision); err != nil {
			return fail(err)
		}
		_ = enc.Encode(map[string]any{"ok": true, "goal_id": *id})
	case "sync":
		return runGoalSync(stdout, stderr, args[1:])
	default:
		fmt.Fprintf(stderr, "fak goal: unknown subcommand %q\n", args[0])
		return 2
	}
	return 0
}

func writeGoalMutation(enc *json.Encoder, fail func(error) int, mutate func() (goalregistry.Goal, error)) int {
	goal, err := mutate()
	if err != nil {
		return fail(err)
	}
	_ = enc.Encode(goal)
	return 0
}

func writeGoalMutationRequiringID(enc *json.Encoder, fail func(error) int, requireID func() error, mutate func() (goalregistry.Goal, error)) int {
	if err := requireID(); err != nil {
		return fail(err)
	}
	return writeGoalMutation(enc, fail, mutate)
}
