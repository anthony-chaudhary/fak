package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/deliverystages"
	"github.com/anthony-chaudhary/fak/internal/workdelivery"
)

func cmdWorkDelivery(args []string) { os.Exit(runWorkDelivery(os.Stdout, os.Stderr, args)) }

func runWorkDelivery(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak work-delivery status|inspect|transition|diagnose|stages [flags]")
		return 2
	}
	switch args[0] {
	case "status":
		return runWorkDeliveryStatus(stdout, stderr, args[1:])
	case "inspect":
		return runWorkDeliveryInspect(stdout, stderr, args[1:])
	case "transition":
		return runWorkDeliveryTransition(stdout, stderr, args[1:])
	case "diagnose":
		return runWorkDeliveryDiagnose(stdout, stderr, args[1:])
	case "stages":
		return runWorkDeliveryStages(stdout, stderr, args[1:])
	default:
		fmt.Fprintf(stderr, "fak work-delivery: unknown mode %q\n", args[0])
		return 2
	}
}

func runWorkDeliveryInspect(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("work-delivery inspect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "work-unit JSON")
	asJSON := fs.Bool("json", false, "JSON output")
	if fs.Parse(args) != nil || *file == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "inspect requires --file")
		return 2
	}
	var unit workdelivery.WorkUnit
	if err := readWorkDeliveryJSON(*file, &unit); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if err := unit.Validate(); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	next := "no final-mile receipt is missing"
	if displayActivation(unit.Axes.Activation) == string(workdelivery.ActivationInactive) {
		next = "capture an activation receipt"
	} else if displayAcceptance(unit.Axes.Acceptance) == string(workdelivery.AcceptanceUnaccepted) {
		next = "capture an operator acceptance receipt"
	}
	if *asJSON {
		return writeWorkDeliveryJSON(stdout, struct {
			Unit workdelivery.WorkUnit `json:"unit"`
			Next string                `json:"next_action"`
		}{Unit: unit, Next: next})
	}
	fmt.Fprintf(stdout, "unit %s\n  release: %s (observed)\n  activation: %s (observed)\n  operator acceptance: %s (observed)\n  next: %s\n", unit.ID, unit.Axes.Release, displayActivation(unit.Axes.Activation), displayAcceptance(unit.Axes.Acceptance), next)
	return 0
}

func runWorkDeliveryStatus(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("work-delivery status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "work-unit JSON")
	asJSON := fs.Bool("json", false, "JSON output")
	if fs.Parse(args) != nil || *file == "" {
		if *file == "" {
			fmt.Fprintln(stderr, "status requires --file")
		}
		return 2
	}
	var unit workdelivery.WorkUnit
	if err := readWorkDeliveryJSON(*file, &unit); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if err := unit.Validate(); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *asJSON {
		return writeWorkDeliveryJSON(stdout, unit)
	}
	fmt.Fprintf(stdout, "unit %s\n  authoring: %s (declared)\n  compile admission: %s (declared)\n  verification: %s (observed)\n  integration: %s (observed)\n  release: %s (observed)\n", unit.ID, unit.Axes.Authoring, unit.Axes.Admission, unit.Axes.Verification, unit.Axes.Integration, unit.Axes.Release)
	return 0
}

func runWorkDeliveryTransition(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("work-delivery transition", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "work-unit JSON")
	axis := fs.String("axis", "", "axis")
	to := fs.String("to", "", "target state")
	gate := fs.String("gate", "manual", "gate")
	owner := fs.String("owner", "", "owner")
	out := fs.String("out", "", "write updated unit")
	asJSON := fs.Bool("json", false, "JSON output")
	if fs.Parse(args) != nil || *file == "" || *axis == "" || *to == "" {
		fmt.Fprintln(stderr, "transition requires --file --axis --to")
		return 2
	}
	var unit workdelivery.WorkUnit
	if err := readWorkDeliveryJSON(*file, &unit); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	from, err := axisState(unit.Axes, workdelivery.Axis(*axis))
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	receipt := workdelivery.Receipt{Schema: workdelivery.Schema, UnitID: unit.ID, Transition: workdelivery.Transition{Axis: workdelivery.Axis(*axis), From: from, To: *to}, Gate: *gate, Owner: *owner, ObservedAt: time.Now().UTC()}
	updated, err := workdelivery.Apply(unit, receipt)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *out != "" {
		data, _ := json.MarshalIndent(updated, "", "  ")
		if err := os.WriteFile(*out, append(data, '\n'), 0o644); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	}
	if *asJSON {
		return writeWorkDeliveryJSON(stdout, struct {
			Receipt workdelivery.Receipt  `json:"receipt"`
			Unit    workdelivery.WorkUnit `json:"unit"`
		}{receipt, updated})
	}
	fmt.Fprintf(stdout, "unit %s: %s %s -> %s\n  gate: %s (observed)\n  next: inspect the next prerequisite stage\n", unit.ID, *axis, from, *to, *gate)
	return 0
}

func runWorkDeliveryDiagnose(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("work-delivery diagnose", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "failure observation JSON")
	asJSON := fs.Bool("json", false, "JSON output")
	if fs.Parse(args) != nil || *file == "" {
		fmt.Fprintln(stderr, "diagnose requires --file")
		return 2
	}
	var observation workdelivery.FailureObservation
	if err := readWorkDeliveryJSON(*file, &observation); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	diagnosis, err := workdelivery.Diagnose(observation)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *asJSON {
		return writeWorkDeliveryJSON(stdout, diagnosis)
	}
	if diagnosis.Unit != nil {
		fmt.Fprintf(stdout, "unit %s blocked at %s (%s)\n", diagnosis.Unit.ID, diagnosis.Gate, diagnosis.Class)
	} else {
		fmt.Fprintf(stdout, "scope %s: %s at %s (%s)\n", diagnosis.ScopeID, diagnosis.Kind, diagnosis.Gate, diagnosis.Class)
	}
	if diagnosis.Blocker != nil {
		fmt.Fprintf(stdout, "  blocker: %s — %s\n", diagnosis.Blocker.Code, diagnosis.Blocker.Detail)
	}
	fmt.Fprintf(stdout, "  next: %s\n", diagnosis.NextAction)
	return 0
}

func runWorkDeliveryStages(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("work-delivery stages", flag.ContinueOnError)
	fs.SetOutput(stderr)
	id := fs.String("id", "", "stage id")
	local := fs.String("local", "", "local status/gate")
	asJSON := fs.Bool("json", false, "JSON output")
	if fs.Parse(args) != nil {
		return 2
	}
	registry := deliverystages.Default()
	if err := registry.Validate(); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *local != "" {
		item, ok := registry.ResolveLocal(*local)
		if !ok {
			fmt.Fprintf(stderr, "unknown local status %q; diagnose as unknown/irreducible\n", *local)
			return 1
		}
		if *asJSON {
			return writeWorkDeliveryJSON(stdout, item)
		}
		fmt.Fprintf(stdout, "%s -> stage %s, bottleneck %s\n", item.Local, item.Stage, item.Bottleneck)
		return 0
	}
	if *id != "" {
		stage, ok := registry.Stage(deliverystages.StageID(*id))
		if !ok {
			fmt.Fprintf(stderr, "unknown stage %q\n", *id)
			return 1
		}
		if *asJSON {
			return writeWorkDeliveryJSON(stdout, stage)
		}
		fmt.Fprintf(stdout, "stage %s: %s\n  owner: %s\n  gate: %s\n  prerequisites: %s\n  retry: %s\n  split: %s\n", stage.ID, stage.Name, stage.Owner, stage.Gate, joinStageIDs(stage.Prerequisites), stage.RetryCommand, strings.Join(stage.SplitDimensions, ", "))
		return 0
	}
	if *asJSON {
		return writeWorkDeliveryJSON(stdout, registry)
	}
	for _, stage := range registry.Stages {
		fmt.Fprintf(stdout, "%-22s %s (gate=%s owner=%s)\n", stage.ID, stage.Name, stage.Gate, stage.Owner)
	}
	return 0
}

func readWorkDeliveryJSON(path string, target any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}
func writeWorkDeliveryJSON(w io.Writer, value any) int {
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return 1
	}
	return 0
}
func axisState(axes workdelivery.Axes, axis workdelivery.Axis) (string, error) {
	switch axis {
	case workdelivery.AxisAuthoring:
		return string(axes.Authoring), nil
	case workdelivery.AxisAdmission:
		return string(axes.Admission), nil
	case workdelivery.AxisVerification:
		return string(axes.Verification), nil
	case workdelivery.AxisIntegration:
		return string(axes.Integration), nil
	case workdelivery.AxisRelease:
		return string(axes.Release), nil
	case workdelivery.AxisActivation:
		return displayActivation(axes.Activation), nil
	case workdelivery.AxisAcceptance:
		return displayAcceptance(axes.Acceptance), nil
	default:
		return "", fmt.Errorf("unknown axis %q", axis)
	}
}
func joinStageIDs(ids []deliverystages.StageID) string {
	values := make([]string, len(ids))
	for i, id := range ids {
		values[i] = string(id)
	}
	return strings.Join(values, ", ")
}

func displayActivation(state workdelivery.ActivationState) string {
	if state == "" {
		return string(workdelivery.ActivationInactive)
	}
	return string(state)
}
func displayAcceptance(state workdelivery.AcceptanceState) string {
	if state == "" {
		return string(workdelivery.AcceptanceUnaccepted)
	}
	return string(state)
}
