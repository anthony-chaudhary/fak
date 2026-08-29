package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/learningobservation"
)

func cmdLearningObservation(argv []string) {
	os.Exit(runLearningObservation(os.Stdout, os.Stderr, argv))
}

func runLearningObservation(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak learning-observation <add|link|trace> [flags]")
		return 2
	}
	sub := argv[0]
	fs := flag.NewFlagSet("learning-observation "+sub, flag.ContinueOnError)
	fs.SetOutput(stderr)
	storePath := fs.String("store", defaultLearningObservationStorePath(), "durable JSON store path")
	if sub == "add" {
		kind := fs.String("kind", "", "record kind: observation, candidate, witness, or verdict")
		source := fs.String("source", "", "source provenance")
		content := fs.String("content", "", "record content")
		outcome := fs.String("outcome", "", "verdict outcome: kept or rejected")
		store, code := parseLearningObservationStore(fs, argv[1:], storePath, stderr)
		if code != 0 {
			return code
		}
		record, created, err := store.Add(learningobservation.Kind(*kind), *source, *content, learningobservation.Outcome(*outcome))
		if err != nil {
			return learningObservationError(stderr, err)
		}
		return finishLearningObservationMutation(stdout, stderr, store, *storePath, created, map[string]any{"record": record, "created": created})
	}
	if sub == "link" {
		from := fs.String("from", "", "source record ID")
		relation := fs.String("relation", "", "closed-enum relation")
		to := fs.String("to", "", "target record ID")
		store, code := parseLearningObservationStore(fs, argv[1:], storePath, stderr)
		if code != 0 {
			return code
		}
		created, err := store.Link(*from, learningobservation.Relation(*relation), *to)
		if err != nil {
			return learningObservationError(stderr, err)
		}
		return finishLearningObservationMutation(stdout, stderr, store, *storePath, created, map[string]any{"edge": learningobservation.Edge{From: *from, Relation: learningobservation.Relation(*relation), To: *to}, "created": created})
	}
	if sub == "trace" {
		candidate := fs.String("candidate", "", "candidate record ID")
		store, code := parseLearningObservationStore(fs, argv[1:], storePath, stderr)
		if code != 0 {
			return code
		}
		records, edges, err := store.Trace(*candidate)
		if err != nil {
			return learningObservationError(stderr, err)
		}
		return writeLearningObservationJSON(stdout, map[string]any{"candidate": *candidate, "records": records, "edges": edges})
	}
	fmt.Fprintf(stderr, "learning-observation: unknown subcommand %q\n", sub)
	return 2
}

func parseLearningObservationStore(fs *flag.FlagSet, args []string, storePath *string, stderr io.Writer) (*learningobservation.Store, int) {
	if err := fs.Parse(args); err != nil {
		return nil, 2
	}
	store, err := learningobservation.Load(*storePath)
	if err != nil {
		return nil, learningObservationError(stderr, err)
	}
	return store, 0
}

func finishLearningObservationMutation(stdout, stderr io.Writer, store *learningobservation.Store, storePath string, created bool, value any) int {
	if created {
		if err := store.Save(storePath); err != nil {
			return learningObservationError(stderr, err)
		}
	}
	return writeLearningObservationJSON(stdout, value)
}

func defaultLearningObservationStorePath() string {
	if value := strings.TrimSpace(os.Getenv("FAK_LEARNING_OBSERVATION_STORE")); value != "" {
		return value
	}
	if dir, err := os.UserConfigDir(); err == nil && strings.TrimSpace(dir) != "" {
		return filepath.Join(dir, "fak", "learning-observations.json")
	}
	return filepath.Join(".fak", "learning-observations.json")
}

func learningObservationError(stderr io.Writer, err error) int {
	fmt.Fprintf(stderr, "learning-observation: %v\n", err)
	return 1
}

func writeLearningObservationJSON(stdout io.Writer, value any) int {
	enc := json.NewEncoder(stdout)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(value); err != nil {
		return 1
	}
	return 0
}
