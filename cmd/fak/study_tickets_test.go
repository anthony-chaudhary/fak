package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studytickets"
)

func TestRunStudyTicketsBuild(t *testing.T) {
	var stdout, stderr bytes.Buffer
	var got studytickets.BuildOptions
	writes := 0
	ops := studyTicketsOperations{
		build: func(opts studytickets.BuildOptions) (studytickets.Ledger, studytickets.Report, error) {
			got = opts
			return studytickets.Ledger{}, studytickets.Report{}, nil
		},
		marshalLedger: func(studytickets.Ledger) ([]byte, error) { return []byte("ledger"), nil },
		marshalReport: func(studytickets.Report) ([]byte, error) { return []byte("report"), nil },
		write:         func(string, []byte) error { writes++; return nil },
	}
	args := []string{"build", "--priority", "p", "--join", "j", "--forge", "f", "--adjacency", "a", "--classification", "c", "--ledger", "l", "--report", "r"}
	if code := runStudyTicketsWithOperations(&stdout, &stderr, args, ops); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if got.PriorityPath != "p" || got.ClassificationPath != "c" || writes != 2 {
		t.Fatalf("opts=%+v writes=%d", got, writes)
	}
}

func TestRunStudyTicketsValidateError(t *testing.T) {
	var stdout, stderr bytes.Buffer
	ops := studyTicketsOperations{validate: func(studytickets.ValidateOptions) error { return errors.New("drift") }}
	args := []string{"validate", "--priority", "p", "--join", "j", "--forge", "f", "--adjacency", "a", "--classification", "c", "--ledger", "l", "--report", "r"}
	if code := runStudyTicketsWithOperations(&stdout, &stderr, args, ops); code != 1 || !strings.Contains(stderr.String(), "drift") {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
}

func TestRunStudyTicketsRejectsMissingInput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runStudyTickets(&stdout, &stderr, []string{"build"}); code != 2 || !strings.Contains(stderr.String(), "--classification") {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
}
