package main

import (
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

func readAuditRows(stderr io.Writer, action, path string) ([]journal.Row, bool) {
	rows, err := journal.ReadAllSegments(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak audit %s: %v\n", action, err)
		return nil, false
	}
	return journal.WithoutCutAnchors(rows), true
}
