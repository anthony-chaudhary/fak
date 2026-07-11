package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

func runResumeCap(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak resume cap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sessionsPath := fs.String("sessions", "", "fleet sessions.json registry")
	floor := fs.Int("floor", 4, "minimum launches per tick")
	ceiling := fs.Int("ceiling", 64, "maximum launches per tick")
	seatCap := fs.Int("seat-cap", 6, "safe concurrent sessions per healthy seat")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *sessionsPath == "" {
		fmt.Fprintln(stderr, "fak resume cap: --sessions is required")
		return 2
	}
	f, err := os.Open(*sessionsPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak resume cap: %v\n", err)
		return 1
	}
	defer f.Close()
	var registry struct {
		Accounts []resume.HeadroomSeat `json:"accounts"`
	}
	if err := json.NewDecoder(f).Decode(&registry); err != nil {
		fmt.Fprintf(stderr, "fak resume cap: decode: %v\n", err)
		return 1
	}
	return writeJSON(stdout, resume.DeriveWatchdogCap(registry.Accounts, *floor, *ceiling, *seatCap))
}
