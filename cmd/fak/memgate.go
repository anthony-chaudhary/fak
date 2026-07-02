package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/memgate"
)

func cmdMemgate(argv []string) { os.Exit(runMemgate(os.Stdout, os.Stderr, argv)) }

func runMemgate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("memgate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	requireGB := fs.Float64("require-gb", 0, "GB the upcoming load needs")
	waitS := fs.Int("wait", 0, "poll up to N seconds for memory to free")
	asJSON := fs.Bool("json", false, "emit only the structured snapshot")
	interval := fs.Float64("interval", 5.0, "poll interval seconds with --wait")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	deadline := time.Now().Add(time.Duration(*waitS) * time.Second)
	for {
		snap, err := memgate.CurrentSnapshot()
		if err != nil {
			fmt.Fprintf(stderr, "memgate: %v\n", err)
			return 2
		}
		if *requireGB > 0 {
			snap = memgate.Evaluate(snap, *requireGB)
		}
		ok := snap.Admit == nil || *snap.Admit
		if ok || *waitS <= 0 || time.Now().After(deadline) {
			data, err := json.MarshalIndent(snap, "", "  ")
			if err != nil {
				fmt.Fprintf(stderr, "memgate: %v\n", err)
				return 2
			}
			if *asJSON || *requireGB == 0 {
				fmt.Fprintln(stdout, string(data))
			} else {
				verdict := "REFUSE"
				if ok {
					verdict = "ADMIT"
				}
				fmt.Fprintf(stdout, "[memgate] %s: need %.2fGB, available %.2fGB (shortfall %.2fGB, wired %.2fGB %s)\n",
					verdict, *requireGB, snap.AvailableGB, snap.ShortfallGB, snap.WiredGB, map[bool]string{true: "HIGH", false: "ok"}[snap.HighWired])
				if !ok && len(snap.Holders) > 0 {
					fmt.Fprintln(stdout, "[memgate] big holders to consider stopping:")
					for i, h := range snap.Holders {
						if i >= 5 {
							break
						}
						fmt.Fprintf(stdout, "           pid %6d  %5.2fGB  %s\n", h.PID, h.RSSGB, h.Comm)
					}
				}
			}
			if ok {
				return 0
			}
			return 1
		}
		time.Sleep(time.Duration(*interval * float64(time.Second)))
	}
}
