package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/issuehygiene"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func runScoreboardIssuePackets(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak scoreboard issue-packets", flag.ContinueOnError)
	fs.SetOutput(stderr)
	from := fs.String("from", "", "read issue-hygiene scorecard JSON from this file (- for stdin)")
	packetSize := fs.Int("packet-size", issuehygiene.DefaultPacketSize, "maximum issue mutations per worker packet")
	unsafeOversized := fs.Bool("unsafe-oversized", false, "allow packet sizes above 1; requires --price-ref and --review-ref")
	priceRef := fs.String("price-ref", "", "oversized plan: DOS/dispatch price receipt reference")
	reviewRef := fs.String("review-ref", "", "oversized plan: explicit operator review reference")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak scoreboard issue-packets: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *from == "" {
		fmt.Fprintln(stderr, "fak scoreboard issue-packets: --from is required")
		return 2
	}
	var raw []byte
	var err error
	if *from == "-" {
		raw, err = io.ReadAll(os.Stdin)
	} else {
		raw, err = os.ReadFile(*from)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak scoreboard issue-packets: read --from: %v\n", err)
		return 1
	}
	var payload scorecard.Payload
	if err := json.Unmarshal(raw, &payload); err != nil {
		fmt.Fprintf(stderr, "fak scoreboard issue-packets: parse scorecard JSON: %v\n", err)
		return 1
	}
	plan, err := issuehygiene.PlanPackets(payload, issuehygiene.PacketOptions{
		PacketSize: *packetSize, UnsafeOversized: *unsafeOversized,
		PriceRef: *priceRef, ReviewRef: *reviewRef,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak scoreboard issue-packets: %v\n", err)
		return 1
	}
	if err := writeIndentedJSONNoEscape(stdout, plan); err != nil {
		fmt.Fprintf(stderr, "fak scoreboard issue-packets: encode JSON: %v\n", err)
		return 1
	}
	return 0
}
