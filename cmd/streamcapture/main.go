// Command streamcapture records a REAL provider stream so a hermetic Go test
// can replay it byte-for-byte.
//
// Why record at all: a hand-authored SSE literal proves the parser, not the
// provider. The claim under test in the continuous-generation lane (#6342,
// parent #6219) is about how a live provider actually chunks a turn — above all
// whether tool-call arguments arrive incrementally or whole, because that sets
// the finest boundary at which an adapter can honestly act on a steering
// directive. That is an empirical fact about a provider, so it has to come from
// one.
//
// This is the Go recorder. It is stdlib-only and imports nothing from the fak
// tree on purpose: the packages that consume its output are under active
// concurrent development, and a recorder that cannot be broken by them is worth
// more than one that shares their types.
//
// Nothing here is hermetic — it needs network and a key. The committed captures
// are the hermetic artifact; -verify re-derives every manifest claim from the
// recorded bytes with no network at all, so a manifest that overstates what it
// captured cannot survive review.
//
// Usage:
//
//	# one capture
//	go run ./cmd/streamcapture -provider groq -model openai/gpt-oss-120b -scenario tool-destructive
//
//	# the whole witness set
//	go run ./cmd/streamcapture -all
//
//	# which models actually fragment tool arguments across deltas?
//	go run ./cmd/streamcapture -probe -provider nvidia -models a,b,c
//
//	# offline: re-derive every manifest row from the bytes on disk
//	go run ./cmd/streamcapture -verify
//
// Keys are read from the environment and are never written into a capture or a
// manifest: FAK_GROQ_API_KEY (groq), NVIDIA_API_KEY (nvidia), OPENAI_API_KEY
// with OPENAI_BASE_URL (any other OpenAI-compatible endpoint, including the
// local fak gateway).
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "streamcapture:", err)
		os.Exit(1)
	}
}

func run(argv []string, out *os.File) error {
	fs := flag.NewFlagSet("streamcapture", flag.ContinueOnError)
	fs.SetOutput(out)
	var (
		provider = fs.String("provider", "", "provider to record: "+strings.Join(providerNames(), ", "))
		model    = fs.String("model", "", "provider model id")
		scenario = fs.String("scenario", "", "scenario to record: "+strings.Join(scenarioNames(), ", "))
		dir      = fs.String("dir", defaultDir, "capture directory")
		all      = fs.Bool("all", false, "record the committed witness set")
		probe    = fs.Bool("probe", false, "record -models against the tool scenario and report which fragment tool arguments")
		models   = fs.String("models", "", "comma-separated model ids for -probe")
		verify   = fs.Bool("verify", false, "offline: re-derive every manifest row from the recorded bytes")
		keep     = fs.Bool("keep-probe-captures", false, "with -probe, keep every probed capture instead of only the manifest-worthy ones")
	)
	if err := fs.Parse(argv); err != nil {
		return err
	}

	switch {
	case *verify:
		return verifyManifest(*dir, out)
	case *probe:
		return runProbe(*dir, *provider, splitList(*models), *keep, out)
	case *all:
		return runCaptures(*dir, captureTargets, out)
	case *provider != "" && *model != "" && *scenario != "":
		return runCaptures(*dir, []target{{Provider: *provider, Model: *model, Scenario: *scenario}}, out)
	default:
		fs.Usage()
		return fmt.Errorf("give -verify, -all, -probe, or all of -provider/-model/-scenario")
	}
}

func splitList(csv string) []string {
	var out []string
	for _, part := range strings.Split(csv, ",") {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}
