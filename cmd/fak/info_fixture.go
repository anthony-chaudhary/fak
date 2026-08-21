package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// info_fixture.go — the OFFLINE render path for `fak info`. It is the deterministic twin of
// `fak console guard --journal`: instead of polling a live gateway's /debug/vars, it reads a
// recorded snapshot (the exact JSON shape `fak info --json` emits) from a file and draws one
// static overlay frame, then exits. That is what lets the tabbed overlay go into the README and
// be regenerated repeatably — the live overlay renders only from a gateway, so without this
// there was no way to capture a fixed, payload-free frame. See visuals/info-overlay-capture.md.
//
// It reuses the SAME renderer the live loop uses (renderGuardInfoInteractiveBlock), so the
// captured frame is byte-identical to what a live pane of the same width/height would draw for
// the same vars — the fixture is the only thing that changes.

// infoViewByName resolves a --tab word to the infoView it selects, matching the tab labels
// infoViewName prints (overview/agents/fleet/accounts/cache/safety/gateway). It returns ok=false for an
// unknown name so the caller can report the valid set rather than silently drawing overview.
func infoViewByName(name string) (infoView, bool) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "overview", "":
		return viewOverview, true
	case "agents":
		return viewAgents, true
	case "fleet":
		return viewFleet, true
	case "accounts", "endpoints":
		// "accounts" is the tab label; "endpoints" is the internal view name — accept both so a
		// caller reading either the UI or the code lands on the same frame.
		return viewEndpoints, true
	case "cache":
		return viewCache, true
	case "safety":
		return viewSafety, true
	case "gateway", "startup":
		return viewStartup, true
	default:
		return viewOverview, false
	}
}

// infoTabNames is the valid --tab set in tab order, for the error message.
func infoTabNames() []string {
	names := make([]string, 0, infoViewCount())
	for i := 0; i < infoViewCount(); i++ {
		names = append(names, infoViewName(infoView(i)))
	}
	return names
}

// runInfoFixtureFrame renders ONE static overlay frame for the given tab from a recorded
// /debug/vars snapshot file, offline. path is the JSON fixture (the `fak info --json` shape);
// tab selects the view; frame must be true (the only mode today); width/height fix the pane
// geometry (0 = roomy). It writes the frame + a trailing newline to stdout and returns 0, or a
// house error to stderr and non-zero.
func runInfoFixtureFrame(stdout, stderr io.Writer, path, tab string, frame bool, width, height int) int {
	if !frame {
		fmt.Fprintln(stderr, "fak info: --from-fixture supports only --frame (one static frame) today; drop --frame=false")
		return 2
	}
	view, ok := infoViewByName(tab)
	if !ok {
		fmt.Fprintf(stderr, "fak info: unknown --tab %q; valid: %s\n", tab, strings.Join(infoTabNames(), ", "))
		return 2
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak info: reading fixture %s: %v\n", path, err)
		return 1
	}
	var v guardInfoVars
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	// Do NOT DisallowUnknownFields: a real /debug/vars snapshot carries many blocks the overlay
	// does not surface (the same tolerance fetchGuardInfoVars relies on), so a snapshot captured
	// with `fak info --json` OR taken raw from the gateway both decode.
	if err := dec.Decode(&v); err != nil {
		fmt.Fprintf(stderr, "fak info: parsing fixture %s as a /debug/vars snapshot: %v\n", path, err)
		return 1
	}
	if v.Observation == nil {
		present := map[string]json.RawMessage{}
		if err := json.Unmarshal(raw, &present); err == nil {
			applyLegacyGuardInfoObservation(&v, present, "FIXTURE_LEGACY_DEBUG_VARS")
		}
	}

	// Seed the trend with a few identical samples so the sparklines render a flat-but-present line
	// rather than an empty gutter — a single static snapshot has no history, and a docs frame that
	// showed blank sparklines would misrepresent the live overlay. A live capture with real motion
	// can instead pass a multi-sample fixture once the format grows a history array; today one
	// snapshot is the contract, so a flat seed is the honest rendering.
	tr := newGuardInfoTrend(guardInfoTrendCap)
	for i := 0; i < guardInfoTrendSeedSamples; i++ {
		tr.push(v)
	}

	block := renderGuardInfoInteractiveBlock(infoViewState{active: view}, v, tr, width, height)
	fmt.Fprintln(stdout, block)
	return 0
}

// guardInfoTrendSeedSamples is how many copies of the single fixture snapshot the offline frame
// pushes into the trend so the sparklines draw. A handful is enough to fill the sparkline gutter;
// the exact count does not affect a flat line's shape.
const guardInfoTrendSeedSamples = 6
