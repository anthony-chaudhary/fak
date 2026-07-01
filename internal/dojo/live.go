package dojo

// live.go is the reader side of the live-episode corpus that `fak guard --dojo` /
// `fak serve --dojo` write under <repoRoot>/.dojo/live-episodes/. The original
// writer dropped START-MARKERS ONLY: a single JSON object carrying {mode,
// command, started, cwd, workspace} and NOTHING billable. The current reader keeps
// those legacy markers visible and unscorable, and also accepts completed-session
// rows that carry scored dojo inputs derived from observed provider usage. It
// never fabricates a Realized number off metadata.
//
// Keeping the discovery + parse + missing-witness diagnosis pure here (the dir
// path is passed in, never resolved) makes it unit-testable without a workspace,
// and lets the cmd/fak shell fold the result into the same report envelope the
// corpus path uses. When the writer side later captures the full scored episode
// (the rest of #1089/#1093, on the guard.go/serve.go lane this reader does NOT
// touch), ScorableLiveEpisodes is the seam that turns those into ScoredInputs.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// LiveEpisodesRel is the corpus directory `--dojo` writes under the repo root.
// It is the read counterpart of the path cmd/fak/guard.go builds; named here so
// the reader and the writer agree on one literal.
const LiveEpisodesRel = ".dojo/live-episodes"

// liveEpisodeFilePrefix / liveEpisodeFileExt bracket the start-marker filenames
// the writer emits (episode_YYYYMMDD_HHMMSS.jsonl), so the scan only folds files
// it actually wrote and ignores anything else dropped in the directory.
const (
	liveEpisodeFilePrefix = "episode_"
	liveEpisodeFileExt    = ".jsonl"
)

// LiveEpisodeMarker is one parsed start-marker — exactly the shape the writer
// encodes. Every field is descriptive metadata about WHEN/WHERE a `--dojo`
// session started; none of it is a billable measurement, which is precisely why
// a marker cannot be scored on its own.
type LiveEpisodeMarker struct {
	// File is the marker's basename, kept so a summary can name which files it
	// folded without leaking the absolute path.
	File string `json:"file"`
	// Mode is the writer's "mode" key ("live" for a real session marker).
	Mode string `json:"mode"`
	// Command is which wrapper wrote it ("guard" | "serve").
	Command string `json:"command"`
	// Started is the RFC3339 session-start timestamp.
	Started string `json:"started"`
	// Workspace is the repo root the session ran under.
	Workspace string `json:"workspace"`
	// Scored carries completed-session prediction/outcome pairs. It is empty on
	// legacy start-only markers.
	Scored []ScoredInput `json:"scored,omitempty"`
}

// LiveCorpus is the folded result of scanning a live-episode directory: the
// markers discovered, and an honest account of what is MISSING to score any
// start-only rows. Scorable is the count of markers that carry enough measured
// inputs to produce scored episodes, so a caller can fold completed sessions while
// still degrading gracefully for legacy markers.
type LiveCorpus struct {
	// Dir is the directory scanned (echoed for the human/JSON surface).
	Dir string `json:"dir"`
	// Present is whether the directory exists at all. A missing dir is NOT an
	// error (fail-open): Present=false with Found=0 is the empty-but-fine state.
	Present bool `json:"present"`
	// Found is how many start-marker files were discovered and parsed.
	Found int `json:"found"`
	// Scorable is how many of those carry enough ground truth to score.
	Scorable int `json:"scorable"`
	// Markers are the parsed start-markers, oldest-first by filename.
	Markers []LiveEpisodeMarker `json:"markers,omitempty"`
	// Missing names, in plain words, what each found-but-unscorable marker still
	// needs before it can be scored (empty when Found==0 or Scorable==Found).
	Missing string `json:"missing,omitempty"`
}

// ReadLiveCorpus scans dir for `--dojo` start-markers and folds them into a
// LiveCorpus. It is fail-open by construction: a missing or empty directory
// returns a valid zero-found corpus (Present reflects existence) with no error,
// so `fak dojo run --live` on a workspace that never enabled --dojo is a clean
// "nothing recorded yet", not a failure. A single unreadable/malformed marker is
// skipped (parity with the corpus scanners), never fatal.
//
// The function is pure aside from the directory read, which is the one I/O the
// discovery inherently needs.
func ReadLiveCorpus(dir string) (LiveCorpus, error) {
	lc := LiveCorpus{Dir: dir}
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			// Fail-open: an absent corpus is the honest empty state, not an error.
			return lc, nil
		}
		return lc, err
	}
	lc.Present = true

	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		n := e.Name()
		if strings.HasPrefix(n, liveEpisodeFilePrefix) && strings.HasSuffix(n, liveEpisodeFileExt) {
			names = append(names, n)
		}
	}
	// episode_YYYYMMDD_HHMMSS.jsonl sorts lexicographically == chronologically.
	sort.Strings(names)

	for _, n := range names {
		m, ok := parseLiveMarker(filepath.Join(dir, n))
		if !ok {
			continue // a malformed marker is skipped, not fatal (corpus-scan parity)
		}
		m.File = n
		lc.Markers = append(lc.Markers, m)
	}
	lc.Found = len(lc.Markers)

	scorable := 0
	for _, m := range lc.Markers {
		if liveMarkerScorable(m) {
			scorable++
		}
	}
	lc.Scorable = scorable
	if lc.Found > 0 && lc.Scorable < lc.Found {
		lc.Missing = "some live rows are start-markers only: they record {mode, command, started, workspace} " +
			"but carry no per-turn provider usage records or AdjudicationSummary, so there is no " +
			"billed reality to score the levers against. Capturing the full episode (turns + " +
			"adjudication) on the --dojo writer side is required before those rows can score."
	}
	return lc, nil
}

// parseLiveMarker reads one start-marker file. The writer emits a single JSON
// object (one line), so the first decodable object is the marker; a read error
// or an undecodable file yields ok=false and is skipped by the caller.
func parseLiveMarker(path string) (LiveEpisodeMarker, bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return LiveEpisodeMarker{}, false
	}
	var m LiveEpisodeMarker
	if err := json.Unmarshal(trimToFirstJSONObject(raw), &m); err != nil {
		return LiveEpisodeMarker{}, false
	}
	// A marker with neither a mode nor a command is not one of ours.
	if m.Mode == "" && m.Command == "" {
		return LiveEpisodeMarker{}, false
	}
	return m, true
}

// trimToFirstJSONObject returns the bytes of the first top-level {...} object in
// raw (the writer emits exactly one per file, but a future multi-line episode
// would append turns below it — this keeps the marker parse robust to that). It
// falls back to the whole input when no object delimiter is found.
func trimToFirstJSONObject(raw []byte) []byte {
	start := -1
	depth := 0
	inStr := false
	esc := false
	for i, b := range raw {
		switch {
		case esc:
			esc = false
		case b == '\\' && inStr:
			esc = true
		case b == '"':
			inStr = !inStr
		case inStr:
			// skip bytes inside a string literal
		case b == '{':
			if depth == 0 {
				start = i
			}
			depth++
		case b == '}':
			depth--
			if depth == 0 && start >= 0 {
				return raw[start : i+1]
			}
		}
	}
	return raw
}

// liveMarkerScorable reports whether a parsed marker carries enough measured
// inputs to produce at least one scored episode.
func liveMarkerScorable(m LiveEpisodeMarker) bool {
	for _, in := range m.Scored {
		if in.Outcome.Measured {
			return true
		}
	}
	return false
}

// ScorableLiveEpisodes adapts the scorable markers of a LiveCorpus into the
// dojo's (prediction, outcome) pairs. It is the seam the corpus path's lever
// adapters mirror: pure, unit-testable, and inventing NO number.
func ScorableLiveEpisodes(lc LiveCorpus) []ScoredInput {
	if lc.Scorable == 0 {
		return nil
	}
	var out []ScoredInput
	for _, m := range lc.Markers {
		if !liveMarkerScorable(m) {
			continue
		}
		for _, in := range m.Scored {
			if in.Outcome.Measured {
				out = append(out, in)
			}
		}
	}
	return out
}
