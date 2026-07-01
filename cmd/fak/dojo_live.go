package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

// persistLiveDojoEpisode writes a completed --dojo live row when the session has
// observed provider-cache turns. The row carries scored vcache-warmth inputs, so
// `fak dojo run --live` can fold billed reality without scraping the transcript.
func persistLiveDojoEpisode(command string, srv *gateway.Server) error {
	if srv == nil {
		return nil
	}
	turns, ok := srv.VCacheTurnsSnapshot()
	if !ok || len(turns) == 0 {
		return nil
	}
	rep := vcacheobserve.Observe(turns, vcacheobserve.DefaultMultipliers())
	inputs := vcacheEpisodesFromObserve(rep.Prediction)
	if len(inputs) == 0 {
		return nil
	}
	return logDojoEpisodeFile(command, inputs)
}

func logDojoEpisodeFile(command string, scored []dojo.ScoredInput) error {
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("getcwd: %w", err)
	}
	workspaceRoot := findRepoRoot(cwd)
	dojoDir := filepath.Join(workspaceRoot, filepath.FromSlash(dojo.LiveEpisodesRel))
	if err := os.MkdirAll(dojoDir, 0o755); err != nil {
		return fmt.Errorf("mkdir dojo corpus: %w", err)
	}

	now := time.Now().UTC()
	episodeFile := filepath.Join(dojoDir, fmt.Sprintf("episode_%s_%09d.jsonl", now.Format("20060102_150405"), now.Nanosecond()))
	f, err := os.OpenFile(episodeFile, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return fmt.Errorf("open episode file: %w", err)
	}
	defer f.Close()

	ep := struct {
		Mode      string             `json:"mode"`
		Command   string             `json:"command"`
		Started   string             `json:"started"`
		CWD       string             `json:"cwd"`
		Workspace string             `json:"workspace"`
		Scored    []dojo.ScoredInput `json:"scored,omitempty"`
	}{
		Mode:      "live",
		Command:   command,
		Started:   now.Format(time.RFC3339),
		CWD:       cwd,
		Workspace: workspaceRoot,
		Scored:    scored,
	}
	if err := json.NewEncoder(f).Encode(ep); err != nil {
		return fmt.Errorf("encode episode: %w", err)
	}
	return nil
}
