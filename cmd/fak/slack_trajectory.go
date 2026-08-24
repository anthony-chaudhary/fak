package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

const trajectoryChannelEnv = "FAK_TRAJECTORY_CHANNEL"

const trajectoryDefaultCooldown = time.Hour

type trajectoryPostState struct {
	Digest     string `json:"digest"`
	PostedUnix int64  `json:"posted_unix"`
}

var trajectoryPost = func(token, apiBase, channel, text string) error {
	c, err := scoreboardClient(token, apiBase)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err = c.Post(ctx, channel, text, nil)
	return err
}

func renderTrajectoryDigest(report trajctl.CurveReport, limit int) string {
	if limit <= 0 {
		limit = 5
	}
	var b strings.Builder
	b.WriteString("trajectory control - worst first\n")
	n := 0
	for _, c := range report.Objectives {
		if c.Signal == trajctl.SignalHealthy {
			continue
		}
		fmt.Fprintf(&b, "- %s - %s - %.2f (%+.2f): %s\n", c.ObjectiveID, c.Signal, c.Latest, c.Delta, c.Detail)
		n++
		if n >= limit {
			break
		}
	}
	if n == 0 {
		b.WriteString("- all open objectives healthy\n")
	}
	return b.String()
}

func runSlackTrajectory(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack trajectory", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", trajctl.DefaultLedgerRel, "trajctl JSONL ledger")
	channel := fs.String("channel", "", "Slack channel (default "+trajectoryChannelEnv+")")
	token := fs.String("token", "", "Slack bot token (default FAK_SCOREBOARD_TOKEN)")
	apiBase := fs.String("api-base", "", "Slack API base override")
	dry := fs.Bool("dry-run", false, "render only; never post")
	statePath := fs.String("state", filepath.Join(".fak", "slack-trajectory-state.json"), "storm-bound post state")
	cooldown := fs.Duration("cooldown", trajectoryDefaultCooldown, "minimum interval for an unchanged digest")
	limit := fs.Int("limit", 5, "maximum non-healthy objectives in one post")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	report := trajctl.Fold(trajctl.ReadLedgerFile(*ledger)).OpenCurves()
	text := renderTrajectoryDigest(report, *limit)
	if *dry {
		fmt.Fprint(stdout, text)
		return 0
	}
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(text)))
	if prior, err := os.ReadFile(*statePath); err == nil {
		var st trajectoryPostState
		if json.Unmarshal(prior, &st) == nil && st.Digest == digest && time.Since(time.Unix(st.PostedUnix, 0)) < *cooldown {
			fmt.Fprintln(stdout, "fak slack trajectory: unchanged digest suppressed by cooldown")
			return 0
		}
	}
	ch := strings.TrimSpace(*channel)
	if ch == "" {
		ch = strings.TrimSpace(os.Getenv(trajectoryChannelEnv))
	}
	if ch == "" {
		fmt.Fprintln(stderr, "fak slack trajectory: channel is not configured; set --channel or "+trajectoryChannelEnv)
		return 2
	}
	tok := strings.TrimSpace(*token)
	if tok == "" {
		tok = scoreboard.ResolveToken()
	}
	if tok == "" {
		fmt.Fprintln(stderr, "fak slack trajectory: token is not configured; set --token or FAK_SCOREBOARD_TOKEN / "+slackenv.EnvFileName)
		return 2
	}
	if err := trajectoryPost(tok, *apiBase, ch, text); err != nil {
		fmt.Fprintf(stderr, "fak slack trajectory: post: %v\n", err)
		return 1
	}
	if b, err := json.Marshal(trajectoryPostState{Digest: digest, PostedUnix: time.Now().Unix()}); err == nil {
		_ = os.MkdirAll(filepath.Dir(*statePath), 0o755)
		_ = os.WriteFile(*statePath, b, 0o644)
	}
	fmt.Fprintf(stdout, "fak slack trajectory: posted %d-byte digest to %s\n", len(text), ch)
	return 0
}
