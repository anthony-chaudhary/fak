package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/studyforge"
)

func runStudyForge(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak study-forge capture|validate [flags]")
		return 2
	}
	switch args[0] {
	case "capture":
		return runStudyForgeCapture(stdout, stderr, args[1:])
	case "validate":
		return runStudyForgeValidate(stdout, stderr, args[1:])
	default:
		fmt.Fprintln(stderr, "usage: fak study-forge capture|validate [flags]")
		return 2
	}
}

func runStudyForgeCapture(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("study-forge capture", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repository := fs.String("repository", "", "GitHub repository as owner/name")
	cutoffText := fs.String("cutoff", "", "inclusive corpus cutoff (RFC3339, required)")
	out := fs.String("out", "", "corpus output path (required)")
	resume := fs.Bool("resume", false, "resume the partial corpus already at --out")
	baseURL := fs.String("base-url", "https://api.github.com", "GitHub REST API base URL")
	tokenEnv := fs.String("token-env", "GITHUB_TOKEN", "environment variable containing a read-only GitHub token")
	retries := fs.Int("retries", 3, "maximum retry attempts for transient responses")
	checkpointPages := fs.Int("checkpoint-pages", 1, "atomically checkpoint after this many accepted pages")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	parts := strings.Split(strings.TrimSpace(*repository), "/")
	if fs.NArg() != 0 || len(parts) != 2 || parts[0] == "" || parts[1] == "" || strings.TrimSpace(*cutoffText) == "" || strings.TrimSpace(*out) == "" || *retries < 0 || *checkpointPages <= 0 {
		fmt.Fprintln(stderr, "usage: fak study-forge capture --repository owner/name --cutoff RFC3339 --out PATH [--resume] [--checkpoint-pages N]")
		return 2
	}
	cutoff, err := time.Parse(time.RFC3339, *cutoffText)
	if err != nil {
		fmt.Fprintf(stderr, "study-forge: invalid --cutoff: %v\n", err)
		return 2
	}
	var prior *studyforge.Corpus
	if *resume {
		corpus, err := studyforge.ReadResume(*out)
		if err != nil {
			fmt.Fprintf(stderr, "study-forge: read resume corpus: %v\n", err)
			return 1
		}
		prior = &corpus
	}
	client := &http.Client{Timeout: 90 * time.Second, Transport: githubStudyForgeTransport{
		base: http.DefaultTransport, token: strings.TrimSpace(os.Getenv(*tokenEnv)),
	}}
	collector := studyforge.NewCollector(client)
	collector.BaseURL = strings.TrimRight(*baseURL, "/")
	collector.MaxRetries = *retries
	corpus, captureErr := collector.Capture(context.Background(), studyforge.CaptureRequest{
		Owner: parts[0], Repository: parts[1], Cutoff: cutoff.UTC(), Resume: prior,
		CheckpointEvery: *checkpointPages,
		Checkpoint: func(corpus studyforge.Corpus) error {
			return studyforge.Write(*out, corpus)
		},
	})
	if captureErr != nil {
		fmt.Fprintf(stderr, "study-forge: capture stopped; checkpoint target %s: %v\n", *out, captureErr)
		return 1
	}
	if err := studyforge.Validate(corpus); err != nil {
		fmt.Fprintf(stderr, "study-forge: invalid completed corpus: %v\n", err)
		return 1
	}
	if corpus.Receipt.Status != studyforge.StatusComplete {
		fmt.Fprintf(stderr, "study-forge: receipt status is %s, not complete\n", corpus.Receipt.Status)
		return 1
	}
	fmt.Fprintf(stdout, "captured complete study forge corpus for %s at %s to %s\n", *repository, cutoff.UTC().Format(time.RFC3339), *out)
	return 0
}

func runStudyForgeValidate(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("study-forge validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	receipt := fs.String("receipt", "", "corpus or receipt JSON path")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || strings.TrimSpace(*receipt) == "" {
		fmt.Fprintln(stderr, "usage: fak study-forge validate --receipt PATH")
		return 2
	}
	corpus, err := studyforge.Read(*receipt)
	if err != nil {
		fmt.Fprintf(stderr, "study-forge: %v\n", err)
		return 1
	}
	if err := studyforge.Validate(corpus); err != nil {
		fmt.Fprintf(stderr, "study-forge: invalid receipt: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "valid study forge receipt for %s at %s\n", corpus.Receipt.Repository, corpus.Receipt.Cutoff)
	return 0
}

type githubStudyForgeTransport struct {
	base  http.RoundTripper
	token string
}

func (t githubStudyForgeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.Header.Set("Accept", "application/vnd.github+json")
	clone.Header.Set("X-GitHub-Api-Version", "2022-11-28")
	clone.Header.Set("User-Agent", "fak-studyforge")
	if t.token != "" {
		clone.Header.Set("Authorization", "Bearer "+t.token)
	}
	return t.base.RoundTrip(clone)
}
