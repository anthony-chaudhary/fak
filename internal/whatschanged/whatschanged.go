package whatschanged

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"sort"
	"strconv"
	"strings"
)

type Options struct {
	Since string
	Paths []string
	Run   Runner
}

type Report struct {
	Since        string   `json:"since"`
	Head         string   `json:"head"`
	Range        string   `json:"range"`
	Paths        []string `json:"paths"`
	Empty        bool     `json:"empty"`
	ChangedFiles []string `json:"changed_files,omitempty"`
	Commits      []Commit `json:"commits,omitempty"`
}

type Commit struct {
	SHA         string   `json:"sha"`
	Short       string   `json:"short"`
	Subject     string   `json:"subject"`
	AuthorName  string   `json:"author_name,omitempty"`
	AuthorEmail string   `json:"author_email,omitempty"`
	UnixTime    int64    `json:"unix_time,omitempty"`
	Files       []string `json:"files"`
}

type Runner func(ctx context.Context, dir string, args ...string) (RunResult, error)

type RunResult struct {
	Stdout []byte
	Stderr []byte
	Code   int
}

func RealRunner(ctx context.Context, dir string, args ...string) (RunResult, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			code = exit.ExitCode()
			err = nil
		}
	}
	return RunResult{Stdout: stdout.Bytes(), Stderr: stderr.Bytes(), Code: code}, err
}

func Preview(ctx context.Context, dir string, opt Options) (Report, error) {
	run := opt.Run
	if run == nil {
		run = RealRunner
	}
	paths := cleanPaths(opt.Paths)
	if len(paths) == 0 {
		return Report{}, fmt.Errorf("whats-changed: at least one --paths value is required")
	}
	sinceRef := strings.TrimSpace(opt.Since)
	if sinceRef == "" {
		sinceRef = "HEAD"
	}
	since, err := revParse(ctx, dir, sinceRef, run)
	if err != nil {
		return Report{}, err
	}
	head, err := revParse(ctx, dir, "HEAD", run)
	if err != nil {
		return Report{}, err
	}
	rep := Report{
		Since: since,
		Head:  head,
		Range: since + ".." + head,
		Paths: paths,
	}
	if since == head {
		rep.Empty = true
		return rep, nil
	}
	commits, err := logCommits(ctx, dir, since, paths, run)
	if err != nil {
		return Report{}, err
	}
	var changed []string
	for i := range commits {
		files, err := commitFiles(ctx, dir, commits[i].SHA, paths, run)
		if err != nil {
			return Report{}, err
		}
		commits[i].Files = files
		changed = append(changed, files...)
	}
	rep.Commits = commits
	rep.ChangedFiles = uniqueSorted(changed)
	rep.Empty = len(rep.Commits) == 0
	return rep, nil
}

func logCommits(ctx context.Context, dir, since string, paths []string, run Runner) ([]Commit, error) {
	args := []string{"log", "--reverse", "--format=%H%x1f%an%x1f%ae%x1f%ct%x1f%s%x1e", since + "..HEAD", "--"}
	args = append(args, paths...)
	out, err := run(ctx, dir, args...)
	if err != nil {
		return nil, err
	}
	if out.Code != 0 {
		return nil, gitError("log", out)
	}
	records := bytes.Split(out.Stdout, []byte{0x1e})
	commits := make([]Commit, 0, len(records))
	for _, rec := range records {
		rec = bytes.TrimSpace(rec)
		if len(rec) == 0 {
			continue
		}
		fields := bytes.SplitN(rec, []byte{0x1f}, 5)
		if len(fields) != 5 {
			return nil, fmt.Errorf("whats-changed: could not parse git log record")
		}
		unixTime, _ := strconv.ParseInt(string(fields[3]), 10, 64)
		sha := string(fields[0])
		commits = append(commits, Commit{
			SHA:         sha,
			Short:       shortSHA(sha),
			AuthorName:  string(fields[1]),
			AuthorEmail: string(fields[2]),
			UnixTime:    unixTime,
			Subject:     string(fields[4]),
		})
	}
	return commits, nil
}

func commitFiles(ctx context.Context, dir, sha string, paths []string, run Runner) ([]string, error) {
	args := []string{"diff-tree", "--no-commit-id", "--name-only", "-r", "-m", "-z", sha, "--"}
	args = append(args, paths...)
	out, err := run(ctx, dir, args...)
	if err != nil {
		return nil, err
	}
	if out.Code != 0 {
		return nil, gitError("diff-tree "+shortSHA(sha), out)
	}
	return uniqueSorted(splitNUL(out.Stdout)), nil
}

func revParse(ctx context.Context, dir, ref string, run Runner) (string, error) {
	out, err := run(ctx, dir, "rev-parse", "--verify", ref+"^{commit}")
	if err != nil {
		return "", err
	}
	if out.Code != 0 {
		return "", gitError("rev-parse "+ref, out)
	}
	return strings.TrimSpace(string(out.Stdout)), nil
}

func gitError(op string, r RunResult) error {
	detail := strings.TrimSpace(string(r.Stderr))
	if detail == "" {
		detail = strings.TrimSpace(string(r.Stdout))
	}
	if detail == "" {
		detail = fmt.Sprintf("exit %d", r.Code)
	}
	return fmt.Errorf("whats-changed: git %s failed: %s", op, detail)
}

func cleanPaths(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func splitNUL(b []byte) []string {
	parts := bytes.Split(b, []byte{0})
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if len(p) > 0 {
			out = append(out, string(p))
		}
	}
	return out
}

func uniqueSorted(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, item := range in {
		if item == "" || seen[item] {
			continue
		}
		seen[item] = true
		out = append(out, item)
	}
	sort.Strings(out)
	return out
}

func shortSHA(sha string) string {
	if len(sha) <= 12 {
		return sha
	}
	return sha[:12]
}
