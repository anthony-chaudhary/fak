package recall

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ErrStale is returned when a recalled page names a concrete artifact that no
// longer verifies against the current git checkout. It is a read-time guard: the
// page bytes are preserved in CAS for audit, but they are withheld from model
// context rather than re-injected as fact.
var ErrStale = errors.New("recall: page stale at read-time artifact reverify")

// ArtifactKind names the concrete artifact surfaces #1158 asks recall to re-check.
type ArtifactKind string

const (
	ArtifactGitSHA ArtifactKind = "git_sha"
	ArtifactPath   ArtifactKind = "path"
	ArtifactFlag   ArtifactKind = "flag"
)

// ArtifactStatus is the read-time verdict for one named artifact.
type ArtifactStatus string

const (
	ArtifactFresh        ArtifactStatus = "fresh"
	ArtifactStale        ArtifactStatus = "stale"
	ArtifactUnverifiable ArtifactStatus = "unverifiable"
)

// ArtifactClaim is one concrete artifact mentioned by recalled text.
type ArtifactClaim struct {
	Kind  ArtifactKind `json:"kind"`
	Value string       `json:"value"`
}

// ArtifactFinding is the verifier's verdict for one claim.
type ArtifactFinding struct {
	Claim  ArtifactClaim  `json:"claim"`
	Status ArtifactStatus `json:"status"`
	Detail string         `json:"detail,omitempty"`
}

// ArtifactVerifier is the seam used by tests and by future DOS-backed readers.
// The default verifier checks git commits, repo-relative paths, and flags in the
// current checkout.
type ArtifactVerifier func(context.Context, []ArtifactClaim) []ArtifactFinding

// WithArtifactVerifier overrides the read-time artifact verifier for this loaded
// session. Passing nil restores the default verifier.
func (s *Session) WithArtifactVerifier(v ArtifactVerifier) *Session {
	s.artifactVerifier = v
	return s
}

func (s *Session) verifyArtifacts(ctx context.Context, p Page, body []byte) error {
	claims := ExtractArtifactClaims(string(body) + "\n" + p.Descriptor)
	if len(claims) == 0 {
		return nil
	}
	verifier := s.artifactVerifier
	if verifier == nil {
		verifier = DefaultArtifactVerifier
	}
	findings := verifier(ctx, claims)
	if len(findings) == 0 {
		findings = []ArtifactFinding{{
			Claim:  claims[0],
			Status: ArtifactUnverifiable,
			Detail: "artifact verifier returned no verdicts",
		}}
	}
	var bad []string
	for _, f := range findings {
		if f.Status == ArtifactFresh {
			continue
		}
		label := fmt.Sprintf("%s %q %s", f.Claim.Kind, f.Claim.Value, f.Status)
		if strings.TrimSpace(f.Detail) != "" {
			label += ": " + strings.TrimSpace(f.Detail)
		}
		bad = append(bad, label)
	}
	if len(bad) == 0 {
		return nil
	}
	sort.Strings(bad)
	return fmt.Errorf("%w: page %d withheld; %s", ErrStale, p.Step, strings.Join(bad, "; "))
}

// ExtractArtifactClaims finds conservative, concrete artifact mentions in recalled
// text. It deliberately ignores arbitrary numbers and URLs: only commit-like SHAs,
// repo-shaped paths, and CLI flags are load-bearing enough to verify.
func ExtractArtifactClaims(text string) []ArtifactClaim {
	seen := map[string]bool{}
	var out []ArtifactClaim
	add := func(kind ArtifactKind, value string) {
		value = strings.TrimSpace(value)
		if value == "" {
			return
		}
		key := string(kind) + "\x00" + strings.ToLower(value)
		if seen[key] {
			return
		}
		seen[key] = true
		out = append(out, ArtifactClaim{Kind: kind, Value: value})
	}
	extractSHAs(text, add)
	extractFlags(text, add)
	extractPaths(text, add)
	return out
}

var (
	shaRe  = regexp.MustCompile(`(?i)\b[0-9a-f]{7,40}\b`)
	flagRe = regexp.MustCompile("(?:^|[\\s\"'`\\[(])(--[A-Za-z][A-Za-z0-9][A-Za-z0-9-]{1,64})\\b")
)

func extractSHAs(text string, add func(ArtifactKind, string)) {
	lower := strings.ToLower(text)
	for _, loc := range shaRe.FindAllStringIndex(text, -1) {
		raw := text[loc[0]:loc[1]]
		if !hasHexLetter(raw) {
			continue
		}
		if len(raw) < 40 && !hasCommitCue(lower, loc[0]) {
			continue
		}
		add(ArtifactGitSHA, strings.ToLower(raw))
	}
}

func hasHexLetter(s string) bool {
	for _, r := range s {
		if r >= 'a' && r <= 'f' || r >= 'A' && r <= 'F' {
			return true
		}
	}
	return false
}

func hasCommitCue(lower string, at int) bool {
	start := at - 32
	if start < 0 {
		start = 0
	}
	prefix := lower[start:at]
	for _, cue := range []string{"commit", "sha", "head", "rev", "revision"} {
		if strings.Contains(prefix, cue) {
			return true
		}
	}
	return false
}

func extractFlags(text string, add func(ArtifactKind, string)) {
	for _, m := range flagRe.FindAllStringSubmatch(text, -1) {
		if len(m) > 1 {
			add(ArtifactFlag, m[1])
		}
	}
}

func extractPaths(text string, add func(ArtifactKind, string)) {
	for _, raw := range strings.Fields(text) {
		tok := cleanArtifactToken(raw)
		if tok == "" || strings.Contains(tok, "://") || strings.Contains(tok, "@") {
			continue
		}
		tok = strings.ReplaceAll(tok, "\\", "/")
		if absolutePathToken(tok) {
			continue
		}
		if i := strings.LastIndexByte(tok, ':'); i > 1 && allDigits(tok[i+1:]) {
			tok = tok[:i]
		}
		if !strings.Contains(tok, "/") || strings.Contains(tok, "..") || strings.ContainsAny(tok, "*?{}") {
			continue
		}
		if repoPrefixed(tok) || knownPathExt(filepath.Ext(tok)) {
			add(ArtifactPath, tok)
		}
	}
}

func absolutePathToken(p string) bool {
	if strings.HasPrefix(p, "/") {
		return true
	}
	if len(p) >= 3 && ((p[0] >= 'A' && p[0] <= 'Z') || (p[0] >= 'a' && p[0] <= 'z')) && p[1] == ':' && p[2] == '/' {
		return true
	}
	return false
}

func cleanArtifactToken(s string) string {
	return strings.TrimFunc(s, func(r rune) bool {
		return unicode.IsSpace(r) || strings.ContainsRune("`'\"[](){}<>,.;", r)
	})
}

func allDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func repoPrefixed(p string) bool {
	for _, prefix := range []string{
		"cmd/", "internal/", "docs/", "tools/", "examples/", "testdata/",
		"pkg/", "scripts/", "experiments/", "visuals/", "plugins/",
	} {
		if strings.HasPrefix(p, prefix) {
			return true
		}
	}
	return false
}

func knownPathExt(ext string) bool {
	switch strings.ToLower(ext) {
	case ".go", ".md", ".json", ".jsonl", ".py", ".ps1", ".sh", ".toml", ".yaml", ".yml", ".txt", ".mod", ".sum", ".c", ".h", ".cu", ".m", ".s", ".css", ".html", ".svg", ".png", ".jpg", ".jpeg":
		return true
	default:
		return false
	}
}

// DefaultArtifactVerifier is the default read-time guard. It is intentionally
// local and deterministic: git SHAs must name commits reachable from HEAD and
// not later reverted, repo-relative paths must exist in the current working
// tree, and CLI flags must still appear in the tracked checkout.
func DefaultArtifactVerifier(ctx context.Context, claims []ArtifactClaim) []ArtifactFinding {
	root, gitOK := gitRoot(ctx)
	out := make([]ArtifactFinding, 0, len(claims))
	for _, c := range claims {
		switch c.Kind {
		case ArtifactGitSHA:
			out = append(out, verifyGitSHA(ctx, root, gitOK, c))
		case ArtifactPath:
			out = append(out, verifyPath(root, c))
		case ArtifactFlag:
			out = append(out, verifyFlag(ctx, root, gitOK, c))
		default:
			out = append(out, ArtifactFinding{Claim: c, Status: ArtifactUnverifiable, Detail: "unknown artifact kind"})
		}
	}
	return out
}

func gitRoot(ctx context.Context) (string, bool) {
	cmd := exec.CommandContext(ctx, "git", "rev-parse", "--show-toplevel")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err == nil {
		if root := strings.TrimSpace(string(out)); root != "" {
			return root, true
		}
	}
	if wd, wdErr := os.Getwd(); wdErr == nil {
		return wd, false
	}
	return "", false
}

func verifyGitSHA(ctx context.Context, root string, gitOK bool, c ArtifactClaim) ArtifactFinding {
	if !gitOK {
		return ArtifactFinding{Claim: c, Status: ArtifactUnverifiable, Detail: "git root unavailable"}
	}
	catFile := exec.CommandContext(ctx, "git", "-C", root, "cat-file", "-e", c.Value+"^{commit}")
	windowgate.ConfigureBackgroundCommand(catFile)
	if err := catFile.Run(); err != nil {
		return ArtifactFinding{Claim: c, Status: ArtifactStale, Detail: "commit does not resolve in git"}
	}
	full := c.Value
	revParse := exec.CommandContext(ctx, "git", "-C", root, "rev-parse", "--verify", c.Value+"^{commit}")
	windowgate.ConfigureBackgroundCommand(revParse)
	if out, err := revParse.Output(); err == nil {
		if resolved := strings.TrimSpace(string(out)); resolved != "" {
			full = resolved
		}
	}
	mergeBase := exec.CommandContext(ctx, "git", "-C", root, "merge-base", "--is-ancestor", full, "HEAD")
	windowgate.ConfigureBackgroundCommand(mergeBase)
	err := mergeBase.Run()
	if err == nil {
		if revertSHA, ok := revertedBy(ctx, root, full); ok {
			return ArtifactFinding{Claim: c, Status: ArtifactStale, Detail: "commit is reachable but later reverted by " + shortArtifactSHA(revertSHA)}
		}
		return ArtifactFinding{Claim: c, Status: ArtifactFresh, Detail: "commit resolves and is reachable from HEAD"}
	}
	return ArtifactFinding{Claim: c, Status: ArtifactStale, Detail: "commit resolves but is not reachable from HEAD"}
}

func revertedBy(ctx context.Context, root, fullSHA string) (string, bool) {
	cmd := exec.CommandContext(ctx, "git", "-C", root, "log", "--format=%H%x00%B%x00", fullSHA+"..HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", false
	}
	parts := strings.Split(string(out), "\x00")
	needle := "this reverts commit " + strings.ToLower(fullSHA)
	for i := 0; i+1 < len(parts); i += 2 {
		if strings.Contains(strings.ToLower(parts[i+1]), needle) {
			return strings.TrimSpace(parts[i]), true
		}
	}
	return "", false
}

func shortArtifactSHA(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

func verifyPath(root string, c ArtifactClaim) ArtifactFinding {
	if root == "" {
		return ArtifactFinding{Claim: c, Status: ArtifactUnverifiable, Detail: "working tree root unavailable"}
	}
	clean := filepath.Clean(filepath.FromSlash(c.Value))
	if filepath.IsAbs(clean) || strings.HasPrefix(clean, ".."+string(filepath.Separator)) || clean == ".." {
		return ArtifactFinding{Claim: c, Status: ArtifactUnverifiable, Detail: "path is not repo-relative"}
	}
	if _, err := os.Stat(filepath.Join(root, clean)); err == nil {
		return ArtifactFinding{Claim: c, Status: ArtifactFresh, Detail: "path exists in working tree"}
	} else if errors.Is(err, os.ErrNotExist) {
		return ArtifactFinding{Claim: c, Status: ArtifactStale, Detail: "path missing from working tree"}
	} else {
		return ArtifactFinding{Claim: c, Status: ArtifactUnverifiable, Detail: err.Error()}
	}
}

func verifyFlag(ctx context.Context, root string, gitOK bool, c ArtifactClaim) ArtifactFinding {
	if !gitOK {
		return ArtifactFinding{Claim: c, Status: ArtifactUnverifiable, Detail: "git grep unavailable"}
	}
	cmd := exec.CommandContext(ctx, "git", "-C", root, "grep", "-F", "--", c.Value, "--", ".")
	windowgate.ConfigureBackgroundCommand(cmd)
	err := cmd.Run()
	if err == nil {
		return ArtifactFinding{Claim: c, Status: ArtifactFresh, Detail: "flag appears in tracked checkout"}
	}
	if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
		return ArtifactFinding{Claim: c, Status: ArtifactStale, Detail: "flag missing from tracked checkout"}
	}
	return ArtifactFinding{Claim: c, Status: ArtifactUnverifiable, Detail: "git grep failed"}
}
