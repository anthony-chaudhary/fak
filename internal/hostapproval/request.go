package hostapproval

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

const (
	Schema         = "fak.host-approval.request.v1"
	Authorization  = "none"
	BrokerRequired = "required"
	maxTTL         = 15 * time.Minute
)

var serverNameRE = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

type Options struct {
	Installer, Policy, Context, CodexConfig, Server, TaskID, SessionID string
	TTL                                                                time.Duration
}

type Artifact struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type FileSnapshot struct {
	Path   string `json:"path"`
	Exists bool   `json:"exists"`
	SHA256 string `json:"sha256,omitempty"`
}

type Adapter struct {
	Executable string   `json:"executable"`
	SHA256     string   `json:"sha256"`
	Args       []string `json:"args"`
}

type PlanStep struct {
	Executable string   `json:"executable"`
	Args       []string `json:"args"`
}

type Request struct {
	Schema        string             `json:"schema"`
	Authorization string             `json:"authorization"`
	BrokerStatus  string             `json:"broker_status"`
	Nonce         string             `json:"request_nonce"`
	PreparedAt    time.Time          `json:"prepared_at"`
	ExpiresAt     time.Time          `json:"expires_at"`
	TTLSeconds    int64              `json:"ttl_seconds"`
	TaskID        string             `json:"task_id"`
	SessionID     string             `json:"session_id"`
	Server        string             `json:"server"`
	Installer     Artifact           `json:"installer"`
	Policy        Artifact           `json:"policy"`
	Context       Artifact           `json:"context"`
	CodexConfig   FileSnapshot       `json:"codex_config"`
	RemoteConfig  remotemaint.Config `json:"remote_config"`
	Adapters      []Adapter          `json:"adapters"`
	Plan          []PlanStep         `json:"plan"`
}

func Prepare(opts Options) (Request, error) {
	if opts.TTL <= 0 || opts.TTL > maxTTL || opts.TTL%time.Second != 0 {
		return Request{}, errors.New("hostapproval: ttl must be whole seconds in (0,15m]")
	}
	if !validLabel(opts.TaskID, 256) || !validLabel(opts.SessionID, 256) || !serverNameRE.MatchString(opts.Server) {
		return Request{}, errors.New("hostapproval: invalid task, session, or server")
	}
	installer, err := artifact(opts.Installer)
	if err != nil {
		return Request{}, fmt.Errorf("hostapproval: installer: %w", err)
	}
	policyArtifact, policyRaw, err := artifactBytes(opts.Policy)
	if err != nil {
		return Request{}, fmt.Errorf("hostapproval: policy: %w", err)
	}
	if _, err := policy.ParsePolicy(policyRaw); err != nil {
		return Request{}, fmt.Errorf("hostapproval: policy: %w", err)
	}
	contextArtifact, contextRaw, err := artifactBytes(opts.Context)
	if err != nil {
		return Request{}, fmt.Errorf("hostapproval: context: %w", err)
	}
	// Reuse the executable-hashing authority loader, not merely the JSON parser.
	if _, err := remotemaint.LoadAuthority(contextArtifact.Path); err != nil {
		return Request{}, fmt.Errorf("hostapproval: context: %w", err)
	}
	cfg, err := remotemaint.ParseConfig(contextRaw)
	if err != nil {
		return Request{}, fmt.Errorf("hostapproval: context: %w", err)
	}
	codexConfig, err := snapshot(opts.CodexConfig)
	if err != nil {
		return Request{}, fmt.Errorf("hostapproval: codex config: %w", err)
	}
	adapters, err := snapshotAdapters(cfg)
	if err != nil {
		return Request{}, err
	}
	nonceBytes := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, nonceBytes); err != nil {
		return Request{}, fmt.Errorf("hostapproval: nonce: %w", err)
	}
	now := time.Now().UTC()
	expires := now.Add(opts.TTL)
	for _, profile := range cfg.Profiles {
		if profile.ExpiresAt.Before(expires) {
			expires = profile.ExpiresAt.UTC()
		}
	}
	r := Request{
		Schema: Schema, Authorization: Authorization, BrokerStatus: BrokerRequired,
		Nonce: hex.EncodeToString(nonceBytes), PreparedAt: now, ExpiresAt: expires,
		TTLSeconds: int64(opts.TTL / time.Second), TaskID: opts.TaskID, SessionID: opts.SessionID,
		Server: opts.Server, Installer: installer, Policy: policyArtifact, Context: contextArtifact,
		CodexConfig: codexConfig, RemoteConfig: cfg, Adapters: adapters,
	}
	r.Plan = expectedPlan(r)
	if err := validateRequest(r); err != nil {
		return Request{}, err
	}
	return r, nil
}

func Verify(r Request, sessionID string) error {
	if err := validateRequest(r); err != nil {
		return err
	}
	if !validLabel(sessionID, 256) || sessionID != r.SessionID {
		return errors.New("hostapproval: session mismatch")
	}
	now := time.Now().UTC()
	if r.PreparedAt.After(now) {
		return errors.New("hostapproval: request prepared in the future")
	}
	if !now.Before(r.ExpiresAt) {
		return errors.New("hostapproval: request expired")
	}
	for name, want := range map[string]Artifact{"installer": r.Installer, "policy": r.Policy, "context": r.Context} {
		got, raw, err := artifactBytes(want.Path)
		if err != nil || got != want {
			return fmt.Errorf("hostapproval: %s changed", name)
		}
		if name == "policy" {
			if _, err := policy.ParsePolicy(raw); err != nil {
				return fmt.Errorf("hostapproval: policy invalid: %w", err)
			}
		}
	}
	contextRaw, err := os.ReadFile(r.Context.Path)
	if err != nil {
		return errors.New("hostapproval: context changed")
	}
	if _, err := remotemaint.LoadAuthority(r.Context.Path); err != nil {
		return fmt.Errorf("hostapproval: context authority changed: %w", err)
	}
	cfg, err := remotemaint.ParseConfig(contextRaw)
	if err != nil || !reflect.DeepEqual(cfg, r.RemoteConfig) {
		return errors.New("hostapproval: remote config changed")
	}
	adapters, err := snapshotAdapters(cfg)
	if err != nil || !reflect.DeepEqual(adapters, r.Adapters) {
		return errors.New("hostapproval: adapter changed")
	}
	config, err := snapshot(r.CodexConfig.Path)
	if err != nil || config != r.CodexConfig {
		return errors.New("hostapproval: codex config changed")
	}
	if !reflect.DeepEqual(expectedPlan(r), r.Plan) {
		return errors.New("hostapproval: plan changed")
	}
	return nil
}

func Decode(raw []byte) (Request, error) {
	var r Request
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil {
		return Request{}, fmt.Errorf("hostapproval: malformed request: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return Request{}, errors.New("hostapproval: trailing request data")
	}
	if err := validateRequest(r); err != nil {
		return Request{}, err
	}
	return r, nil
}

func Digest(r Request) (string, error) {
	if err := validateRequest(r); err != nil {
		return "", err
	}
	raw, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}

func validateRequest(r Request) error {
	if r.Schema != Schema || r.Authorization != Authorization || r.BrokerStatus != BrokerRequired {
		return errors.New("hostapproval: request is not non-authorizing")
	}
	if len(r.Nonce) != 64 {
		return errors.New("hostapproval: invalid request nonce")
	}
	if _, err := hex.DecodeString(r.Nonce); err != nil || strings.ToLower(r.Nonce) != r.Nonce {
		return errors.New("hostapproval: invalid request nonce")
	}
	if !validLabel(r.TaskID, 256) || !validLabel(r.SessionID, 256) || !serverNameRE.MatchString(r.Server) {
		return errors.New("hostapproval: invalid identity")
	}
	if r.TTLSeconds <= 0 || r.TTLSeconds > int64(maxTTL/time.Second) || r.PreparedAt.IsZero() || r.ExpiresAt.IsZero() ||
		!r.PreparedAt.Equal(r.PreparedAt.UTC()) || !r.ExpiresAt.Equal(r.ExpiresAt.UTC()) || !r.ExpiresAt.After(r.PreparedAt) ||
		r.ExpiresAt.After(r.PreparedAt.Add(time.Duration(r.TTLSeconds)*time.Second)) {
		return errors.New("hostapproval: invalid request lifetime")
	}
	for _, a := range []Artifact{r.Installer, r.Policy, r.Context} {
		if !canonicalAbs(a.Path) || !isDigest(a.SHA256) {
			return errors.New("hostapproval: invalid artifact")
		}
	}
	if !canonicalAbs(r.CodexConfig.Path) || (r.CodexConfig.Exists != (r.CodexConfig.SHA256 != "")) || (r.CodexConfig.Exists && !isDigest(r.CodexConfig.SHA256)) {
		return errors.New("hostapproval: invalid codex config snapshot")
	}
	if len(r.RemoteConfig.Profiles) == 0 || len(r.Adapters) != len(r.RemoteConfig.Profiles) || len(r.Plan) != 2 {
		return errors.New("hostapproval: incomplete reviewed scope")
	}
	if err := remotemaint.ValidateConfig(r.RemoteConfig); err != nil {
		return fmt.Errorf("hostapproval: invalid reviewed remote config: %w", err)
	}
	for i, a := range r.Adapters {
		if r.ExpiresAt.After(r.RemoteConfig.Profiles[i].ExpiresAt) {
			return errors.New("hostapproval: request exceeds remote context expiry")
		}
		if !canonicalAbs(a.Executable) || !isDigest(a.SHA256) || a.Executable != r.RemoteConfig.Profiles[i].Authority.Executable ||
			a.SHA256 != r.RemoteConfig.Profiles[i].Authority.SHA256 || !reflect.DeepEqual(a.Args, r.RemoteConfig.Profiles[i].Authority.Args) {
			return errors.New("hostapproval: invalid adapter snapshot")
		}
	}
	if !reflect.DeepEqual(expectedPlan(r), r.Plan) {
		return errors.New("hostapproval: invalid plan")
	}
	return nil
}

func expectedPlan(r Request) []PlanStep {
	ttl := fmt.Sprintf("%ds", r.TTLSeconds)
	return []PlanStep{
		{Executable: r.Installer.Path, Args: []string{"codex", "mcp", "install", "--source", r.Installer.Path, "--policy", r.Policy.Path, "--config", r.CodexConfig.Path, "--server", r.Server}},
		{Executable: r.Installer.Path, Args: []string{"guard", "allow", "--remote-maintenance", r.Context.Path, "--ttl", ttl}},
	}
}

func artifact(path string) (Artifact, error) { a, _, err := artifactBytes(path); return a, err }

func artifactBytes(path string) (Artifact, []byte, error) {
	p, err := canonicalPath(path)
	if err != nil {
		return Artifact{}, nil, err
	}
	info, err := os.Lstat(p)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return Artifact{}, nil, errors.New("not a regular file")
	}
	raw, err := os.ReadFile(p)
	if err != nil {
		return Artifact{}, nil, err
	}
	sum := sha256.Sum256(raw)
	return Artifact{Path: p, SHA256: hex.EncodeToString(sum[:])}, raw, nil
}

func snapshot(path string) (FileSnapshot, error) {
	p, err := canonicalPath(path)
	if err != nil {
		return FileSnapshot{}, err
	}
	info, err := os.Lstat(p)
	if errors.Is(err, os.ErrNotExist) {
		return FileSnapshot{Path: p}, nil
	}
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return FileSnapshot{}, errors.New("not a regular file")
	}
	raw, err := os.ReadFile(p)
	if err != nil {
		return FileSnapshot{}, err
	}
	sum := sha256.Sum256(raw)
	return FileSnapshot{Path: p, Exists: true, SHA256: hex.EncodeToString(sum[:])}, nil
}

func snapshotAdapters(cfg remotemaint.Config) ([]Adapter, error) {
	out := make([]Adapter, 0, len(cfg.Profiles))
	for _, p := range cfg.Profiles {
		a, err := artifact(p.Authority.Executable)
		if err != nil {
			return nil, fmt.Errorf("hostapproval: adapter: %w", err)
		}
		if a.SHA256 != p.Authority.SHA256 {
			return nil, errors.New("hostapproval: adapter digest mismatch")
		}
		out = append(out, Adapter{Executable: a.Path, SHA256: a.SHA256, Args: append([]string(nil), p.Authority.Args...)})
	}
	return out, nil
}

func canonicalPath(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", errors.New("empty path")
	}
	p, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	return filepath.Clean(p), nil
}
func canonicalAbs(path string) bool { return filepath.IsAbs(path) && filepath.Clean(path) == path }
func isDigest(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size && strings.ToLower(s) == s
}
func validLabel(s string, max int) bool {
	if s == "" || len(s) > max || strings.TrimSpace(s) != s {
		return false
	}
	for _, r := range s {
		if r < 0x20 || r == 0x7f {
			return false
		}
	}
	return true
}
