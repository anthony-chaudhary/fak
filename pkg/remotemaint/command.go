package remotemaint

import (
	"bytes"
	"context"
	"crypto/sha256"
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	maxAdapterOutput       = 1 << 20
	resolveValidateTimeout = 60 * time.Second
	executeTimeout         = 180 * time.Second
)

type commandAuthority struct {
	configPath string
	mu         sync.Mutex
	bindings   map[string]string
}

type adapterRequest struct {
	Schema          string `json:"schema"`
	Phase           string `json:"phase"`
	ProfileID       string `json:"profile_id"`
	OperationDigest string `json:"operation_digest"`
	Grant           *Grant `json:"grant,omitempty"`
}

type adapterError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type adapterResponse struct {
	Schema  string        `json:"schema"`
	Grant   *Grant        `json:"grant,omitempty"`
	Receipt *Receipt      `json:"receipt,omitempty"`
	Error   *adapterError `json:"error,omitempty"`
}

func LoadAuthority(configPath string) (Authority, error) {
	abs, err := filepath.Abs(configPath)
	if err != nil {
		return nil, fmt.Errorf("%w: invalid context path", ErrUnavailable)
	}
	if _, _, err := loadConfig(abs, time.Now().UTC()); err != nil {
		return nil, err
	}
	return &commandAuthority{configPath: abs, bindings: make(map[string]string)}, nil
}

func (a *commandAuthority) Resolve(ctx context.Context, req Request) (Grant, error) {
	if err := validateRequest(req); err != nil {
		return Grant{}, err
	}
	profile, configDigest, err := a.currentProfile(req.ProfileID)
	if err != nil {
		return Grant{}, err
	}
	response, err := runAdapter(ctx, profile.Authority, adapterRequest{
		Schema: RequestEnvelopeSchema, Phase: "resolve", ProfileID: req.ProfileID, OperationDigest: req.OperationDigest,
	})
	if err != nil {
		return Grant{}, err
	}
	if response.Grant == nil || response.Receipt != nil {
		return Grant{}, fmt.Errorf("%w: resolve response shape", ErrAuthority)
	}
	grant := normalizeGrant(*response.Grant)
	if grant.ProfileID != req.ProfileID || grant.OperationDigest != req.OperationDigest {
		return Grant{}, fmt.Errorf("%w: resolve identity", ErrChanged)
	}
	if err := validateGrant(grant, time.Now().UTC()); err != nil {
		return Grant{}, err
	}
	current, currentDigest, err := a.currentProfile(req.ProfileID)
	if err != nil {
		return Grant{}, err
	}
	if currentDigest != configDigest {
		return Grant{}, fmt.Errorf("%w: operator context changed during resolve", ErrChanged)
	}
	if err := matchProfile(grant, current); err != nil {
		return Grant{}, err
	}
	a.mu.Lock()
	key := GrantDigest(grant)
	if previous, ok := a.bindings[key]; ok && previous != configDigest {
		a.bindings[key] = ""
		a.mu.Unlock()
		return Grant{}, fmt.Errorf("%w: conflicting operator context", ErrChanged)
	}
	a.bindings[key] = configDigest
	a.mu.Unlock()
	return grant, nil
}

func (a *commandAuthority) Validate(ctx context.Context, grant Grant) error {
	if err := validateGrant(grant, time.Now().UTC()); err != nil {
		return err
	}
	profile, _, err := a.boundProfile(grant)
	if err != nil {
		return err
	}
	response, err := runAdapter(ctx, profile.Authority, adapterRequest{
		Schema: RequestEnvelopeSchema, Phase: "validate", ProfileID: grant.ProfileID,
		OperationDigest: grant.OperationDigest, Grant: ptrGrant(grant),
	})
	if err != nil {
		return err
	}
	if response.Grant == nil || response.Receipt != nil {
		return fmt.Errorf("%w: validate response shape", ErrAuthority)
	}
	got := cloneGrant(*response.Grant)
	if err := validateGrant(got, time.Now().UTC()); err != nil {
		return err
	}
	if !equalGrantBytes(grant, got) {
		return fmt.Errorf("%w: validate changed grant", ErrChanged)
	}
	if _, _, err := a.boundProfile(grant); err != nil {
		return err
	}
	return nil
}

func (a *commandAuthority) Execute(ctx context.Context, grant Grant) (Receipt, error) {
	if err := validateGrant(grant, time.Now().UTC()); err != nil {
		return Receipt{}, err
	}
	profile, _, err := a.boundProfile(grant)
	if err != nil {
		return Receipt{}, err
	}
	response, err := runAdapter(ctx, profile.Authority, adapterRequest{
		Schema: RequestEnvelopeSchema, Phase: "execute", ProfileID: grant.ProfileID,
		OperationDigest: grant.OperationDigest, Grant: ptrGrant(grant),
	})
	if err != nil {
		if response.Receipt == nil || response.Error == nil || response.Grant != nil || validateFailureReceipt(*response.Receipt, grant, time.Now().UTC()) != nil {
			return Receipt{}, fmt.Errorf("%w: adapter execution denied", ErrAuthority)
		}
		if _, _, bindErr := a.boundProfile(grant); bindErr != nil {
			return Receipt{}, bindErr
		}
		switch response.Receipt.Outcome {
		case OutcomeUnknownEffect:
			return *response.Receipt, ErrUnknownEffect
		case OutcomeRollbackFailed:
			return *response.Receipt, ErrRollbackFailed
		default:
			return Receipt{}, fmt.Errorf("%w: invalid failure outcome", ErrAuthority)
		}
	}
	if response.Receipt == nil || response.Grant != nil {
		return Receipt{}, fmt.Errorf("%w: execute response shape", ErrAuthority)
	}
	if err := validateReceipt(*response.Receipt, grant, time.Now().UTC()); err != nil {
		return Receipt{}, err
	}
	if _, _, err := a.boundProfile(grant); err != nil {
		return Receipt{}, err
	}
	return *response.Receipt, nil
}

func (a *commandAuthority) currentProfile(id string) (Profile, string, error) {
	cfg, digest, err := loadConfig(a.configPath, time.Now().UTC())
	if err != nil {
		return Profile{}, "", err
	}
	for _, profile := range cfg.Profiles {
		if profile.ProfileID == id {
			if err := rehashExecutable(profile.Authority); err != nil {
				return Profile{}, "", err
			}
			return profile, digest, nil
		}
	}
	return Profile{}, "", fmt.Errorf("%w: profile not configured", ErrUnavailable)
}

func (a *commandAuthority) boundProfile(grant Grant) (Profile, string, error) {
	profile, digest, err := a.currentProfile(grant.ProfileID)
	if err != nil {
		return Profile{}, "", err
	}
	if err := matchProfile(grant, profile); err != nil {
		return Profile{}, "", err
	}
	a.mu.Lock()
	bound, ok := a.bindings[GrantDigest(grant)]
	a.mu.Unlock()
	if !ok || bound != digest {
		return Profile{}, "", fmt.Errorf("%w: operator context changed", ErrChanged)
	}
	return profile, digest, nil
}

func matchProfile(g Grant, p Profile) error {
	if g.ProfileID != p.ProfileID || g.Target != p.Target || g.HostKey != p.HostKey || g.Operation != p.Operation ||
		g.PayloadSHA256 != p.PayloadSHA256 || g.Holder != p.Holder || g.ExpiresAt.After(p.ExpiresAt) ||
		!equalStrings(g.Paths, p.Paths) || !equalPathPrefixes(g.PathPrefixes, p.PathPrefixes) || !equalStrings(g.Units, p.Units) {
		return fmt.Errorf("%w: grant outside trusted profile", ErrScope)
	}
	if len(g.LeasePins) != len(p.LeaseNodes) {
		return fmt.Errorf("%w: lease node set mismatch", ErrScope)
	}
	for i := range p.LeaseNodes {
		if g.LeasePins[i].NodeID != p.LeaseNodes[i] {
			return fmt.Errorf("%w: lease node mismatch", ErrScope)
		}
	}
	return nil
}

func runAdapter(parent context.Context, command Command, request adapterRequest) (adapterResponse, error) {
	if err := rehashExecutable(command); err != nil {
		return adapterResponse{}, err
	}
	raw, err := json.Marshal(request)
	if err != nil {
		return adapterResponse{}, fmt.Errorf("%w: request encoding", ErrAuthority)
	}
	deadline := resolveValidateTimeout
	if request.Phase == "execute" {
		deadline = executeTimeout
	}
	ctx, cancel := context.WithTimeout(parent, deadline)
	defer cancel()
	cmd := exec.CommandContext(ctx, command.Executable, command.Args...)
	cmd.Stdin = bytes.NewReader(raw)
	cmd.Dir = filepath.Dir(command.Executable)
	cmd.Env = trustedHostEnvironment()
	stdout := newBoundedBuffer(maxAdapterOutput)
	stderr := newBoundedBuffer(maxAdapterOutput)
	cmd.Stdout, cmd.Stderr = stdout, stderr
	runErr := cmd.Run()
	if err := rehashExecutable(command); err != nil {
		return adapterResponse{}, err
	}
	if stdout.overflow || stderr.overflow {
		return adapterResponse{}, fmt.Errorf("%w: adapter output exceeded limit", ErrAuthority)
	}
	var response adapterResponse
	if err := decodeStrict(stdout.Bytes(), &response); err != nil || response.Schema != ResponseEnvelopeSchema {
		if ctx.Err() != nil {
			return adapterResponse{}, fmt.Errorf("%w: adapter deadline", ErrAuthority)
		}
		if runErr != nil {
			return adapterResponse{}, fmt.Errorf("%w: adapter failed", ErrAuthority)
		}
		return adapterResponse{}, fmt.Errorf("%w: malformed adapter response", ErrAuthority)
	}
	if response.Error != nil && (!validName(response.Error.Code, 64) || len(response.Error.Message) > 256) {
		return adapterResponse{}, fmt.Errorf("%w: malformed adapter error", ErrAuthority)
	}
	if runErr != nil || response.Error != nil {
		return response, fmt.Errorf("%w: adapter denied", ErrAuthority)
	}
	return response, nil
}

func trustedHostEnvironment() []string {
	allowed := map[string]struct{}{
		"PATH": {}, "SYSTEMROOT": {}, "WINDIR": {}, "HOME": {}, "USERPROFILE": {},
		"HOMEDRIVE": {}, "HOMEPATH": {}, "TEMP": {}, "TMP": {}, "TMPDIR": {},
	}
	byName := make(map[string]string, len(allowed))
	for _, entry := range os.Environ() {
		name, _, ok := strings.Cut(entry, "=")
		if !ok || name == "" {
			continue
		}
		canonical := strings.ToUpper(name)
		if _, ok := allowed[canonical]; ok {
			byName[canonical] = entry
		}
	}
	names := make([]string, 0, len(byName))
	for name := range byName {
		names = append(names, name)
	}
	sort.Strings(names)
	env := make([]string, 0, len(names))
	for _, name := range names {
		env = append(env, byName[name])
	}
	return env
}

func validateCommand(command Command) error {
	if !filepath.IsAbs(command.Executable) || !isDigest(command.SHA256) || len(command.Args) > 64 {
		return fmt.Errorf("%w: invalid command registration", ErrAuthority)
	}
	base := strings.ToLower(strings.TrimSuffix(filepath.Base(command.Executable), filepath.Ext(command.Executable)))
	interpreters := map[string]bool{"sh": true, "bash": true, "dash": true, "zsh": true, "fish": true, "cmd": true, "powershell": true, "pwsh": true, "python": true, "python3": true, "perl": true, "ruby": true, "node": true}
	if interpreters[base] {
		return fmt.Errorf("%w: interpreter registration denied", ErrAuthority)
	}
	for _, arg := range command.Args {
		if len(arg) > 4096 || strings.ContainsRune(arg, 0) {
			return fmt.Errorf("%w: invalid fixed argument", ErrAuthority)
		}
	}
	return rehashExecutable(command)
}

func rehashExecutable(command Command) error {
	info, err := os.Lstat(command.Executable)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("%w: executable unavailable", ErrUnavailable)
	}
	f, err := os.Open(command.Executable)
	if err != nil {
		return fmt.Errorf("%w: executable unreadable", ErrUnavailable)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return fmt.Errorf("%w: executable unreadable", ErrUnavailable)
	}
	if hex.EncodeToString(h.Sum(nil)) != command.SHA256 {
		return fmt.Errorf("%w: executable digest", ErrChanged)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("%w: executable unreadable", ErrUnavailable)
	}
	if err := verifyNativeBinary(f); err != nil {
		return fmt.Errorf("%w: executable is not a native binary", ErrAuthority)
	}
	return nil
}

func verifyNativeBinary(r io.ReaderAt) error {
	if f, err := elf.NewFile(r); err == nil {
		f.Close()
		return nil
	}
	if f, err := pe.NewFile(r); err == nil {
		f.Close()
		return nil
	}
	if f, err := macho.NewFile(r); err == nil {
		f.Close()
		return nil
	}
	return errors.New("unknown executable format")
}

type boundedBuffer struct {
	bytes.Buffer
	limit    int
	overflow bool
}

func newBoundedBuffer(limit int) *boundedBuffer { return &boundedBuffer{limit: limit} }

func (b *boundedBuffer) Write(p []byte) (int, error) {
	want := len(p)
	remaining := b.limit - b.Len()
	if remaining <= 0 {
		b.overflow = true
		return want, nil
	}
	if len(p) > remaining {
		b.overflow = true
		p = p[:remaining]
	}
	_, _ = b.Buffer.Write(p)
	return want, nil
}

func ptrGrant(g Grant) *Grant { c := cloneGrant(g); return &c }

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalPathPrefixes(a, b []PathPrefix) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalGrantBytes(a, b Grant) bool {
	ra, _ := json.Marshal(a)
	rb, _ := json.Marshal(b)
	return bytes.Equal(ra, rb)
}
