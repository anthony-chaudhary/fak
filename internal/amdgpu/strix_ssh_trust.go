package amdgpu

import (
	"bufio"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	StrixKnownHostsAlias       = "strix-halo-fak.local"
	StrixKnownHostsKeyType     = "ssh-ed25519"
	StrixKnownHostsFingerprint = "SHA256:EnLBzJUqXmehYkHiniCkd/6m92HfS13memYJB2jbWOc"
	StrixKnownHostsFileEnv     = "FAK_STRIX_KNOWN_HOSTS_FILE"
	StrixKnownHostsOperand     = "__fak_strix_known_hosts"

	strixKnownHostsMaxFileBytes = 4096
	strixKnownHostsMaxReads     = 4
	strixKnownHostsLifetime     = 30 * time.Second
	strixKnownHostsIOTimeout    = 2 * time.Second
)

var ErrStrixHostTrustRefused = errors.New("STRIX_HOST_TRUST_REFUSED")

type strixHostTrustError struct{ reason string }

func (e *strixHostTrustError) Error() string {
	return ErrStrixHostTrustRefused.Error() + ": " + e.reason
}

func (e *strixHostTrustError) Unwrap() error { return ErrStrixHostTrustRefused }

func strixTrustRefused(reason string) error { return &strixHostTrustError{reason: reason} }

// StrixKnownHostsBroker is an invocation-scoped, read-only holder for a validated
// Strix host-key line. It never retains the source path after construction.
type StrixKnownHostsBroker struct {
	listener net.Listener
	endpoint string
	cap      string
	line     string
	expires  time.Time
	maxReads int
	now      func() time.Time

	mu           sync.Mutex
	reads        int
	closed       bool
	closeErr     error
	done         chan struct{}
	expiryCancel chan struct{}
	once         sync.Once
}

// StrixKnownHostsChildEnvironment returns a fresh environment with every exact
// or case-variant trust-file ingress removed. It never mutates env or the process
// environment, so a broker child cannot inherit the parent-only file authority.
func StrixKnownHostsChildEnvironment(env []string) []string {
	clean := make([]string, 0, len(env))
	for _, item := range env {
		name, _, ok := strings.Cut(item, "=")
		if ok && strings.EqualFold(name, StrixKnownHostsFileEnv) {
			continue
		}
		clean = append(clean, item)
	}
	return clean
}

// StartStrixKnownHostsBroker reads the dedicated trust file exactly once and
// starts a short-lived loopback broker pinned to the production fingerprint.
func StartStrixKnownHostsBroker() (*StrixKnownHostsBroker, error) {
	path, err := strixKnownHostsPath(os.Environ())
	if err != nil {
		return nil, err
	}
	line, err := loadStrixKnownHosts(path, StrixKnownHostsFingerprint)
	if err != nil {
		return nil, err
	}
	exe, err := os.Executable()
	if err != nil {
		return nil, strixTrustRefused("unsafe executable")
	}
	if err := validateStrixExecutable(exe); err != nil {
		return nil, err
	}
	return startStrixKnownHostsBroker(line, strixKnownHostsMaxReads, strixKnownHostsLifetime, time.Now)
}

func strixKnownHostsPath(env []string) (string, error) {
	var value string
	seen := false
	for _, item := range env {
		name, val, ok := strings.Cut(item, "=")
		if !ok || !strings.EqualFold(name, StrixKnownHostsFileEnv) {
			continue
		}
		if seen || name != StrixKnownHostsFileEnv || val == "" {
			return "", strixTrustRefused("invalid trust ingress")
		}
		seen, value = true, val
	}
	if !seen || !filepath.IsAbs(value) || filepath.Clean(value) != value {
		return "", strixTrustRefused("invalid trust ingress")
	}
	return value, nil
}

func loadStrixKnownHosts(path, expectedFingerprint string) (string, error) {
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() {
		return "", strixTrustRefused("unsafe trust file")
	}
	f, err := os.Open(path)
	if err != nil {
		return "", strixTrustRefused("trust read failed")
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		return "", strixTrustRefused("unsafe trust file")
	}
	data, err := io.ReadAll(io.LimitReader(f, strixKnownHostsMaxFileBytes+1))
	if err != nil || len(data) == 0 || len(data) > strixKnownHostsMaxFileBytes {
		return "", strixTrustRefused("invalid trust snapshot")
	}
	after, err := os.Lstat(path)
	if err != nil || !after.Mode().IsRegular() || !os.SameFile(opened, after) {
		return "", strixTrustRefused("unsafe trust file")
	}
	line, err := canonicalizeStrixKnownHosts(string(data), expectedFingerprint)
	if err != nil {
		return "", err
	}
	return line, nil
}

func canonicalizeStrixKnownHosts(raw, expectedFingerprint string) (string, error) {
	if strings.Contains(raw, "\r") || strings.Count(raw, "\n") != 1 || !strings.HasSuffix(raw, "\n") {
		return "", strixTrustRefused("noncanonical trust entry")
	}
	line := strings.TrimSuffix(raw, "\n")
	fields := strings.Split(line, " ")
	if len(fields) != 3 || fields[0] != StrixKnownHostsAlias || fields[1] != StrixKnownHostsKeyType || fields[2] == "" {
		return "", strixTrustRefused("invalid trust entry")
	}
	blob, err := base64.StdEncoding.Strict().DecodeString(fields[2])
	if err != nil || base64.StdEncoding.EncodeToString(blob) != fields[2] || !validED25519SSHBlob(blob) {
		return "", strixTrustRefused("invalid trust key")
	}
	digest := sha256.Sum256(blob)
	fingerprint := "SHA256:" + base64.RawStdEncoding.EncodeToString(digest[:])
	if subtle.ConstantTimeCompare([]byte(fingerprint), []byte(expectedFingerprint)) != 1 {
		return "", strixTrustRefused("host-key pin mismatch")
	}
	return line + "\n", nil
}

func validED25519SSHBlob(blob []byte) bool {
	if len(blob) != 4+len(StrixKnownHostsKeyType)+4+32 {
		return false
	}
	nameLen := int(binary.BigEndian.Uint32(blob[:4]))
	if nameLen != len(StrixKnownHostsKeyType) || string(blob[4:4+nameLen]) != StrixKnownHostsKeyType {
		return false
	}
	keyOffset := 4 + nameLen
	return binary.BigEndian.Uint32(blob[keyOffset:keyOffset+4]) == 32
}

func startStrixKnownHostsBroker(line string, maxReads int, lifetime time.Duration, now func() time.Time) (*StrixKnownHostsBroker, error) {
	return startStrixKnownHostsBrokerWithExpiry(line, maxReads, lifetime, now, time.After)
}

func startStrixKnownHostsBrokerWithExpiry(line string, maxReads int, lifetime time.Duration, now func() time.Time, after func(time.Duration) <-chan time.Time) (*StrixKnownHostsBroker, error) {
	if maxReads < 1 || maxReads > strixKnownHostsMaxReads || lifetime <= 0 || now == nil {
		return nil, strixTrustRefused("invalid broker limits")
	}
	if after == nil {
		return nil, strixTrustRefused("invalid broker limits")
	}
	expiry := after(lifetime)
	if expiry == nil {
		return nil, strixTrustRefused("invalid broker limits")
	}
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		return nil, strixTrustRefused("broker listener failed")
	}
	capBytes := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, capBytes); err != nil {
		_ = ln.Close()
		return nil, strixTrustRefused("broker capability failed")
	}
	b := &StrixKnownHostsBroker{
		listener:     ln,
		endpoint:     ln.Addr().String(),
		cap:          base64.RawURLEncoding.EncodeToString(capBytes),
		line:         line,
		expires:      now().Add(lifetime),
		maxReads:     maxReads,
		now:          now,
		done:         make(chan struct{}),
		expiryCancel: make(chan struct{}),
	}
	go b.serve()
	go b.expire(expiry)
	return b, nil
}

// KnownHostsCommand returns the fixed self-exec invocation consumed by OpenSSH.
// Only the executable, hidden operand, loopback endpoint, and capability appear.
func (b *StrixKnownHostsBroker) KnownHostsCommand() (string, error) {
	if b == nil {
		return "", strixTrustRefused("invalid broker")
	}
	exe, err := os.Executable()
	if err != nil {
		return "", strixTrustRefused("unsafe executable")
	}
	if err := validateStrixExecutable(exe); err != nil {
		return "", err
	}
	b.mu.Lock()
	if !b.validLocked() || b.closed || !validBrokerCapability(b.cap) || b.line == "" || !b.now().Before(b.expires) {
		b.mu.Unlock()
		return "", strixTrustRefused("broker unavailable")
	}
	endpoint, capability := b.endpoint, b.cap
	b.mu.Unlock()
	return strings.Join([]string{
		quoteOpenSSHArg(exe),
		quoteOpenSSHArg(StrixKnownHostsOperand),
		quoteOpenSSHArg(endpoint),
		quoteOpenSSHArg(capability),
	}, " "), nil
}

func validateStrixExecutable(path string) error {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || strings.ContainsAny(path, "\x00\r\n%") {
		return strixTrustRefused("unsafe executable")
	}
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() {
		return strixTrustRefused("unsafe executable")
	}
	return nil
}

func quoteOpenSSHArg(value string) string {
	value = strings.ReplaceAll(value, "\\", "\\\\")
	value = strings.ReplaceAll(value, "\"", "\\\"")
	return "\"" + value + "\""
}

func (b *StrixKnownHostsBroker) serve() {
	defer close(b.done)
	for {
		conn, err := b.listener.Accept()
		if err != nil {
			return
		}
		b.handle(conn)
	}
}

func (b *StrixKnownHostsBroker) expire(expiry <-chan time.Time) {
	select {
	case <-expiry:
		_ = b.shutdown()
	case <-b.expiryCancel:
	}
}

func (b *StrixKnownHostsBroker) handle(conn net.Conn) {
	defer conn.Close()
	remote, ok := conn.RemoteAddr().(*net.TCPAddr)
	if !ok || !remote.IP.IsLoopback() {
		return
	}
	_ = conn.SetDeadline(time.Now().Add(strixKnownHostsIOTimeout))
	reader := bufio.NewReader(io.LimitReader(conn, 256))
	request, err := reader.ReadString('\n')
	if err != nil || len(request) > 128 || !strings.HasSuffix(request, "\n") {
		return
	}
	provided := strings.TrimSuffix(request, "\n")
	b.mu.Lock()
	allowed := !b.closed && b.now().Before(b.expires) && b.reads < b.maxReads
	if allowed {
		b.reads++
	}
	expectedCapability, line := b.cap, b.line
	b.mu.Unlock()
	if !allowed || len(provided) != len(expectedCapability) || subtle.ConstantTimeCompare([]byte(provided), []byte(expectedCapability)) != 1 {
		return
	}
	_, _ = io.WriteString(conn, line)
}

// Close destroys the invocation capability and stops the listener. It is idempotent.
func (b *StrixKnownHostsBroker) Close() error {
	if b == nil {
		return strixTrustRefused("invalid broker")
	}
	b.mu.Lock()
	valid := b.validLocked()
	b.mu.Unlock()
	if !valid {
		return strixTrustRefused("invalid broker")
	}
	return b.shutdown()
}

func (b *StrixKnownHostsBroker) validLocked() bool {
	return b.listener != nil && b.endpoint != "" && validBrokerEndpoint(b.endpoint) &&
		b.maxReads >= 1 && b.maxReads <= strixKnownHostsMaxReads && b.now != nil &&
		!b.expires.IsZero() && b.done != nil && b.expiryCancel != nil
}

func (b *StrixKnownHostsBroker) shutdown() error {
	b.once.Do(func() {
		b.mu.Lock()
		b.closed = true
		b.cap = ""
		b.line = ""
		b.mu.Unlock()
		close(b.expiryCancel)
		if err := b.listener.Close(); err != nil && !errors.Is(err, net.ErrClosed) {
			b.closeErr = strixTrustRefused("broker cleanup failed")
		}
		<-b.done
	})
	return b.closeErr
}

// RunStrixKnownHostsBrokerChild fetches one entry from the invocation broker,
// independently checks the production pin, and writes exactly one canonical line.
func RunStrixKnownHostsBrokerChild(endpoint, capability string, stdout io.Writer) error {
	return runStrixKnownHostsBrokerChild(endpoint, capability, stdout, StrixKnownHostsFingerprint)
}

func runStrixKnownHostsBrokerChild(endpoint, capability string, stdout io.Writer, expectedFingerprint string) error {
	if stdout == nil || !validBrokerEndpoint(endpoint) || !validBrokerCapability(capability) {
		return strixTrustRefused("invalid broker request")
	}
	conn, err := net.DialTimeout("tcp4", endpoint, strixKnownHostsIOTimeout)
	if err != nil {
		return strixTrustRefused("broker unavailable")
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(strixKnownHostsIOTimeout))
	if _, err := io.WriteString(conn, capability+"\n"); err != nil {
		return strixTrustRefused("broker request failed")
	}
	data, err := io.ReadAll(io.LimitReader(conn, strixKnownHostsMaxFileBytes+1))
	if err != nil || len(data) == 0 || len(data) > strixKnownHostsMaxFileBytes {
		return strixTrustRefused("broker response refused")
	}
	line, err := canonicalizeStrixKnownHosts(string(data), expectedFingerprint)
	if err != nil {
		return err
	}
	if _, err := io.WriteString(stdout, line); err != nil {
		return strixTrustRefused("broker output failed")
	}
	return nil
}

func validBrokerEndpoint(endpoint string) bool {
	if len(endpoint) < 3 || len(endpoint) > 64 || strings.ContainsAny(endpoint, "\x00\r\n \t") {
		return false
	}
	host, portText, err := net.SplitHostPort(endpoint)
	if err != nil || host != "127.0.0.1" {
		return false
	}
	port, err := strconv.Atoi(portText)
	return err == nil && port >= 1 && port <= 65535 && strconv.Itoa(port) == portText
}

func validBrokerCapability(capability string) bool {
	if len(capability) != 43 {
		return false
	}
	decoded, err := base64.RawURLEncoding.Strict().DecodeString(capability)
	return err == nil && len(decoded) == 32 && base64.RawURLEncoding.EncodeToString(decoded) == capability
}
