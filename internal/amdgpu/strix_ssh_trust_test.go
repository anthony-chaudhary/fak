package amdgpu

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestStrixKnownHostsBrokerProtocol(t *testing.T) {
	key := bytes.Repeat([]byte{0x5a}, 32)
	blob := make([]byte, 4+len(StrixKnownHostsKeyType)+4+len(key))
	binary.BigEndian.PutUint32(blob[:4], uint32(len(StrixKnownHostsKeyType)))
	copy(blob[4:], StrixKnownHostsKeyType)
	off := 4 + len(StrixKnownHostsKeyType)
	binary.BigEndian.PutUint32(blob[off:off+4], uint32(len(key)))
	copy(blob[off+4:], key)
	digest := sha256.Sum256(blob)
	fingerprint := "SHA256:" + base64.RawStdEncoding.EncodeToString(digest[:])
	line := StrixKnownHostsAlias + " " + StrixKnownHostsKeyType + " " + base64.StdEncoding.EncodeToString(blob) + "\n"

	t.Run("canonical success exact stdout and cleanup", func(t *testing.T) {
		path := writeTrustFile(t, line)
		loaded, err := loadStrixKnownHosts(path, fingerprint)
		if err != nil || loaded != line {
			t.Fatalf("load = %q, %v", loaded, err)
		}
		broker, err := startStrixKnownHostsBroker(loaded, 2, time.Minute, time.Now)
		if err != nil {
			t.Fatal(err)
		}
		var stdout bytes.Buffer
		if err := runStrixKnownHostsBrokerChild(broker.endpoint, broker.cap, &stdout, fingerprint); err != nil {
			t.Fatal(err)
		}
		if stdout.String() != line {
			t.Fatalf("stdout = %q, want exact canonical line", stdout.String())
		}
		if err := broker.Close(); err != nil {
			t.Fatal(err)
		}
		if err := broker.Close(); err != nil {
			t.Fatalf("idempotent close: %v", err)
		}
		stdout.Reset()
		if err := runStrixKnownHostsBrokerChild(broker.endpoint, broker.cap, &stdout, fingerprint); err == nil || stdout.Len() != 0 {
			t.Fatalf("closed broker err=%v stdout=%q", err, stdout.String())
		}
	})

	badEntries := map[string]string{
		"missing newline":    strings.TrimSuffix(line, "\n"),
		"duplicate":          line + line,
		"wrong alias":        "strix1 " + strings.TrimPrefix(line, StrixKnownHostsAlias+" "),
		"wrong type":         strings.Replace(line, StrixKnownHostsKeyType, "ssh-rsa", 1),
		"tampered blob":      strings.Replace(line, "Wlpa", "WVpa", 1),
		"comment":            strings.TrimSuffix(line, "\n") + " comment\n",
		"noncanonical space": strings.Replace(line, " ", "  ", 1),
	}
	for name, raw := range badEntries {
		t.Run(name, func(t *testing.T) {
			_, err := canonicalizeStrixKnownHosts(raw, fingerprint)
			assertTrustRefusedRedacted(t, err, raw, fingerprint)
		})
	}
	t.Run("wrong fingerprint", func(t *testing.T) {
		_, err := canonicalizeStrixKnownHosts(line, "SHA256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
		assertTrustRefusedRedacted(t, err, line, fingerprint)
	})
	t.Run("missing and unsafe trust file", func(t *testing.T) {
		_, err := loadStrixKnownHosts(filepath.Join(t.TempDir(), "missing"), fingerprint)
		assertTrustRefusedRedacted(t, err, "missing", fingerprint)
		if runtime.GOOS != "windows" {
			target := writeTrustFile(t, line)
			link := filepath.Join(t.TempDir(), "link")
			if err := os.Symlink(target, link); err != nil {
				t.Fatal(err)
			}
			_, err = loadStrixKnownHosts(link, fingerprint)
			assertTrustRefusedRedacted(t, err, link, fingerprint)
		}
	})
	t.Run("ingress duplicate case variant and relative", func(t *testing.T) {
		for _, env := range [][]string{
			{},
			{StrixKnownHostsFileEnv + "=relative"},
			{StrixKnownHostsFileEnv + "=C:\\one", "fak_strix_known_hosts_file=C:\\two"},
		} {
			_, err := strixKnownHostsPath(env)
			assertTrustRefusedRedacted(t, err, strings.Join(env, ";"), fingerprint)
		}
	})
	t.Run("wrong capability and budget exhaustion", func(t *testing.T) {
		broker, err := startStrixKnownHostsBroker(line, 2, time.Minute, time.Now)
		if err != nil {
			t.Fatal(err)
		}
		defer broker.Close()
		wrong := strings.Repeat("A", 43)
		for i := 0; i < 3; i++ {
			var stdout bytes.Buffer
			capability := broker.cap
			if i == 0 {
				capability = wrong
			}
			err := runStrixKnownHostsBrokerChild(broker.endpoint, capability, &stdout, fingerprint)
			if i == 1 && err != nil {
				t.Fatalf("one remaining authorized read refused: %v", err)
			}
			if i != 1 && (err == nil || stdout.Len() != 0) {
				t.Fatalf("attempt %d err=%v stdout=%q", i, err, stdout.String())
			}
		}
	})
	t.Run("expired capability", func(t *testing.T) {
		now := time.Unix(100, 0)
		clock := func() time.Time { return now }
		broker, err := startStrixKnownHostsBroker(line, 1, time.Second, clock)
		if err != nil {
			t.Fatal(err)
		}
		defer broker.Close()
		now = now.Add(2 * time.Second)
		var stdout bytes.Buffer
		err = runStrixKnownHostsBrokerChild(broker.endpoint, broker.cap, &stdout, fingerprint)
		if err == nil || stdout.Len() != 0 {
			t.Fatalf("expired capability err=%v stdout=%q", err, stdout.String())
		}
	})
	t.Run("bounded malformed endpoint capability and hostile executable", func(t *testing.T) {
		for _, endpoint := range []string{"localhost:22", "127.0.0.1:0", "127.0.0.1:01", "[::1]:22", "127.0.0.1:22 injected"} {
			var stdout bytes.Buffer
			err := runStrixKnownHostsBrokerChild(endpoint, strings.Repeat("A", 43), &stdout, fingerprint)
			if err == nil || stdout.Len() != 0 {
				t.Fatalf("endpoint %q err=%v stdout=%q", endpoint, err, stdout.String())
			}
		}
		for _, path := range []string{"relative", filepath.Join(t.TempDir(), "bad%h"), filepath.Join(t.TempDir(), "bad\npath")} {
			assertTrustRefusedRedacted(t, validateStrixExecutable(path), path, fingerprint)
		}
	})
	t.Run("production wrapper is hard pinned", func(t *testing.T) {
		var _ func(string, string, io.Writer) error = RunStrixKnownHostsBrokerChild
		var stdout bytes.Buffer
		err := RunStrixKnownHostsBrokerChild("127.0.0.1:1", strings.Repeat("A", 43), &stdout)
		if err == nil || stdout.Len() != 0 {
			t.Fatalf("production client err=%v stdout=%q", err, stdout.String())
		}
	})
	t.Run("self exec command contains only bounded broker operands", func(t *testing.T) {
		broker, err := startStrixKnownHostsBroker(line, 1, time.Minute, time.Now)
		if err != nil {
			t.Fatal(err)
		}
		command, err := broker.KnownHostsCommand()
		if err != nil {
			t.Fatal(err)
		}
		for _, want := range []string{StrixKnownHostsOperand, broker.endpoint, broker.cap} {
			if !strings.Contains(command, want) {
				t.Fatalf("command lacks fixed broker operand")
			}
		}
		if strings.Contains(command, line) || strings.Contains(command, StrixKnownHostsFileEnv) {
			t.Fatalf("command leaked trust material")
		}
		if err := broker.Close(); err != nil {
			t.Fatal(err)
		}
		if command, err := broker.KnownHostsCommand(); err == nil || command != "" {
			t.Fatalf("closed command = %q, %v", command, err)
		}
	})
	t.Run("child environment removes all trust ingress without mutation", func(t *testing.T) {
		parent := []string{
			"KEEP=one",
			StrixKnownHostsFileEnv + "=C:\\secret-one",
			"fak_strix_known_hosts_file=C:\\secret-two",
			"KEEP_TWO=two",
			"MALFORMED",
		}
		before := append([]string(nil), parent...)
		got := StrixKnownHostsChildEnvironment(parent)
		want := []string{"KEEP=one", "KEEP_TWO=two", "MALFORMED"}
		if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
			t.Fatalf("child environment = %#v, want %#v", got, want)
		}
		if strings.Join(parent, "\x00") != strings.Join(before, "\x00") {
			t.Fatalf("parent environment mutated: %#v", parent)
		}
		got[0] = "CHANGED=yes"
		if parent[0] != "KEEP=one" {
			t.Fatal("returned environment aliases parent storage")
		}
	})
	t.Run("nil zero and malformed broker lifecycle refuses without panic", func(t *testing.T) {
		var nilBroker *StrixKnownHostsBroker
		if command, err := nilBroker.KnownHostsCommand(); command != "" || !errors.Is(err, ErrStrixHostTrustRefused) {
			t.Fatalf("nil command = %q, %v", command, err)
		}
		if err := nilBroker.Close(); !errors.Is(err, ErrStrixHostTrustRefused) {
			t.Fatalf("nil close = %v", err)
		}
		zero := &StrixKnownHostsBroker{}
		if command, err := zero.KnownHostsCommand(); command != "" || !errors.Is(err, ErrStrixHostTrustRefused) {
			t.Fatalf("zero command = %q, %v", command, err)
		}
		if err := zero.Close(); !errors.Is(err, ErrStrixHostTrustRefused) {
			t.Fatalf("zero close = %v", err)
		}
		malformed := &StrixKnownHostsBroker{endpoint: "external.example:22"}
		if command, err := malformed.KnownHostsCommand(); command != "" || !errors.Is(err, ErrStrixHostTrustRefused) {
			t.Fatalf("malformed command = %q, %v", command, err)
		}
		if err := malformed.Close(); !errors.Is(err, ErrStrixHostTrustRefused) {
			t.Fatalf("malformed close = %v", err)
		}
	})
	t.Run("expiry closes listener and remains idempotently closable", func(t *testing.T) {
		expiry := make(chan time.Time, 1)
		after := func(time.Duration) <-chan time.Time { return expiry }
		broker, err := startStrixKnownHostsBrokerWithExpiry(line, 1, time.Minute, time.Now, after)
		if err != nil {
			t.Fatal(err)
		}
		endpoint, capability := broker.endpoint, broker.cap
		expiry <- time.Now()
		select {
		case <-broker.done:
		case <-time.After(time.Second):
			t.Fatal("broker listener did not close at expiry")
		}
		var stdout bytes.Buffer
		if err := runStrixKnownHostsBrokerChild(endpoint, capability, &stdout, fingerprint); err == nil || stdout.Len() != 0 {
			t.Fatalf("expired listener err=%v stdout=%q", err, stdout.String())
		}
		if err := broker.Close(); err != nil {
			t.Fatalf("close after expiry: %v", err)
		}
		if err := broker.Close(); err != nil {
			t.Fatalf("repeated close after expiry: %v", err)
		}
	})
}

func writeTrustFile(t *testing.T, line string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "known_hosts")
	if err := os.WriteFile(path, []byte(line), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func assertTrustRefusedRedacted(t *testing.T, err error, secrets ...string) {
	t.Helper()
	if !errors.Is(err, ErrStrixHostTrustRefused) {
		t.Fatalf("err = %v, want typed refusal", err)
	}
	message := err.Error()
	for _, secret := range secrets {
		if secret != "" && strings.Contains(message, secret) {
			t.Fatalf("error leaked sensitive input: %q", message)
		}
	}
}
