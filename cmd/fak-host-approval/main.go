// Command fak-host-approval prepares and verifies immutable host-approval
// requests. It never approves or executes a request: authorization requires a
// protected broker that is deliberately outside this process.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hostapproval"
)

const maxRequestBytes = 1 << 20

type verificationResult struct {
	Schema          string `json:"schema"`
	Authorized      bool   `json:"authorized"`
	BrokerRequired  bool   `json:"broker_required"`
	BrokerAvailable bool   `json:"broker_available"`
	RequestSHA256   string `json:"request_sha256"`
	SessionID       string `json:"session_id"`
}

type statusResult struct {
	Schema                                      string `json:"schema"`
	Authorized                                  bool   `json:"authorized"`
	BrokerRequired                              bool   `json:"broker_required"`
	BrokerAvailable                             bool   `json:"broker_available"`
	Platform                                    string `json:"platform"`
	WebAuthnAPIAvailable                        bool   `json:"webauthn_api_available"`
	WebAuthnAPIVersion                          uint32 `json:"webauthn_api_version,omitempty"`
	UserVerifyingPlatformAuthenticatorAvailable bool   `json:"user_verifying_platform_authenticator_available"`
	ProbeError                                  string `json:"probe_error,omitempty"`
}

type platformProbe struct {
	apiAvailable       bool
	apiVersion         uint32
	authenticatorReady bool
	err                error
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		usage(stderr)
		return 2
	}
	switch args[0] {
	case "prepare":
		return runPrepare(args[1:], stdout, stderr)
	case "verify":
		return runVerify(args[1:], stdin, stdout, stderr)
	case "status":
		return runStatus(args[1:], stdout, stderr)
	default:
		fmt.Fprintf(stderr, "fak-host-approval: unknown command %q\n", args[0])
		usage(stderr)
		return 2
	}
}

func runPrepare(args []string, stdout, stderr io.Writer) int {
	if err := rejectDuplicateFlags(args); err != nil {
		fmt.Fprintf(stderr, "fak-host-approval prepare: %v\n", err)
		return 2
	}
	fs := flag.NewFlagSet("fak-host-approval prepare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	installer := fs.String("installer", "", "immutable installer identity")
	policy := fs.String("policy", "", "immutable policy identity")
	context := fs.String("context", "", "immutable context identity")
	codexConfig := fs.String("codex-config", "", "immutable Codex config identity")
	server := fs.String("server", "", "server identity")
	taskID := fs.String("task", "", "task identity")
	sessionID := fs.String("session", "", "session identity")
	ttl := fs.Duration("ttl", 0, "bounded request lifetime")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak-host-approval prepare: positional arguments are not accepted")
		return 2
	}
	req, err := hostapproval.Prepare(hostapproval.Options{
		Installer:   *installer,
		Policy:      *policy,
		Context:     *context,
		CodexConfig: *codexConfig,
		Server:      *server,
		TaskID:      *taskID,
		SessionID:   *sessionID,
		TTL:         *ttl,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak-host-approval prepare: %v\n", err)
		return 1
	}
	digest, err := hostapproval.Digest(req)
	if err != nil {
		fmt.Fprintf(stderr, "fak-host-approval prepare: digest: %v\n", err)
		return 1
	}
	if err := json.NewEncoder(stdout).Encode(req); err != nil {
		fmt.Fprintf(stderr, "fak-host-approval prepare: encode: %v\n", err)
		return 1
	}
	fmt.Fprintf(stderr, "request_sha256=%s\n", digest)
	fmt.Fprintln(stderr, "authorized=false broker_required=true broker_available=false nonce_status=untrusted_pending_broker")
	return 0
}

func runVerify(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if err := rejectDuplicateFlags(args); err != nil {
		fmt.Fprintf(stderr, "fak-host-approval verify: %v\n", err)
		return 2
	}
	fs := flag.NewFlagSet("fak-host-approval verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sessionID := fs.String("session", "", "expected session identity")
	wantDigest := fs.String("request-sha256", "", "expected immutable request SHA-256")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || strings.TrimSpace(*sessionID) == "" || strings.TrimSpace(*wantDigest) == "" {
		fmt.Fprintln(stderr, "usage: fak-host-approval verify --session ID --request-sha256 SHA256 < request.json")
		return 2
	}
	raw, err := readBounded(stdin, maxRequestBytes)
	if err != nil {
		fmt.Fprintf(stderr, "fak-host-approval verify: %v\n", err)
		return 1
	}
	if err := rejectDuplicateJSONFields(raw); err != nil {
		fmt.Fprintf(stderr, "fak-host-approval verify: decode: %v\n", err)
		return 1
	}
	req, err := hostapproval.Decode(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak-host-approval verify: decode: %v\n", err)
		return 1
	}
	digest, err := hostapproval.Digest(req)
	if err != nil {
		fmt.Fprintf(stderr, "fak-host-approval verify: digest: %v\n", err)
		return 1
	}
	if digest != strings.TrimSpace(*wantDigest) {
		fmt.Fprintln(stderr, "fak-host-approval verify: request digest mismatch")
		return 1
	}
	if err := hostapproval.Verify(req, *sessionID); err != nil {
		fmt.Fprintf(stderr, "fak-host-approval verify: %v\n", err)
		return 1
	}
	result := verificationResult{
		Schema:          "fak.host-approval.verification.v1",
		Authorized:      false,
		BrokerRequired:  true,
		BrokerAvailable: false,
		RequestSHA256:   digest,
		SessionID:       *sessionID,
	}
	if err := json.NewEncoder(stdout).Encode(result); err != nil {
		fmt.Fprintf(stderr, "fak-host-approval verify: encode: %v\n", err)
		return 1
	}
	return 0
}

// rejectDuplicateFlags removes "last value wins" ambiguity from the authority
// envelope. Every supported flag takes exactly one value.
func rejectDuplicateFlags(args []string) error {
	seen := make(map[string]bool)
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			break
		}
		if !strings.HasPrefix(arg, "-") || arg == "-" {
			continue
		}
		nameValue := strings.TrimLeft(arg, "-")
		name, _, inline := strings.Cut(nameValue, "=")
		if seen[name] {
			return fmt.Errorf("duplicate flag --%s", name)
		}
		seen[name] = true
		if !inline && i+1 < len(args) {
			i++
		}
	}
	return nil
}

func rejectDuplicateJSONFields(raw []byte) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	var scan func() error
	scan = func() error {
		tok, err := dec.Token()
		if err != nil {
			return err
		}
		delim, ok := tok.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := make(map[string]bool)
			for dec.More() {
				keyToken, err := dec.Token()
				if err != nil {
					return err
				}
				key, ok := keyToken.(string)
				if !ok {
					return errors.New("object key is not a string")
				}
				if seen[key] {
					return fmt.Errorf("duplicate JSON field %q", key)
				}
				seen[key] = true
				if err := scan(); err != nil {
					return err
				}
			}
			_, err := dec.Token()
			return err
		case '[':
			for dec.More() {
				if err := scan(); err != nil {
					return err
				}
			}
			_, err := dec.Token()
			return err
		default:
			return fmt.Errorf("unexpected JSON delimiter %q", delim)
		}
	}
	return scan()
}

func runStatus(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("fak-host-approval status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak-host-approval status: arguments are not accepted")
		return 2
	}
	p := probePlatform()
	result := statusResult{
		Schema:               "fak.host-approval.status.v1",
		Authorized:           false,
		BrokerRequired:       true,
		BrokerAvailable:      false,
		Platform:             runtime.GOOS,
		WebAuthnAPIAvailable: p.apiAvailable,
		WebAuthnAPIVersion:   p.apiVersion,
		UserVerifyingPlatformAuthenticatorAvailable: p.authenticatorReady,
	}
	if p.err != nil {
		result.ProbeError = p.err.Error()
	}
	if err := json.NewEncoder(stdout).Encode(result); err != nil {
		fmt.Fprintf(stderr, "fak-host-approval status: encode: %v\n", err)
		return 1
	}
	return 0
}

func readBounded(r io.Reader, limit int64) ([]byte, error) {
	if r == nil {
		return nil, errors.New("request input is required")
	}
	raw, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) > limit {
		return nil, fmt.Errorf("request exceeds %d bytes", limit)
	}
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, errors.New("request input is empty")
	}
	return raw, nil
}

func usage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak-host-approval prepare|verify|status")
}
