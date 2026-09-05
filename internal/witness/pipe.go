package witness

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// PipeReceiptSchema identifies the schema version of PipeReceipt.
const PipeReceiptSchema = "fak.pipe-receipt.v1"

// PipeReceipt captures the unforgeable stdout/stderr SHA-256 digests, byte
// counts, duration, and exit code from a subprocess execution.
type PipeReceipt struct {
	Schema        string   `json:"schema"`
	Command       []string `json:"command"`
	ExitCode      int      `json:"exit_code"`
	StdoutSHA256  string   `json:"stdout_sha256"`
	StderrSHA256  string   `json:"stderr_sha256"`
	StdoutBytes   int64    `json:"stdout_bytes"`
	StderrBytes   int64    `json:"stderr_bytes"`
	DurationNanos int64    `json:"duration_nanos"`
	Error         string   `json:"error,omitempty"`
	Attestation   string   `json:"attestation,omitempty"`
	Signature     string   `json:"signature,omitempty"`

	executed bool
}

// WitnessOutcome maps the receipt outcome into the kernel's three-way witness contract.
func (r PipeReceipt) WitnessOutcome() abi.WitnessOutcome {
	if r.Error != "" {
		return abi.WitnessAbstain
	}
	if !r.executed && !VerifyPipeReceiptAttestation(&r, nil) {
		return abi.WitnessAbstain
	}
	if r.ExitCode == 0 {
		return abi.WitnessConfirmed
	}
	return abi.WitnessRefuted
}

var (
	pipeAuthKeyOnce sync.Once
	pipeAuthKey     []byte
)

func getPipeAuthKey() []byte {
	pipeAuthKeyOnce.Do(func() {
		if envSec := os.Getenv("FAK_PIPE_AUTH_KEY"); envSec != "" {
			h := sha256.Sum256([]byte(envSec))
			pipeAuthKey = h[:]
			return
		}
		pipeAuthKey = make([]byte, 32)
		if _, err := rand.Read(pipeAuthKey); err != nil {
			h := sha256.Sum256([]byte("fak.pipe.attestation.default.secret"))
			pipeAuthKey = h[:]
		}
	})
	return pipeAuthKey
}

// SignPipeReceipt generates an HMAC-SHA256 attestation over the receipt execution fields.
// If key is empty or nil, the internal auth key is used.
func SignPipeReceipt(receipt *PipeReceipt, key []byte) string {
	if receipt == nil {
		return ""
	}
	if len(key) == 0 {
		key = getPipeAuthKey()
	}
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(PipeReceiptSchema))
	mac.Write([]byte{0})
	for _, arg := range receipt.Command {
		mac.Write([]byte(arg))
		mac.Write([]byte{0})
	}
	fmt.Fprintf(mac, "%d:%s:%s:%d:%d:%d:%s",
		receipt.ExitCode,
		receipt.StdoutSHA256,
		receipt.StderrSHA256,
		receipt.StdoutBytes,
		receipt.StderrBytes,
		receipt.DurationNanos,
		receipt.Error,
	)
	return hex.EncodeToString(mac.Sum(nil))
}

// VerifyPipeReceiptAttestation checks whether receipt carries a valid signature or attestation.
// If key is empty or nil, the internal witness secret is used.
func VerifyPipeReceiptAttestation(receipt *PipeReceipt, key []byte) bool {
	if receipt == nil {
		return false
	}
	sig := receipt.Signature
	if sig == "" {
		sig = receipt.Attestation
	}
	if sig == "" {
		return false
	}
	sig = strings.TrimPrefix(sig, "hmac-sha256:")
	expected := SignPipeReceipt(receipt, key)
	return hmac.Equal([]byte(strings.ToLower(sig)), []byte(strings.ToLower(expected)))
}

func sha256Hex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// PipeClaimSpec describes expected subprocess execution properties or an already-captured receipt.
type PipeClaimSpec struct {
	Command           []string     `json:"command,omitempty"`
	ExpectedExitCode  int          `json:"expected_exit_code"`
	ExpectedStdoutSHA string       `json:"expected_stdout_sha,omitempty"`
	ExpectedStderrSHA string       `json:"expected_stderr_sha,omitempty"`
	Receipt           *PipeReceipt `json:"receipt,omitempty"`
}

type countingWriter struct {
	w     io.Writer
	count int64
}

func (c *countingWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.count += int64(n)
	return n, err
}

// boundedBuffer caps retained memory buffer to prevent unbounded OOM attacks
// while allowing the hashing stream to process all bytes.
type boundedBuffer struct {
	buf   bytes.Buffer
	limit int
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	remaining := b.limit - b.buf.Len()
	if remaining > 0 {
		if len(p) <= remaining {
			b.buf.Write(p)
		} else {
			b.buf.Write(p[:remaining])
		}
	}
	return len(p), nil
}

func (b *boundedBuffer) Bytes() []byte {
	return b.buf.Bytes()
}

// MaxPipeBufferSize is the maximum bytes retained in memory per stream buffer (4MB).
const MaxPipeBufferSize = 4 << 20 // 4 MiB

// MaxRetainedStreamBytes is the maximum bytes retained in memory per stream.
const MaxRetainedStreamBytes = MaxPipeBufferSize

// RunPipeWitness executes argv[0] with args argv[1:] in dir, connecting stdout and stderr
// to separate SHA-256 hash streams, byte counters, and memory buffers. It returns
// the PipeReceipt, stdout bytes, stderr bytes, and any execution error.
func RunPipeWitness(ctx context.Context, dir string, argv ...string) (PipeReceipt, []byte, []byte, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if len(argv) == 0 || strings.TrimSpace(argv[0]) == "" {
		receipt := PipeReceipt{
			Schema:   PipeReceiptSchema,
			Command:  argv,
			ExitCode: -1,
			Error:    "empty command",
		}
		return receipt, nil, nil, errors.New("empty command")
	}

	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	windowgate.ConfigureBackgroundCommand(cmd)
	procguard.ConfigureProcessTreeCancel(cmd)
	cmd.WaitDelay = 10 * time.Second
	if dir != "" {
		cmd.Dir = dir
		cleanGitEnv(cmd)
	}

	stdoutBuf := &boundedBuffer{limit: MaxPipeBufferSize}
	stderrBuf := &boundedBuffer{limit: MaxPipeBufferSize}
	stdoutHasher := sha256.New()
	stderrHasher := sha256.New()

	stdoutCounter := &countingWriter{w: io.MultiWriter(stdoutBuf, stdoutHasher)}
	stderrCounter := &countingWriter{w: io.MultiWriter(stderrBuf, stderrHasher)}

	cmd.Stdout = stdoutCounter
	cmd.Stderr = stderrCounter

	start := time.Now()
	err := cmd.Run()
	durationNanos := time.Since(start).Nanoseconds()

	exitCode := 0
	errStr := ""
	var retErr error

	if err != nil {
		if ctx.Err() != nil {
			exitCode = -1
			errStr = ctx.Err().Error()
			retErr = ctx.Err()
		} else if ee, ok := err.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
			errStr = ""
			retErr = nil
		} else {
			exitCode = -1
			errStr = err.Error()
			retErr = err
		}
	}

	receipt := PipeReceipt{
		Schema:        PipeReceiptSchema,
		Command:       argv,
		ExitCode:      exitCode,
		StdoutSHA256:  hex.EncodeToString(stdoutHasher.Sum(nil)),
		StderrSHA256:  hex.EncodeToString(stderrHasher.Sum(nil)),
		StdoutBytes:   stdoutCounter.count,
		StderrBytes:   stderrCounter.count,
		DurationNanos: durationNanos,
		Error:         errStr,
		executed:      true,
	}
	receipt.Attestation = SignPipeReceipt(&receipt, nil)
	receipt.Signature = receipt.Attestation

	return receipt, stdoutBuf.Bytes(), stderrBuf.Bytes(), retErr
}

// ValidatePipeReceipt checks whether receipt satisfies spec's expected exit code and output digests.
func ValidatePipeReceipt(receipt *PipeReceipt, spec PipeClaimSpec) (abi.WitnessOutcome, error) {
	if receipt == nil {
		return abi.WitnessAbstain, fmt.Errorf("nil pipe receipt")
	}
	// Verify receipt authenticity: must have genuine OS execution or valid attestation/signature.
	if !receipt.executed && !VerifyPipeReceiptAttestation(receipt, nil) {
		return abi.WitnessAbstain, fmt.Errorf("unverified pipe receipt: requires verified OS execution or valid attestation/signature")
	}
	// If the claim spec includes an inlined receipt, it must match the execution without tampering.
	if spec.Receipt != nil {
		if !spec.Receipt.executed && !VerifyPipeReceiptAttestation(spec.Receipt, nil) {
			return abi.WitnessRefuted, fmt.Errorf("inlined pipe receipt untrusted: unverified self-authored receipt")
		}
		if spec.Receipt.ExitCode != receipt.ExitCode ||
			!strings.EqualFold(spec.Receipt.StdoutSHA256, receipt.StdoutSHA256) ||
			!strings.EqualFold(spec.Receipt.StderrSHA256, receipt.StderrSHA256) {
			return abi.WitnessRefuted, fmt.Errorf("pipe receipt tampered: inlined receipt contradicts execution")
		}
	}
	if receipt.Error != "" {
		return abi.WitnessAbstain, fmt.Errorf("pipe execution failed: %s", receipt.Error)
	}
	if receipt.ExitCode != spec.ExpectedExitCode {
		return abi.WitnessRefuted, fmt.Errorf("exit code mismatch: got %d, want %d", receipt.ExitCode, spec.ExpectedExitCode)
	}
	if spec.ExpectedStdoutSHA != "" && !strings.EqualFold(receipt.StdoutSHA256, strings.TrimSpace(spec.ExpectedStdoutSHA)) {
		return abi.WitnessRefuted, fmt.Errorf("stdout sha256 mismatch: got %s, want %s", receipt.StdoutSHA256, spec.ExpectedStdoutSHA)
	}
	if spec.ExpectedStderrSHA != "" && !strings.EqualFold(receipt.StderrSHA256, strings.TrimSpace(spec.ExpectedStderrSHA)) {
		return abi.WitnessRefuted, fmt.Errorf("stderr sha256 mismatch: got %s, want %s", receipt.StderrSHA256, spec.ExpectedStderrSHA)
	}
	return abi.WitnessConfirmed, nil
}
