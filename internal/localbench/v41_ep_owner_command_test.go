package localbench

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fak-test:runtime fast est=25ms
func TestV41EPOwnerCommandContract(t *testing.T) {
	req := v41EPOwnerRequestFixture()
	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "request.json")
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	validate := []string{"--validate", "--request", path, "--rank", "0"}
	t.Run("valid-input-is-diagnostic-only", func(t *testing.T) {
		for rank, rankArg := range []string{"0", "1", "2"} {
			t.Run(rankArg, func(t *testing.T) {
				var stdout, stderr bytes.Buffer
				if code := RunV41EPOwnerCommand(t.Context(), &stdout, &stderr, []string{"--validate", "--request", path, "--rank", rankArg}); code != 0 || stderr.Len() != 0 {
					t.Fatalf("code=%d stderr=%s", code, stderr.String())
				}
				var got v41EPOwnerCommandDiagnostic
				if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
					t.Fatal(err)
				}
				digest, _ := V41EPOwnerRequestDigest(req)
				if got.Schema != v41EPOwnerCommandSchema || got.Status != "request_valid" || got.RequestSHA256 == nil || *got.RequestSHA256 != digest || got.Rank == nil || *got.Rank != rank ||
					got.DeclaredSourceSHA256 == nil || *got.DeclaredSourceSHA256 != req.SourceSHA256 || got.DeclaredShardSetSHA256 == nil || *got.DeclaredShardSetSHA256 != req.ExpectedShardSetSHA256 ||
					got.DeclaredBackend == nil || *got.DeclaredBackend != req.Owners[rank].Backend || got.DeclaredDevice == nil || *got.DeclaredDevice != req.Owners[rank].ExpectedDevice {
					t.Fatalf("request agreement changed: %+v", got)
				}
				if got.ExecutionAvailable || got.ExecutionPerformed || got.NativeFactoriesEntered || got.InheritedDescriptorsAdopted || got.ObservedExecutableSHA256 != nil ||
					got.ArtifactAuthentication != "unobserved" || got.SupervisorEnforcement != "unobserved" || got.SourceAttestation != "unavailable" || len(got.Unavailable) != 2 {
					t.Fatalf("diagnostic acquired execution/provenance credit: %+v", got)
				}
				for _, forbidden := range []string{V41EPOwnerReceiptSchema, "prefill_ns", "decode_loop_ns", "completed_step_calls", `"status":"complete"`, `"status":"pass"`} {
					if strings.Contains(stdout.String(), forbidden) {
						t.Fatalf("validation emitted measurement %q: %s", forbidden, stdout.String())
					}
				}
			})
		}
	})
	t.Run("supervised-mode-is-unconditionally-refused", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := RunV41EPOwnerCommand(t.Context(), &stdout, &stderr, []string{"--supervised-owner"})
		var got v41EPOwnerCommandDiagnostic
		if code != 1 || json.Unmarshal(stdout.Bytes(), &got) != nil || got.Status != "refused" || got.Operation != "supervised-owner" || got.RequestSHA256 != nil || got.Rank != nil ||
			got.NativeFactoriesEntered || got.InheritedDescriptorsAdopted || !strings.Contains(stderr.String(), "artifact issuer and owner binding are missing") {
			t.Fatalf("reserved mode changed: code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
		}
	})
	t.Run("strict-arguments", func(t *testing.T) {
		cases := [][]string{
			nil, {"--validate"}, {"--rank", "0"}, {"--validate", "--request", path},
			{"--validate", "--request", path, "--rank", "3"}, {"--validate", "--request", path, "--rank", "00"},
			{"--validate", "--request", path, "--rank", "+0"}, {"--validate", "--request", path, "--rank"},
			{"--validate", "--request", "", "--rank", "0"}, {"--validate", "--request", strings.Repeat("x", V41EPOwnerMaxPathBytes+1), "--rank", "0"},
			{"--validate", "--validate", "--request", path, "--rank", "0"}, {"--validate", "--request", path, "--request", path, "--rank", "0"},
			{"--validate", "--request", path, "--rank", "0", "--rank", "1"}, {"--supervised-owner", "--supervised-owner"},
			{"--validate", "--supervised-owner"}, {"--supervised-owner", "--request", path}, {"--supervised-owner", "--rank", "0"},
			{"--validate", "--request", path, "--rank", "0", "--run"}, {"--supervised-owner", "--approve-native-artifacts"},
			{"--help", "--validate"}, {"--validate", "--request=" + path, "--rank", "0"},
		}
		for _, args := range cases {
			var stdout, stderr bytes.Buffer
			if code := RunV41EPOwnerCommand(t.Context(), &stdout, &stderr, args); code != 2 || stdout.Len() != 0 || stderr.Len() == 0 {
				t.Fatalf("args=%q code=%d stdout=%s stderr=%s", args, code, stdout.String(), stderr.String())
			}
		}
	})
	t.Run("file-and-json-refusals", func(t *testing.T) {
		dir := t.TempDir()
		empty, large, bad, link := filepath.Join(dir, "empty"), filepath.Join(dir, "large"), filepath.Join(dir, "bad"), filepath.Join(dir, "link")
		for p, data := range map[string][]byte{empty: {}, large: bytes.Repeat([]byte{' '}, V41EPOwnerMaxRequestBytes+1), bad: append(append([]byte(nil), raw...), []byte(` {}`)...)} {
			if err := os.WriteFile(p, data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		paths := []string{dir, empty, large, bad, filepath.Join(dir, "missing")}
		if err := os.Symlink(path, link); err == nil {
			paths = append(paths, link)
		} else if runtime.GOOS == "linux" {
			t.Fatalf("required Linux symlink refusal fixture: %v", err)
		} else {
			t.Logf("symlink fixture unavailable on %s: %v", runtime.GOOS, err)
		}
		for _, p := range paths {
			var stdout, stderr bytes.Buffer
			if code := RunV41EPOwnerCommand(t.Context(), &stdout, &stderr, []string{"--validate", "--request", p, "--rank", "1"}); code != 1 || stdout.Len() != 0 || stderr.Len() == 0 {
				t.Fatalf("path=%s code=%d stdout=%s stderr=%s", p, code, stdout.String(), stderr.String())
			}
		}
	})
	t.Run("cancellation-before-and-after-decode", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		var stdout, stderr bytes.Buffer
		if code := RunV41EPOwnerCommand(ctx, &stdout, &stderr, validate); code != 1 || stdout.Len() != 0 || !strings.Contains(stderr.String(), context.Canceled.Error()) {
			t.Fatalf("canceled command accepted: code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
		}
		ctx, cancel = context.WithCancel(context.Background())
		defer cancel()
		reader := &v41EPOwnerCancelReader{Reader: bytes.NewReader(raw), cancel: cancel}
		got, err := v41EPOwnerDecodeCommandRequest(ctx, reader)
		if !errors.Is(err, context.Canceled) || got.RunID != "" {
			t.Fatalf("cancellation after reading accepted a request: %+v %v", got, err)
		}
	})
	t.Run("close-errors-preserve-primary-errors", func(t *testing.T) {
		closeErr, readErr := errors.New("close failed"), errors.New("read failed")
		for _, primary := range []error{nil, readErr} {
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			info, err := f.Stat()
			if err != nil {
				f.Close()
				t.Fatal(err)
			}
			owned := &v41EPOwnerErrorFile{File: f, closeErr: closeErr, readErr: primary}
			_, err = v41EPOwnerReadOpenedCommandRequest(t.Context(), info, owned)
			if !errors.Is(err, closeErr) || primary != nil && !errors.Is(err, primary) || owned.closes != 1 {
				t.Fatalf("release error lost: %v closes=%d", err, owned.closes)
			}
		}
	})
	t.Run("diagnostic-write-failure-and-help", func(t *testing.T) {
		var stderr bytes.Buffer
		if code := RunV41EPOwnerCommand(t.Context(), v41EPOwnerErrorWriter{}, &stderr, validate); code != 1 || stderr.Len() == 0 {
			t.Fatalf("write failure accepted: code=%d stderr=%s", code, stderr.String())
		}
		var stdout bytes.Buffer
		stderr.Reset()
		if code := RunV41EPOwnerCommand(t.Context(), &stdout, &stderr, []string{"--help"}); code != 0 || !strings.Contains(stdout.String(), "native execution is unavailable") || !strings.Contains(stdout.String(), "package initialization") {
			t.Fatalf("help changed: code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
		}
	})
}

type v41EPOwnerCancelReader struct {
	io.Reader
	cancel context.CancelFunc
}

func (r *v41EPOwnerCancelReader) Read(p []byte) (int, error) {
	n, err := r.Reader.Read(p)
	r.cancel()
	return n, err
}

type v41EPOwnerErrorFile struct {
	*os.File
	closeErr, readErr error
	closes            int
}

func (f *v41EPOwnerErrorFile) Read(p []byte) (int, error) {
	if f.readErr != nil {
		return 0, f.readErr
	}
	return f.File.Read(p)
}

func (f *v41EPOwnerErrorFile) Close() error {
	f.closes++
	return errors.Join(f.File.Close(), f.closeErr)
}

type v41EPOwnerErrorWriter struct{}

func (v41EPOwnerErrorWriter) Write([]byte) (int, error) {
	return 0, errors.New("diagnostic write failed")
}
