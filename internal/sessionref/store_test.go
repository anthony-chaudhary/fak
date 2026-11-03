package sessionref

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// fakeRunner is the recording stand-in for Runner used across this package's
// tests: it answers each verb from a canned table keyed by args[0] and records
// every call, so assertions verify what the code actually invoked (args, dir,
// tmp-file bytes), not just what the fake returned.
type fakeRunner struct {
	calls   []fakeCall
	answers map[string]fakeAnswer
	// lastHash* captures the tmp blob path and its on-disk bytes for the most
	// recent hash-object call, read while the file still exists (WriteBlob only
	// removes it after the runner returns).
	lastHashPath  string
	lastHashBytes []byte
}

type fakeCall struct {
	dir  string
	args []string
}

type fakeAnswer struct {
	out  string
	code int
	err  error
}

func newFakeRunner() *fakeRunner {
	return &fakeRunner{answers: map[string]fakeAnswer{}}
}

func (f *fakeRunner) Run(ctx context.Context, dir string, args ...string) (string, int, error) {
	f.calls = append(f.calls, fakeCall{dir: dir, args: append([]string(nil), args...)})
	if args[0] == "hash-object" && len(args) >= 3 {
		if b, err := os.ReadFile(args[2]); err == nil {
			f.lastHashPath = args[2]
			f.lastHashBytes = b
		}
	}
	ans, ok := f.answers[args[0]]
	if !ok {
		return "", 0, nil
	}
	return ans.out, ans.code, ans.err
}

// gitExecSentinel stands in for "git could not be executed at all" (the err
// channel of the Runner contract, as opposed to a non-zero exit code).
var gitExecSentinel = errors.New("sentinel: git not executable")

const sha40 = "0123456789abcdef0123456789abcdef01234567" // 40-hex SHA-1

func TestPutBlobRefDisabled(t *testing.T) {
	t.Parallel()
	fr := newFakeRunner()
	const ref = "refs/fak/locks/session-x"
	got, err := PutBlobRef(context.Background(), fr.Run, "", ref, SessionDescriptor{ID: "x"}, true)
	if err != nil {
		t.Fatalf("disabled put = %v, want nil", err)
	}
	if got != ref {
		t.Errorf("ref = %q, want the requested %q back (fail-open)", got, ref)
	}
	if len(fr.calls) != 0 {
		t.Errorf("runner called %d times, want 0 (nothing written when disabled)", len(fr.calls))
	}
}

func TestPutBlobRefSuccess(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name         string
		desc         SessionDescriptor
		wantAgentKey bool
	}{
		{name: "empty agent uuid omitted", desc: SessionDescriptor{ID: "abc", Host: "h", UpdatedAt: 1700000000}},
		{name: "set agent uuid present", desc: SessionDescriptor{ID: "abc", AgentUUID: "uuid-9"}, wantAgentKey: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			fr := newFakeRunner()
			fr.answers["hash-object"] = fakeAnswer{out: sha40 + "\n"}
			fr.answers["update-ref"] = fakeAnswer{code: 0}
			ref, err := PutBlobRef(context.Background(), fr.Run, "/some/dir", "refs/fak/locks/session-abc", tc.desc, false)
			if err != nil {
				t.Fatal(err)
			}
			if ref != "refs/fak/locks/session-abc" {
				t.Errorf("ref = %q, want refs/fak/locks/session-abc", ref)
			}
			if len(fr.calls) != 2 {
				t.Fatalf("runner called %d times, want exactly 2", len(fr.calls))
			}
			first, second := fr.calls[0], fr.calls[1]
			if first.dir != "/some/dir" {
				t.Errorf("first call dir = %q, want /some/dir", first.dir)
			}
			if len(first.args) != 3 || first.args[0] != "hash-object" || first.args[1] != "-w" {
				t.Fatalf("first call args = %v, want [hash-object -w <tmp>]", first.args)
			}
			tmp := first.args[2]
			if fr.lastHashPath != tmp {
				t.Fatalf("captured tmp path %q, want %q", fr.lastHashPath, tmp)
			}
			if second.args[0] != "update-ref" || second.args[1] != ref || second.args[2] != sha40 {
				t.Errorf("second call args = %v, want [update-ref %s %s]", second.args, ref, sha40)
			}
			want, err := json.Marshal(tc.desc)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(fr.lastHashBytes, want) {
				t.Errorf("tmp blob bytes = %s, want json.Marshal(descriptor) = %s", fr.lastHashBytes, want)
			}
			// omitempty wire contract: legacy readers see byte-identical blobs when
			// no Claude UUID is set, so the key must be absent, not "".
			if hasKey := strings.Contains(string(fr.lastHashBytes), `"agent_uuid"`); hasKey != tc.wantAgentKey {
				t.Errorf(`blob contains "agent_uuid" key = %v, want %v (blob: %s)`, hasKey, tc.wantAgentKey, fr.lastHashBytes)
			}
		})
	}
}

func TestPutBlobRefUpdateRefFails(t *testing.T) {
	t.Parallel()
	fr := newFakeRunner()
	fr.answers["hash-object"] = fakeAnswer{out: sha40 + "\n"}
	fr.answers["update-ref"] = fakeAnswer{code: 128}
	ref, err := PutBlobRef(context.Background(), fr.Run, "", "refs/fak/locks/session-abc", SessionDescriptor{ID: "abc"}, false)
	if err == nil {
		t.Fatal("want error when update-ref exits non-zero")
	}
	if ref != "" {
		t.Errorf("ref = %q, want empty on error", ref)
	}
}

func TestPutBlobRefGitNotExecutable(t *testing.T) {
	t.Parallel()
	fr := newFakeRunner()
	fr.answers["hash-object"] = fakeAnswer{err: gitExecSentinel}
	_, err := PutBlobRef(context.Background(), fr.Run, "", "refs/fak/locks/session-abc", SessionDescriptor{ID: "abc"}, false)
	if !errors.Is(err, gitExecSentinel) {
		t.Errorf("error %v does not satisfy errors.Is(_, gitExecSentinel)", err)
	}
}

func TestWriteBlob(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	blob := []byte(`{"id":"abc","updated_at":1700000000}`)

	t.Run("success writes blob and cleans up tmp", func(t *testing.T) {
		t.Parallel()
		fr := newFakeRunner()
		fr.answers["hash-object"] = fakeAnswer{out: sha40 + "\n"}
		sha, err := WriteBlob(ctx, fr.Run, "", blob)
		if err != nil {
			t.Fatal(err)
		}
		if sha != sha40 {
			t.Errorf("sha = %q, want %q", sha, sha40)
		}
		if !bytes.Equal(fr.lastHashBytes, blob) {
			t.Errorf("tmp blob bytes = %s, want %s", fr.lastHashBytes, blob)
		}
		if _, statErr := os.Stat(fr.lastHashPath); !errors.Is(statErr, fs.ErrNotExist) {
			t.Errorf("tmp file %s still present after success: %v", fr.lastHashPath, statErr)
		}
	})

	t.Run("non-zero exit fails", func(t *testing.T) {
		t.Parallel()
		fr := newFakeRunner()
		fr.answers["hash-object"] = fakeAnswer{code: 1}
		if _, err := WriteBlob(ctx, fr.Run, "", blob); err == nil {
			t.Error("want error when hash-object exits 1")
		}
	})

	t.Run("short object id fails", func(t *testing.T) {
		t.Parallel()
		fr := newFakeRunner()
		fr.answers["hash-object"] = fakeAnswer{out: "deadbeef\n"}
		if _, err := WriteBlob(ctx, fr.Run, "", blob); err == nil {
			t.Error("want error for 8-char object id")
		}
	})

	t.Run("all-zero object id fails", func(t *testing.T) {
		t.Parallel()
		fr := newFakeRunner()
		fr.answers["hash-object"] = fakeAnswer{out: strings.Repeat("0", 40) + "\n"}
		if _, err := WriteBlob(ctx, fr.Run, "", blob); err == nil {
			t.Error("want error for all-zero object id")
		}
	})

	t.Run("sha256 length accepted", func(t *testing.T) {
		t.Parallel()
		sha64 := strings.Repeat(sha40[:16], 4) // 64 hex chars
		fr := newFakeRunner()
		fr.answers["hash-object"] = fakeAnswer{out: sha64 + "\n"}
		got, err := WriteBlob(ctx, fr.Run, "", blob)
		if err != nil {
			t.Fatal(err)
		}
		if got != sha64 {
			t.Errorf("sha = %q, want %q", got, sha64)
		}
	})
}

func TestValidObjectID(t *testing.T) {
	t.Parallel()
	sha64 := strings.Repeat(sha40[:16], 4)
	for _, tc := range []struct {
		id    string
		valid bool
	}{
		{id: sha40, valid: true},
		{id: sha64, valid: true},
		{id: strings.ToUpper(sha40), valid: true},
		{id: sha40[:39]},
		{id: sha40 + "a"},
		{id: sha64[:63]},
		{id: sha64 + "a"},
		{id: strings.Repeat("0", 40)},
		{id: sha40[:39] + "g"},
		{id: ""},
	} {
		if got := ValidObjectID(tc.id); got != tc.valid {
			label := tc.id
			if len(label) > 24 {
				label = label[:24] + "...(len=" + strconv.Itoa(len(tc.id)) + ")"
			}
			t.Errorf("ValidObjectID(%q) = %v, want %v", label, got, tc.valid)
		}
	}
}

func TestHasObjectDB(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	for _, tc := range []struct {
		name   string
		answer fakeAnswer
		want   bool
	}{
		{name: "probe succeeds", answer: fakeAnswer{code: 0}, want: true},
		{name: "probe exits non-zero", answer: fakeAnswer{code: 128}},
		{name: "git not executable", answer: fakeAnswer{err: gitExecSentinel, code: 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			fr := newFakeRunner()
			fr.answers["rev-parse"] = tc.answer
			if got := HasObjectDB(ctx, fr.Run, ""); got != tc.want {
				t.Errorf("HasObjectDB = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestPublishingDisabled(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("non-real-git never probes", func(t *testing.T) {
		t.Parallel()
		fr := newFakeRunner()
		var once sync.Once
		var off bool
		if got := PublishingDisabled(ctx, fr.Run, "", false, &once, &off); got {
			t.Errorf("PublishingDisabled = true, want false when realGit=false")
		}
		if off {
			t.Error("publishOff flipped even though no probe ran")
		}
		if len(fr.calls) != 0 {
			t.Errorf("runner probed %d times, want 0", len(fr.calls))
		}
	})

	t.Run("probe success keeps publishing on", func(t *testing.T) {
		t.Parallel()
		fr := newFakeRunner()
		fr.answers["rev-parse"] = fakeAnswer{code: 0}
		var once sync.Once
		var off bool
		if got := PublishingDisabled(ctx, fr.Run, "", true, &once, &off); got {
			t.Errorf("PublishingDisabled = true, want false when the object DB probe succeeds")
		}
		if off {
			t.Error("publishOff flipped despite a successful probe")
		}
		if len(fr.calls) != 1 {
			t.Errorf("runner probed %d times, want 1", len(fr.calls))
		}
	})

	t.Run("probe failure disables, exactly once", func(t *testing.T) {
		t.Parallel()
		fr := newFakeRunner()
		fr.answers["rev-parse"] = fakeAnswer{code: 128}
		var once sync.Once
		var off bool
		if got := PublishingDisabled(ctx, fr.Run, "", true, &once, &off); !got {
			t.Fatalf("PublishingDisabled = false, want true when the object DB probe fails")
		}
		if !off {
			t.Error("publishOff not set after a failed probe")
		}
		// Second call must observe the cached verdict without re-probing:
		// sync.Once means exactly one rev-parse for the Store's whole lifetime.
		if got := PublishingDisabled(ctx, fr.Run, "", true, &once, &off); !got {
			t.Errorf("second call = false, want cached true")
		}
		if len(fr.calls) != 1 {
			t.Errorf("runner probed %d times total, want 1 (sync.Once)", len(fr.calls))
		}
	})
}

func TestStorePublishSessionEndToEnd(t *testing.T) {
	t.Parallel()
	fr := newFakeRunner()
	fr.answers["rev-parse"] = fakeAnswer{code: 0}
	fr.answers["hash-object"] = fakeAnswer{out: sha40 + "\n"}
	fr.answers["update-ref"] = fakeAnswer{code: 0}
	s := &Store{run: fr.Run, dir: ""}
	const want = "refs/fak/locks/session-abc-123"
	ref, err := s.PublishSession(context.Background(), SessionDescriptor{ID: "abc-123"})
	if err != nil {
		t.Fatal(err)
	}
	if ref != want {
		t.Errorf("ref = %q, want %q", ref, want)
	}
	if len(fr.calls) != 3 {
		t.Fatalf("runner called %d times, want 3 (probe, hash-object, update-ref): %v", len(fr.calls), fr.calls)
	}
	if fr.calls[0].args[0] != "rev-parse" || fr.calls[1].args[0] != "hash-object" || fr.calls[2].args[0] != "update-ref" {
		t.Errorf("call sequence = %v, want [rev-parse hash-object update-ref]", verbsOf(fr.calls))
	}
	if fr.calls[2].args[1] != want || fr.calls[2].args[2] != sha40 {
		t.Errorf("update-ref args = %v, want [update-ref %s %s]", fr.calls[2].args, want, sha40)
	}

	frBad := newFakeRunner()
	sBad := &Store{run: frBad.Run, dir: ""}
	if _, err := sBad.PublishSession(context.Background(), SessionDescriptor{ID: "bad/id"}); err == nil {
		t.Error("want error for invalid id")
	}
	if len(frBad.calls) != 0 {
		t.Errorf("invalid-id publish made %d runner calls, want 0", len(frBad.calls))
	}
}

func TestStorePublishSessionNonRepoFallback(t *testing.T) {
	t.Parallel()
	fr := newFakeRunner()
	fr.answers["rev-parse"] = fakeAnswer{code: 128} // no object DB: feature simply off on this box
	s := &Store{run: fr.Run, dir: ""}
	ref, err := s.PublishSession(context.Background(), SessionDescriptor{ID: "abc-123"})
	if err != nil {
		t.Fatalf("fail-open publish = %v, want nil", err)
	}
	if want := "refs/fak/locks/session-abc-123"; ref != want {
		t.Errorf("ref = %q, want %q", ref, want)
	}
	if len(fr.calls) != 1 || fr.calls[0].args[0] != "rev-parse" {
		t.Errorf("calls = %v, want exactly the single object-DB probe; nothing may be written", fr.calls)
	}
}

func verbsOf(calls []fakeCall) []string {
	out := make([]string, 0, len(calls))
	for _, c := range calls {
		out = append(out, c.args[0])
	}
	return out
}
