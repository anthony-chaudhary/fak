package leaseref

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

// The hook is the native test executable, avoiding a shell-script test helper.
func init() {
	if marker := os.Getenv("FAK_LEASEREF_PUBLISH_HOOK"); marker != "" {
		_ = os.WriteFile(marker, []byte("invoked"), 0600)
		os.Exit(1)
	}
}

type publishFixture struct {
	t             *testing.T
	local, remote string
	store         *Store
	record        Record
	opts          PublishOptions
}

func newPublishFixture(t *testing.T) *publishFixture {
	t.Helper()
	t.Setenv("GIT_CONFIG_GLOBAL", os.DevNull)
	t.Setenv("GIT_CONFIG_NOSYSTEM", "1")
	root := t.TempDir()
	f := &publishFixture{t: t, local: filepath.Join(root, "local"), remote: filepath.Join(root, "remote.git")}
	f.git(root, "", "init", "-q", f.local)
	f.git(root, "", "init", "-q", "--bare", f.remote)
	f.git(f.local, "", "remote", "add", "origin", f.remote)
	f.store = NewInDir(f.local)
	f.record = Record{ID: "owned", Holder: "holder", SessionID: "session", Generation: 2,
		AcquiredAt: time.Now().Add(-time.Minute).Unix(), TTLSeconds: 3600}
	f.opts = PublishOptions{Ref: f.record.Ref(), Holder: f.record.Holder, SessionID: f.record.SessionID,
		Generation: f.record.Generation, Remote: "origin", ExpectAbsent: true,
		TrustedDirectRefs: "authorized-server-control:disposable direct-reference fixture"}
	f.setLocal(f.record)
	unrelated := f.blob(f.remote, map[string]string{"fixture": "unrelated"})
	f.git(f.remote, "", "update-ref", "refs/fak/locks/unrelated", unrelated)
	return f
}

func (f *publishFixture) git(dir, stdin string, args ...string) string {
	f.t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir, cmd.Stdin = dir, strings.NewReader(stdin)
	out, err := cmd.CombinedOutput()
	if err != nil {
		f.t.Fatalf("fixture git %v: %v: %s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func (f *publishFixture) blob(dir string, value any) string {
	f.t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		f.t.Fatal(err)
	}
	return f.git(dir, string(body), "hash-object", "-w", "--stdin")
}

func (f *publishFixture) setLocal(value any) {
	f.opts.PinnedOID = f.blob(f.local, value)
	f.git(f.local, "", "update-ref", f.opts.Ref, f.opts.PinnedOID)
}

func (f *publishFixture) predecessor(value any) string {
	oid := f.blob(f.remote, value)
	if local := f.blob(f.local, value); local != oid {
		f.t.Fatal("fixture object mismatch")
	}
	f.git(f.remote, "", "update-ref", f.opts.Ref, oid)
	f.opts.ExpectAbsent, f.opts.ExpectedRemoteOID = false, oid
	return oid
}

func (f *publishFixture) refs(dir string) string {
	return f.git(dir, "", "for-each-ref", "--format=%(refname) %(objectname)")
}

// fak-test:justify why=integration when=changed:internal/leaseref/publish* witness=publish.go#PublishOwnedRef
// fak-test:runtime integration est=5s lane=default
func TestLeaserefPublish(t *testing.T) {
	for _, kind := range []string{"absent-intent", "renewed-lease", "expired-predecessor"} {
		t.Run(kind, func(t *testing.T) {
			f := newPublishFixture(t)
			switch kind {
			case "absent-intent":
				intent := IntentRecord{Target: "#13845", Key: IntentKey("#13845"), Holder: f.opts.Holder,
					SessionID: f.opts.SessionID, AcquiredAt: f.record.AcquiredAt, TTLSeconds: 3600}
				f.opts.Ref, f.opts.Generation = intent.Ref(), 0
				f.setLocal(intent)
			case "renewed-lease":
				f.predecessor(f.record)
				f.record.RenewedAt = time.Now().Unix()
				f.setLocal(f.record)
			case "expired-predecessor":
				old := f.record
				old.Holder, old.SessionID, old.Generation = "peer", "peer-session", 1
				old.AcquiredAt, old.TTLSeconds = 1, 1
				f.predecessor(old)
			}
			beforeLocal := f.refs(f.local)
			unrelated := f.git(f.remote, "", "rev-parse", "refs/fak/locks/unrelated")
			// Fetch URL deliberately differs: only the single push URL is relevant.
			f.git(f.local, "", "remote", "set-url", "origin", filepath.Join(t.TempDir(), "missing"))
			f.git(f.local, "", "remote", "set-url", "--push", "origin", f.remote)
			f.git(f.local, "", "config", "push.followTags", "true")
			f.git(f.local, "", "config", "push.recurseSubmodules", "on-demand")
			f.git(f.local, "", "config", "push.negotiate", "true")
			trace := filepath.Join(t.TempDir(), "git-trace.json")
			t.Setenv("GIT_TRACE2_EVENT", trace)
			var push []string
			var deadline time.Time
			f.store.run = func(ctx context.Context, dir string, args ...string) (string, int, error) {
				got, ok := ctx.Deadline()
				if !ok || (!deadline.IsZero() && got != deadline) {
					t.Fatal("not one shared bounded context")
				}
				deadline = got
				if !slices.Contains(args, "--no-replace-objects") || !slices.Contains(args, "--no-lazy-fetch") {
					t.Fatalf("object substitution or implicit fetch allowed: %v", args)
				}
				if slices.Contains(args, "fetch") {
					t.Fatal("publication fetched")
				}
				if slices.Contains(args, "push") {
					push = slices.Clone(args)
				}
				return gitRunner(ctx, dir, args...)
			}
			r := f.store.PublishOwnedRef(context.Background(), f.opts)
			if !r.ConfirmedPublication || !r.TransportAttempted || !r.TransportCompleted || r.Indeterminate || r.RemoteReadbackOID != f.opts.PinnedOID || r.LocalReadbackOID != f.opts.PinnedOID {
				t.Fatalf("publication receipt: %+v", r)
			}
			want := []string{"--no-replace-objects", "--no-lazy-fetch", "-c", "push.negotiate=false", "push", "--no-follow-tags", "--recurse-submodules=no",
				"--force-with-lease=" + f.opts.Ref + ":" + f.opts.ExpectedRemoteOID, "--", f.remote, f.opts.PinnedOID + ":" + f.opts.Ref}
			if !reflect.DeepEqual(push, want) {
				t.Fatalf("push=%v want=%v", push, want)
			}
			traceData, err := os.ReadFile(trace)
			if err != nil {
				t.Fatal(err)
			}
			for _, line := range strings.Split(strings.TrimSpace(string(traceData)), "\n") {
				var event struct {
					Argv []string `json:"argv"`
				}
				if err := json.Unmarshal([]byte(line), &event); err != nil {
					t.Fatal(err)
				}
				if slices.Contains(event.Argv, "fetch") {
					t.Fatalf("Git spawned an implicit fetch: %v", event.Argv)
				}
			}
			if f.refs(f.local) != beforeLocal || f.git(f.remote, "", "rev-parse", "refs/fak/locks/unrelated") != unrelated {
				t.Fatal("publication changed local or unrelated remote refs")
			}
		})
	}
	for _, kind := range []string{"holder", "session", "generation", "expired", "local-moved", "local-symbolic", "unknown-predecessor", "foreign-live", "foreign-intent", "intent-key", "generation-regression", "namespace", "oid", "absence-both", "absence-neither", "mirror", "pushurls", "url-rewrite", "destination-alias", "config-env", "assurance", "replacement-object", "malformed-json", "unknown-field", "trailing-json", "duplicate-field", "unicode-field", "cancelled"} {
		t.Run("refuse-"+kind, func(t *testing.T) {
			f := newPublishFixture(t)
			ctx := context.Background()
			switch kind {
			case "cancelled":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			case "holder":
				f.opts.Holder = "peer"
			case "session":
				f.opts.SessionID = "other"
			case "generation":
				f.opts.Generation++
			case "expired":
				f.record.AcquiredAt, f.record.TTLSeconds = 1, 1
				f.setLocal(f.record)
			case "local-moved":
				f.record.RenewedAt = time.Now().Unix()
				oid := f.blob(f.local, f.record)
				f.git(f.local, "", "update-ref", f.opts.Ref, oid)
			case "local-symbolic":
				f.git(f.local, "", "update-ref", "refs/fak/locks/alias-target", f.opts.PinnedOID)
				f.git(f.local, "", "symbolic-ref", f.opts.Ref, "refs/fak/locks/alias-target")
			case "unknown-predecessor":
				f.opts.ExpectAbsent = false
				f.opts.ExpectedRemoteOID = strings.Repeat("a", len(f.opts.PinnedOID))
			case "foreign-live":
				old := f.record
				old.Holder = "peer"
				f.predecessor(old)
			case "generation-regression":
				old := f.record
				old.Generation++
				f.predecessor(old)
			case "namespace":
				f.opts.Ref = "refs/heads/main"
			case "oid":
				f.opts.PinnedOID = f.opts.PinnedOID[:8]
			case "absence-both":
				f.opts.ExpectedRemoteOID = f.opts.PinnedOID
			case "absence-neither":
				f.opts.ExpectAbsent = false
			case "foreign-intent", "intent-key":
				in := IntentRecord{Target: "#13845", Key: IntentKey("#13845"), Holder: f.opts.Holder,
					SessionID: f.opts.SessionID, AcquiredAt: f.record.AcquiredAt, TTLSeconds: 3600}
				f.opts.Ref, f.opts.Generation = in.Ref(), 0
				if kind == "foreign-intent" {
					old := in
					old.Holder, old.AcquiredAt, old.TTLSeconds = "peer", 1, 1
					f.predecessor(old)
				} else {
					in.Key = "issue-1"
				}
				f.setLocal(in)
			case "mirror":
				f.git(f.local, "", "config", "remote.origin.mirror", "true")
			case "pushurls":
				f.git(f.local, "", "config", "--add", "remote.origin.pushurl", f.remote)
				f.git(f.local, "", "config", "--add", "remote.origin.pushurl", f.remote)
			case "url-rewrite":
				f.git(f.local, "", "config", "url."+f.remote+".pushInsteadOf", "fixture:")
			case "destination-alias":
				destination := "file://" + f.remote
				f.git(f.local, "", "remote", "set-url", "--push", "origin", destination)
				f.git(f.local, "", "config", "remote."+destination+".url", filepath.Join(t.TempDir(), "redirected.git"))
				f.git(f.local, "", "config", "remote."+destination+".mirror", "true")
			case "config-env":
				t.Setenv("GIT_CONFIG", os.DevNull)
			case "assurance":
				f.opts.TrustedDirectRefs = "true"
			case "replacement-object":
				bad := f.blob(f.local, map[string]string{"not": "owned"})
				f.git(f.local, "", "replace", bad, f.opts.PinnedOID)
				f.opts.PinnedOID = bad
				f.git(f.local, "", "update-ref", f.opts.Ref, bad)
			case "malformed-json", "unknown-field", "trailing-json", "duplicate-field", "unicode-field":
				body, err := json.Marshal(f.record)
				if err != nil {
					t.Fatal(err)
				}
				text := string(body)
				switch kind {
				case "malformed-json":
					text = text[:len(text)-1]
				case "unknown-field":
					text = text[:len(text)-1] + `,"unknown":0}`
				case "trailing-json":
					text += `{}`
				case "duplicate-field":
					text = text[:len(text)-1] + `,"HOLDER":"holder"}`
				case "unicode-field":
					text = text[:len(text)-1] + `,"ſeſſion_id":"session"}`
				}
				f.opts.PinnedOID = f.git(f.local, text, "hash-object", "-w", "--stdin")
				f.git(f.local, "", "update-ref", f.opts.Ref, f.opts.PinnedOID)
			}
			beforeLocal, beforeRemote := f.refs(f.local), f.refs(f.remote)
			var networks int
			f.store.run = func(ctx context.Context, dir string, args ...string) (string, int, error) {
				if slices.Contains(args, "push") || slices.Contains(args, "ls-remote") {
					networks++
				}
				return gitRunner(ctx, dir, args...)
			}
			r := f.store.PublishOwnedRef(ctx, f.opts)
			if r.ConfirmedPublication || r.TransportAttempted || r.Indeterminate || networks != 0 {
				t.Fatalf("unsafe admission: %+v networks=%d", r, networks)
			}
			if kind == "cancelled" && r.Reason != context.Canceled.Error() {
				t.Fatalf("cancellation reason=%q", r.Reason)
			}
			if f.refs(f.local) != beforeLocal || f.refs(f.remote) != beforeRemote {
				t.Fatal("refusal changed refs")
			}
		})
	}
	for _, kind := range []string{"local-moved-in-preflight", "advertised-symref"} {
		t.Run(kind, func(t *testing.T) {
			f := newPublishFixture(t)
			beforeRemote := f.refs(f.remote)
			var pushes int
			f.store.run = func(ctx context.Context, dir string, args ...string) (string, int, error) {
				if slices.Contains(args, "push") {
					pushes++
				}
				out, code, err := gitRunner(ctx, dir, args...)
				if slices.Contains(args, "ls-remote") {
					if kind == "advertised-symref" {
						return "ref: refs/fak/locks/unrelated\t" + f.opts.Ref + "\n" + f.opts.PinnedOID + "\t" + f.opts.Ref + "\n", 0, nil
					}
					peer := f.record
					peer.Holder = "peer"
					f.git(f.local, "", "update-ref", f.opts.Ref, f.blob(f.local, peer))
				}
				return out, code, err
			}
			r := f.store.PublishOwnedRef(context.Background(), f.opts)
			if r.ConfirmedPublication || r.TransportAttempted || pushes != 0 || f.refs(f.remote) != beforeRemote {
				t.Fatalf("preflight did not fence publication: %+v pushes=%d", r, pushes)
			}
		})
	}
	for _, kind := range []string{"peer-race", "readback-failure", "remote-after-push", "local-after-push", "timeout", "hook"} {
		t.Run(kind, func(t *testing.T) {
			f := newPublishFixture(t)
			old := f.predecessor(f.record)
			f.record.RenewedAt = time.Now().Unix()
			f.setLocal(f.record)
			beforeRemote := f.refs(f.remote)
			unrelated := f.git(f.remote, "", "rev-parse", "refs/fak/locks/unrelated")
			peer := f.record
			peer.Holder = "peer"
			peerOID := f.blob(f.remote, peer)
			localPeerOID := f.blob(f.local, peer)
			if kind == "hook" {
				executable, err := os.Executable()
				if err != nil {
					t.Fatal(err)
				}
				hook := filepath.Join(f.local, ".git", "hooks", "pre-push")
				if err := os.Link(executable, hook); err != nil {
					in, err := os.Open(executable)
					if err != nil {
						t.Fatal(err)
					}
					defer in.Close()
					out, err := os.OpenFile(hook, os.O_CREATE|os.O_WRONLY, 0700)
					if err != nil {
						t.Fatal(err)
					}
					_, err = io.Copy(out, in)
					closeErr := out.Close()
					if err != nil || closeErr != nil {
						t.Fatalf("copy hook: %v %v", err, closeErr)
					}
				}
				t.Setenv("FAK_LEASEREF_PUBLISH_HOOK", filepath.Join(t.TempDir(), "marker"))
			}
			var pushes int
			f.opts.Timeout = time.Second
			f.store.run = func(ctx context.Context, dir string, args ...string) (string, int, error) {
				if slices.Contains(args, "push") {
					pushes++
					if kind == "peer-race" {
						f.git(f.remote, "", "update-ref", f.opts.Ref, peerOID, old)
					}
					if kind == "timeout" {
						<-ctx.Done()
						return "credential-secret", -1, errors.New("credential-secret")
					}
					out, code, err := gitRunner(ctx, dir, args...)
					if kind == "remote-after-push" {
						f.git(f.remote, "", "update-ref", f.opts.Ref, peerOID)
					}
					if kind == "local-after-push" {
						f.git(f.local, "", "update-ref", f.opts.Ref, localPeerOID)
					}
					return out, code, err
				}
				if kind == "readback-failure" && pushes > 0 && slices.Contains(args, "ls-remote") {
					return "credential-secret", 1, nil
				}
				return gitRunner(ctx, dir, args...)
			}
			r := f.store.PublishOwnedRef(context.Background(), f.opts)
			if r.ConfirmedPublication || !r.TransportAttempted || !r.Indeterminate || pushes != 1 {
				t.Fatalf("uncertain result: %+v pushes=%d", r, pushes)
			}
			if kind == "timeout" && r.Reason != context.DeadlineExceeded.Error() {
				t.Fatalf("deadline reason=%q", r.Reason)
			}
			body, _ := json.Marshal(r)
			if strings.Contains(string(body), "credential-secret") || strings.Contains(string(body), f.remote) {
				t.Fatal("receipt disclosed transport content")
			}
			if kind == "peer-race" || kind == "remote-after-push" {
				if got := f.git(f.remote, "", "rev-parse", f.opts.Ref); got != peerOID {
					t.Fatalf("peer overwritten: %s", got)
				}
			} else if kind == "timeout" || kind == "hook" {
				if f.refs(f.remote) != beforeRemote {
					t.Fatal("failed transport changed refs")
				}
			}
			if got := f.git(f.remote, "", "rev-parse", "refs/fak/locks/unrelated"); got != unrelated {
				t.Fatal("uncertain publication changed unrelated remote ref")
			}
			if kind == "hook" {
				if _, err := os.Stat(os.Getenv("FAK_LEASEREF_PUBLISH_HOOK")); err != nil {
					t.Fatal("normal pre-push hook did not execute")
				}
			}
		})
	}
}
