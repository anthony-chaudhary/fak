package amdgpu

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestStrixControllerAuthorityRequiresSemanticEpochAndSameHandleProvenance(t *testing.T) {
	const literalEpoch = "c3d5ac66e4bfdfa7eee783bd7b06abb8dbaec584"
	if strixGitSemanticEpoch != literalEpoch {
		t.Fatalf("semantic epoch = %q, want %q", strixGitSemanticEpoch, literalEpoch)
	}
	now := time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)
	valid := strixControllerProvenance{
		revision:     strings.Repeat("a", 40),
		binarySHA256: strings.Repeat("b", 64),
		binaryBytes:  4096,
	}

	for _, tc := range []struct {
		name        string
		provenance  strixControllerProvenance
		ancestry    strixGitAncestry
		ancestryErr error
		wantCode    string
		want        bool
	}{
		{name: "epoch equality", provenance: valid, ancestry: strixGitAncestryEpochEqual, want: true},
		{name: "epoch equality with classifier error", provenance: valid, ancestry: strixGitAncestryEpochEqual, ancestryErr: errors.New("uncertain"), wantCode: strixGitSnapshotUnattestedToken},
		{name: "descendant", provenance: valid, ancestry: strixGitAncestryDescendant, want: true},
		{name: "descendant with classifier error", provenance: valid, ancestry: strixGitAncestryDescendant, ancestryErr: errors.New("cleanup failed at /private/snapshot"), wantCode: strixGitSnapshotUnattestedToken},
		{name: "pre epoch", provenance: valid, ancestry: strixGitAncestryPreEpoch, wantCode: strixControllerStaleToken},
		{name: "pre epoch with classifier error", provenance: valid, ancestry: strixGitAncestryPreEpoch, ancestryErr: errors.New("uncertain"), wantCode: strixGitSnapshotUnattestedToken},
		{name: "unrelated", provenance: valid, ancestry: strixGitAncestryUnrelated, wantCode: strixGitSnapshotUnattestedToken},
		{name: "unrelated with classifier error", provenance: valid, ancestry: strixGitAncestryUnrelated, ancestryErr: errors.New("uncertain"), wantCode: strixGitSnapshotUnattestedToken},
		{name: "uncertain ancestry", provenance: valid, ancestry: strixGitAncestryUnattested, wantCode: strixGitSnapshotUnattestedToken},
		{name: "uncertain ancestry with classifier error", provenance: valid, ancestry: strixGitAncestryUnattested, ancestryErr: errors.New("uncertain"), wantCode: strixGitSnapshotUnattestedToken},
		{name: "missing revision", provenance: strixControllerProvenance{binarySHA256: valid.binarySHA256, binaryBytes: valid.binaryBytes}, ancestry: strixGitAncestryDescendant, wantCode: strixGitSnapshotUnattestedToken},
		{name: "dirty provenance", provenance: strixControllerProvenance{revision: valid.revision, dirty: true, binarySHA256: valid.binarySHA256, binaryBytes: valid.binaryBytes}, ancestry: strixGitAncestryDescendant, wantCode: strixGitSnapshotUnattestedToken},
		{name: "malformed revision", provenance: strixControllerProvenance{revision: "abc1234", binarySHA256: valid.binarySHA256, binaryBytes: valid.binaryBytes}, ancestry: strixGitAncestryDescendant, wantCode: strixGitSnapshotUnattestedToken},
		{name: "malformed digest", provenance: strixControllerProvenance{revision: valid.revision, binarySHA256: "not-a-digest", binaryBytes: valid.binaryBytes}, ancestry: strixGitAncestryDescendant, wantCode: strixGitSnapshotUnattestedToken},
		{name: "zero byte executable", provenance: strixControllerProvenance{revision: valid.revision, binarySHA256: valid.binarySHA256}, ancestry: strixGitAncestryDescendant, wantCode: strixGitSnapshotUnattestedToken},
		{name: "negative byte executable", provenance: strixControllerProvenance{revision: valid.revision, binarySHA256: valid.binarySHA256, binaryBytes: -1}, ancestry: strixGitAncestryDescendant, wantCode: strixGitSnapshotUnattestedToken},
	} {
		t.Run(tc.name, func(t *testing.T) {
			evidence, err := evaluateStrixControllerAuthority(tc.provenance, tc.ancestry, tc.ancestryErr, now)
			if tc.want {
				if err != nil {
					t.Fatalf("evaluate: %v", err)
				}
				authority := StrixControllerAuthority{evidence: evidence, seal: &strixControllerAuthoritySealValue}
				if authority.Epoch() != literalEpoch || authority.ObservedRevision() != valid.revision ||
					authority.ExecutableSHA256() != valid.binarySHA256 || authority.ExecutableBytes() != valid.binaryBytes ||
					!authority.AdmittedAt().Equal(now) {
					t.Fatalf("authority evidence = epoch %q revision %q digest %q bytes %d time %v",
						authority.Epoch(), authority.ObservedRevision(), authority.ExecutableSHA256(), authority.ExecutableBytes(), authority.AdmittedAt())
				}
				return
			}
			if got := strixControllerRefusalCode(err); got != tc.wantCode {
				t.Fatalf("refusal code = %q, want %q (err=%v)", got, tc.wantCode, err)
			}
			if evidence != (strixControllerAuthorityEvidence{}) {
				t.Fatalf("refusal returned evidence: %+v", evidence)
			}
		})
	}

	t.Run("constructor and authority are opaque", func(t *testing.T) {
		ctor := reflect.TypeOf(NewStrixControllerAuthority)
		contextType := reflect.TypeOf((*context.Context)(nil)).Elem()
		if ctor.NumIn() != 2 || ctor.In(0) != contextType || ctor.In(1).Kind() != reflect.String {
			t.Fatalf("constructor = %v, want only context and resolved repository root", ctor)
		}
		production := productionStrixControllerAuthorityDependencies()
		for name, pair := range map[string][2]any{
			"observer":    {production.observe, observeStrixControllerProvenance},
			"snapshotter": {production.snapshot, newStrixGitObjectSnapshot},
			"classifier":  {production.classify, classifyStrixGitSnapshotAncestry},
			"clock":       {production.clock, time.Now},
		} {
			if reflect.ValueOf(pair[0]).Pointer() != reflect.ValueOf(pair[1]).Pointer() {
				t.Fatalf("production %s is not fixed to the audited function", name)
			}
		}
		if production.goos != runtime.GOOS {
			t.Fatalf("production GOOS = %q, want %q", production.goos, runtime.GOOS)
		}
		typ := reflect.TypeOf(StrixControllerAuthority{})
		for i := 0; i < typ.NumField(); i++ {
			if typ.Field(i).IsExported() {
				t.Fatalf("authority field %q is exported", typ.Field(i).Name)
			}
		}
		for _, forbidden := range []string{"MarshalJSON", "UnmarshalJSON", "MarshalText", "UnmarshalText"} {
			if _, ok := typ.MethodByName(forbidden); ok {
				t.Fatalf("authority exposes %s", forbidden)
			}
		}
		authority := StrixControllerAuthority{evidence: strixControllerAuthorityEvidence{
			epoch: literalEpoch, revision: valid.revision, binarySHA256: valid.binarySHA256, binaryBytes: valid.binaryBytes, admittedAt: now,
		}, seal: &strixControllerAuthoritySealValue}
		encoded, err := json.Marshal(authority)
		if err != nil || string(encoded) != "{}" {
			t.Fatalf("opaque JSON = %q, %v; want no serializable evidence", encoded, err)
		}
		var decoded StrixControllerAuthority
		if err := json.Unmarshal(encoded, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.Epoch() != "" || decoded.ObservedRevision() != "" || decoded.ExecutableSHA256() != "" || decoded.ExecutableBytes() != 0 || !decoded.AdmittedAt().IsZero() {
			t.Fatal("serialization minted authority")
		}
		partial := StrixControllerAuthority{evidence: authority.evidence}
		if partial.Epoch() != "" || partial.ObservedRevision() != "" || partial.ExecutableSHA256() != "" || partial.ExecutableBytes() != 0 || !partial.AdmittedAt().IsZero() {
			t.Fatal("copied partial state validated without the authority seal")
		}
		partialEvidence := authority.evidence
		partialEvidence.binaryBytes = 0
		partialEvidence.admittedAt = time.Time{}
		partial = StrixControllerAuthority{evidence: partialEvidence, seal: authority.seal}
		if partial.valid() || partial.Epoch() != "" || partial.ExecutableBytes() != 0 || !partial.AdmittedAt().IsZero() {
			t.Fatal("sealed partial evidence validated")
		}
	})

	t.Run("refusals are local and redacted", func(t *testing.T) {
		secret := `/private/repository/.git /proc/secret/exe GIT_OBJECT_DIRECTORY=/attacker raw command output`
		for _, err := range []error{
			newStrixControllerAuthorityRefusal(strixGitSnapshotUnattestedToken, errors.New(secret), ""),
			strixControllerObservationRefusal("linux", errors.New(secret)),
		} {
			if strings.Contains(err.Error(), secret) || strings.Contains(err.Error(), "/private/") || strings.Contains(err.Error(), "GIT_OBJECT_DIRECTORY") {
				t.Fatalf("refusal leaked local detail: %v", err)
			}
		}
	})

	t.Run("native Windows is unsupported with sanctioned recovery", func(t *testing.T) {
		err := strixControllerObservationRefusal("windows", errors.New(`C:\private\fak.exe`))
		if got := strixControllerRefusalCode(err); got != strixControllerUnsupportedToken {
			t.Fatalf("code = %q, want %q", got, strixControllerUnsupportedToken)
		}
		var refusal *StrixControllerAuthorityRefusal
		if !errors.As(err, &refusal) || !strings.Contains(refusal.Recovery(), "current stamped WSL build") || !strings.Contains(refusal.Recovery(), "/proc/self/exe") {
			t.Fatalf("recovery = %q", refusal.Recovery())
		}
		if strings.Contains(err.Error(), `C:\private`) {
			t.Fatalf("Windows refusal leaked executable path: %v", err)
		}
	})

	t.Run("nil context refuses before any snapshot or transport", func(t *testing.T) {
		authority, err := NewStrixControllerAuthority(nil, `/private/repository`)
		if got := strixControllerRefusalCode(err); got != strixGitSnapshotUnattestedToken {
			t.Fatalf("code = %q, want %q", got, strixGitSnapshotUnattestedToken)
		}
		if authority != (StrixControllerAuthority{}) {
			t.Fatal("nil context minted authority")
		}
		if strings.Contains(err.Error(), "/private/") {
			t.Fatalf("nil-context refusal leaked path: %v", err)
		}
	})

	t.Run("non-nil production orchestration is ordered and cleanup-safe", func(t *testing.T) {
		baseDeps := func() strixControllerAuthorityDependencies {
			return strixControllerAuthorityDependencies{
				observe: func() (strixControllerProvenance, error) { return valid, nil },
				snapshot: func(context.Context, string) (*strixGitObjectSnapshot, error) {
					t.Fatal("unexpected snapshot call")
					return nil, nil
				},
				classify: func(context.Context, *strixGitObjectSnapshot, string) (strixGitAncestry, error) {
					t.Fatal("unexpected classifier call")
					return strixGitAncestryUnattested, nil
				},
				clock: func() time.Time {
					t.Fatal("unexpected clock call")
					return time.Time{}
				},
				goos: "linux",
			}
		}
		asRefusal := func(t *testing.T, err error, code string) *StrixControllerAuthorityRefusal {
			t.Helper()
			var refusal *StrixControllerAuthorityRefusal
			if !errors.As(err, &refusal) || refusal.Code() != code {
				t.Fatalf("refusal = %T %v, want %q", err, err, code)
			}
			return refusal
		}

		t.Run("observer precedes snapshot and Windows refuses", func(t *testing.T) {
			var events []string
			deps := baseDeps()
			deps.goos = "windows"
			deps.observe = func() (strixControllerProvenance, error) {
				events = append(events, "observe")
				return strixControllerProvenance{}, errors.New(`C:\private\fak.exe`)
			}
			deps.snapshot = func(context.Context, string) (*strixGitObjectSnapshot, error) {
				events = append(events, "snapshot")
				return nil, nil
			}
			authority, err := newStrixControllerAuthorityWith(context.Background(), `/private/repository`, deps)
			refusal := asRefusal(t, err, strixControllerUnsupportedToken)
			if authority.valid() || strings.Join(events, ",") != "observe" || !strings.Contains(refusal.Recovery(), "/proc/self/exe") {
				t.Fatalf("authority/events/recovery = %v %q %q", authority.valid(), events, refusal.Recovery())
			}
		})

		t.Run("snapshot creation error retains cleanup retry", func(t *testing.T) {
			calls := 0
			snapshot := &strixGitObjectSnapshot{root: "opaque-owned-snapshot", removeAll: func(string) error {
				calls++
				if calls == 1 {
					return errors.New("private cleanup path")
				}
				return nil
			}}
			deps := baseDeps()
			deps.snapshot = func(context.Context, string) (*strixGitObjectSnapshot, error) {
				return snapshot, errors.New("creation failed at /private/snapshot")
			}
			authority, err := newStrixControllerAuthorityWith(context.Background(), `/private/repository`, deps)
			refusal := asRefusal(t, err, strixGitSnapshotUnattestedToken)
			if authority.valid() || calls != 1 || snapshot.gitDir() == "" {
				t.Fatalf("creation cleanup ownership lost: authority=%v calls=%d root=%q", authority.valid(), calls, snapshot.gitDir())
			}
			if err := refusal.RetryCleanup(); err != nil || calls != 2 || snapshot.gitDir() != "" {
				t.Fatalf("creation cleanup retry: err=%v calls=%d root=%q", err, calls, snapshot.gitDir())
			}
		})

		t.Run("classifier cleanup error retains retry", func(t *testing.T) {
			calls := 0
			snapshot := &strixGitObjectSnapshot{root: "opaque-owned-snapshot", removeAll: func(string) error {
				calls++
				if calls < 3 {
					return errors.New("private cleanup path")
				}
				return nil
			}}
			deps := baseDeps()
			deps.snapshot = func(context.Context, string) (*strixGitObjectSnapshot, error) { return snapshot, nil }
			deps.classify = func(context.Context, *strixGitObjectSnapshot, string) (strixGitAncestry, error) {
				if err := snapshot.close(); err == nil {
					t.Fatal("injected classifier cleanup unexpectedly succeeded")
				}
				return strixGitAncestryDescendant, errors.New("classifier cleanup failed")
			}
			authority, err := newStrixControllerAuthorityWith(context.Background(), `/private/repository`, deps)
			refusal := asRefusal(t, err, strixGitSnapshotUnattestedToken)
			if authority.valid() || calls != 2 || snapshot.gitDir() == "" {
				t.Fatalf("classifier cleanup ownership lost: authority=%v calls=%d root=%q", authority.valid(), calls, snapshot.gitDir())
			}
			if err := refusal.RetryCleanup(); err != nil || calls != 3 || snapshot.gitDir() != "" {
				t.Fatalf("classifier cleanup retry: err=%v calls=%d root=%q", err, calls, snapshot.gitDir())
			}
		})

		t.Run("success samples clock only after cleanup", func(t *testing.T) {
			var events []string
			snapshot := &strixGitObjectSnapshot{root: "opaque-owned-snapshot", removeAll: func(string) error {
				events = append(events, "cleanup")
				return nil
			}}
			deps := baseDeps()
			deps.observe = func() (strixControllerProvenance, error) {
				events = append(events, "observe")
				return valid, nil
			}
			deps.snapshot = func(context.Context, string) (*strixGitObjectSnapshot, error) {
				events = append(events, "snapshot")
				return snapshot, nil
			}
			deps.classify = func(context.Context, *strixGitObjectSnapshot, string) (strixGitAncestry, error) {
				events = append(events, "classify")
				if err := snapshot.close(); err != nil {
					return strixGitAncestryUnattested, err
				}
				return strixGitAncestryDescendant, nil
			}
			deps.clock = func() time.Time {
				if snapshot.gitDir() != "" {
					t.Fatal("clock sampled before snapshot cleanup")
				}
				events = append(events, "clock")
				return now
			}
			authority, err := newStrixControllerAuthorityWith(context.Background(), `/private/repository`, deps)
			if err != nil || !authority.valid() || strings.Join(events, ",") != "observe,snapshot,classify,cleanup,clock" || !authority.AdmittedAt().Equal(now) {
				t.Fatalf("success = authority %v err %v events %q time %v", authority.valid(), err, events, authority.AdmittedAt())
			}
		})

		t.Run("classifier success without cleanup cannot mint", func(t *testing.T) {
			clockCalls := 0
			snapshot := &strixGitObjectSnapshot{root: "opaque-owned-snapshot", removeAll: func(string) error { return nil }}
			deps := baseDeps()
			deps.snapshot = func(context.Context, string) (*strixGitObjectSnapshot, error) { return snapshot, nil }
			deps.classify = func(context.Context, *strixGitObjectSnapshot, string) (strixGitAncestry, error) {
				return strixGitAncestryDescendant, nil
			}
			deps.clock = func() time.Time { clockCalls++; return now }
			authority, err := newStrixControllerAuthorityWith(context.Background(), `/private/repository`, deps)
			asRefusal(t, err, strixGitSnapshotUnattestedToken)
			if authority.valid() || clockCalls != 0 || snapshot.gitDir() != "" {
				t.Fatalf("unclean classifier minted: authority=%v clocks=%d root=%q", authority.valid(), clockCalls, snapshot.gitDir())
			}
		})
	})
}

func strixControllerRefusalCode(err error) string {
	var refusal *StrixControllerAuthorityRefusal
	if errors.As(err, &refusal) {
		return refusal.Code()
	}
	return ""
}
