package codexresume

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestFindRolloutExactAndUniquePrefix(t *testing.T) {
	home := t.TempDir()
	firstID := "11111111-2222-3333-4444-555555555555"
	secondID := "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
	first := writeSyntheticRollout(t, home, "2026/08/28", "rollout-first.jsonl", firstID, "first")
	writeSyntheticRollout(t, home, "2026/08/29", "rollout-second.jsonl", secondID, "second")
	writeFile(t, filepath.Join(home, "sessions", "not-a-year", "rollout-ignored.jsonl"), rolloutSessionMeta(firstID))
	writeFile(t, filepath.Join(home, "auth.json"), []byte(`{"access_token":"must-not-be-read"}`))

	match, err := FindRollout(home, firstID)
	if err != nil {
		t.Fatalf("exact lookup: %v", err)
	}
	if match.ThreadID != firstID || !samePath(match.Path, first) {
		t.Fatalf("exact match = %#v, want id %q path %q", match, firstID, first)
	}
	if match.RelativePath != "sessions/2026/08/28/rollout-first.jsonl" {
		t.Fatalf("relative path = %q", match.RelativePath)
	}

	match, err = FindRollout(home, "aaaaaaaa-bbbb")
	if err != nil {
		t.Fatalf("unique prefix lookup: %v", err)
	}
	if match.ThreadID != secondID {
		t.Fatalf("prefix matched %q, want %q", match.ThreadID, secondID)
	}

	if _, err := FindRollout(home, "missing"); !errors.Is(err, ErrRolloutNotFound) {
		t.Fatalf("missing lookup error = %v, want ErrRolloutNotFound", err)
	}
}

func TestFindRolloutRefusesAmbiguousPrefix(t *testing.T) {
	home := t.TempDir()
	writeSyntheticRollout(t, home, "2026/08/28", "rollout-one.jsonl", "12345678-1111-1111-1111-111111111111", "one")
	writeSyntheticRollout(t, home, "2026/08/29", "rollout-two.jsonl", "12345678-2222-2222-2222-222222222222", "two")

	_, err := FindRollout(home, "12345678")
	if !errors.Is(err, ErrRolloutAmbiguous) {
		t.Fatalf("error = %v, want ErrRolloutAmbiguous", err)
	}
	var ambiguous *AmbiguousRolloutError
	if !errors.As(err, &ambiguous) || len(ambiguous.Matches) != 2 {
		t.Fatalf("ambiguous detail = %#v", ambiguous)
	}
}

func TestBindingStoreRoundTripIsCredentialFree(t *testing.T) {
	home := t.TempDir()
	rollout := writeSyntheticRollout(t, home, "2026/08/29", "rollout.jsonl", "thread-1", "payload")
	secretAccountKey := "account-key-that-must-not-be-stored"
	observed := time.Date(2026, 8, 29, 12, 34, 56, 0, time.FixedZone("offset", -7*60*60))
	binding, err := NewThreadBinding("thread-1", home, secretAccountKey, rollout, observed)
	if err != nil {
		t.Fatal(err)
	}
	if binding.SchemaVersion != BindingSchemaVersion {
		t.Fatalf("schema version = %d", binding.SchemaVersion)
	}
	if binding.AccountKeyDigest != AccountKeyDigest(secretAccountKey) || binding.AccountKeyDigest == secretAccountKey {
		t.Fatalf("unexpected account digest %q", binding.AccountKeyDigest)
	}
	if binding.ObservedAt.Location() != time.UTC {
		t.Fatalf("observed time location = %v, want UTC", binding.ObservedAt.Location())
	}
	if !filepath.IsAbs(binding.CanonicalHome) {
		t.Fatalf("canonical home is not absolute: %q", binding.CanonicalHome)
	}
	if binding.RelativeRolloutPath != "sessions/2026/08/29/rollout.jsonl" {
		t.Fatalf("relative rollout = %q", binding.RelativeRolloutPath)
	}

	storeRoot := filepath.Join(t.TempDir(), "bindings")
	store := BindingStore{Root: storeRoot}
	if err := store.Save(binding); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.Load(binding.ThreadID)
	if err != nil {
		t.Fatal(err)
	}
	if loaded != binding {
		t.Fatalf("loaded binding = %#v, want %#v", loaded, binding)
	}
	payload, err := os.ReadFile(filepath.Join(storeRoot, "thread-1.json"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(payload), secretAccountKey) || strings.Contains(string(payload), "access_token") || strings.Contains(string(payload), "refresh_token") {
		t.Fatalf("binding persisted credential material: %s", payload)
	}
	if _, err := store.Load("unbound"); !errors.Is(err, ErrBindingNotFound) {
		t.Fatalf("missing binding error = %v", err)
	}
}

func TestClassifyRehome(t *testing.T) {
	homeA := t.TempDir()
	homeB := t.TempDir()
	binding := &ThreadBinding{
		SchemaVersion:       BindingSchemaVersion,
		ThreadID:            "thread",
		CanonicalHome:       canonicalForTest(t, homeA),
		AccountKeyDigest:    AccountKeyDigest("account-a"),
		RelativeRolloutPath: "sessions/2026/08/29/rollout.jsonl",
		ObservedAt:          time.Now().UTC(),
	}
	tests := []struct {
		name       string
		binding    *ThreadBinding
		home       string
		accountKey string
		ambiguous  bool
		want       RehomeClass
	}{
		{"same home same account", binding, homeA, "account-a", false, RehomeSameHomeSameAccount},
		{"different home same account", binding, homeB, "account-a", false, RehomeDifferentHomeSameAccount},
		{"same home different account", binding, homeA, "account-b", false, RehomeSameHomeDifferentAccount},
		{"different home different account", binding, homeB, "account-b", false, RehomeDifferentHomeDifferentAccount},
		{"unknown account", binding, homeA, "", false, RehomeUnknown},
		{"unbound", nil, homeA, "account-a", false, RehomeUnbound},
		{"ambiguous", binding, homeA, "account-a", true, RehomeAmbiguous},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := ClassifyRehome(test.binding, test.home, test.accountKey, test.ambiguous); got != test.want {
				t.Fatalf("ClassifyRehome() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestCopyRolloutAtomicIdempotentAndCollisionSafe(t *testing.T) {
	sourceHome := t.TempDir()
	targetHome := t.TempDir()
	source := writeSyntheticRollout(t, sourceHome, "2026/08/29", "rollout.jsonl", "thread-copy", "original payload")
	sourceBefore, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(sourceHome, "auth.json"), []byte("oauth-secret"))
	writeFile(t, filepath.Join(sourceHome, "history.jsonl"), []byte("global-state"))

	result, err := CopyRollout(sourceHome, targetHome, source)
	if err != nil {
		t.Fatal(err)
	}
	if result.Idempotent {
		t.Fatal("first copy reported idempotent")
	}
	destination := filepath.Join(canonicalForTest(t, targetHome), "sessions", "2026", "08", "29", "rollout.jsonl")
	if result.Path != destination {
		t.Fatalf("destination = %q, want %q", result.Path, destination)
	}
	destinationData, err := os.ReadFile(destination)
	if err != nil {
		t.Fatal(err)
	}
	if string(destinationData) != string(sourceBefore) {
		t.Fatal("destination differs from selected rollout")
	}
	sourceAfter, err := os.ReadFile(source)
	if err != nil || string(sourceAfter) != string(sourceBefore) {
		t.Fatalf("source changed: data=%q err=%v", sourceAfter, err)
	}
	for _, global := range []string{"auth.json", "history.jsonl"} {
		if _, err := os.Stat(filepath.Join(targetHome, global)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("global state %s was copied: %v", global, err)
		}
	}

	result, err = CopyRollout(sourceHome, targetHome, source)
	if err != nil || !result.Idempotent {
		t.Fatalf("identical repeat = %#v, %v", result, err)
	}

	writeFile(t, destination, []byte("different destination"))
	if _, err := CopyRollout(sourceHome, targetHome, source); !errors.Is(err, ErrDestinationExists) {
		t.Fatalf("collision error = %v, want ErrDestinationExists", err)
	}
	if data, err := os.ReadFile(destination); err != nil || string(data) != "different destination" {
		t.Fatalf("collision overwrote destination: %q, %v", data, err)
	}
	if data, err := os.ReadFile(source); err != nil || string(data) != string(sourceBefore) {
		t.Fatalf("collision changed source: %q, %v", data, err)
	}
}

func TestCopyRolloutRejectsNonRolloutState(t *testing.T) {
	sourceHome := t.TempDir()
	targetHome := t.TempDir()
	auth := filepath.Join(sourceHome, "auth.json")
	writeFile(t, auth, []byte("secret"))
	if _, err := CopyRollout(sourceHome, targetHome, auth); err == nil {
		t.Fatal("CopyRollout accepted auth.json")
	}
	if _, err := os.Stat(filepath.Join(targetHome, "auth.json")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("target auth.json exists: %v", err)
	}
}

func writeSyntheticRollout(t *testing.T, home, datePath, name, threadID, payload string) string {
	t.Helper()
	path := filepath.Join(append([]string{home, "sessions"}, append(strings.Split(datePath, "/"), name)...)...)
	writeFile(t, path, append(rolloutSessionMeta(threadID), []byte(`{"type":"event_msg","payload":{"text":"`+payload+`"}}`+"\n")...))
	return path
}

func rolloutSessionMeta(threadID string) []byte {
	return []byte(`{"type":"session_meta","payload":{"id":"` + threadID + `"}}` + "\n")
}

func writeFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func canonicalForTest(t *testing.T, path string) string {
	t.Helper()
	canonical, err := canonicalHomePath(path)
	if err != nil {
		t.Fatal(err)
	}
	return canonical
}
