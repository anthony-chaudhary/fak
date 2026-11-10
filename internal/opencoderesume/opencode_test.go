package opencoderesume

import (
	"context"
	"encoding/json"
	"errors"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

const observedID = "ses_e3861b794ffeKK5UOg25IFE1mR" // live-observed, 26-char body

func TestIsSessionID(t *testing.T) {
	body := func(n int) string { return strings.Repeat("a", n) }
	cases := []struct {
		id   string
		want bool
	}{
		{observedID, true},
		{"ses_" + body(16), true},  // lower boundary
		{"ses_" + body(64), true},  // upper boundary
		{"ses_" + body(15), false}, // too short
		{"ses_" + body(65), false}, // too long
		{"sess_" + body(20), false},
		{"SES_" + body(20), false},
		{"ss_" + body(20), false},
		{"ses_abc/def" + body(12), false},  // path separator
		{"ses_abc\\def" + body(12), false}, // backslash
		{" ses_" + body(20), false},
		{"ses_" + body(20) + " ", false},
		{"ses_a b" + body(18), false}, // embedded space
		{"ses_" + body(10) + "&" + body(8), false},
		{`ses_` + body(10) + `"` + body(8), false},
		{"ses_" + body(10) + "\n" + body(8), false},
		{"ses_" + body(10) + ";" + body(8), false},
		{"ses_" + body(10) + "$(" + body(6), false},
		{"ses_" + body(10) + "-" + body(8), false},
		{"ses_" + body(10) + "." + body(8), false},
		{"ses__" + body(19), false}, // underscore in body
		{"", false},
	}
	for _, tc := range cases {
		if got := IsSessionID(tc.id); got != tc.want {
			t.Errorf("IsSessionID(%q) = %v, want %v", tc.id, got, tc.want)
		}
	}
}

// listCallRecorder captures what Verify would have executed against the CLI.
type listCallRecorder struct {
	bin      string
	dir      string
	maxCount int
	ctx      context.Context
	calls    int
}

// withFakeList swaps runListCommand for the duration of the test and returns
// a recorder of whatever the production Verify would have executed.
func withFakeList(t *testing.T, out []byte, runErr error) *listCallRecorder {
	t.Helper()
	rec := &listCallRecorder{}
	orig := runListCommand
	runListCommand = func(ctx context.Context, bin, dir string, maxCount int) ([]byte, error) {
		rec.bin, rec.dir, rec.maxCount, rec.ctx = bin, dir, maxCount, ctx
		rec.calls++
		return out, runErr
	}
	t.Cleanup(func() { runListCommand = orig })
	return rec
}

func listingJSON(ids ...string) []byte {
	rows := make([]string, len(ids))
	for i, id := range ids {
		rows[i] = `{"id":"` + id + `","directory":"/proj","updated":1794351470961,"created":1794350403068}`
	}
	return []byte("[" + strings.Join(rows, ",") + "]")
}

func TestVerifyUnit(t *testing.T) {
	validMissing := "ses_zzzz0000000000000000000000000000"

	t.Run("hit", func(t *testing.T) {
		rec := withFakeList(t, listingJSON(observedID), nil)
		got, err := Verify(context.Background(), Options{SessionID: observedID, Dir: "/proj"})
		if err != nil {
			t.Fatalf("Verify hit: unexpected error: %v", err)
		}
		if got != VerdictResumable {
			t.Fatalf("Verify hit = %q, want %q", got, VerdictResumable)
		}
		if rec.bin != "opencode" || rec.dir != "/proj" || rec.maxCount != defaultMaxCount || rec.calls != 1 {
			t.Fatalf("runner got bin=%q dir=%q maxCount=%d calls=%d, want opencode /proj %d 1", rec.bin, rec.dir, rec.maxCount, rec.calls, defaultMaxCount)
		}
		deadline, ok := rec.ctx.Deadline()
		if !ok {
			t.Fatal("runner ctx has no deadline; execution must be bounded")
		}
		if remaining := time.Until(deadline); remaining <= 0 || remaining > listTimeout {
			t.Fatalf("runner ctx deadline off by %v; want within (0, %s]", remaining, listTimeout)
		}
	})

	t.Run("miss", func(t *testing.T) {
		withFakeList(t, listingJSON(observedID), nil)
		got, err := Verify(context.Background(), Options{SessionID: validMissing, Dir: "/proj"})
		if err != nil {
			t.Fatalf("Verify miss: unexpected error: %v", err)
		}
		if got != VerdictNotFound {
			t.Fatalf("Verify miss = %q, want %q", got, VerdictNotFound)
		}
	})

	t.Run("malformed_json_object", func(t *testing.T) {
		withFakeList(t, []byte(`{"sessions":[]}`), nil)
		got, err := Verify(context.Background(), Options{SessionID: observedID})
		if got != VerdictStoreUnreadable || err == nil {
			t.Fatalf("Verify object stdout = (%q, %v), want store_unreadable with error", got, err)
		}
	})

	t.Run("null_stdout", func(t *testing.T) {
		withFakeList(t, []byte("null"), nil)
		got, err := Verify(context.Background(), Options{SessionID: observedID})
		if got != VerdictStoreUnreadable || err == nil {
			t.Fatalf("Verify null stdout = (%q, %v), want store_unreadable with error", got, err)
		}
	})

	t.Run("garbage_stdout", func(t *testing.T) {
		withFakeList(t, []byte("not json at all"), nil)
		got, err := Verify(context.Background(), Options{SessionID: observedID})
		if got != VerdictStoreUnreadable || err == nil {
			t.Fatalf("Verify garbage stdout = (%q, %v), want store_unreadable with error", got, err)
		}
	})

	t.Run("empty_stdout", func(t *testing.T) {
		withFakeList(t, nil, nil)
		got, err := Verify(context.Background(), Options{SessionID: observedID})
		if got != VerdictStoreUnreadable || err == nil {
			t.Fatalf("Verify empty stdout = (%q, %v), want store_unreadable with error", got, err)
		}
	})

	t.Run("nonzero_exit", func(t *testing.T) {
		withFakeList(t, nil, errors.New("exit status 1"))
		got, err := Verify(context.Background(), Options{SessionID: observedID})
		if got != VerdictStoreUnreadable || err == nil {
			t.Fatalf("Verify nonzero exit = (%q, %v), want store_unreadable with wrapped error", got, err)
		}
	})

	t.Run("empty_id_skips_cli", func(t *testing.T) {
		rec := withFakeList(t, listingJSON(observedID), nil)
		got, err := Verify(context.Background(), Options{SessionID: "   "})
		if got != VerdictInvalidIdentity || err != nil {
			t.Fatalf("Verify whitespace id = (%q, %v), want invalid_identity without error", got, err)
		}
		if rec.calls != 0 {
			t.Fatalf("CLI executed %d times for an invalid id; must not run", rec.calls)
		}
	})

	t.Run("bad_shape_id_skips_cli", func(t *testing.T) {
		rec := withFakeList(t, listingJSON(observedID), nil)
		got, err := Verify(context.Background(), Options{SessionID: "ses_short"})
		if got != VerdictInvalidIdentity || err != nil {
			t.Fatalf("Verify short id = (%q, %v), want invalid_identity without error", got, err)
		}
		if rec.calls != 0 {
			t.Fatalf("CLI executed %d times for an invalid id; must not run", rec.calls)
		}
	})

	t.Run("explicit_bin_dir_maxcount", func(t *testing.T) {
		rec := withFakeList(t, listingJSON(observedID), nil)
		got, err := Verify(context.Background(), Options{SessionID: observedID, Bin: "oc-bin", Dir: "d", MaxCount: 7})
		if got != VerdictResumable || err != nil {
			t.Fatalf("Verify explicit opts = (%q, %v), want resumable", got, err)
		}
		if rec.bin != "oc-bin" || rec.dir != "d" || rec.maxCount != 7 {
			t.Fatalf("runner got bin=%q dir=%q maxCount=%d, want oc-bin d 7", rec.bin, rec.dir, rec.maxCount)
		}
	})

	t.Run("negative_maxcount_falls_back", func(t *testing.T) {
		rec := withFakeList(t, listingJSON(observedID), nil)
		if _, err := Verify(context.Background(), Options{SessionID: observedID, MaxCount: -1}); err != nil {
			t.Fatalf("Verify negative maxcount: %v", err)
		}
		if rec.maxCount != defaultMaxCount {
			t.Fatalf("runner got maxCount=%d, want default %d", rec.maxCount, defaultMaxCount)
		}
	})
}

func TestResumeArgv(t *testing.T) {
	cases := []struct {
		name        string
		bin         string
		sessionID   string
		prompt      string
		interactive bool
		want        []string
	}{
		{
			name:      "headless_no_prompt",
			bin:       "",
			sessionID: observedID,
			want:      []string{"opencode", "run", "--session", observedID},
		},
		{
			name:      "headless_with_prompt",
			bin:       "",
			sessionID: observedID,
			prompt:    "continue where you left off",
			want:      []string{"opencode", "run", "--session", observedID, "continue where you left off"},
		},
		{
			name:        "interactive",
			bin:         "",
			sessionID:   observedID,
			interactive: true,
			want:        []string{"opencode", "--session", observedID},
		},
		{
			name:      "custom_bin",
			bin:       "oc",
			sessionID: observedID,
			prompt:    "p",
			want:      []string{"oc", "run", "--session", observedID, "p"},
		},
		{
			name:      "trimmed_bin",
			bin:       "  opencode  ",
			sessionID: observedID,
			want:      []string{"opencode", "run", "--session", observedID},
		},
		{
			name:      "whitespace_prompt_is_empty",
			bin:       "",
			sessionID: observedID,
			prompt:    "   ",
			want:      []string{"opencode", "run", "--session", observedID},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResumeArgv(tc.bin, tc.sessionID, tc.prompt, tc.interactive)
			if err != nil {
				t.Fatalf("ResumeArgv: unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("ResumeArgv = %v, want %v", got, tc.want)
			}
		})
	}

	badIDs := []string{"", "   ", "ses_short", "ses_abc/def", "ses_" + strings.Repeat("a", 10) + "&"}
	for _, id := range badIDs {
		if _, err := ResumeArgv("", id, "", false); err == nil {
			t.Errorf("ResumeArgv(%q): expected error, got none", id)
		} else if !errors.Is(err, ErrInvalidSessionID) {
			t.Errorf("ResumeArgv(%q): error %v does not wrap ErrInvalidSessionID", id, err)
		}
	}
}

// fak-test:runtime integration est=5s lane=optin
func TestVerifyRealOpenCodeCLI(t *testing.T) {
	if _, err := exec.LookPath("opencode"); err != nil {
		t.Skipf("skipping live witness: opencode CLI not on PATH (%v)", err)
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	// Derive a real id from the same oracle under test, then re-verify it
	// through the public API. Two CLI invocations total.
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	out, err := runListCommand(ctx, "opencode", root, 5)
	if err != nil {
		t.Skipf("skipping live witness: opencode session list failed: %v", err)
	}
	var rows []sessionRow
	if err := json.Unmarshal(out, &rows); err != nil {
		t.Skipf("skipping live witness: unparseable live listing: %v", err)
	}
	var realID string
	for _, row := range rows {
		if IsSessionID(row.ID) {
			realID = row.ID
			break
		}
	}
	if realID == "" {
		t.Skipf("skipping live witness: no verifiable session id in the live listing")
	}

	got, err := Verify(context.Background(), Options{SessionID: realID, Dir: root})
	if err != nil {
		t.Fatalf("Verify(real id %s): unexpected error: %v", realID, err)
	}
	if got != VerdictResumable {
		t.Fatalf("Verify(real id %s) = %q, want %q", realID, got, VerdictResumable)
	}

	missing := "ses_zzzz0000000000000000000000000000"
	got, err = Verify(context.Background(), Options{SessionID: missing, Dir: root})
	if err != nil {
		t.Fatalf("Verify(missing id): unexpected error: %v", err)
	}
	if got != VerdictNotFound {
		t.Fatalf("Verify(missing id) = %q, want %q", got, VerdictNotFound)
	}
}
