package validate

import (
	"context"
	"reflect"
	"testing"
	"time"
)

// TestValidateGoTestTimeoutFollowsDeadline pins #13558: the isolated go test must
// inherit the validate budget instead of Go's 10m default, and its alarm must fire
// inside that budget so a hung test is named by Go's goroutine dump.
func TestValidateGoTestTimeoutFollowsDeadline(t *testing.T) {
	now := time.Date(2026, 9, 27, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		name      string
		remaining time.Duration
		want      time.Duration
	}{
		{name: "90m build-check budget keeps a capped 5m margin", remaining: 90 * time.Minute, want: 85 * time.Minute},
		{name: "default 4m budget keeps a tenth", remaining: 4 * time.Minute, want: 3*time.Minute + 36*time.Second},
		{name: "sub-second remainder truncates to whole seconds", remaining: 100*time.Minute + 999*time.Millisecond, want: 95 * time.Minute},
		{name: "exhausted budget never disables the alarm", remaining: -time.Second, want: time.Second},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithDeadline(context.Background(), now.Add(tc.remaining))
			defer cancel()
			got := validateGoTestTimeout(ctx, now)
			if got != tc.want {
				t.Fatalf("validateGoTestTimeout(remaining %s) = %s, want %s", tc.remaining, got, tc.want)
			}
			if tc.remaining > time.Second && got >= tc.remaining {
				t.Fatalf("timeout %s does not fire before the %s validate deadline", got, tc.remaining)
			}
			args := validateTestArgs("TestX", got, []string{"./cmd/fak"})
			want := []string{"test", "-count=1", "-timeout", tc.want.String(), "-run", "TestX", "./cmd/fak"}
			if !reflect.DeepEqual(args, want) {
				t.Fatalf("validateTestArgs = %q, want %q", args, want)
			}
			parsed, err := time.ParseDuration(args[3])
			if err != nil || parsed != tc.want {
				t.Fatalf("go test cannot read -timeout %q back as %s: %v", args[3], tc.want, err)
			}
		})
	}
}

func TestValidateTestArgsWithoutDeadlineKeepsGoDefault(t *testing.T) {
	if got := validateGoTestTimeout(context.Background(), time.Now()); got != 0 {
		t.Fatalf("validateGoTestTimeout(no deadline) = %s, want 0 (omit -timeout)", got)
	}
	args := validateTestArgs("", 0, []string{"./internal/validate"})
	if want := []string{"test", "-count=1", "./internal/validate"}; !reflect.DeepEqual(args, want) {
		t.Fatalf("validateTestArgs(no timeout) = %q, want %q", args, want)
	}
}

// TestValidateJSONTestArgsKeepsTimeout covers the test_audit_full argv shape.
func TestValidateJSONTestArgsKeepsTimeout(t *testing.T) {
	args := validateJSONTestArgs(validateTestArgs("", 85*time.Minute, []string{"./cmd/fak"}))
	want := []string{"test", "-json", "-count=1", "-timeout", "1h25m0s", "./cmd/fak"}
	if !reflect.DeepEqual(args, want) {
		t.Fatalf("validateJSONTestArgs(validateTestArgs) = %q, want %q", args, want)
	}
}
