package codexresume

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"time"
)

// LoggedOutRefreshFailure is the exact provider-authored terminal message that
// means a rollout lost the Codex account which originally owned it.
const LoggedOutRefreshFailure = "Your access token could not be refreshed because you have since logged out or signed in to another account. Please sign in again."

var ErrRecoveryRetryFailed = errors.New("codexresume: recovery retry did not complete")

// RecoveryHome describes a caller-enrolled Codex home. AccountKey is a stable,
// non-secret account identity; auth remains resident in Home and is never read
// or copied by this package.
type RecoveryHome struct {
	Home       string
	AccountKey string
	Eligible   bool
	Healthy    bool
}

// RecoveryTarget is the credential-free target handed to the single retry.
type RecoveryTarget struct {
	Home        string
	RolloutPath string
	Binding     ThreadBinding
}

// RecoveryAttempt preserves the authoritative original failure even when a
// rehome or retry fails. Result changes only after a completed retry.
type RecoveryAttempt struct {
	Original    Result
	Result      Result
	RetryResult *Result
	Attempted   bool
	Recovered   bool
	TargetHome  string
	RolloutPath string
	Binding     *ThreadBinding
}

// IsLoggedOutRefreshFailure recognizes only the exact provider-authored task
// failure. Process stderr/stdout and similar-looking errors are not inputs.
func IsLoggedOutRefreshFailure(result Result) bool {
	return result.Outcome == OutcomeTurnFailed &&
		result.TurnError != nil &&
		result.TurnError.Message == LoggedOutRefreshFailure
}

// RecoverLoggedOutRefresh rehomes one rollout to the first different eligible,
// healthy account/home and invokes retry exactly once. It never reads or copies
// auth state. If no safe target exists, copying fails, or the retry does not
// complete, Result remains the original authoritative failure.
func RecoverLoggedOutRefresh(
	ctx context.Context,
	failed Result,
	binding ThreadBinding,
	homes []RecoveryHome,
	retry func(context.Context, RecoveryTarget) (Result, error),
) (RecoveryAttempt, error) {
	attempt := RecoveryAttempt{Original: failed, Result: failed}
	if !IsLoggedOutRefreshFailure(failed) {
		return attempt, nil
	}
	if retry == nil {
		return attempt, errors.New("codexresume: recovery retry is required")
	}
	if err := validateBinding(binding); err != nil {
		return attempt, fmt.Errorf("codexresume: recovery binding: %w", err)
	}

	home, ok := selectRecoveryHome(binding, homes)
	if !ok {
		return attempt, nil
	}
	attempt.Attempted = true
	attempt.TargetHome = home.Home

	sourcePath := filepath.Join(binding.CanonicalHome, filepath.FromSlash(binding.RelativeRolloutPath))
	copied, err := CopyRollout(binding.CanonicalHome, home.Home, sourcePath)
	if err != nil {
		return attempt, fmt.Errorf("codexresume: recovery copy rollout: %w", err)
	}
	attempt.RolloutPath = copied.Path

	targetBinding, err := NewThreadBinding(binding.ThreadID, home.Home, home.AccountKey, copied.Path, time.Now().UTC())
	if err != nil {
		return attempt, fmt.Errorf("codexresume: recovery target binding: %w", err)
	}
	attempt.Binding = &targetBinding
	target := RecoveryTarget{Home: targetBinding.CanonicalHome, RolloutPath: copied.Path, Binding: targetBinding}

	retried, retryErr := retry(ctx, target)
	attempt.RetryResult = &retried
	if retryErr != nil {
		return attempt, fmt.Errorf("codexresume: recovery retry: %w", retryErr)
	}
	if !retried.TaskCompleted || (retried.Outcome != OutcomeCompleted && retried.Outcome != OutcomeCompletedReclaimed) {
		return attempt, ErrRecoveryRetryFailed
	}
	attempt.Result = retried
	attempt.Recovered = true
	return attempt, nil
}

func selectRecoveryHome(binding ThreadBinding, homes []RecoveryHome) (RecoveryHome, bool) {
	for _, home := range homes {
		if !home.Eligible || !home.Healthy || home.AccountKey == "" {
			continue
		}
		if ClassifyRehome(&binding, home.Home, home.AccountKey, false) != RehomeDifferentHomeDifferentAccount {
			continue
		}
		canonical, err := canonicalHomePath(home.Home)
		if err != nil {
			continue
		}
		home.Home = canonical
		return home, true
	}
	return RecoveryHome{}, false
}
