package microagent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

// Lineage binds in-process microagent executions to the same durable ancestry graph as
// process-backed guard and dispatchworker children. It deliberately records an execution
// identity, not a process identity: many microagents may share one PID while retaining
// distinct registration/session ids and parent edges.
type Lineage struct {
	Store                sessionregistry.Store
	ParentRegistrationID string
	ParentAttemptID      string
	RootRegistrationID   string
	RootIssue            string
	TaskID               string
	GoalID               string
	HostID               string
	Now                  func() time.Time
}

// LineageFromEnv inherits the registration edge exported to a nested launcher. It returns
// nil when the caller is not itself registered, so standalone library users do not
// accidentally write an unrooted graph. A partially-present edge is an error rather than
// silently losing the starting goal.
func LineageFromEnv() (*Lineage, error) {
	parent := firstLineageEnv("FAK_REGISTRATION_ID", "FAK_PARENT_REGISTRATION_ID")
	root := strings.TrimSpace(os.Getenv("FAK_ROOT_REGISTRATION_ID"))
	if parent == "" && root == "" {
		return nil, nil
	}
	if parent == "" || root == "" {
		return nil, errors.New("microagent lineage requires both parent and root registration ids")
	}
	path := strings.TrimSpace(os.Getenv("FAK_SESSION_REGISTRY"))
	if path == "" {
		path = sessionregistry.DefaultPath()
	}
	return &Lineage{
		Store:                sessionregistry.Store{Path: path},
		ParentRegistrationID: parent,
		ParentAttemptID:      firstLineageEnv("FAK_ATTEMPT_ID", "FAK_PARENT_ATTEMPT_ID"),
		RootRegistrationID:   root,
		RootIssue:            firstLineageEnv("FAK_ROOT_ISSUE", "DISPATCH_ISSUE"),
		TaskID:               firstLineageEnv("FAK_TASK_ID", "DISPATCH_ISSUE"),
		GoalID:               firstLineageEnv("FAK_GOAL_ID"),
		HostID:               firstLineageEnv("COMPUTERNAME", "HOSTNAME"),
	}, nil
}

// Child returns the lineage inherited by a nested microagent launched from logicalID.
// The registration id is deterministic from the parent edge and logical id, so the edge
// can be handed down before execution while Store.Register still detects conflicting
// replay. The parent must register before the nested child begins.
func (l *Lineage) Child(logicalID string) *Lineage {
	if l == nil {
		return nil
	}
	child := *l
	child.ParentRegistrationID = microagentRegistrationID(l.ParentRegistrationID, strings.TrimSpace(logicalID))
	child.ParentAttemptID = child.ParentRegistrationID
	return &child
}

// WithLineage wraps one logical agent. Registration is persisted before its first Step;
// completion, error, or cancellation is terminalized before Step returns. Registration
// failures are fail-closed, matching process-backed child launch semantics.
func WithLineage(agentID string, inner Microagent, lineage *Lineage) Microagent {
	if lineage == nil {
		return inner
	}
	return &lineageAgent{id: strings.TrimSpace(agentID), inner: inner, lineage: lineage}
}

type lineageAgent struct {
	id       string
	inner    Microagent
	lineage  *Lineage
	record   sessionregistry.Record
	started  bool
	terminal bool
}

func (a *lineageAgent) Step(ctx context.Context, gateway Gateway) (bool, error) {
	if a.terminal {
		return true, nil
	}
	if !a.started {
		if err := a.start(); err != nil {
			return false, err
		}
	}
	done, runErr := a.inner.Step(ctx, gateway)
	if runErr == nil && !done {
		return false, nil
	}
	state, reason := sessionregistry.StateCompleted, ""
	if runErr != nil {
		state, reason = sessionregistry.StateFailed, runErr.Error()
		if errors.Is(runErr, context.Canceled) || errors.Is(ctx.Err(), context.Canceled) {
			state = sessionregistry.StateCancelled
		}
	}
	if _, err := a.lineage.Store.Terminal(a.record.RegistrationID, state, reason, "", a.now()); err != nil {
		return false, fmt.Errorf("terminalize microagent %q: %w", a.id, err)
	}
	a.terminal = true
	return done, runErr
}

func (a *lineageAgent) start() error {
	if a.inner == nil {
		return errors.New("microagent lineage: inner agent is nil")
	}
	if a.id == "" {
		return errors.New("microagent lineage: logical agent id is required")
	}
	if strings.TrimSpace(a.lineage.ParentRegistrationID) == "" || strings.TrimSpace(a.lineage.RootRegistrationID) == "" {
		return errors.New("microagent lineage: parent and root registration ids are required")
	}
	now := a.now()
	record, err := sessionregistry.New(sessionregistry.NewInput{
		RegistrationID:       microagentRegistrationID(a.lineage.ParentRegistrationID, a.id),
		ParentRegistrationID: a.lineage.ParentRegistrationID,
		ParentAttemptID:      a.lineage.ParentAttemptID,
		RootRegistrationID:   a.lineage.RootRegistrationID,
		RootIssue:            a.lineage.RootIssue,
		TaskID:               a.lineage.TaskID,
		GoalID:               a.lineage.GoalID,
		LaunchKind:           "in_process_microagent",
		Runtime:              "microagent",
		SessionID:            microagentSessionID(a.lineage.ParentRegistrationID, a.id),
		HostID:               a.lineage.HostID,
		Now:                  now,
	})
	if err != nil {
		return fmt.Errorf("create microagent registration %q: %w", a.id, err)
	}
	if err := a.lineage.Store.Register(record); err != nil {
		return fmt.Errorf("register microagent %q: %w", a.id, err)
	}
	active, err := a.lineage.Store.Start(record.RegistrationID, os.Getpid(), now)
	if err != nil {
		return fmt.Errorf("start microagent registration %q: %w", a.id, err)
	}
	a.record, a.started = active, true
	return nil
}

func (a *lineageAgent) now() time.Time {
	if a.lineage != nil && a.lineage.Now != nil {
		return a.lineage.Now().UTC()
	}
	return time.Now().UTC()
}

func microagentRegistrationID(parent, logical string) string {
	return microagentIdentity("micro-reg-", parent, logical)
}
func microagentSessionID(parent, logical string) string {
	return microagentIdentity("micro-session-", parent, logical)
}
func microagentIdentity(prefix, parent, logical string) string {
	sum := sha256.Sum256([]byte(parent + "\x00" + logical))
	return prefix + hex.EncodeToString(sum[:8])
}

func firstLineageEnv(names ...string) string {
	for _, name := range names {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			return value
		}
	}
	return ""
}
