package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

type guardRegistration struct {
	Store  sessionregistry.Store
	Record sessionregistry.Record
}

var guardRegistrations sync.Map   // *exec.Cmd -> guardRegistration
var guardTerminalIntents sync.Map // *exec.Cmd -> reason

func prepareGuardChildRegistration(meta guardChildSpawnMetadata, grant toolprocgate.SpawnGrant) (guardRegistration, error) {
	env := envMapFromGrant(grant.Env)
	path := strings.TrimSpace(meta.RegistryPath)
	if path == "" {
		path = sessionregistry.DefaultPath()
	}
	attempt := firstGuardEnv(env, "FAK_CHILD_ATTEMPT_ID")
	if attempt == "" {
		attempt = grant.GrantID
	}
	parent := firstGuardEnv(env, "FAK_REGISTRATION_ID", "FAK_PARENT_REGISTRATION_ID")
	parentAttempt := firstGuardEnv(env, "FAK_ATTEMPT_ID", "FAK_PARENT_ATTEMPT_ID")
	root := firstGuardEnv(env, "FAK_ROOT_REGISTRATION_ID")
	if parent != "" && root == "" {
		root = parent
	}
	rec, err := sessionregistry.New(sessionregistry.NewInput{RegistrationID: firstGuardEnv(env, "FAK_CHILD_REGISTRATION_ID"), ParentRegistrationID: parent, ParentAttemptID: parentAttempt, RootRegistrationID: root, RootOutcome: firstGuardEnv(env, "FAK_ROOT_OUTCOME"), RootIssue: firstGuardEnv(env, "FAK_ROOT_ISSUE", "DISPATCH_ISSUE"), TaskID: firstGuardEnv(env, "FAK_TASK_ID", "DISPATCH_ISSUE"), GoalID: firstGuardEnv(env, "FAK_GOAL_ID"), AttemptID: attempt, ResumeOfAttemptID: firstGuardEnv(env, "FAK_RESUME_OF_ATTEMPT_ID"), LaunchKind: guardLaunchKind(env, grant), Scope: guardScope(env), Lane: firstGuardEnv(env, "FAK_LANE", "DISPATCH_LANE"), LeaseID: firstGuardEnv(env, "FAK_LEASE_ID"), Runtime: guardRuntime(grant), SessionID: firstGuardEnv(env, "FAK_SESSION_ID"), ThreadID: firstGuardEnv(env, "FAK_THREAD_ID"), HostID: firstGuardEnv(env, "COMPUTERNAME", "HOSTNAME")})
	if err != nil {
		return guardRegistration{}, err
	}
	g := guardRegistration{Store: sessionregistry.Store{Path: path}, Record: rec}
	if err = g.Store.Register(rec); err != nil {
		return guardRegistration{}, fmt.Errorf("guard child registration persist failed (child not started): %w", err)
	}
	return g, nil
}
func withGuardRegistrationEnv(g toolprocgate.SpawnGrant, reg guardRegistration) toolprocgate.SpawnGrant {
	m := envMapFromGrant(g.Env)
	r := reg.Record
	m["FAK_REGISTRATION_ID"] = r.RegistrationID
	m["FAK_ATTEMPT_ID"] = r.AttemptID
	m["FAK_PARENT_REGISTRATION_ID"] = r.ParentRegistrationID
	m["FAK_PARENT_ATTEMPT_ID"] = r.ParentAttemptID
	m["FAK_ROOT_REGISTRATION_ID"] = r.RootRegistrationID
	m["FAK_ROOT_OUTCOME"] = r.RootOutcome
	m["FAK_ROOT_ISSUE"] = r.RootIssue
	m["FAK_TASK_ID"] = r.TaskID
	m["FAK_GOAL_ID"] = r.GoalID
	g.Env = envVarsFromMap(m)
	return g
}
func startGuardRegistration(g guardRegistration, child *exec.Cmd) error {
	if child == nil || child.Process == nil {
		return fmt.Errorf("guard child launcher returned no started process")
	}
	_, err := g.Store.Start(g.Record.RegistrationID, child.Process.Pid, time.Now().UTC())
	return err
}
func startBoundGuardRegistration(child *exec.Cmd) error {
	v, ok := guardRegistrations.Load(child)
	if !ok {
		return fmt.Errorf("guard child registration binding missing")
	}
	g := v.(guardRegistration)
	if err := startGuardRegistration(g, child); err != nil {
		terminalGuardRegistration(g, sessionregistry.StateFailed, "start_readback_failed", "")
		guardRegistrations.Delete(child)
		return err
	}
	return nil
}

func bindGuardRegistration(child *exec.Cmd, g guardRegistration) {
	if child != nil {
		guardRegistrations.Store(child, g)
	}
}
func terminalGuardRegistration(g guardRegistration, state sessionregistry.State, reason, witness string) {
	_, _ = g.Store.Terminal(g.Record.RegistrationID, state, reason, witness, time.Now().UTC())
}
func markGuardChildTerminalIntent(child *exec.Cmd, reason string) {
	if child != nil {
		guardTerminalIntents.Store(child, reason)
	}
}
func terminalGuardChild(child *exec.Cmd, runErr error, reason string) {
	if v, ok := guardTerminalIntents.LoadAndDelete(child); ok && reason == "" {
		reason = v.(string)
	}
	v, ok := guardRegistrations.LoadAndDelete(child)
	if !ok {
		return
	}
	g := v.(guardRegistration)
	state := sessionregistry.StateCompleted
	if runErr != nil {
		state = sessionregistry.StateFailed
	}
	if reason == "restart" || reason == "cancelled" || reason == "time_budget" {
		state = sessionregistry.StateCancelled
	}
	if reason == "lost" {
		state = sessionregistry.StateLost
	}
	if reason == "" && runErr != nil {
		reason = "worker_exit_failed"
		if ee, ok := runErr.(*exec.ExitError); ok {
			reason = fmt.Sprintf("worker_exit_%d", ee.ExitCode())
		}
	}
	terminalGuardRegistration(g, state, reason, firstGuardEnv(envMapFromStrings(child.Env), "FAK_WITNESS_REF"))
}
func envMapFromGrant(in []toolprocgate.EnvVar) map[string]string {
	m := map[string]string{}
	for _, v := range in {
		m[v.Name] = v.Value
	}
	return m
}
func envMapFromStrings(in []string) map[string]string {
	m := map[string]string{}
	for _, v := range in {
		if i := strings.IndexByte(v, '='); i > 0 {
			m[v[:i]] = v[i+1:]
		}
	}
	return m
}
func envVarsFromMap(m map[string]string) []toolprocgate.EnvVar {
	out := make([]toolprocgate.EnvVar, 0, len(m))
	for k, v := range m {
		out = append(out, toolprocgate.EnvVar{Name: k, Value: v})
	}
	return out
}
func firstGuardEnv(m map[string]string, n ...string) string {
	for _, k := range n {
		if v := strings.TrimSpace(m[k]); v != "" {
			return v
		}
	}
	return ""
}
func guardRuntime(g toolprocgate.SpawnGrant) string {
	if v := strings.TrimSpace(g.Backend); v != "" {
		return v
	}
	return "guarded-agent"
}
func guardLaunchKind(env map[string]string, g toolprocgate.SpawnGrant) string {
	if firstGuardEnv(env, "FAK_RESUME_OF_ATTEMPT_ID") != "" {
		return "resume_wrapper"
	}
	return "guarded_tui"
}
func guardScope(env map[string]string) []string {
	if v := firstGuardEnv(env, "DISPATCH_WORKSPACE", "FAK_WORKSPACE"); v != "" {
		return []string{v}
	}
	if cwd, err := os.Getwd(); err == nil {
		return []string{cwd}
	}
	return nil
}
