//go:build linux

// Command v41epguard is the driver-free, one-owner supervisor for the manual
// v41-ep-owner benchmark. It cannot select an arbitrary child command or rank set.
// Registration through the existing benchmark route is a separate prerequisite.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

const (
	guardPlanSchema    = "fak.v41-ep-owner.launch/v1"
	guardReceiptSchema = "fak.v41-ep-owner.supervisor/v1"
	guardMagic         = "FKEPGRD1"
	guardMaxPlan       = 2 << 20
	guardMaxRequest    = 1 << 20
	guardMaxControl    = 4 << 10
	guardMaxReceipt    = 2 << 20
	guardMaxOutput     = 1 << 20
	guardMaxImage      = 512 << 20
	guardMaxTerminate  = 600 * time.Second
	guardMaxKillGrace  = 5 * time.Second
	guardStartup       = 5 * time.Second
	guardRetireGrace   = 5 * time.Second
	guardMemoryFloor   = 5 * (1 << 29)
	guardDiskFloor     = 3 * (1 << 30)
)

// The request is opaque canonical JSON here. Importing localbench to decode it
// would pull GPU package initializers into this otherwise driver-free program.
// The owner independently decodes it and checks the exact canonical digest.
type guardPlan struct {
	Schema             string        `json:"schema"`
	HelperSHA256       string        `json:"helper_sha256"`
	OwnerPath          string        `json:"owner_path"`
	OwnerSHA256        string        `json:"owner_sha256"`
	Request            []byte        `json:"request"`
	RequestSHA256      string        `json:"request_sha256"`
	Rank               *int          `json:"rank"`
	ContractID         string        `json:"contract_id"`
	TerminateNS        time.Duration `json:"terminate_ns"`
	KillGraceNS        time.Duration `json:"kill_grace_ns"`
	Environment        []string      `json:"environment"`
	ResourcePath       string        `json:"resource_path"`
	MinimumMemoryBytes uint64        `json:"minimum_memory_bytes"`
	MinimumDiskBytes   uint64        `json:"minimum_disk_bytes"`
}

type guardHello struct {
	AdmissionUnixNS int64  `json:"admission_unix_ns"`
	Version         int    `json:"version"`
	Rank            int    `json:"rank"`
	RequestSHA256   string `json:"request_sha256"`
	ContractID      string `json:"contract_id"`
	ParentPID       int    `json:"parent_pid"`
	ParentStart     string `json:"parent_start"`
	ParentSHA256    string `json:"parent_sha256"`
	OwnerSHA256     string `json:"owner_sha256"`
	TerminateUnixNS int64  `json:"terminate_unix_ns"`
	KillUnixNS      int64  `json:"kill_unix_ns"`
	ReadPipeInode   uint64 `json:"read_pipe_inode"`
	WritePipeInode  uint64 `json:"write_pipe_inode"`
}

type guardChallenge struct {
	Nonce                 string `json:"nonce"`
	ContractID            string `json:"contract_id"`
	RequestSHA256         string `json:"request_sha256"`
	Rank                  int    `json:"rank"`
	PID                   int    `json:"pid"`
	ExecutableSHA256      string `json:"executable_sha256"`
	LatestTerminateUnixNS int64  `json:"latest_terminate_unix_ns"`
	KillGraceNS           int64  `json:"kill_grace_ns"`
}

type guardAdmission struct {
	Challenge       guardChallenge `json:"challenge"`
	ParentPID       int            `json:"parent_pid"`
	ParentStart     string         `json:"parent_start"`
	OwnerStart      string         `json:"owner_start"`
	TerminateUnixNS int64          `json:"terminate_unix_ns"`
	KillUnixNS      int64          `json:"kill_unix_ns"`
	EvidenceSHA256  string         `json:"evidence_sha256"`
}

type guardSignal struct {
	Signal string    `json:"signal"`
	At     time.Time `json:"at"`
	Error  string    `json:"error,omitempty"`
}

// This value has no JSON/env/flag constructor. The test issuer is confined to
// _test.go and qualifies only its actual held test image. Production has no
// issuer yet, so main refuses before Start. A later native issuer requires a
// separately reviewed artifact receipt, independently pinned outside the plan,
// binding actual owner/helper image hashes, the exact ordered environment hash,
// source/build identity, explicit backend, and qualified loading-artifact hashes.
// The issuer must check the receipt's scope and actual held artifacts before
// constructing this value; caller-declared digest strings are not evidence.
type guardArtifactAdmission struct {
	ownerSHA, environmentSHA, evidenceSHA, kind string
}

type guardArtifactEvidence struct {
	Kind              string `json:"kind"`
	OwnerSHA256       string `json:"owner_sha256"`
	EnvironmentSHA256 string `json:"environment_sha256"`
	EvidenceSHA256    string `json:"evidence_sha256"`
}

func guardEnvironmentSHA(environment []string) string {
	raw, _ := json.Marshal(environment)
	return guardSHA(raw)
}

func (a guardArtifactAdmission) observe(owner *guardImage, environment []string) (*guardArtifactEvidence, error) {
	if owner == nil || a.kind != "test-image-qualification" || a.ownerSHA != owner.digest || a.environmentSHA != guardEnvironmentSHA(environment) || !guardDigest(a.evidenceSHA) {
		return nil, errors.New("v41epguard: independently qualified loading artifacts unavailable; native launch refused")
	}
	return &guardArtifactEvidence{a.kind, a.ownerSHA, a.environmentSHA, a.evidenceSHA}, nil
}

type guardResult struct {
	ArtifactQualification      *guardArtifactEvidence `json:"artifact_qualification"`
	NativeLoadingQualification string                 `json:"native_loading_qualification"`
	AdmissionDeadline          time.Time              `json:"admission_deadline"`
	TerminatingSignal          *string                `json:"terminating_signal"`
	ResourceSamples            []guardResource        `json:"resource_samples"`
	PlanSHA256                 string                 `json:"plan_sha256"`
	Schema                     string                 `json:"schema"`
	Status                     string                 `json:"status"`
	RequestSHA256              string                 `json:"request_sha256"`
	Rank                       int                    `json:"rank"`
	HelperSHA256               string                 `json:"helper_sha256"`
	OwnerSHA256                string                 `json:"owner_sha256"`
	PID                        int                    `json:"pid"`
	PGID                       int                    `json:"pgid"`
	ProcessStart               string                 `json:"process_start"`
	StartedAt                  time.Time              `json:"started_at"`
	TerminateAt                time.Time              `json:"terminate_at"`
	KillAt                     time.Time              `json:"kill_at"`
	FinishedAt                 time.Time              `json:"finished_at"`
	AdmissionSHA256            string                 `json:"admission_sha256,omitempty"`
	Signals                    []guardSignal          `json:"signals"`
	WaitJoined                 bool                   `json:"wait_joined"`
	ExitCode                   *int                   `json:"exit_code"`
	GroupReleased              *bool                  `json:"group_released"`
	ChildReceipt               json.RawMessage        `json:"child_receipt,omitempty"`
	ChildReceiptSHA256         string                 `json:"child_receipt_sha256,omitempty"`
	Stdout                     []byte                 `json:"stdout"`
	Stderr                     []byte                 `json:"stderr"`
	Failure                    string                 `json:"failure,omitempty"`
}

func main() {
	if len(os.Args) != 3 || os.Args[1] != "--plan" {
		fmt.Fprintln(os.Stderr, "usage: v41epguard --plan APPROVED_LAUNCH.json")
		os.Exit(2)
	}
	raw, err := guardReadFile(os.Args[2], guardMaxPlan)
	var plan guardPlan
	if err == nil {
		err = guardStrict(raw, &plan)
	}
	if err == nil {
		err = plan.validate()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	result := guardRun(context.Background(), plan, guardArtifactAdmission{})
	if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
		os.Exit(1)
	}
	if result.Status != "complete" {
		os.Exit(1)
	}
}

func (p guardPlan) validate() error {
	if p.Schema != guardPlanSchema || !guardDigest(p.HelperSHA256) || !guardDigest(p.OwnerSHA256) || !guardDigest(p.RequestSHA256) ||
		p.Rank == nil || *p.Rank < 0 || *p.Rank > 2 || len(p.ContractID) == 0 || len(p.ContractID) > 128 || strings.TrimSpace(p.ContractID) != p.ContractID ||
		len(p.Request) == 0 || len(p.Request) > guardMaxRequest || !json.Valid(p.Request) || guardSHA(p.Request) != p.RequestSHA256 ||
		!guardAbsolute(p.OwnerPath) || !guardAbsolute(p.ResourcePath) ||
		p.TerminateNS <= 0 || p.TerminateNS > guardMaxTerminate || p.KillGraceNS <= 0 || p.KillGraceNS > guardMaxKillGrace ||
		p.MinimumMemoryBytes < guardMemoryFloor || p.MinimumDiskBytes < guardDiskFloor {
		return errors.New("v41epguard: invalid bounded launch plan")
	}
	// Only launch-local runtime inputs are accepted. No inherited credentials,
	// shell expansion or unrelated behavioral FAK_* namespace is copied.
	// Driver/shader code-loading variables are refused in this leaf. Device
	// selectors are not artifact qualification or authentication of system drivers.
	allowed := map[string]bool{"LANG": true, "LC_ALL": true, "TZ": true, "ROCR_VISIBLE_DEVICES": true, "CUDA_VISIBLE_DEVICES": true}
	seen := map[string]bool{}
	for _, item := range p.Environment {
		key, value, ok := strings.Cut(item, "=")
		if !ok || !allowed[key] || seen[key] || value == "" || len(value) > 4096 || strings.ContainsAny(value, "\x00\r\n") {
			return errors.New("v41epguard: invalid explicit environment")
		}
		seen[key] = true
	}
	if !seen["LANG"] || !seen["LC_ALL"] || !seen["TZ"] {
		return errors.New("v41epguard: explicit locale and timezone are required")
	}
	return nil
}

type guardProcess struct {
	pid, ppid, pgid int
	start           string
}
type guardImage struct {
	file   *os.File
	stat   syscall.Stat_t
	digest string
}
type guardIOResult struct {
	data []byte
	err  error
}
type guardExchange struct {
	receipt  []byte
	evidence string
	err      error
}

// guardRun owns exactly one direct child and its new process group. There is no
// escaped-descendant containment, nor an assurance against a hostile same-UID
// process. Kernel-uninterruptible tasks may remain after SIGKILL; no release is
// reported until it is actually observed. A host guard remains the final boundary.
func guardRun(ctx context.Context, p guardPlan, admission guardArtifactAdmission) (result guardResult) {
	result = guardResult{Schema: guardReceiptSchema, Status: "refused", NativeLoadingQualification: "unresolved", RequestSHA256: p.RequestSHA256, HelperSHA256: p.HelperSHA256, OwnerSHA256: p.OwnerSHA256}
	if err := p.validate(); err != nil {
		result.Failure = err.Error()
		return
	}
	result.Rank = *p.Rank
	planBytes, _ := json.Marshal(p)
	result.PlanSHA256 = guardSHA(planBytes)
	var failures []error
	add := func(err error) {
		if err != nil {
			failures = append(failures, err)
		}
	}
	defer func() {
		result.FinishedAt = time.Now()
		if len(failures) != 0 {
			result.Failure = errors.Join(failures...).Error()
			if result.Status != "refused" {
				result.Status = "incomplete"
			}
		}
	}()
	if ctx == nil || ctx.Err() != nil {
		add(errors.New("v41epguard: live context required"))
		return
	}
	self, err := guardOpenImage("/proc/self/exe", p.HelperSHA256, true)
	if err != nil {
		add(err)
		return
	}
	defer func() { add(self.file.Close()) }()
	owner, err := guardOpenImage(p.OwnerPath, p.OwnerSHA256, false)
	if err != nil {
		add(err)
		return
	}
	defer func() { add(owner.file.Close()) }()
	result.ArtifactQualification, err = admission.observe(owner, p.Environment)
	if err != nil {
		add(err)
		return
	}
	parent, err := guardObserveProcess(os.Getpid())
	if err != nil {
		add(err)
		return
	}
	initial, err := guardResources(p.ResourcePath)
	if err != nil {
		add(err)
		return
	}
	if err := initial.check(p, initial); err != nil {
		add(err)
		return
	}
	var files []*os.File
	defer func() {
		for _, f := range files {
			_ = f.Close()
		}
	}()
	pipe := func() (*os.File, *os.File, error) {
		r, w, e := os.Pipe()
		if e == nil {
			files = append(files, r, w)
		}
		return r, w, e
	}
	controlR, controlW, err := pipe()
	if err != nil {
		add(err)
		return
	}
	eventR, eventW, err := pipe()
	if err != nil {
		add(err)
		return
	}
	stdoutR, stdoutW, err := pipe()
	if err != nil {
		add(err)
		return
	}
	stderrR, stderrW, err := pipe()
	if err != nil {
		add(err)
		return
	}
	stdin, err := os.Open(os.DevNull)
	if err != nil {
		add(err)
		return
	}
	files = append(files, stdin)
	readInode, err := guardPipeInode(controlR)
	if err != nil {
		add(err)
		return
	}
	writeInode, err := guardPipeInode(eventW)
	if err != nil {
		add(err)
		return
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	result.StartedAt = time.Now()
	result.TerminateAt = result.StartedAt.Add(p.TerminateNS)
	result.KillAt = result.TerminateAt.Add(p.KillGraceNS)
	result.AdmissionDeadline = guardEarlier(result.StartedAt.Add(guardStartup), result.TerminateAt)
	// A wedged Start/Wait must not keep an unbounded live supervisor. The owned
	// child's pre-exec PDEATHSIG also covers this abrupt failure. No terminal or
	// release receipt is manufactured when this last backstop exits the helper.
	backstop := time.AfterFunc(time.Until(result.KillAt.Add(guardRetireGrace)), func() { os.Exit(124) })
	defer backstop.Stop()
	watchdog := newGuardDeadline(result.AdmissionDeadline, result.TerminateAt, result.KillAt, p.KillGraceNS)
	defer watchdog.close()
	monitor := newGuardResourceMonitor(p, initial, watchdog.abort)
	var cmd *exec.Cmd
	startedChild, waited := false, false
	defer func() {
		if startedChild {
			if !waited {
				add(watchdog.retire(cmd))
				waited = true
			}
			result.WaitJoined = true
			if cmd.ProcessState != nil {
				code := cmd.ProcessState.ExitCode()
				result.ExitCode = &code
				if status, ok := cmd.ProcessState.Sys().(syscall.WaitStatus); ok && status.Signaled() {
					signal := status.Signal().String()
					result.TerminatingSignal = &signal
				}
			}
			if err := syscall.Kill(-cmd.Process.Pid, 0); errors.Is(err, syscall.ESRCH) {
				released := true
				result.GroupReleased = &released
			} else {
				add(errors.New("v41epguard: process-group release not established"))
			}
		}
		result.ResourceSamples, err = monitor.close()
		add(err)
		result.Signals, err = watchdog.snapshot()
		add(err)
	}()
	if err := monitor.failure(); err != nil {
		add(err)
		return
	}
	pidfd := -1
	cmd = exec.Command("/proc/self/fd/3", "benchmarks", "run", "v41-ep-owner", "--", "--supervised-owner")
	cmd.Env = append([]string(nil), p.Environment...)
	cmd.Dir = "/"
	cmd.Stdin, cmd.Stdout, cmd.Stderr = stdin, stdoutW, stderrW
	cmd.ExtraFiles = []*os.File{owner.file, controlR, eventW}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL, PidFD: &pidfd}
	if err := cmd.Start(); err != nil {
		add(err)
		return
	}
	startedChild = true
	result.Status, result.PID, result.PGID = "incomplete", cmd.Process.Pid, cmd.Process.Pid
	_ = controlR.Close()
	_ = eventW.Close()
	_ = stdoutW.Close()
	_ = stderrW.Close()
	_ = stdin.Close()
	if pidfd >= 0 {
		defer syscall.Close(pidfd)
	}
	watchdog.publish(cmd)
	signal := watchdog.signal
	if pidfd < 0 {
		add(errors.New("v41epguard: atomic pidfd unavailable"))
		return
	}
	if _, err := guardTerminal(pidfd); err != nil {
		add(err)
		return
	}
	actual, err := guardObserveProcess(cmd.Process.Pid)
	if err != nil {
		add(err)
		return
	}
	if actual.ppid != parent.pid || actual.pgid != actual.pid {
		add(errors.New("v41epguard: child ownership disagrees"))
		return
	}
	result.ProcessStart = actual.start
	actualImage, err := guardOpenImage(fmt.Sprintf("/proc/%d/exe", actual.pid), p.OwnerSHA256, true)
	if err != nil {
		add(err)
		return
	}
	imageMatch := actualImage.stat.Dev == owner.stat.Dev && actualImage.stat.Ino == owner.stat.Ino
	add(actualImage.file.Close())
	after, err := guardObserveProcess(actual.pid)
	if err != nil || after != actual || !imageMatch {
		add(errors.Join(err, errors.New("v41epguard: actual held child image changed")))
		return
	}
	hello := guardHello{AdmissionUnixNS: result.AdmissionDeadline.UnixNano(), Version: 1, Rank: *p.Rank, RequestSHA256: p.RequestSHA256, ContractID: p.ContractID, ParentPID: parent.pid,
		ParentStart: parent.start, ParentSHA256: self.digest, OwnerSHA256: owner.digest, TerminateUnixNS: result.TerminateAt.UnixNano(),
		KillUnixNS: result.KillAt.UnixNano(), ReadPipeInode: readInode, WritePipeInode: writeInode}
	stdoutDone, stderrDone := make(chan guardIOResult, 1), make(chan guardIOResult, 1)
	go func() {
		data, err := io.ReadAll(io.LimitReader(stdoutR, guardMaxOutput+1))
		if len(data) > guardMaxOutput {
			data, err = data[:guardMaxOutput], errors.New("stdout cap exceeded")
		}
		stdoutDone <- guardIOResult{data, err}
	}()
	go func() {
		data, err := io.ReadAll(io.LimitReader(stderrR, guardMaxOutput+1))
		if len(data) > guardMaxOutput {
			data, err = data[:guardMaxOutput], errors.New("stderr cap exceeded")
		}
		stderrDone <- guardIOResult{data, err}
	}()
	exchangeDone := make(chan guardExchange, 1)
	go func() {
		exchangeDone <- guardProtocol(controlW, eventR, p, hello, actual, result.StartedAt, result.TerminateAt, result.KillAt, watchdog.admit)
	}()
	var exchange *guardExchange
	var stdout, stderr *guardIOResult
	stopAt, killAt := result.TerminateAt, result.KillAt
	termSent, killSent := false, false
	for {
		now := time.Now()
		if ctx.Err() != nil && stopAt.After(now) {
			add(ctx.Err())
			stopAt = now
			killAt = guardEarlier(killAt, now.Add(p.KillGraceNS))
		}
		select {
		case got := <-exchangeDone:
			exchange = &got
			if got.err != nil {
				add(got.err)
				stopAt = now
				killAt = guardEarlier(killAt, now.Add(p.KillGraceNS))
			}
		default:
		}
		select {
		case got := <-stdoutDone:
			stdout = &got
			if got.err != nil {
				add(got.err)
				stopAt = now
				killAt = guardEarlier(killAt, now.Add(p.KillGraceNS))
			}
		default:
		}
		select {
		case got := <-stderrDone:
			stderr = &got
			if got.err != nil {
				add(got.err)
				stopAt = now
				killAt = guardEarlier(killAt, now.Add(p.KillGraceNS))
			}
		default:
		}
		if !now.Before(stopAt) && !termSent {
			add(context.DeadlineExceeded)
			signal(syscall.SIGTERM)
			termSent = true
			_ = controlW.Close()
		}
		if !now.Before(killAt) && !killSent {
			signal(syscall.SIGKILL)
			killSent = true
		}
		terminal, err := guardTerminal(pidfd)
		if err != nil {
			add(err)
			signal(syscall.SIGKILL)
			break
		}
		if terminal {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	waitErr := watchdog.retire(cmd)
	waited, result.WaitJoined = true, true
	add(waitErr)
	if cmd.ProcessState != nil {
		exitCode := cmd.ProcessState.ExitCode()
		result.ExitCode = &exitCode
	}
	// Descendants retaining inherited output ends cannot hold reader goroutines.
	// Bounded drain preserves already-buffered terminal bytes; then Close and join.
	drain := time.NewTimer(100 * time.Millisecond)
	for exchange == nil || stdout == nil || stderr == nil {
		select {
		case got := <-exchangeDone:
			exchange = &got
		case got := <-stdoutDone:
			stdout = &got
		case got := <-stderrDone:
			stderr = &got
		case <-drain.C:
			_ = eventR.Close()
			_ = controlW.Close()
			_ = stdoutR.Close()
			_ = stderrR.Close()
		}
	}
	drain.Stop()
	add(exchange.err)
	add(stdout.err)
	add(stderr.err)
	result.Stdout, result.Stderr, result.AdmissionSHA256 = stdout.data, stderr.data, exchange.evidence
	if exchange.receipt != nil {
		result.ChildReceipt = exchange.receipt
		result.ChildReceiptSHA256 = guardSHA(exchange.receipt)
		add(guardReceiptBinding(exchange.receipt, p, exchange.evidence))
	}
	if len(failures) == 0 && result.ChildReceipt != nil {
		result.Status = "complete"
	}
	return
}

// The independent timers are armed before Start and never rescheduled. A
// missing published child at the hard-kill boundary exits the creating parent,
// relying on the pre-exec PDEATHSIG/PPID race check rather than an unknown PID.
// The mutex makes the last group signal and Wait indivisible to timer callbacks.
type guardDeadline struct {
	mu                                                     sync.Mutex
	cmd                                                    *exec.Cmd
	retired, term, kill, expired, admitted, startupExpired bool
	startupAt, killAt                                      time.Time
	grace                                                  time.Duration
	records                                                []guardSignal
	err                                                    error
	timers                                                 []*time.Timer
}

func newGuardDeadline(startup, term, kill time.Time, grace time.Duration) *guardDeadline {
	d := &guardDeadline{startupAt: startup, killAt: kill, grace: grace}
	// A tiny/expired deadline can dispatch immediately; publish the timer list
	// under the same mutex used by callbacks that append an early-stop timer.
	d.mu.Lock()
	d.timers = []*time.Timer{
		time.AfterFunc(time.Until(startup), func() {
			d.mu.Lock()
			if d.retired || d.admitted {
				d.mu.Unlock()
				return
			}
			d.startupExpired = true
			d.mu.Unlock()
			d.abort(errors.New("v41epguard: startup/admission deadline exceeded"))
		}),
		time.AfterFunc(time.Until(term), func() { d.expire(syscall.SIGTERM) }),
		time.AfterFunc(time.Until(kill), func() {
			d.mu.Lock()
			if d.cmd == nil && !d.retired {
				os.Exit(124)
			}
			d.mu.Unlock()
			d.expire(syscall.SIGKILL)
		}),
	}
	d.mu.Unlock()
	return d
}
func (d *guardDeadline) abort(cause error) {
	d.mu.Lock()
	if d.retired {
		d.mu.Unlock()
		return
	}
	d.err = errors.Join(d.err, cause)
	if !d.term {
		d.term = true
		if d.cmd != nil {
			d.sendLocked(syscall.SIGTERM)
		}
		delay := time.Until(guardEarlier(d.killAt, time.Now().Add(d.grace)))
		d.timers = append(d.timers, time.AfterFunc(delay, func() { d.signal(syscall.SIGKILL) }))
	}
	d.mu.Unlock()
}
func (d *guardDeadline) admit() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.retired || d.admitted || d.startupExpired || d.term || d.kill || !time.Now().Before(d.startupAt) {
		return errors.New("v41epguard: startup/admission deadline or stop won")
	}
	d.admitted = true
	d.timers[0].Stop()
	return nil
}
func (d *guardDeadline) expire(sig syscall.Signal) {
	d.mu.Lock()
	if !d.retired {
		d.expired = true
	}
	d.mu.Unlock()
	d.signal(sig)
}
func (d *guardDeadline) publish(cmd *exec.Cmd) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.cmd = cmd
	if d.kill {
		d.sendLocked(syscall.SIGKILL)
	} else if d.term {
		d.sendLocked(syscall.SIGTERM)
	}
}
func (d *guardDeadline) signal(sig syscall.Signal) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.retired || sig == syscall.SIGTERM && d.term || sig == syscall.SIGKILL && d.kill {
		return
	}
	if sig == syscall.SIGTERM {
		d.term = true
	} else {
		d.kill = true
	}
	if d.cmd != nil {
		d.sendLocked(sig)
	}
}
func (d *guardDeadline) sendLocked(sig syscall.Signal) {
	event := guardSignal{Signal: sig.String(), At: time.Now()}
	if err := syscall.Kill(-d.cmd.Process.Pid, sig); err != nil && !errors.Is(err, syscall.ESRCH) {
		event.Error = err.Error()
		d.err = errors.Join(d.err, err, d.cmd.Process.Kill())
	}
	d.records = append(d.records, event)
}
func (d *guardDeadline) retire(cmd *exec.Cmd) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.retired {
		return errors.New("v41epguard: duplicate child retirement")
	}
	if !d.kill {
		d.kill = true
		d.sendLocked(syscall.SIGKILL)
	}
	err := cmd.Wait()
	d.retired = true
	return err
}
func (d *guardDeadline) snapshot() ([]guardSignal, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	err := d.err
	if d.startupExpired {
		err = errors.Join(err, errors.New("v41epguard: startup/admission deadline exceeded"))
	}
	if d.expired {
		err = errors.Join(err, context.DeadlineExceeded)
	}
	return append([]guardSignal(nil), d.records...), err
}
func (d *guardDeadline) close() {
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, timer := range d.timers {
		timer.Stop()
	}
	// Also retire a never-started attempt, so already-dispatched callbacks cannot
	// signal or exit the helper after a pre-Start refusal has returned.
	d.retired = true
}

func guardReceiptBinding(raw []byte, p guardPlan, evidence string) error {
	var envelope struct {
		Schema                   string `json:"schema"`
		Status                   string `json:"status"`
		Rank                     int    `json:"rank"`
		RequestSHA256            string `json:"request_sha256"`
		ExecutableSHA256         string `json:"executable_sha256"`
		SupervisorEvidenceSHA256 string `json:"supervisor_evidence_sha256"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return err
	}
	if envelope.Schema != "fak.v41-ep-owner.receipt/v1" || envelope.Status != "complete" || envelope.Rank != *p.Rank || envelope.RequestSHA256 != p.RequestSHA256 || envelope.ExecutableSHA256 != p.OwnerSHA256 || envelope.SupervisorEvidenceSHA256 != evidence {
		return errors.New("v41epguard: child receipt is incomplete or unbound")
	}
	return nil
}

func guardProtocol(write, read *os.File, p guardPlan, hello guardHello, child guardProcess, started, terminate, kill time.Time, admit func() error) guardExchange {
	var out guardExchange
	deadline := guardEarlier(started.Add(guardStartup), terminate)
	if err := write.SetWriteDeadline(deadline); err != nil {
		out.err = err
		return out
	}
	if err := read.SetReadDeadline(deadline); err != nil {
		out.err = err
		return out
	}
	helloRaw, err := json.Marshal(hello)
	if err == nil {
		err = guardWriteFrame(write, 1, helloRaw, guardMaxControl)
	}
	if err == nil {
		err = guardWriteFrame(write, 2, p.Request, guardMaxRequest)
	}
	var challengeRaw []byte
	if err == nil {
		challengeRaw, err = guardReadFrame(read, 3, guardMaxControl)
	}
	var ch guardChallenge
	if err == nil {
		err = guardStrict(challengeRaw, &ch)
	}
	if err == nil && (!guardDigest(ch.Nonce) || ch.Nonce == strings.Repeat("0", 64) || ch.ContractID != p.ContractID || ch.RequestSHA256 != p.RequestSHA256 || ch.Rank != *p.Rank ||
		ch.PID != child.pid || ch.ExecutableSHA256 != p.OwnerSHA256 || ch.LatestTerminateUnixNS < terminate.UnixNano() || ch.KillGraceNS < int64(p.KillGraceNS) || !time.Now().Before(deadline)) {
		err = errors.New("v41epguard: live child challenge disagrees")
	}
	if err != nil {
		out.err = err
		return out
	}
	admission := guardAdmission{Challenge: ch, ParentPID: hello.ParentPID, ParentStart: hello.ParentStart, OwnerStart: child.start, TerminateUnixNS: terminate.UnixNano(), KillUnixNS: kill.UnixNano()}
	unsigned, _ := json.Marshal(admission)
	admission.EvidenceSHA256 = guardSHA(bytes.Join([][]byte{helloRaw, challengeRaw, unsigned}, []byte{0}))
	admissionRaw, _ := json.Marshal(admission)
	if err = guardWriteFrame(write, 4, admissionRaw, guardMaxControl); err != nil {
		out.err = err
		return out
	}
	if err = admit(); err != nil {
		out.err = err
		return out
	}
	out.evidence = admission.EvidenceSHA256
	if err = read.SetReadDeadline(kill); err != nil {
		out.err = err
		return out
	}
	out.receipt, err = guardReadFrame(read, 5, guardMaxReceipt)
	if err == nil && !json.Valid(out.receipt) {
		err = errors.New("v41epguard: invalid final receipt JSON")
	}
	if err != nil {
		out.receipt = nil
		out.err = err
	}
	return out
}

func guardWriteFrame(w io.Writer, kind byte, data []byte, max int) error {
	if len(data) == 0 || len(data) > max {
		return errors.New("v41epguard: frame exceeds cap")
	}
	var header [13]byte
	copy(header[:8], guardMagic)
	header[8] = kind
	binary.BigEndian.PutUint32(header[9:], uint32(len(data)))
	for _, part := range [][]byte{header[:], data} {
		for len(part) > 0 {
			n, err := w.Write(part)
			if n < 0 || n > len(part) {
				return errors.New("v41epguard: invalid write count")
			}
			if err != nil {
				return err
			}
			if n == 0 {
				return io.ErrNoProgress
			}
			part = part[n:]
		}
	}
	return nil
}

func guardReadFrame(r io.Reader, kind byte, max int) ([]byte, error) {
	var header [13]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, err
	}
	n := binary.BigEndian.Uint32(header[9:])
	if string(header[:8]) != guardMagic || header[8] != kind || n == 0 || uint64(n) > uint64(max) {
		return nil, errors.New("v41epguard: invalid frame")
	}
	data := make([]byte, int(n))
	_, err := io.ReadFull(r, data)
	return data, err
}

func guardOpenImage(path, expected string, proc bool) (*guardImage, error) {
	flags := syscall.O_RDONLY | syscall.O_CLOEXEC
	if !proc {
		flags |= syscall.O_NOFOLLOW
	}
	fd, err := syscall.Open(path, flags, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	fail := func(err error) (*guardImage, error) { _ = f.Close(); return nil, err }
	var before, after syscall.Stat_t
	if err := syscall.Fstat(fd, &before); err != nil {
		return fail(err)
	}
	if before.Mode&syscall.S_IFMT != syscall.S_IFREG || before.Mode&06000 != 0 || before.Mode&0111 == 0 || before.Size < 4 || before.Size > guardMaxImage {
		return fail(errors.New("v41epguard: unsupported executable image"))
	}
	capability := make([]byte, 256)
	if n, err := syscall.Getxattr(fmt.Sprintf("/proc/self/fd/%d", fd), "security.capability", capability); n > 0 || err != nil && !errors.Is(err, syscall.ENODATA) && !errors.Is(err, syscall.ENOTSUP) {
		return fail(errors.New("v41epguard: executable capability status unavailable or privileged"))
	}
	var magic [4]byte
	if _, err := f.ReadAt(magic[:], 0); err != nil || magic != [4]byte{0x7f, 'E', 'L', 'F'} {
		return fail(errors.New("v41epguard: native ELF image required"))
	}
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(f, guardMaxImage+1))
	if err != nil {
		return fail(err)
	}
	if err := syscall.Fstat(fd, &after); err != nil {
		return fail(err)
	}
	digest := hex.EncodeToString(h.Sum(nil))
	if n != before.Size || before.Dev != after.Dev || before.Ino != after.Ino || before.Size != after.Size || before.Mtim != after.Mtim || before.Ctim != after.Ctim || digest != expected {
		return fail(errors.New("v41epguard: held image digest or identity changed"))
	}
	return &guardImage{file: f, stat: after, digest: digest}, nil
}

func guardObserveProcess(pid int) (guardProcess, error) {
	raw, err := guardReadFile(fmt.Sprintf("/proc/%d/stat", pid), 8192)
	if err != nil {
		return guardProcess{}, err
	}
	end := bytes.LastIndexByte(raw, ')')
	if end < 0 {
		return guardProcess{}, errors.New("invalid proc stat")
	}
	fields := strings.Fields(string(raw[end+1:]))
	if len(fields) < 20 {
		return guardProcess{}, errors.New("short proc stat")
	}
	ppid, e1 := strconv.Atoi(fields[1])
	pgid, e2 := strconv.Atoi(fields[2])
	start, e3 := strconv.ParseUint(fields[19], 10, 64)
	if e1 != nil || e2 != nil || e3 != nil || start == 0 {
		return guardProcess{}, errors.New("invalid proc identity")
	}
	return guardProcess{pid: pid, ppid: ppid, pgid: pgid, start: fields[19]}, nil
}

func guardTerminal(fd int) (bool, error) {
	var read syscall.FdSet
	bits := int(unsafe.Sizeof(read.Bits[0]) * 8)
	if fd < 0 || fd >= len(read.Bits)*bits {
		return false, errors.New("v41epguard: pidfd outside bounded select set")
	}
	read.Bits[fd/bits] |= 1 << uint(fd%bits)
	timeout := syscall.Timeval{}
	n, err := syscall.Select(fd+1, &read, nil, nil, &timeout)
	if errors.Is(err, syscall.EINTR) {
		return false, nil
	}
	return n > 0, err
}

func guardPipeInode(f *os.File) (uint64, error) {
	var st syscall.Stat_t
	if err := syscall.Fstat(int(f.Fd()), &st); err != nil {
		return 0, err
	}
	if st.Mode&syscall.S_IFMT != syscall.S_IFIFO {
		return 0, errors.New("v41epguard: anonymous pipe required")
	}
	return st.Ino, nil
}
func guardEarlier(a, b time.Time) time.Time {
	if a.Before(b) {
		return a
	}
	return b
}
func guardSHA(raw []byte) string { sum := sha256.Sum256(raw); return hex.EncodeToString(sum[:]) }
func guardDigest(value string) bool {
	if len(value) != 64 || strings.ToLower(value) != value {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
func guardAbsolute(value string) bool {
	return len(value) <= 4096 && filepath.IsAbs(value) && filepath.Clean(value) == value && !strings.ContainsAny(value, "\x00\r\n")
}
func guardReadFile(path string, cap int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, cap+1))
	if len(data) > int(cap) {
		return nil, errors.New("v41epguard: input cap exceeded")
	}
	return data, err
}

// Requiring canonical encoding rejects duplicate, unknown, omitted, aliased and
// trailing fields without treating encoding/json's null-to-zero coercion as input.
func guardStrict(raw []byte, value any) error {
	probe := json.NewDecoder(bytes.NewReader(raw))
	depth := 0
	for {
		token, err := probe.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if token == nil {
			return errors.New("v41epguard: null is not an explicit value")
		}
		if delimiter, ok := token.(json.Delim); ok {
			if delimiter == '{' || delimiter == '[' {
				depth++
			} else {
				depth--
			}
			if depth > 16 {
				return errors.New("v41epguard: JSON nesting cap exceeded")
			}
		}
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(value); err != nil {
		return err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return errors.New("v41epguard: trailing JSON")
	}
	canonical, err := json.Marshal(value)
	if err != nil || !bytes.Equal(raw, canonical) {
		return errors.New("v41epguard: exact canonical JSON required")
	}
	return nil
}

type guardResource struct {
	Available bool              `json:"available"`
	At        time.Time         `json:"at"`
	Memory    uint64            `json:"memory_available"`
	Disk      uint64            `json:"disk_available"`
	Pressure  map[string]uint64 `json:"pressure"`
}

func guardResources(path string) (guardResource, error) {
	out := guardResource{At: time.Now()}
	mem, err := guardReadFile("/proc/meminfo", 64<<10)
	if err != nil {
		return out, err
	}
	out.Memory, err = guardMemoryAvailable(mem)
	if err != nil {
		return out, err
	}
	vm, err := guardReadFile("/proc/vmstat", 64<<10)
	if err != nil {
		return out, err
	}
	out.Pressure = map[string]uint64{}
	for _, line := range strings.Split(string(vm), "\n") {
		f := strings.Fields(line)
		if len(f) == 2 && (f[0] == "oom_kill" || strings.HasPrefix(f[0], "allocstall") || strings.HasPrefix(f[0], "pgscan_direct")) {
			n, err := strconv.ParseUint(f[1], 10, 64)
			if err != nil {
				return out, err
			}
			out.Pressure[f[0]] = n
		}
	}
	var disk syscall.Statfs_t
	if err := syscall.Statfs(path, &disk); err != nil {
		return out, err
	}
	if disk.Bsize <= 0 || disk.Bavail > ^uint64(0)/uint64(disk.Bsize) {
		return out, errors.New("invalid free disk observation")
	}
	out.Disk = disk.Bavail * uint64(disk.Bsize)
	out.Available = true
	return out, nil
}

// Missing or malformed fields remain unavailable, never an observed zero.
func guardMemoryAvailable(raw []byte) (uint64, error) {
	var value uint64
	seen := false
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 || fields[0] != "MemAvailable:" {
			continue
		}
		if seen || len(fields) != 3 || fields[2] != "kB" {
			return 0, errors.New("invalid MemAvailable observation")
		}
		n, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil || n > ^uint64(0)/1024 {
			return 0, errors.New("invalid MemAvailable observation")
		}
		value, seen = n*1024, true
	}
	if !seen {
		return 0, errors.New("MemAvailable observation unavailable")
	}
	return value, nil
}

func (r guardResource) check(p guardPlan, initial guardResource) error {
	alloc, scan := false, false
	for key := range r.Pressure {
		alloc = alloc || strings.HasPrefix(key, "allocstall")
		scan = scan || strings.HasPrefix(key, "pgscan_direct")
	}
	_, oom := r.Pressure["oom_kill"]
	if !r.Available || r.Memory < p.MinimumMemoryBytes || r.Disk < p.MinimumDiskBytes || !oom || !alloc || !scan || !reflect.DeepEqual(r.Pressure, initial.Pressure) {
		return errors.New("v41epguard: resource floor or pressure refusal")
	}
	return nil
}

// The monitor is armed before Start and continues through process retirement.
// It owns its bounded sample list; close joins it before those observations are
// copied into a receipt. Failure never depends on the owner advancing startup.
type guardResourceMonitor struct {
	mu         sync.Mutex
	stop, done chan struct{}
	once       sync.Once
	samples    []guardResource
	err        error
}

func newGuardResourceMonitor(p guardPlan, initial guardResource, abort func(error)) *guardResourceMonitor {
	m := &guardResourceMonitor{stop: make(chan struct{}), done: make(chan struct{})}
	ready := make(chan struct{})
	go func() {
		defer close(m.done)
		sample := func() {
			resource, err := guardResources(p.ResourcePath)
			if err == nil {
				err = resource.check(p, initial)
			}
			m.mu.Lock()
			if len(m.samples) < 1024 {
				m.samples = append(m.samples, resource)
			} else {
				err = errors.Join(err, errors.New("v41epguard: resource sample cap exceeded"))
			}
			firstFailure := m.err == nil && err != nil
			if firstFailure {
				m.err = err
			}
			m.mu.Unlock()
			if firstFailure {
				abort(err)
			}
		}
		sample()
		close(ready)
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				sample()
			case <-m.stop:
				sample()
				return
			}
		}
	}()
	<-ready
	return m
}
func (m *guardResourceMonitor) failure() error { m.mu.Lock(); defer m.mu.Unlock(); return m.err }
func (m *guardResourceMonitor) close() ([]guardResource, error) {
	m.once.Do(func() { close(m.stop); <-m.done })
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]guardResource(nil), m.samples...), m.err
}
