//go:build linux

package localbench

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
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

// The reviewed helper is built first. Its actual binary digest is independently
// pinned into the owner build with -ldflags -X; an unset pin refuses admission.
// No request field, environment variable, or prior receipt can override it.
var v41EPOwnerGuardBinarySHA256 string

const (
	v41EPHMagic      = "FKEPGRD1"
	v41EPHControlCap = 4 << 10
	v41EPHReceiptCap = 2 << 20
	v41EPHImageCap   = 512 << 20
	v41EPHStartup    = 5 * time.Second
)

// These wire structures deliberately match the driver-free tools/v41epguard
// program without importing it or making it import GPU-initializing packages.
type v41EPHHello struct {
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

type v41EPHChallenge struct {
	Nonce                 string `json:"nonce"`
	ContractID            string `json:"contract_id"`
	RequestSHA256         string `json:"request_sha256"`
	Rank                  int    `json:"rank"`
	PID                   int    `json:"pid"`
	ExecutableSHA256      string `json:"executable_sha256"`
	LatestTerminateUnixNS int64  `json:"latest_terminate_unix_ns"`
	KillGraceNS           int64  `json:"kill_grace_ns"`
}

type v41EPHAdmission struct {
	Challenge       v41EPHChallenge `json:"challenge"`
	ParentPID       int             `json:"parent_pid"`
	ParentStart     string          `json:"parent_start"`
	OwnerStart      string          `json:"owner_start"`
	TerminateUnixNS int64           `json:"terminate_unix_ns"`
	KillUnixNS      int64           `json:"kill_unix_ns"`
	EvidenceSHA256  string          `json:"evidence_sha256"`
}

type v41EPHProcess struct {
	pid, ppid, pgid int
	start           string
}

// This is invocation-local ownership, not an independently transferable receipt.
// The helper's pinned code supplies external enforcement. Matching a same-UID
// process image and pipe exchange is not remote attestation or a hostile-host
// security boundary. Go package initialization precedes this model-resource gate
// but must already be contained by the externally launched helper.
type v41EPOwnerLiveHandoff struct {
	Request                  V41EPOwnerRequest
	Rank                     int
	Context                  context.Context
	process                  v41EPHProcess
	parent                   v41EPHProcess
	hello                    v41EPHHello
	helloRaw                 []byte
	image, read, write       *os.File
	startup, terminate, kill time.Time
	cancel                   context.CancelFunc
	verifyUsed               atomic.Bool
	closed                   atomic.Bool
	closeOnce                sync.Once
	watchDone                chan struct{}
	watchStarted             bool
	evidence                 string
	closeErr                 error
}

func openV41EPOwnerLiveHandoff(ctx context.Context) (_ *v41EPOwnerLiveHandoff, err error) {
	if ctx == nil || ctx.Err() != nil || !v41EPOwnerSHA256(v41EPOwnerGuardBinarySHA256, true) {
		return nil, errors.New("v41 EP owner: a live context and independently pinned supervisor are required")
	}
	h := &v41EPOwnerLiveHandoff{startup: time.Now().Add(v41EPHStartup), watchDone: make(chan struct{})}
	// Calling this private entry transfers exactly descriptors 3/4/5. Until
	// os.File adopts them, failures must also retire partially configured FDs.
	rawOwned := true
	defer func() {
		if rawOwned {
			for _, fd := range []int{3, 4, 5} {
				err = errors.Join(err, syscall.Close(fd))
			}
		}
		if err != nil {
			err = errors.Join(err, h.Close())
		}
	}()
	// ExtraFiles descriptors arrive in blocking mode. Make only these owned
	// anonymous pipe descriptions pollable before adopting them into os.File.
	for _, fd := range []int{4, 5} {
		if e := syscall.SetNonblock(fd, true); e != nil {
			return nil, e
		}
	}
	h.image, h.read, h.write = os.NewFile(3, "v41-owner-image"), os.NewFile(4, "v41-owner-control"), os.NewFile(5, "v41-owner-events")
	rawOwned = false
	for _, fd := range []int{3, 4, 5} {
		syscall.CloseOnExec(fd)
	}
	readInode, err := v41EPHPipe(4, syscall.O_RDONLY)
	if err != nil {
		return nil, err
	}
	writeInode, err := v41EPHPipe(5, syscall.O_WRONLY)
	if err != nil {
		return nil, err
	}
	var deathSignal int32
	if _, _, errno := syscall.RawSyscall6(syscall.SYS_PRCTL, syscall.PR_GET_PDEATHSIG, uintptr(unsafe.Pointer(&deathSignal)), 0, 0, 0, 0); errno != 0 || deathSignal != int32(syscall.SIGKILL) {
		return nil, errors.New("v41 EP owner: kernel parent-death SIGKILL is required")
	}
	h.process, err = v41EPHObserveProcess(os.Getpid())
	if err != nil {
		return nil, err
	}
	if h.process.ppid <= 1 || h.process.pgid != h.process.pid {
		return nil, errors.New("v41 EP owner: owned owner process group required")
	}
	h.parent, err = v41EPHObserveProcess(h.process.ppid)
	if err != nil {
		return nil, err
	}
	parentHash, _, err := v41EPHImage(fmt.Sprintf("/proc/%d/exe", h.parent.pid), nil)
	if err != nil || parentHash != v41EPOwnerGuardBinarySHA256 {
		return nil, errors.Join(err, errors.New("v41 EP owner: actual supervisor image pin disagrees"))
	}
	selfHash, selfStat, err := v41EPHImage("/proc/self/exe", nil)
	if err != nil {
		return nil, err
	}
	heldHash, heldStat, err := v41EPHImage("", h.image)
	if err != nil {
		return nil, err
	}
	if selfHash != heldHash || selfStat.Dev != heldStat.Dev || selfStat.Ino != heldStat.Ino {
		return nil, errors.New("v41 EP owner: running image is not the held launch inode")
	}
	if err := h.read.SetReadDeadline(v41EPHContextDeadline(ctx, h.startup)); err != nil {
		return nil, err
	}
	h.helloRaw, err = v41EPHRead(h.read, 1, v41EPHControlCap)
	if err != nil {
		return nil, err
	}
	if err := v41EPHStrict(h.helloRaw, &h.hello); err != nil {
		return nil, err
	}
	hello := h.hello
	if hello.Version != 1 || hello.Rank < 0 || hello.Rank > 2 || !v41EPOwnerSHA256(hello.RequestSHA256, true) || hello.ParentPID != h.parent.pid ||
		hello.ParentStart != h.parent.start || hello.ParentSHA256 != parentHash || hello.OwnerSHA256 != selfHash || hello.ReadPipeInode != readInode || hello.WritePipeInode != writeInode {
		return nil, errors.New("v41 EP owner: actual process or inherited channel identity disagrees")
	}
	h.terminate, h.kill = time.Unix(0, hello.TerminateUnixNS), time.Unix(0, hello.KillUnixNS)
	h.startup = v41EPHEarlier(h.startup, time.Unix(0, hello.AdmissionUnixNS))
	if !time.Now().Before(h.startup) || h.startup.After(h.terminate) {
		return nil, errors.New("v41 EP owner: invalid absolute admission ceiling")
	}
	if !time.Now().Before(h.terminate) || !h.kill.After(h.terminate) || h.kill.Sub(h.terminate) > 5*time.Second {
		return nil, errors.New("v41 EP owner: expired or unbounded supervisor times")
	}
	if err := h.read.SetReadDeadline(v41EPHContextDeadline(ctx, v41EPHEarlier(h.startup, h.terminate))); err != nil {
		return nil, err
	}
	raw, err := v41EPHRead(h.read, 2, V41EPOwnerMaxRequestBytes)
	if err != nil {
		return nil, err
	}
	h.Request, err = DecodeV41EPOwnerRequest(bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	canonical, err := json.Marshal(h.Request)
	if err != nil || !bytes.Equal(raw, canonical) || v41EPHSHA(raw) != hello.RequestSHA256 || h.Request.Supervisor.ContractID != hello.ContractID ||
		h.kill.Sub(h.terminate) > h.Request.Supervisor.KillGraceNS || h.terminate.After(time.Now().Add(h.Request.Supervisor.TerminateNS)) {
		return nil, errors.New("v41 EP owner: canonical request and bounded supervisor contract disagree")
	}
	h.Rank = hello.Rank
	if expected := h.Request.Owners[h.Rank].ExpectedExecutableSHA256; expected != "" && expected != selfHash {
		return nil, errors.New("v41 EP owner: request binary pin disagrees")
	}
	if err := h.recheckParent(); err != nil {
		return nil, err
	}
	if ctx.Err() != nil || !time.Now().Before(v41EPHEarlier(h.startup, h.terminate)) {
		return nil, errors.New("v41 EP owner: initial handoff deadline expired")
	}
	h.Context, h.cancel = context.WithDeadline(ctx, h.terminate)
	return h, nil
}

func (h *v41EPOwnerLiveHandoff) ObserveProcess(ctx context.Context) (v41EPOwnerProcessIdentity, error) {
	if h == nil || h.closed.Load() || ctx == nil || ctx.Err() != nil {
		return v41EPOwnerProcessIdentity{}, errors.New("v41 EP owner: process observation is unavailable")
	}
	actual, err := v41EPHObserveProcess(os.Getpid())
	if err != nil || actual != h.process {
		return v41EPOwnerProcessIdentity{}, errors.Join(err, errors.New("v41 EP owner: process identity changed"))
	}
	digest, _, err := v41EPHImage("/proc/self/exe", nil)
	if err != nil || digest != h.hello.OwnerSHA256 {
		return v41EPOwnerProcessIdentity{}, errors.Join(err, errors.New("v41 EP owner: executable image changed"))
	}
	return v41EPOwnerProcessIdentity{PID: actual.pid, ExecutableSHA256: digest}, ctx.Err()
}

func (h *v41EPOwnerLiveHandoff) Verify(ctx context.Context, challenge v41EPOwnerSupervisorChallenge) (_ v41EPOwnerSupervisorHandoff, err error) {
	if h == nil || h.closed.Load() || !h.verifyUsed.CompareAndSwap(false, true) {
		return v41EPOwnerSupervisorHandoff{}, errors.New("v41 EP owner: supervisor handoff is single-use")
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, h.Close())
		}
	}()
	if ctx == nil || ctx.Err() != nil || h.Context.Err() != nil || challenge.Nonce == ([32]byte{}) || challenge.ContractID != h.hello.ContractID ||
		challenge.RequestSHA256 != h.hello.RequestSHA256 || challenge.Rank != h.Rank || challenge.Process.PID != h.process.pid || challenge.Process.ExecutableSHA256 != h.hello.OwnerSHA256 ||
		challenge.LatestTerminateAt.Before(h.terminate) || challenge.KillGrace < h.kill.Sub(h.terminate) {
		return v41EPOwnerSupervisorHandoff{}, errors.New("v41 EP owner: invocation challenge disagrees with owned process")
	}
	deadline := v41EPHContextDeadline(ctx, v41EPHEarlier(h.startup, h.terminate))
	if !time.Now().Before(deadline) {
		return v41EPOwnerSupervisorHandoff{}, context.DeadlineExceeded
	}
	if err := h.read.SetReadDeadline(deadline); err != nil {
		return v41EPOwnerSupervisorHandoff{}, err
	}
	if err := h.write.SetWriteDeadline(deadline); err != nil {
		return v41EPOwnerSupervisorHandoff{}, err
	}
	wire := v41EPHChallenge{Nonce: hex.EncodeToString(challenge.Nonce[:]), ContractID: challenge.ContractID, RequestSHA256: challenge.RequestSHA256,
		Rank: challenge.Rank, PID: challenge.Process.PID, ExecutableSHA256: challenge.Process.ExecutableSHA256, LatestTerminateUnixNS: challenge.LatestTerminateAt.UnixNano(), KillGraceNS: int64(challenge.KillGrace)}
	wireRaw, _ := json.Marshal(wire)
	if err := v41EPHWrite(h.write, 3, wireRaw, v41EPHControlCap); err != nil {
		return v41EPOwnerSupervisorHandoff{}, err
	}
	raw, err := v41EPHRead(h.read, 4, v41EPHControlCap)
	if err != nil {
		return v41EPOwnerSupervisorHandoff{}, err
	}
	var admission v41EPHAdmission
	if err := v41EPHStrict(raw, &admission); err != nil {
		return v41EPOwnerSupervisorHandoff{}, err
	}
	evidence := admission.EvidenceSHA256
	admission.EvidenceSHA256 = ""
	unsigned, _ := json.Marshal(admission)
	if admission.Challenge != wire || admission.ParentPID != h.parent.pid || admission.ParentStart != h.parent.start || admission.OwnerStart != h.process.start ||
		admission.TerminateUnixNS != h.hello.TerminateUnixNS || admission.KillUnixNS != h.hello.KillUnixNS || !v41EPOwnerSHA256(evidence, true) ||
		evidence != v41EPHSHA(bytes.Join([][]byte{h.helloRaw, wireRaw, unsigned}, []byte{0})) {
		return v41EPOwnerSupervisorHandoff{}, errors.New("v41 EP owner: live admission identity or deadline disagrees")
	}
	if err := h.recheckParent(); err != nil {
		return v41EPOwnerSupervisorHandoff{}, err
	}
	if ctx.Err() != nil || h.Context.Err() != nil || !time.Now().Before(deadline) {
		return v41EPOwnerSupervisorHandoff{}, errors.New("v41 EP owner: admission completed after cancellation/deadline")
	}
	if err := h.read.SetReadDeadline(h.kill); err != nil {
		return v41EPOwnerSupervisorHandoff{}, err
	}
	h.evidence, h.watchStarted = evidence, true
	go func() {
		defer close(h.watchDone)
		var unexpected [1]byte
		_, _ = h.read.Read(unexpected[:])
		// No further parent frame is legal. EOF, extra data, deadline, or read
		// failure all cancel cooperative work; the external timer owns hard stop.
		if !h.closed.Load() {
			h.cancel()
		}
	}()
	// All serialized fields were checked above. Retaining the original challenge
	// preserves its Go monotonic metadata; absolute wire deadlines are not shifted
	// forward by handshake latency. The parent retains its own monotonic timers.
	return v41EPOwnerSupervisorHandoff{Challenge: challenge, TerminateAt: h.terminate, KillAt: h.kill, EvidenceSHA256: evidence}, nil
}

func (h *v41EPOwnerLiveHandoff) Finish(receipt V41EPOwnerReceipt) (err error) {
	if h == nil || h.closed.Load() || h.evidence == "" {
		return errors.New("v41 EP owner: no admitted receipt channel")
	}
	defer func() { err = errors.Join(err, h.Close()) }()
	if receipt.Schema != V41EPOwnerReceiptSchema || receipt.Rank != h.Rank || receipt.RequestSHA256 != h.hello.RequestSHA256 || receipt.ExecutableSHA256 == nil || *receipt.ExecutableSHA256 != h.hello.OwnerSHA256 ||
		receipt.SupervisorEvidenceSHA256 == nil || *receipt.SupervisorEvidenceSHA256 != h.evidence {
		return errors.New("v41 EP owner: receipt does not bind this invocation")
	}
	raw, err := json.Marshal(receipt)
	if err != nil {
		return err
	}
	if err := h.write.SetWriteDeadline(v41EPHEarlier(time.Now().Add(v41EPHStartup), h.kill)); err != nil {
		return err
	}
	return v41EPHWrite(h.write, 5, raw, v41EPHReceiptCap)
}

// Close is serialized by the owner after Verify/Finish. It joins the only
// channel watcher and never closes a Session, group, model, or backend itself.
func (h *v41EPOwnerLiveHandoff) Close() error {
	if h == nil {
		return nil
	}
	h.closeOnce.Do(func() {
		h.closed.Store(true)
		for _, f := range []*os.File{h.read, h.write, h.image} {
			if f != nil {
				h.closeErr = errors.Join(h.closeErr, f.Close())
			}
		}
		if h.cancel != nil {
			h.cancel()
		}
		if h.watchStarted {
			<-h.watchDone
		}
	})
	return h.closeErr
}

func (h *v41EPOwnerLiveHandoff) recheckParent() error {
	self, err := v41EPHObserveProcess(os.Getpid())
	if err != nil || self != h.process || os.Getppid() != h.parent.pid {
		return errors.Join(err, errors.New("v41 EP owner: parent ownership changed"))
	}
	parent, err := v41EPHObserveProcess(h.parent.pid)
	if err != nil || parent != h.parent {
		return errors.Join(err, errors.New("v41 EP owner: supervisor process identity changed"))
	}
	digest, _, err := v41EPHImage(fmt.Sprintf("/proc/%d/exe", parent.pid), nil)
	if err != nil || digest != v41EPOwnerGuardBinarySHA256 {
		return errors.Join(err, errors.New("v41 EP owner: supervisor image changed"))
	}
	if os.Getppid() != parent.pid {
		return errors.New("v41 EP owner: supervisor disappeared during observation")
	}
	return nil
}

func v41EPHPipe(fd, direction int) (uint64, error) {
	var stat syscall.Stat_t
	if err := syscall.Fstat(fd, &stat); err != nil {
		return 0, err
	}
	flags, _, errno := syscall.Syscall(syscall.SYS_FCNTL, uintptr(fd), syscall.F_GETFL, 0)
	if errno != 0 || stat.Mode&syscall.S_IFMT != syscall.S_IFIFO || int(flags)&syscall.O_ACCMODE != direction {
		return 0, errors.New("v41 EP owner: expected inherited anonymous pipe direction")
	}
	return stat.Ino, nil
}

func v41EPHObserveProcess(pid int) (v41EPHProcess, error) {
	f, err := os.Open(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return v41EPHProcess{}, err
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, 8193))
	if err != nil || len(raw) > 8192 {
		return v41EPHProcess{}, errors.New("v41 EP owner: bounded process identity unavailable")
	}
	end := bytes.LastIndexByte(raw, ')')
	if end < 0 {
		return v41EPHProcess{}, errors.New("invalid process stat")
	}
	fields := strings.Fields(string(raw[end+1:]))
	if len(fields) < 20 {
		return v41EPHProcess{}, errors.New("short process stat")
	}
	ppid, e1 := strconv.Atoi(fields[1])
	pgid, e2 := strconv.Atoi(fields[2])
	start, e3 := strconv.ParseUint(fields[19], 10, 64)
	if e1 != nil || e2 != nil || e3 != nil || start == 0 {
		return v41EPHProcess{}, errors.New("invalid process identity")
	}
	return v41EPHProcess{pid: pid, ppid: ppid, pgid: pgid, start: fields[19]}, nil
}

func v41EPHImage(path string, held *os.File) (string, syscall.Stat_t, error) {
	f := held
	if f == nil {
		var err error
		f, err = os.Open(path)
		if err != nil {
			return "", syscall.Stat_t{}, err
		}
		defer f.Close()
	}
	var before, after syscall.Stat_t
	if err := syscall.Fstat(int(f.Fd()), &before); err != nil {
		return "", after, err
	}
	if before.Mode&syscall.S_IFMT != syscall.S_IFREG || before.Size < 4 || before.Size > v41EPHImageCap || before.Mode&06000 != 0 {
		return "", after, errors.New("v41 EP owner: unsupported observed image")
	}
	hash := sha256.New()
	n, err := io.Copy(hash, io.NewSectionReader(f, 0, before.Size))
	if err != nil {
		return "", after, err
	}
	if err := syscall.Fstat(int(f.Fd()), &after); err != nil {
		return "", after, err
	}
	if n != before.Size || before.Dev != after.Dev || before.Ino != after.Ino || before.Size != after.Size || before.Mtim != after.Mtim || before.Ctim != after.Ctim {
		return "", after, errors.New("v41 EP owner: observed image changed during hash")
	}
	return hex.EncodeToString(hash.Sum(nil)), after, nil
}

func v41EPHSHA(raw []byte) string { digest := sha256.Sum256(raw); return hex.EncodeToString(digest[:]) }
func v41EPHEarlier(a, b time.Time) time.Time {
	if a.Before(b) {
		return a
	}
	return b
}
func v41EPHContextDeadline(ctx context.Context, cap time.Time) time.Time {
	if deadline, ok := ctx.Deadline(); ok {
		return v41EPHEarlier(cap, deadline)
	}
	return cap
}
func v41EPHStrict(raw []byte, value any) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		return err
	}
	canonical, err := json.Marshal(value)
	if err != nil || !bytes.Equal(raw, canonical) {
		return errors.New("v41 EP owner: exact canonical control frame required")
	}
	return nil
}
func v41EPHRead(reader io.Reader, kind byte, cap int) ([]byte, error) {
	var header [13]byte
	if err := v41EPHReadExact(reader, header[:]); err != nil {
		return nil, err
	}
	size := binary.BigEndian.Uint32(header[9:])
	if string(header[:8]) != v41EPHMagic || header[8] != kind || size == 0 || uint64(size) > uint64(cap) {
		return nil, errors.New("v41 EP owner: invalid control frame")
	}
	data := make([]byte, int(size))
	if err := v41EPHReadExact(reader, data); err != nil {
		return nil, err
	}
	return data, nil
}
func v41EPHReadExact(reader io.Reader, data []byte) error {
	for len(data) > 0 {
		n, err := reader.Read(data)
		if n < 0 || n > len(data) {
			return errors.New("invalid control read count")
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrNoProgress
		}
		data = data[n:]
	}
	return nil
}
func v41EPHWrite(writer io.Writer, kind byte, raw []byte, cap int) error {
	if len(raw) == 0 || len(raw) > cap {
		return errors.New("v41 EP owner: control frame exceeds cap")
	}
	var header [13]byte
	copy(header[:8], v41EPHMagic)
	header[8] = kind
	binary.BigEndian.PutUint32(header[9:], uint32(len(raw)))
	for _, data := range [][]byte{header[:], raw} {
		for len(data) > 0 {
			n, err := writer.Write(data)
			if n < 0 || n > len(data) {
				return errors.New("invalid control write count")
			}
			if err != nil {
				return err
			}
			if n == 0 {
				return io.ErrNoProgress
			}
			data = data[n:]
		}
	}
	return nil
}
