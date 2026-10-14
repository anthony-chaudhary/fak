//go:build linux

package localbench

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
)

// This process fixture verifies the real inherited-FD/process binding only. It
// is not the production supervisor and does not establish its enforcement. The
// separate stdlib guard witness owns the TERM/KILL/parent-death qualification.
func TestMain(m *testing.M) {
	if len(os.Args) == 3 && os.Args[1] == "--v41-ep-handoff-test-owner" {
		if err := v41EPHTestOwner(os.Args[2]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(3)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// fak-test:runtime medium est=2s lane=default
func TestV41EPOwnerLiveHandoffContract(t *testing.T) {
	t.Run("unset-pin-refuses-without-descriptor-adoption", func(t *testing.T) {
		if v41EPOwnerGuardBinarySHA256 != "" {
			t.Fatal("test invocation unexpectedly pinned")
		}
		if h, err := openV41EPOwnerLiveHandoff(context.Background()); h != nil || err == nil {
			t.Fatal("untrusted launch admitted")
		}
	})
	t.Run("canonical-control-and-bounded-io", func(t *testing.T) {
		value := v41EPHChallenge{Nonce: strings.Repeat("a", 64), ContractID: "test", Rank: 0}
		raw, _ := json.Marshal(value)
		var got v41EPHChallenge
		if err := v41EPHStrict(raw, &got); err != nil || got != value {
			t.Fatal("canonical challenge failed", err)
		}
		for _, malformed := range [][]byte{
			append(append([]byte(nil), raw...), ' '),
			bytes.Replace(raw, []byte(`"rank":0`), []byte(`"rank":null`), 1),
			bytes.Replace(raw, []byte(`"rank":0`), []byte(`"rank":0,"rank":0`), 1),
			bytes.Replace(raw, []byte(`"rank":0`), []byte(`"rank":0,"unknown":1`), 1),
		} {
			if v41EPHStrict(malformed, &got) == nil {
				t.Fatal("ambiguous challenge accepted")
			}
		}
		var output v41EPHShortWriter
		if err := v41EPHWrite(&output, 3, raw, v41EPHControlCap); err != nil {
			t.Fatal(err)
		}
		// Independent wire layout check, not a roundtrip-only oracle.
		encoded := output.Bytes()
		if string(encoded[:8]) != "FKEPGRD1" || encoded[8] != 3 || binary.BigEndian.Uint32(encoded[9:13]) != uint32(len(raw)) || !bytes.Equal(encoded[13:], raw) {
			t.Fatal("wire layout changed")
		}
		result, err := v41EPHRead(&v41EPHShortReader{data: append([]byte(nil), encoded...)}, 3, v41EPHControlCap)
		if err != nil || !bytes.Equal(result, raw) {
			t.Fatal("short I/O failed", err)
		}
		if _, err := v41EPHRead(bytes.NewReader(encoded), 4, v41EPHControlCap); err == nil {
			t.Fatal("wrong order accepted")
		}
		if v41EPHWrite(io.Discard, 3, make([]byte, v41EPHControlCap+1), v41EPHControlCap) == nil {
			t.Fatal("oversized frame accepted")
		}
		sentinel := errors.New("partial transfer failure")
		if !errors.Is(v41EPHReadExact(v41EPHFailReader{sentinel}, make([]byte, 1)), sentinel) {
			t.Fatal("final n+error read accepted")
		}
		if !errors.Is(v41EPHWrite(v41EPHFailWriter{sentinel}, 3, raw, v41EPHControlCap), sentinel) {
			t.Fatal("n+error write accepted")
		}
		if !errors.Is(v41EPHReadExact(v41EPHFailReader{}, make([]byte, 1)), io.ErrNoProgress) {
			t.Fatal("zero-progress read accepted")
		}
	})
	for _, mode := range []string{"normal", "wrong-admission", "channel-loss", "wrong-parent-pin", "no-parent-death", "wrong-pipe-direction", "partial-fd-setup", "expired-admission"} {
		t.Run(mode, func(t *testing.T) { v41EPHTestExchange(t, mode) })
	}
}

type v41EPHShortWriter struct{ bytes.Buffer }

func (w *v41EPHShortWriter) Write(p []byte) (int, error) {
	if len(p) > 3 {
		p = p[:3]
	}
	return w.Buffer.Write(p)
}

type v41EPHShortReader struct{ data []byte }

func (r *v41EPHShortReader) Read(p []byte) (int, error) {
	if len(r.data) == 0 {
		return 0, io.EOF
	}
	if len(p) > 2 {
		p = p[:2]
	}
	n := copy(p, r.data)
	r.data = r.data[n:]
	return n, nil
}

type v41EPHFailReader struct{ err error }

func (r v41EPHFailReader) Read(p []byte) (int, error) {
	if r.err == nil {
		return 0, nil
	}
	return len(p), r.err
}

type v41EPHFailWriter struct{ err error }

func (w v41EPHFailWriter) Write(p []byte) (int, error) { return len(p), w.err }

type v41EPHBoundedOutput struct{ bytes.Buffer }

func (w *v41EPHBoundedOutput) Write(p []byte) (int, error) {
	if len(p) > 4096-w.Len() {
		return 0, errors.New("test output cap")
	}
	return w.Buffer.Write(p)
}

func v41EPHTestExchange(t *testing.T, mode string) {
	t.Helper()
	image, err := os.Open("/proc/self/exe")
	if err != nil {
		t.Fatal(err)
	}
	defer image.Close()
	digest, _, err := v41EPHImage("", image)
	if err != nil {
		t.Fatal(err)
	}
	parent, err := v41EPHObserveProcess(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	controlR, controlW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer controlR.Close()
	defer controlW.Close()
	eventR, eventW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer eventR.Close()
	defer eventW.Close()
	readInode, err := v41EPHPipe(int(controlR.Fd()), syscall.O_RDONLY)
	if err != nil {
		t.Fatal(err)
	}
	writeInode, err := v41EPHPipe(int(eventW.Fd()), syscall.O_WRONLY)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "/proc/self/fd/3", "--v41-ep-handoff-test-owner", mode)
	cmd.Env = []string{"LANG=C", "LC_ALL=C", "TZ=UTC"}
	cmd.ExtraFiles = []*os.File{image, controlR, eventW}
	if mode == "wrong-pipe-direction" {
		cmd.ExtraFiles[1] = controlW
	}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL}
	if mode == "no-parent-death" {
		cmd.SysProcAttr.Pdeathsig = 0
	}
	var stdout, stderr v41EPHBoundedOutput
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	cmd.WaitDelay = 100 * time.Millisecond
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	joined := false
	defer func() {
		if !joined {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()
	_ = controlR.Close()
	_ = eventW.Close()
	early := map[string]string{"wrong-parent-pin": "actual supervisor image pin disagrees", "no-parent-death": "kernel parent-death SIGKILL is required", "wrong-pipe-direction": "expected inherited anonymous pipe direction", "partial-fd-setup": "bad file descriptor"}
	if want := early[mode]; want != "" {
		err := cmd.Wait()
		joined = true
		if err != nil || !strings.Contains(stdout.String(), want) {
			t.Fatalf("expected exact pre-admission refusal: err=%v stdout=%s stderr=%s", err, stdout.String(), stderr.String())
		}
		return
	}
	until := time.Now().Add(3 * time.Second)
	if err := controlW.SetWriteDeadline(until); err != nil {
		t.Fatal(err)
	}
	if err := eventR.SetReadDeadline(until); err != nil {
		t.Fatal(err)
	}
	child, err := v41EPHObserveProcess(cmd.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	req := v41EPOwnerRequestFixture()
	req.Owners[0].ExpectedExecutableSHA256 = digest
	requestRaw, _ := json.Marshal(req)
	now := time.Now()
	hello := v41EPHHello{AdmissionUnixNS: now.Add(2 * time.Second).UnixNano(), Version: 1, Rank: 0, RequestSHA256: v41EPHSHA(requestRaw), ContractID: req.Supervisor.ContractID,
		ParentPID: parent.pid, ParentStart: parent.start, ParentSHA256: digest, OwnerSHA256: digest, TerminateUnixNS: until.UnixNano(), KillUnixNS: until.Add(500 * time.Millisecond).UnixNano(), ReadPipeInode: readInode, WritePipeInode: writeInode}
	if mode == "expired-admission" {
		hello.AdmissionUnixNS = now.Add(-time.Second).UnixNano()
	}
	helloRaw, _ := json.Marshal(hello)
	if err := v41EPHWrite(controlW, 1, helloRaw, v41EPHControlCap); err != nil {
		t.Fatal(err)
	}
	if mode == "expired-admission" {
		err := cmd.Wait()
		joined = true
		if err != nil || !strings.Contains(stdout.String(), "invalid absolute admission ceiling") {
			t.Fatalf("expired handshake: %v %s %s", err, stdout.String(), stderr.String())
		}
		return
	}
	if err := v41EPHWrite(controlW, 2, requestRaw, V41EPOwnerMaxRequestBytes); err != nil {
		t.Fatal(err)
	}
	challengeRaw, err := v41EPHRead(eventR, 3, v41EPHControlCap)
	if err != nil {
		t.Fatal(err)
	}
	var challenge v41EPHChallenge
	if err := v41EPHStrict(challengeRaw, &challenge); err != nil {
		t.Fatal(err)
	}
	if challenge.PID != child.pid || challenge.ExecutableSHA256 != digest || challenge.RequestSHA256 != hello.RequestSHA256 || challenge.Rank != 0 {
		t.Fatal("unbound child challenge")
	}
	admission := v41EPHAdmission{Challenge: challenge, ParentPID: parent.pid, ParentStart: parent.start, OwnerStart: child.start, TerminateUnixNS: hello.TerminateUnixNS, KillUnixNS: hello.KillUnixNS}
	if mode == "wrong-admission" {
		admission.Challenge.Rank = 2
	}
	unsigned, _ := json.Marshal(admission)
	admission.EvidenceSHA256 = v41EPHSHA(bytes.Join([][]byte{helloRaw, challengeRaw, unsigned}, []byte{0}))
	admissionRaw, _ := json.Marshal(admission)
	if err := v41EPHWrite(controlW, 4, admissionRaw, v41EPHControlCap); err != nil {
		t.Fatal(err)
	}
	if mode == "wrong-admission" {
		err := cmd.Wait()
		joined = true
		if err != nil || !strings.Contains(stdout.String(), "live admission identity or deadline disagrees") {
			t.Fatalf("bad admission: %v %s %s", err, stdout.String(), stderr.String())
		}
		return
	}
	if mode == "channel-loss" {
		_ = controlW.Close()
	}
	receiptRaw, err := v41EPHRead(eventR, 5, v41EPHReceiptCap)
	if err != nil {
		t.Fatal(err)
	}
	var receipt V41EPOwnerReceipt
	if err := json.Unmarshal(receiptRaw, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.SupervisorEvidenceSHA256 == nil || *receipt.SupervisorEvidenceSHA256 != admission.EvidenceSHA256 || receipt.RequestSHA256 != hello.RequestSHA256 || receipt.ExecutableSHA256 == nil || *receipt.ExecutableSHA256 != digest {
		t.Fatal("receipt binding changed")
	}
	err = cmd.Wait()
	joined = true
	if err != nil || stdout.String() != "HANDOFF_CHECKED\n" {
		t.Fatalf("handoff owner: %v %s %s", err, stdout.String(), stderr.String())
	}
}

func v41EPHTestOwner(mode string) error {
	digest, _, err := v41EPHImage("/proc/self/exe", nil)
	if err != nil {
		return err
	}
	// Test-only pin of this exact built fixture. Production has no request/env pin.
	v41EPOwnerGuardBinarySHA256 = digest
	if mode == "wrong-parent-pin" {
		v41EPOwnerGuardBinarySHA256 = strings.Repeat("a", 64)
	}
	if mode == "partial-fd-setup" {
		_ = syscall.Close(5)
	}
	h, err := openV41EPOwnerLiveHandoff(context.Background())
	if err != nil {
		for _, fd := range []int{3, 4, 5} {
			var stat syscall.Stat_t
			if !errors.Is(syscall.Fstat(fd, &stat), syscall.EBADF) {
				return errors.New("failed constructor retained an owned FD")
			}
		}
		fmt.Println(err)
		return nil
	}
	defer h.Close()
	actual, err := h.ObserveProcess(h.Context)
	if err != nil {
		return err
	}
	var nonce [32]byte
	if _, err = rand.Read(nonce[:]); err != nil {
		return err
	}
	challenge := v41EPOwnerSupervisorChallenge{Nonce: nonce, ContractID: h.Request.Supervisor.ContractID, RequestSHA256: h.hello.RequestSHA256, Rank: h.Rank, Process: actual,
		LatestTerminateAt: time.Now().Add(h.Request.Supervisor.TerminateNS), KillGrace: h.Request.Supervisor.KillGraceNS}
	handoff, err := h.Verify(h.Context, challenge)
	if err != nil {
		if !h.closed.Load() {
			return errors.New("failed verification retained owned link")
		}
		fmt.Println(err)
		return nil
	}
	if handoff.Challenge != challenge || !handoff.TerminateAt.Equal(h.terminate) || !handoff.KillAt.Equal(h.kill) {
		return errors.New("original invocation or deadline was replaced")
	}
	if _, err := h.Verify(h.Context, challenge); err == nil {
		return errors.New("handoff reused")
	}
	if mode == "channel-loss" {
		select {
		case <-h.Context.Done():
		case <-time.After(time.Second):
			return errors.New("parent channel loss did not cancel")
		}
	}
	receipt := V41EPOwnerReceipt{Schema: V41EPOwnerReceiptSchema, Status: "incomplete", Rank: h.Rank, RequestSHA256: h.hello.RequestSHA256, ExecutableSHA256: &digest, SupervisorEvidenceSHA256: &handoff.EvidenceSHA256}
	if err := h.Finish(receipt); err != nil {
		return err
	}
	if err := h.Close(); err != nil {
		return err
	}
	fmt.Println("HANDOFF_CHECKED")
	return nil
}
