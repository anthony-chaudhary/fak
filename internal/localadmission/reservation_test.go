package localadmission

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/memgate"
)

func reservationRequest(pid int, peak, steady, capacity int64, pressure memgate.Pressure) ReservationRequest {
	return ReservationRequest{
		OwnerPID: pid,
		Plan:     MemoryPlan{StartupPeakBytes: peak, SteadyBytes: steady},
		Host:     memgate.AdmissionSample{TotalBytes: capacity, AllocatableBytes: capacity, Pressure: pressure},
	}
}

func TestReservationLifecycleAccountsStartupAndSteadySeparately(t *testing.T) {
	ctx := context.Background()
	store := NewReservationStore(t.TempDir())
	store.alive = func(int) bool { return true }

	first, err := store.Reserve(ctx, reservationRequest(101, 60, 30, 100, memgate.PressureNormal))
	if err != nil || !first.Admit || first.Reservation.HeldBytes != 60 || first.Reservation.Phase != "startup" {
		t.Fatalf("first=%+v err=%v", first, err)
	}
	blocked, err := store.Reserve(ctx, reservationRequest(102, 50, 20, 100, memgate.PressureNormal))
	if err != nil || blocked.Admit || blocked.Reason != "aggregate_capacity" || blocked.ReservedBytes != 60 {
		t.Fatalf("blocked=%+v err=%v", blocked, err)
	}
	steady, err := store.MarkSteady(ctx, first.Reservation.ID)
	if err != nil || steady.HeldBytes != 30 || steady.Phase != "steady" {
		t.Fatalf("steady=%+v err=%v", steady, err)
	}
	second, err := store.Reserve(ctx, reservationRequest(102, 50, 20, 100, memgate.PressureNormal))
	if err != nil || !second.Admit || second.ReservedBytes != 30 {
		t.Fatalf("second=%+v err=%v", second, err)
	}
	if err := store.Release(ctx, first.Reservation.ID); err != nil {
		t.Fatal(err)
	}
	if err := store.Release(ctx, first.Reservation.ID); !errors.Is(err, ErrReservationNotFound) {
		t.Fatalf("second release err=%v", err)
	}
}

func TestReservationFailsClosedBeforeCreatingLedger(t *testing.T) {
	for _, pressure := range []memgate.Pressure{memgate.PressureUnknown, memgate.PressureCritical} {
		t.Run(string(pressure), func(t *testing.T) {
			dir := t.TempDir()
			store := NewReservationStore(dir)
			got, err := store.Reserve(context.Background(), reservationRequest(os.Getpid(), 1, 1, 100, pressure))
			if err != nil || got.Admit || got.Reason != "pressure_"+string(pressure) {
				t.Fatalf("decision=%+v err=%v", got, err)
			}
			if _, err := os.Stat(filepath.Join(dir, "reservations.json")); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("ledger created before refusal: %v", err)
			}
		})
	}
}

func TestReservationReapsDeadOwner(t *testing.T) {
	ctx := context.Background()
	store := NewReservationStore(t.TempDir())
	alive := map[int]bool{101: true, 102: true}
	store.alive = func(pid int) bool { return alive[pid] }
	first, err := store.Reserve(ctx, reservationRequest(101, 80, 60, 100, memgate.PressureNormal))
	if err != nil || !first.Admit {
		t.Fatalf("first=%+v err=%v", first, err)
	}
	alive[101] = false
	second, err := store.Reserve(ctx, reservationRequest(102, 80, 60, 100, memgate.PressureNormal))
	if err != nil || !second.Admit || second.Reaped != 1 || second.ReservedBytes != 0 {
		t.Fatalf("second=%+v err=%v", second, err)
	}
}

func TestReservationStoresSerializeConcurrentCallers(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	const callers = 16
	var wg sync.WaitGroup
	results := make(chan ReservationDecision, callers)
	errs := make(chan error, callers)
	for range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			store := NewReservationStore(dir)
			store.alive = func(int) bool { return true }
			got, err := store.Reserve(ctx, reservationRequest(os.Getpid(), 40, 20, 100, memgate.PressureNormal))
			if err != nil {
				errs <- err
				return
			}
			results <- got
		}()
	}
	wg.Wait()
	close(results)
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	admitted := 0
	for got := range results {
		if got.Admit {
			admitted++
		}
	}
	if admitted != 2 {
		t.Fatalf("admitted=%d want 2", admitted)
	}
}

func TestReservationProcessesCannotOvercommit(t *testing.T) {
	if testing.Short() {
		t.Skip("subprocess witness")
	}
	dir := t.TempDir()
	barrier := filepath.Join(dir, "start")
	const callers = 8
	type child struct {
		cmd *exec.Cmd
		out *lockedBuffer
	}
	children := make([]child, 0, callers)
	for i := 0; i < callers; i++ {
		out := &lockedBuffer{}
		cmd := exec.Command(os.Args[0], "-test.run=^TestReservationProcessHelper$")
		cmd.Env = append(os.Environ(),
			"FAK_RESERVATION_HELPER=1",
			"FAK_RESERVATION_DIR="+dir,
			"FAK_RESERVATION_BARRIER="+barrier,
			"FAK_RESERVATION_OWNER="+strconv.Itoa(os.Getpid()),
		)
		cmd.Stdout, cmd.Stderr = out, out
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		children = append(children, child{cmd: cmd, out: out})
	}
	if err := os.WriteFile(barrier, []byte("go"), 0o600); err != nil {
		t.Fatal(err)
	}
	admitted := 0
	for _, child := range children {
		if err := child.cmd.Wait(); err != nil {
			t.Fatalf("helper: %v\n%s", err, child.out.String())
		}
		var got ReservationDecision
		line := strings.SplitN(child.out.String(), "\n", 2)[0]
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("decode %q: %v", child.out.String(), err)
		}
		if got.Admit {
			admitted++
		}
	}
	if admitted != 2 {
		t.Fatalf("admitted=%d want 2", admitted)
	}
}

func TestReservationProcessHelper(t *testing.T) {
	if os.Getenv("FAK_RESERVATION_HELPER") != "1" {
		return
	}
	barrier := os.Getenv("FAK_RESERVATION_BARRIER")
	deadline := time.Now().Add(10 * time.Second)
	for {
		if _, err := os.Stat(barrier); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("barrier timeout")
		}
		time.Sleep(time.Millisecond)
	}
	owner, err := strconv.Atoi(os.Getenv("FAK_RESERVATION_OWNER"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := NewReservationStore(os.Getenv("FAK_RESERVATION_DIR")).Reserve(context.Background(), reservationRequest(owner, 40, 20, 100, memgate.PressureNormal))
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(string(b))
}

type lockedBuffer struct {
	mu sync.Mutex
	b  []byte
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.b = append(b.b, p...)
	return len(p), nil
}
func (b *lockedBuffer) Bytes() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]byte(nil), b.b...)
}
func (b *lockedBuffer) String() string { return string(b.Bytes()) }
