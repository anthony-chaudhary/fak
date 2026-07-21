package guardsessions

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// TestExternalSessionWitnessCrossProcess proves issue #3461 end-to-end with TWO
// real OS processes and the shipped fak.guard-session.v2 schema.
//
// A child process plays the guarded session: it binds an OS-picked loopback
// gateway on 127.0.0.1:0, serves /debug/vars behind a read-scoped bearer, and
// publishes its resolved gateway URL + bearer into the durable session index
// (Row.WithGateway → Record). The parent process plays a SECOND, independent
// process (an operator's `fak session status`): with NO prior knowledge of the
// child's port, it Loads the index, reads the published gateway_url + bearer,
// and fetches the child's live status. The read is admitted WITH the bearer and
// refused WITHOUT it; once the child is gone the endpoint is unreachable (stale).
//
// This is the witness the ticket asks for: a live guard session made discoverable
// and cross-process queryable with no port scraping.
func TestExternalSessionWitnessCrossProcess(t *testing.T) {
	if os.Getenv("FAK_WITNESS_ROLE") == "server" {
		runWitnessGuardChild()
		return
	}

	regDir := t.TempDir()

	// Spawn THIS test binary again as the guard child (a genuine second process).
	child := exec.Command(os.Args[0], "-test.run", "^TestExternalSessionWitnessCrossProcess$", "-test.v")
	child.Env = append(os.Environ(), "FAK_WITNESS_ROLE=server", "FAK_WITNESS_REGDIR="+regDir)
	stdout, err := child.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	killed := false
	defer func() {
		if !killed {
			_ = child.Process.Kill()
		}
		_, _ = child.Process.Wait()
	}()

	// Wait for the child to announce it is registered and serving.
	scanner := bufio.NewScanner(stdout)
	deadline := time.Now().Add(20 * time.Second)
	ready := false
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "WITNESS_READY") {
			ready = true
			break
		}
		if time.Now().After(deadline) {
			break
		}
	}
	if !ready {
		t.Fatal("guard child never announced WITNESS_READY")
	}

	// --- SECOND PROCESS: discover the session with NO prior port knowledge. ---
	var row Row
	found := false
	for i := 0; i < 50; i++ {
		for _, r := range Load(regDir) {
			if r.TraceID == "trace-witness" && strings.TrimSpace(r.GatewayURL) != "" {
				row, found = r, true
				break
			}
		}
		if found {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if !found {
		t.Fatal("operator could not discover the guard session's published gateway from the index")
	}
	if row.Bearer == "" {
		t.Fatalf("discovered row carries no read bearer: %+v", row)
	}
	t.Logf("operator discovered session %s at %s (bearer len=%d) with no prior port knowledge",
		row.Handle, row.GatewayURL, len(row.Bearer))

	// Authenticated read is admitted.
	body, code := witnessGet(t, row.GatewayURL+"/debug/vars", row.Bearer)
	if code != http.StatusOK {
		t.Fatalf("authenticated /debug/vars = %d, want 200 (body=%q)", code, body)
	}
	if !strings.Contains(body, "startup_report") {
		t.Fatalf("status body missing startup_report: %q", body)
	}
	t.Logf("authenticated cross-process read OK: %s", body)

	// Unauthenticated read is refused (bearer is required, read-scope).
	_, code = witnessGet(t, row.GatewayURL+"/debug/vars", "")
	if code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated /debug/vars = %d, want 401", code)
	}
	// A wrong bearer is refused too.
	_, code = witnessGet(t, row.GatewayURL+"/debug/vars", "not-the-token")
	if code != http.StatusUnauthorized {
		t.Fatalf("wrong-bearer /debug/vars = %d, want 401", code)
	}
	t.Log("unauthenticated and wrong-bearer reads correctly refused")

	// --- Kill the guard: the session becomes stale (endpoint unreachable). ---
	_ = child.Process.Kill()
	_, _ = child.Process.Wait()
	killed = true
	unreachable := false
	for i := 0; i < 50; i++ {
		if _, _, err := witnessTry(row.GatewayURL+"/debug/vars", row.Bearer); err != nil {
			unreachable = true
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if !unreachable {
		t.Fatal("gateway still reachable after the guard was killed; stale session not detectable")
	}
	t.Log("after guard exit the published gateway is unreachable — stale session detectable by the operator")
}

// runWitnessGuardChild is the guard side: bind a loopback gateway behind a
// read-scoped bearer, publish it into the index, announce READY, and stay alive.
func runWitnessGuardChild() {
	regDir := os.Getenv("FAK_WITNESS_REGDIR")
	bearer := fmt.Sprintf("readscope-%d-abc123", os.Getpid())

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		fmt.Println("WITNESS_ERR", err)
		os.Exit(1)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/debug/vars", func(w http.ResponseWriter, r *http.Request) {
		// Loopback + read-scoped bearer. No bearer (or a wrong one) => refused.
		if r.Header.Get("Authorization") != "Bearer "+bearer {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		_, _ = io.WriteString(w, `{"startup_report":{"agent":"witness-agent"},"drive":{"state":"running"}}`)
	})
	go func() { _ = http.Serve(ln, mux) }()

	gwURL := "http://" + ln.Addr().String()
	row := NewRow("trace-witness", "witness-agent", os.Getpid(), "/witness/cwd", "audit.jsonl", "nonce-w", time.Now().UTC()).
		WithGateway(gwURL, bearer)
	if err := Record(regDir, row); err != nil {
		fmt.Println("WITNESS_ERR", err)
		os.Exit(1)
	}
	fmt.Println("WITNESS_READY", gwURL)
	time.Sleep(60 * time.Second)
}

func witnessGet(t *testing.T, url, bearer string) (string, int) {
	t.Helper()
	body, code, err := witnessTry(url, bearer)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	return body, code
}

func witnessTry(url, bearer string) (string, int, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return "", 0, err
	}
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	resp, err := (&http.Client{Timeout: 2 * time.Second}).Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return string(b), resp.StatusCode, nil
}
