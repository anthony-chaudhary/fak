package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
)

const (
	serveControlIngressJournalEnv = "FAK_CONTROL_INGRESS_JOURNAL"
	serveControlDirectivePath     = "/v1/fak/control/directives"
	serveControlIngressBodyLimit  = int64(1 << 20)
	windowsJournalProvisionReason = "WINDOWS_CONTROL_JOURNAL_NOT_PREPROVISIONED"
)

// openServeControlIngress provisions and opens the production control journal before
// the listener can bind. The HTTP control surface stays unavailable when no credential
// door exists; a bearer/keyset authenticates a caller but is not evidence of human origin.
func openServeControlIngress(stdio bool, requiredKey string, keyPrincipals map[string]string, env func(string) string) (gateway.ControlIngress, *gateway.DurableControlIngress, error) {
	if stdio || (strings.TrimSpace(requiredKey) == "" && len(keyPrincipals) == 0) {
		return nil, nil, nil
	}
	path := ""
	if env != nil {
		path = strings.TrimSpace(env(serveControlIngressJournalEnv))
	}
	if path == "" {
		return nil, nil, nil
	}
	if err := provisionServeControlJournal(path); err != nil {
		return nil, nil, err
	}
	durable, err := gateway.OpenDurableControlIngress(gateway.DurableControlIngressOptions{
		JournalPath: path,
		Deliver:     serveControlDirectiveDeliver,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("open durable journal %q: %w", path, err)
	}
	return &serveControlIngress{durable: durable}, durable, nil
}

var serveControlDirectiveDeliver = deliverServeControlDirective

// provisionServeControlJournal creates the dedicated journal with private permissions
// and syncs both file and parent directory before OpenDurableControlIngress recovers it.
// Windows cannot establish first-create directory-entry durability through Go's portable
// file API, so it fails closed unless deployment already provisioned the regular file.
func provisionServeControlJournal(path string) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return errors.New("journal path is empty")
	}
	dir := filepath.Dir(path)
	pathInfo, pathErr := os.Lstat(path)
	if runtime.GOOS == "windows" && errors.Is(pathErr, fs.ErrNotExist) {
		return fmt.Errorf("%s: journal %q must be durably provisioned as a regular file before fak serve starts", windowsJournalProvisionReason, path)
	}
	if pathErr != nil && !errors.Is(pathErr, fs.ErrNotExist) {
		return fmt.Errorf("inspect journal path %q: %w", path, pathErr)
	}
	dirInfo, dirErr := os.Lstat(dir)
	dirCreated := false
	if errors.Is(dirErr, fs.ErrNotExist) {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return fmt.Errorf("create journal directory %q: %w", dir, err)
		}
		dirCreated = true
		dirInfo, dirErr = os.Lstat(dir)
	}
	if dirErr != nil {
		return fmt.Errorf("inspect journal directory %q: %w", dir, dirErr)
	}
	if dirInfo.Mode()&os.ModeSymlink != 0 || !dirInfo.IsDir() {
		return fmt.Errorf("journal directory %q must be a non-symlink directory", dir)
	}
	// Change permissions only on the dedicated directory this call created. An
	// operator-supplied existing parent may contain unrelated state and must never
	// be chmodded as a side effect of enabling control ingress.
	if dirCreated {
		if err := os.Chmod(dir, 0o700); err != nil {
			return fmt.Errorf("restrict new journal directory %q: %w", dir, err)
		}
	}
	if pathErr == nil {
		if pathInfo.Mode()&os.ModeSymlink != 0 || !pathInfo.Mode().IsRegular() {
			return fmt.Errorf("journal %q must be a regular non-symlink file", path)
		}
	}
	openFlags := os.O_RDWR
	fileCreated := errors.Is(pathErr, fs.ErrNotExist)
	if fileCreated {
		openFlags |= os.O_CREATE | os.O_EXCL
	}
	f, err := os.OpenFile(path, openFlags, 0o600)
	if err != nil {
		return fmt.Errorf("provision journal %q: %w", path, err)
	}
	closeWith := func(base error) error {
		if closeErr := f.Close(); base == nil && closeErr != nil {
			return closeErr
		}
		return base
	}
	info, err := f.Stat()
	if err != nil {
		return fmt.Errorf("stat journal %q: %w", path, closeWith(err))
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("journal %q is not a regular file: %w", path, closeWith(fs.ErrInvalid))
	}
	boundInfo, err := os.Lstat(path)
	if err != nil {
		return fmt.Errorf("recheck journal path %q: %w", path, closeWith(err))
	}
	if boundInfo.Mode()&os.ModeSymlink != 0 || !boundInfo.Mode().IsRegular() || !os.SameFile(info, boundInfo) {
		return fmt.Errorf("journal path %q changed or resolves through a symlink: %w", path, closeWith(fs.ErrInvalid))
	}
	// Unix mode bits express the required private file posture. Chmod is allowed
	// only for the inode this call created with O_EXCL. Existing files are verified,
	// never changed, so an alias cannot make bootstrap chmod unrelated state.
	// Windows ACLs are part of external pre-provisioning; chmod there would neither
	// prove that ACL nor justify mutating a preexisting file.
	if runtime.GOOS != "windows" {
		if fileCreated {
			if err := f.Chmod(0o600); err != nil {
				return fmt.Errorf("restrict new journal %q: %w", path, closeWith(err))
			}
		} else if info.Mode().Perm() != 0o600 {
			return fmt.Errorf("preexisting journal %q has mode %04o, want 0600", path, info.Mode().Perm())
		}
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("sync journal %q: %w", path, closeWith(err))
	}
	if err := closeWith(nil); err != nil {
		return fmt.Errorf("close journal %q: %w", path, err)
	}
	// Windows reached this point only for a file that existed before bootstrap;
	// no directory entry was created here. The file flush above is sufficient for
	// its current bytes, while the deployment owns proof of the older entry's
	// durability. Do not turn an unsupported directory Sync into a false witness.
	if runtime.GOOS == "windows" {
		return nil
	}
	parent, err := os.Open(dir)
	if err != nil {
		return fmt.Errorf("open journal directory %q for sync: %w", dir, err)
	}
	syncErr := parent.Sync()
	closeErr := parent.Close()
	if syncErr != nil {
		return fmt.Errorf("sync journal directory %q: %w", dir, syncErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close journal directory %q: %w", dir, closeErr)
	}
	return nil
}

// deliverServeControlDirective uses the same durable ControlSession write as the
// existing management endpoint. Only cancel has proven semantics here: it transitions
// the target to Terminating, which closes the table's level-triggered terminate signal.
func deliverServeControlDirective(ctx context.Context, directive gateway.ControlDirective) error {
	if directive.Action != "cancel" {
		return fmt.Errorf("control action %q is unsupported by fak serve", directive.Action)
	}
	_, ok, err := controlSession(ctx, directive.Target, "run", gateway.SessionControlRequest{
		Run:    "terminating",
		Reason: session.ReasonTerminated,
	})
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("cancel target %q refused the terminating transition", directive.Target)
	}
	return nil
}

// serveControlIngress rejects actions whose production delivery semantics are not yet
// proven before they enter the durable accepted journal. Malformed requests still flow
// to the canonical ingress parser and retain its exact validation contract.
type serveControlIngress struct {
	durable *gateway.DurableControlIngress
}

func (s *serveControlIngress) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost && r.URL.Path == serveControlDirectivePath {
		prefix, err := io.ReadAll(io.LimitReader(r.Body, serveControlIngressBodyLimit+1))
		if err == nil {
			if int64(len(prefix)) > serveControlIngressBodyLimit {
				r.Body = io.NopCloser(io.MultiReader(bytes.NewReader(prefix), r.Body))
			} else {
				r.Body = io.NopCloser(bytes.NewReader(prefix))
				dec := json.NewDecoder(bytes.NewReader(prefix))
				dec.DisallowUnknownFields()
				var directive gateway.ControlDirective
				var trailing any
				if dec.Decode(&directive) == nil && dec.Decode(&trailing) == io.EOF && directive.Action != "cancel" && isKnownServeControlAction(directive.Action) {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusServiceUnavailable)
					_ = json.NewEncoder(w).Encode(gateway.ControlReceipt{
						ID: directive.ID, Target: directive.Target, Action: directive.Action,
						Generation: directive.Generation, State: "unavailable", Reason: "unsupported_action",
					})
					return
				}
			}
		}
	}
	s.durable.ServeHTTP(w, r)
}

func isKnownServeControlAction(action string) bool {
	switch action {
	case "pause", "resume", "steer", "reprioritize", "redirect", "stop":
		return true
	default:
		return false
	}
}
