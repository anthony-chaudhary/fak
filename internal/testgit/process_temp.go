package testgit

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

// processTempParentEnvPrefix names the variable ParentProcessTempDir exports so
// a re-executed child of the same test binary recognizes its parent's dir.
const processTempParentEnvPrefix = "FAK_PROCESS_TEMP_PARENT_"

var processTemp struct {
	sync.Mutex
	dirs     []string
	prefixes []string
}

// ProcessTempDir creates a directory under os.TempDir() owned by this process
// and registers it for RemoveProcessTempDirs / ExitProcessTemp.
//
// A TestMain cannot rely on defer: os.Exit skips it, and a re-executed helper
// child usually exits inside its test. The dir name carries the owner pid, so
// creating one first reaps same-prefix siblings whose owner is dead.
func ProcessTempDir(prefix string) (string, error) {
	if prefix == "" {
		return "", fmt.Errorf("testgit: ProcessTempDir needs a prefix")
	}
	dir, err := processalive.OwnedTempDir(prefix)
	if err != nil {
		return "", err
	}
	processTemp.Lock()
	processTemp.dirs = append(processTemp.dirs, dir)
	if !slices.Contains(processTemp.prefixes, prefix) {
		processTemp.prefixes = append(processTemp.prefixes, prefix)
	}
	processTemp.Unlock()
	return dir, nil
}

// MustProcessTempDir is ProcessTempDir for TestMain, which has no *testing.T.
func MustProcessTempDir(prefix string) string {
	dir, err := ProcessTempDir(prefix)
	if err != nil {
		panic(err)
	}
	return dir
}

// ParentProcessTempDir is ProcessTempDir for state a re-executed child of the
// test binary must neither share nor duplicate. In such a child (the
// environment names a live other process as the prefix's owner) it creates
// nothing and returns ("", nil); otherwise it creates the dir and exports this
// process as the owner.
func ParentProcessTempDir(prefix string) (string, error) {
	key := processTempParentEnv(prefix)
	if pid, err := strconv.Atoi(os.Getenv(key)); err == nil && pid != os.Getpid() && processalive.Check(pid) {
		return "", nil
	}
	dir, err := ProcessTempDir(prefix)
	if err != nil {
		return "", err
	}
	_ = os.Setenv(key, strconv.Itoa(os.Getpid()))
	return dir, nil
}

// RemoveProcessTempDirs removes every dir this process created through
// ProcessTempDir, then reaps dead-owner siblings of the same prefixes (the
// dirs of re-executed children that exited inside their test).
func RemoveProcessTempDirs() {
	processTemp.Lock()
	dirs, prefixes := processTemp.dirs, processTemp.prefixes
	processTemp.dirs, processTemp.prefixes = nil, nil
	processTemp.Unlock()
	for _, dir := range dirs {
		_ = os.RemoveAll(dir)
	}
	for _, prefix := range prefixes {
		processalive.ReapDeadOwnerTemp(prefix)
	}
}

// ExitProcessTemp removes this process's ProcessTempDir dirs and exits with
// code. It is the TestMain tail: `testgit.ExitProcessTemp(m.Run())`.
func ExitProcessTemp(code int) {
	RemoveProcessTempDirs()
	os.Exit(code)
}

func processTempParentEnv(prefix string) string {
	var b strings.Builder
	b.WriteString(processTempParentEnvPrefix)
	for _, r := range strings.ToUpper(prefix) {
		if (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	return b.String()
}
