package gateway

import (
	"net/http"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/director"
)

var (
	directorEngineMu   sync.RWMutex
	directorEngineInst *director.RollupEngine
	directorEngineOnce sync.Once
)

// GetDirectorEngine returns the package-level RollupEngine instance, initializing it if necessary.
func GetDirectorEngine() *director.RollupEngine {
	directorEngineMu.RLock()
	eng := directorEngineInst
	directorEngineMu.RUnlock()
	if eng != nil {
		return eng
	}

	directorEngineOnce.Do(func() {
		directorEngineMu.Lock()
		if directorEngineInst == nil {
			directorEngineInst = director.NewRollupEngine()
		}
		directorEngineMu.Unlock()
	})

	directorEngineMu.RLock()
	defer directorEngineMu.RUnlock()
	return directorEngineInst
}

// SetDirectorEngine overrides the package-level RollupEngine instance (for tests or custom configuration).
func SetDirectorEngine(eng *director.RollupEngine) {
	directorEngineMu.Lock()
	defer directorEngineMu.Unlock()
	directorEngineInst = eng
}

// handleA2AGetDirectorDigest implements GET /a2a/v1/director/digest.
// It serves the autonomous multi-agent roll-up digest for zero-self-report supervisor steering (#11411).
func (s *Server) handleA2AGetDirectorDigest(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	engine := GetDirectorEngine()
	digest := engine.CompileDigest()
	writeJSON(w, http.StatusOK, digest)
}
