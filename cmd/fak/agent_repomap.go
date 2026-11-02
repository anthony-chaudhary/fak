package main

import (
	"errors"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

type repoMapFileList []string

func (v *repoMapFileList) String() string { return strings.Join(*v, ",") }
func (v *repoMapFileList) Set(p string) error {
	if len(*v) >= ctxmmu.RepoMapMaxFiles {
		return errors.New("REPOMAP_FILES")
	}
	if err := ctxmmu.ValidateRepoMapPath(p); err != nil {
		return err
	}
	for _, old := range *v {
		if old == p {
			return errors.New("REPOMAP_DUPLICATE")
		}
	}
	*v = append(*v, p)
	return nil
}

// resolveAgentRepoMapOption is deliberately limited to a fresh native one-shot
// arm. It does not change default invocation mode, arm tools, alter policy,
// configure a provider, or implicitly enable a map for child agents.
func resolveAgentRepoMapOption(af *agentFlags, native, interactive bool) (agent.RunOption, error) {
	if len(af.repoMapFiles) == 0 {
		if *af.repoMapTokens != 0 {
			return nil, errors.New("REPOMAP_FILES_REQUIRED")
		}
		return nil, nil
	}
	if !native || interactive || !*af.codeTools || *af.resume != "" || *af.workflow != "" {
		return nil, errors.New("REPOMAP_REQUIRES_FRESH_NATIVE_CODE_ARM")
	}
	return agent.WithRepoMapPreload(agent.RepoMapPreload{
		Files: append([]string(nil), af.repoMapFiles...), Tokens: *af.repoMapTokens,
	})
}
