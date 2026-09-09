// completion.go — the `fak completion` verb.
// Generates shell completion scripts for bash, zsh, and fish.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

func cmdCompletion(args []string) {
	code := runCompletion(os.Stdout, os.Stderr, args)
	if code != 0 {
		os.Exit(code)
	}
}

func runCompletion(stdout, stderr io.Writer, args []string) int {
	for _, arg := range args {
		if arg == "-h" || arg == "--help" || arg == "help" {
			fmt.Fprintf(stdout, "usage: fak completion <bash|zsh|fish>\n\nGenerate shell completion scripts for fak.\n\nOptions:\n  --shell string  shell type (bash, zsh, fish)\n")
			return 0
		}
	}

	fs := flag.NewFlagSet("completion", flag.ContinueOnError)
	shellFlag := fs.String("shell", "", "shell type (bash, zsh, fish)")
	fs.SetOutput(stderr)
	if err := fs.Parse(args); err != nil {
		return 2
	}

	shell := strings.ToLower(strings.TrimSpace(*shellFlag))
	if shell == "" && fs.NArg() > 0 {
		shell = strings.ToLower(strings.TrimSpace(fs.Arg(0)))
	}

	if shell == "" {
		fmt.Fprintf(stdout, "usage: fak completion <bash|zsh|fish>\n\nGenerate shell completion scripts for fak.\n")
		return 0
	}

	switch shell {
	case "bash":
		writeBashCompletion(stdout)
	case "zsh":
		writeZshCompletion(stdout)
	case "fish":
		writeFishCompletion(stdout)
	default:
		fmt.Fprintf(stderr, "fak completion: unsupported shell %q (supported: bash, zsh, fish)\n", shell)
		return 2
	}
	return 0
}

func writeBashCompletion(w io.Writer) {
	const bashScript = `# bash completion for fak                         -*- shell-script -*-

_fak() {
    local cur prev words cword
    if type -t _init_completion >/dev/null 2>&1; then
        _init_completion -n = || return
    else
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
        words=("${COMP_WORDS[@]}")
        cword=$COMP_CWORD
    fi

    local commands="ablate agent api-host architecture armbench attest audit bench build capabilities catchup codex codex-resume completion component compute config coordinate disambiguation doctor egress fanout glm52-prefill-sweep godsplit-plan harness help hook info info-fleet launch lifecycle llmd-smoke ls manage model opencode pack pi policy preflight progress ps pull quantbench question-ledger recall recover redteam replay resume run scratch-janitor self-update serve session session-audit sessionjournal signal stale-work study task tasks temp-artifacts test-quality tier-calibrate tool-width top tree-doctor trunk-build-probe ultracode value-chain version windows-setup wip work-delivery workspin"

    if [[ $cword -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
        return
    fi

    case "${words[1]}" in
        help)
            COMPREPLY=( $(compgen -W "$commands --all --full" -- "$cur") )
            return
            ;;
        completion)
            COMPREPLY=( $(compgen -W "bash zsh fish" -- "$cur") )
            return
            ;;
    esac
}

complete -o default -F _fak fak
`
	fmt.Fprint(w, bashScript)
}

func writeZshCompletion(w io.Writer) {
	const zshScript = `#compdef fak

_fak() {
    local -a commands
    commands=(
        'up:boot the unified agent runtime, gateway, policy, metrics, and session API'
        'manage:wrap an agent harness: manage every tool call in-process'
        'serve:the OpenAI-compatible gateway in front of a local or remote model'
        'agent:run one managed-agent task end to end'
        'ultracode:plan, launch, and observe a bounded concurrent coding-agent fleet'
        'run:run an agent turn (or a recorded trace) through the kernel'
        'claude:launch Claude Code directly against fak serve backend on Mac'
        'codex:launch OpenAI Codex routed through the kernel'
        'opencode:launch OpenCode routed through the kernel'
        'pi:launch Pi coding agent directly targeting fak serve backend'
        'build:build fak from source'
        'session:budget turns/tokens/context; steer or stop'
        'info:live reused-token, effective-cost, and total-savings overlay'
        'resume:price full replay vs cut/reset when resuming a long context'
        'ablate:same-trace cache ablation'
        'capabilities:query token, turn, cache, routing outcomes'
        'progress:one query for recent commits, local WIP, and issue movement'
        'ps:live served-session process table'
        'signal:job control for a running session'
        'doctor:diagnose runtime, kernel admission, and default launch posture'
        'recover:map a refusal reason token to concrete recovery commands'
        'preflight:adjudicate one tool call against a policy'
        'policy:dump / check the deployable capability floor'
        'attest:compliance attestation: prove the policy floor'
        'audit:verify / export a guard decision journal hash chain'
        'egress:prove the network-egress floor'
        'model:resolve / cache an hf:// model'
        'self-update:converge a built-from-source fak binary on origin/main'
        'version:print the fak version'
        'help:overview and help for commands'
        'completion:generate shell completion script for bash, zsh, fish'
    )

    _arguments -C \
        '1: :->command' \
        '*:: :->args'

    case $state in
        command)
            _describe -t commands 'fak command' commands
            ;;
        args)
            case $words[1] in
                completion)
                    _values 'shell' 'bash' 'zsh' 'fish'
                    ;;
                help)
                    _describe -t commands 'fak command' commands
                    ;;
            esac
            ;;
    esac
}

_fak "$@"
`
	fmt.Fprint(w, zshScript)
}

func writeFishCompletion(w io.Writer) {
	const fishScript = `# fish completion for fak

function __fak_needs_command
    set -l cmd (commandline -opc)
    if test (count $cmd) -eq 1
        return 0
    end
    return 1
end

function __fak_using_command
    set -l cmd (commandline -opc)
    if test (count $cmd) -gt 1
        if test $cmd[2] = $argv[1]
            return 0
        end
    end
    return 1
end

# Commands
complete -c fak -n "__fak_needs_command" -a "up" -d "boot the unified agent runtime"
complete -c fak -n "__fak_needs_command" -a "manage" -d "wrap an agent harness"
complete -c fak -n "__fak_needs_command" -a "serve" -d "OpenAI-compatible gateway"
complete -c fak -n "__fak_needs_command" -a "agent" -d "run one managed-agent task"
complete -c fak -n "__fak_needs_command" -a "ultracode" -d "plan, launch, and observe fleet"
complete -c fak -n "__fak_needs_command" -a "run" -d "run an agent turn"
complete -c fak -n "__fak_needs_command" -a "claude" -d "launch Claude Code"
complete -c fak -n "__fak_needs_command" -a "codex" -d "launch OpenAI Codex"
complete -c fak -n "__fak_needs_command" -a "opencode" -d "launch OpenCode"
complete -c fak -n "__fak_needs_command" -a "pi" -d "launch Pi coding agent"
complete -c fak -n "__fak_needs_command" -a "build" -d "build fak from source"
complete -c fak -n "__fak_needs_command" -a "session" -d "budget turns/tokens/context"
complete -c fak -n "__fak_needs_command" -a "info" -d "live reused-token overlay"
complete -c fak -n "__fak_needs_command" -a "resume" -d "price full replay vs cut/reset"
complete -c fak -n "__fak_needs_command" -a "ablate" -d "same-trace cache ablation"
complete -c fak -n "__fak_needs_command" -a "capabilities" -d "query outcomes"
complete -c fak -n "__fak_needs_command" -a "progress" -d "recent commits and WIP"
complete -c fak -n "__fak_needs_command" -a "ps" -d "live served-session process table"
complete -c fak -n "__fak_needs_command" -a "signal" -d "job control for running session"
complete -c fak -n "__fak_needs_command" -a "doctor" -d "diagnose runtime"
complete -c fak -n "__fak_needs_command" -a "recover" -d "map refusal reason to recovery"
complete -c fak -n "__fak_needs_command" -a "preflight" -d "adjudicate tool call against policy"
complete -c fak -n "__fak_needs_command" -a "policy" -d "dump / check capability floor"
complete -c fak -n "__fak_needs_command" -a "attest" -d "compliance attestation"
complete -c fak -n "__fak_needs_command" -a "audit" -d "verify decision journal"
complete -c fak -n "__fak_needs_command" -a "egress" -d "prove network-egress floor"
complete -c fak -n "__fak_needs_command" -a "model" -d "resolve / cache hf:// model"
complete -c fak -n "__fak_needs_command" -a "self-update" -d "converge on origin/main"
complete -c fak -n "__fak_needs_command" -a "version" -d "print fak version"
complete -c fak -n "__fak_needs_command" -a "help" -d "overview and help"
complete -c fak -n "__fak_needs_command" -a "completion" -d "generate shell completions"

# Completion subcommands
complete -c fak -n "__fak_using_command completion" -a "bash zsh fish"
`
	fmt.Fprint(w, fishScript)
}
