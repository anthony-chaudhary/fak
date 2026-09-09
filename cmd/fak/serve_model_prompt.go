package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
)

// partitionServeArgs partitions argv into flag arguments (to be passed to FlagSet.Parse)
// and positional arguments, respecting non-boolean flags with separate values and "--".
func partitionServeArgs(fs *flag.FlagSet, argv []string) (flagArgs []string, posArgs []string) {
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if arg == "--" {
			posArgs = append(posArgs, argv[i+1:]...)
			break
		}
		if strings.HasPrefix(arg, "-") {
			flagArgs = append(flagArgs, arg)
			flagName := strings.TrimLeft(arg, "-")
			if !strings.Contains(flagName, "=") {
				f := fs.Lookup(flagName)
				if f != nil {
					if bf, ok := f.Value.(interface{ IsBoolFlag() bool }); !ok || !bf.IsBoolFlag() {
						if i+1 < len(argv) && !strings.HasPrefix(argv[i+1], "-") {
							i++
							flagArgs = append(flagArgs, argv[i])
						}
					}
				}
			}
		} else {
			posArgs = append(posArgs, arg)
		}
	}
	return flagArgs, posArgs
}

// parseServeArgs partitions argv, executes FlagSet.Parse on the flag arguments,
// and interprets positional arguments (e.g. `model`, model references, or unexpected arguments).
func parseServeArgs(fs *flag.FlagSet, sf *serveFlags, argv []string) error {
	flagArgs, posArgs := partitionServeArgs(fs, argv)
	if err := fs.Parse(flagArgs); err != nil {
		return err
	}

	// Consume leading "model" keyword if present (e.g. `fak serve model --gguf ...` or `fak serve model <ref>`)
	if len(posArgs) > 0 && strings.EqualFold(posArgs[0], "model") {
		posArgs = posArgs[1:]
	}

	// Positional model argument: e.g. `fak serve qwen38:27b-q4` or `fak serve model qwen38:27b-q4`
	if len(posArgs) == 1 {
		candidate := strings.TrimSpace(posArgs[0])
		if candidate != "" {
			if strings.EqualFold(candidate, "mock") {
				if sf.mock != nil {
					*sf.mock = true
				}
				if sf.model != nil {
					*sf.model = "mock"
				}
			} else if strings.HasPrefix(candidate, "http://") || strings.HasPrefix(candidate, "https://") {
				if sf.baseURL != nil && *sf.baseURL == "" {
					*sf.baseURL = candidate
				}
			} else {
				if sf.ggufPath != nil {
					if *sf.ggufPath == "" {
						*sf.ggufPath = candidate
					} else if *sf.ggufPath != candidate {
						return fmt.Errorf("fak serve: model specified both as positional argument %q and via --gguf %q", candidate, *sf.ggufPath)
					}
				}
			}
		}
		posArgs = posArgs[1:]
	}

	if len(posArgs) > 0 {
		return fmt.Errorf("fak serve: unexpected argument %q", posArgs[0])
	}

	// If explicit --model was provided with a non-mock model name and no upstream / gguf / mock:
	explicit := explicitFlagNames(fs)
	if explicit["model"] && sf.model != nil && *sf.model != "" && !strings.EqualFold(*sf.model, "mock") {
		if sf.ggufPath != nil && *sf.ggufPath == "" && (sf.baseURL == nil || *sf.baseURL == "") && (sf.mock == nil || !*sf.mock) {
			*sf.ggufPath = *sf.model
		}
	}

	return nil
}

// resolveServeModelOrPrompt ensures a model or provider target is selected for
// `fak serve`. If none was provided via flags or positional arguments, it prompts
// the operator interactively for a model path or alias instead of silently
// falling back to the offline mock planner. In non-interactive environments, it
// fails closed with a clear error unless --mock is explicitly requested.
func resolveServeModelOrPrompt(sf *serveFlags, explicit map[string]bool, in io.Reader, out io.Writer, isInteractive bool) error {
	if sf == nil {
		return nil
	}
	// Early exits or non-chat surfaces that don't boot the live chat gateway.
	if (sf.printEffectiveConfig != nil && *sf.printEffectiveConfig) ||
		(sf.policyCheck != nil && *sf.policyCheck) ||
		(sf.opencodeConfig != nil && *sf.opencodeConfig) ||
		(sf.writeOpencodeConfig != nil && *sf.writeOpencodeConfig) ||
		(sf.claudeConfig != nil && *sf.claudeConfig) ||
		(sf.writeClaudeConfig != nil && *sf.writeClaudeConfig) ||
		(sf.codexConfig != nil && *sf.codexConfig) ||
		(sf.writeCodexConfig != nil && *sf.writeCodexConfig) ||
		(sf.sizingJSON != nil && *sf.sizingJSON) ||
		(sf.stdio != nil && *sf.stdio) {
		return nil
	}

	// 1. Explicit --mock flag.
	if sf.mock != nil && *sf.mock {
		return nil
	}

	// 2. Explicit --model mock.
	if explicit["model"] && sf.model != nil && strings.EqualFold(strings.TrimSpace(*sf.model), "mock") {
		if sf.mock != nil {
			*sf.mock = true
		}
		return nil
	}

	// 3. Model/target already specified via --gguf or upstream URLs.
	hasGGUF := sf.ggufPath != nil && strings.TrimSpace(*sf.ggufPath) != ""
	hasBaseURL := sf.baseURL != nil && strings.TrimSpace(*sf.baseURL) != ""
	hasReplicas := len(sf.replicaBaseURLs.Values()) > 0
	if hasGGUF || hasBaseURL || hasReplicas {
		return nil
	}

	// 4. Explicit --model <name> where name is not "mock".
	if explicit["model"] && sf.model != nil && strings.TrimSpace(*sf.model) != "" && !strings.EqualFold(strings.TrimSpace(*sf.model), "mock") {
		if sf.ggufPath != nil {
			*sf.ggufPath = strings.TrimSpace(*sf.model)
		}
		return nil
	}

	// 5. No model provided: prompt user if interactive, else fail closed.
	if isInteractive {
		if out != nil {
			fmt.Fprint(out, "Enter model path or alias (e.g. smollm2, path/to/model.gguf, or hf://...): ")
		}
		if in == nil {
			return errors.New("no model provided and stdin is unavailable")
		}
		reader := bufio.NewReader(in)
		line, err := reader.ReadString('\n')
		input := strings.TrimSpace(line)
		input = strings.Trim(input, `"'`)
		if err != nil && (err != io.EOF || input == "") {
			if err == io.EOF && input == "" {
				return errors.New("no model provided (end of input)")
			}
			return fmt.Errorf("read model path: %w", err)
		}
		if input == "" {
			return errors.New("no model provided; pass a model path/alias, --gguf, --base-url, or --mock")
		}
		if strings.EqualFold(input, "mock") {
			if sf.mock != nil {
				*sf.mock = true
			}
			if sf.model != nil {
				*sf.model = "mock"
			}
			return nil
		}
		if strings.HasPrefix(input, "http://") || strings.HasPrefix(input, "https://") {
			if sf.baseURL != nil {
				*sf.baseURL = input
			}
			return nil
		}
		if sf.ggufPath != nil {
			*sf.ggufPath = input
		}
		if sf.model != nil && (*sf.model == "mock" || *sf.model == "") {
			*sf.model = ""
		}
		return nil
	}

	return errors.New("no model provided; specify --gguf <path>, a model alias/path argument, --base-url <url>, or pass --mock for offline mock")
}
