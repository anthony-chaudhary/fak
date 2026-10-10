package main

import "strings"

// splitCommandLine splits a command line into words on unquoted spaces and tabs,
// without a shell; double quotes group a word and are removed.
func splitCommandLine(cmd string) []string {
	var args []string
	var current strings.Builder
	inQuotes := false
	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		if c == '"' {
			inQuotes = !inQuotes
			continue
		}
		if !inQuotes && (c == ' ' || c == '\t') {
			if current.Len() > 0 {
				args = append(args, current.String())
				current.Reset()
			}
			continue
		}
		current.WriteByte(c)
	}
	if current.Len() > 0 {
		args = append(args, current.String())
	}
	return args
}
