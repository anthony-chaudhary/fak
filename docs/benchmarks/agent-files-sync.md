# Agent instruction files stay byte-identical

When the same agent instructions are published under several file names, the copies must
match byte for byte; one differing byte, including a CRLF, is drift. `agentFileSyncDrift`
in `cmd/fak/agent_file_sync.go` compares every mirror with the first path and lists the
ones that differ. A mirror that cannot be read is an error, not silent agreement.

Direct port of ai-dynamo/aiperf `tools/check_agent_files_sync.py:45`
(`ab610add4e8876fca223197596c728e17365f381`, Apache-2.0). Witness:
`go test ./cmd/fak -run TestAgentFileSync -count=1`.
