## Current state

A live OpenCode coding task completed its read, then rejected an admitted edit because the proposal used file_path, old_string, and new_string while the advertised client schema requires filePath, oldString, and newString. The existing adapter covers only read path spelling.

## Parent context

GitHub issue: https://github.com/anthony-chaudhary/fak/issues/12855

Related to #12307 and the read-only repair in #12824.

## Scope

Bind the standard read, edit, and write tool family to the exact client-advertised name and argument spellings in the gateway proposal path.

## Working spine

1. Preserve an exactly advertised source tool name.
2. Map read_file, edit_file, or write_file only when that source is unadvertised and exactly one corresponding lowercase standard tool is advertised.
3. Reject differing aliases before adjudication.
4. Project path and edit-string keys to the sole spelling declared by the selected schema.
5. Preserve all other arguments and existing policy verdicts.

## Gold-plating boundary

Do not fuzzy-match names, map custom tools, change argument values, expand the capability floor, or change HTTP routes and serving configuration.

## Likely files

- internal/gateway/adjudicate_proposed.go
- internal/gateway/opencode_tool_schema_test.go
- docs/tickets/gateway/TICKET-12855-opencode-tool-schema-adapter.md

## Acceptance gate

- Camel and snake client schemas receive their declared path and edit-string keys.
- Exact advertised names take precedence over file-suffix mapping.
- File-suffix mapping requires one unambiguous advertised standard target.
- Unknown optional arguments survive.
- Differing aliases are denied before execution; identical duplicates may normalize.
- Denied and custom calls do not gain authority.

## Witness

go test ./internal/gateway -run 'TestOpenCode.*Advertised.*Schema' -count=1

## Done condition

The one-touch OpenCode loop can execute standard read, edit, and write calls in the schema it advertised without weakening the gateway floor.
