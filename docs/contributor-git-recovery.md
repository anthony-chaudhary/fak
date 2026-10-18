# Contributor Git access and publication recovery

**Audience:** contributors diagnosing authentication or transport failures while
publishing an already-authorized change. This guide adds diagnostic checks; it
does not replace the [contributor landing gates](../CONTRIBUTING.md) or authorize
changing permissions, bypassing a refusal, or publishing to another repository.

**Next action:** capture the failing command, selected Git executable/version,
canonical destination, and redacted error. Read earlier recovery receipts before
resetting authentication or changing configuration.

## Identify which boundary failed

- **Account access:** a user's repository push permission establishes that
  account's access. It does not establish a connector's installation scope or
  the permissions of its token.
- **Integration access:** inspect the actual installed connector's tool schema,
  repository selection, and read-only access result. A selected fork is not its
  upstream. General documentation for a read-only product surface does not prove
  that every installed integration is read-only, and an exposed write tool does
  not prove that its token can write the target repository.
- **GitHub API refusal:** inspect the response body and rate-limit/permission
  headers. A `403` can reflect different causes; do not classify it from the
  status alone. GitHub documents insufficient token permissions for its
  integration-access errors in [REST API troubleshooting][api].
- **Transport refusal:** an HTTP proxy rejecting `CONNECT` before the GitHub
  connection is established is a network-policy result, not proof that GitHub
  rejected the credential. Preserve the proxy, canonical remote, authentication
  mechanism, and TLS verification. Use the environment's formal approval route
  if it supports the exact authorized operation; respect a denial. Do not
  switch hosts or protocols to evade the restriction.

## Make CLI authentication durable before retrying

Before a user-approved CLI login, select an isolated writable configuration
directory outside the repository. GitHub CLI supports `GH_CONFIG_DIR`; its
default location can be unwritable on managed hosts. Confirm writability before
starting the device flow. Browser acceptance alone does not prove the CLI saved
usable credentials. See [CLI environment configuration][environment].

Use restrictive directory/file permissions appropriate to the platform and the
approved credential store. GitHub CLI may fall back to plaintext storage when
the system store is unavailable, so check the selected storage method without
printing its contents. Do not place credentials or device codes in source,
remote URLs, command-line arguments, logs, patches, or backup bundles. Verify
authentication status and use the approved Git credential helper rather than
copying a token into a push command. See [CLI login storage behavior][login].

Confirm the selected Git build actually supports HTTPS. A local-only build made
without curl support can inspect objects but cannot replace an HTTPS-capable
Git executable for network operations. A version string alone is insufficient.

## Read back uncertain operations

After a timeout or ambiguous mutation result, perform read-only inspection before
retrying. Check the target branch and expected object identity; if the change
already landed, do not duplicate it. An inaccessible readback leaves the outcome
unknown. Retry only after the state and authorization for the same operation
are established, through the sanctioned workflow.

Keep local and remote evidence separate:

1. Record the intended local commit, tree, parent(s), and explicit changed paths.
2. Query the actual canonical remote branch, rather than trusting a stale local
   remote-tracking ref. Confirm the published commit identity and inspect its
   tree and parent(s) through the authorized read/fetch route. If peers advanced
   the tip, prove the intended commit is reachable; do not overwrite their work.
3. Report local commit creation, remote publication, software test execution,
   and physical-device qualification separately. A successful push does not
   prove CI or hardware behavior; unavailable or never-run gates are not passes.

For proof-depth requirements, return to [developer tooling](dev-tooling.md#match-proof-to-activation-depth).
Deployment-specific credentials, account selections, recovery transcripts, and
private worker-control details stay behind the [private boundary](dev-process-private-boundary.md).

[api]: https://docs.github.com/en/rest/using-the-rest-api/troubleshooting-the-rest-api
[environment]: https://cli.github.com/manual/gh_help_environment
[login]: https://cli.github.com/manual/gh_auth_login
