# Captured output

```text
$ examples/remote-vm-guard/run.sh
[vm-guard] BLOCKED - the cloud-credential-theft SSRF class (every reachable IMDS address/name):
  PASS AWS/GCP/Azure IMDS via WebFetch                    -> BLOCK
  PASS GCP metadata by NAME (DNS, not IP)                 -> BLOCK
  PASS AWS ECS task-metadata address                      -> BLOCK
  PASS bare metadata IP straight to curl                  -> BLOCK

[vm-guard] ALLOWED - ordinary destinations a real session needs (NOT egress-blocked):
  PASS the Anthropic API (the provider)                   -> ALLOW
  PASS a public git clone over https                      -> ALLOW
  PASS a public host                                      -> ALLOW

[vm-guard] all witnesses passed - the metadata/link-local SSRF class is refused by structure.
```
