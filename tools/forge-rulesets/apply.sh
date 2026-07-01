#!/usr/bin/env bash
# apply.sh — push the fak forge-side trunk-law mirror to a GitHub or GitLab repo.
#
# These rulesets are the NON-BYPASSABLE backstop to fak's client-side hook floor
# (tools/githooks/). A client can defeat the hooks with --no-verify, a core.hooksPath
# override, or shell-laundering; a forge ruleset evaluates the ref update server-side,
# after laundering collapses into a concrete <old-sha> <new-sha> <refname>, so no client
# flag can disarm it. See docs/fak/deployment-guide.md "Forge-side enforcement (required)".
#
# Usage:
#   GitHub:  ./apply.sh github  <owner>/<repo> [current|dev|main] (needs `gh auth login`)
#   GitLab:  ./apply.sh gitlab  <project-id-or-url-encoded>       (needs GITLAB_TOKEN + glab/curl)
#
# Both commands are idempotent-ish: re-applying replaces the named ruleset / overwrites
# the push rules. Edit the selected GitHub ruleset's status-check contexts before applying.

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
forge="${1:-}"; target="${2:-}"; github_role="${3:-current}"

usage() { sed -n '2,18p' "$0" >&2; exit 2; }
[ -z "$forge" ] || [ -z "$target" ] && usage

case "$forge" in
  github)
    case "$github_role" in
      current|single|trunk) ruleset="github-ruleset.json" ;;
      dev|development) ruleset="github-dev-ruleset.json" ;;
      main|release|front-door) ruleset="github-main-ruleset.json" ;;
      *.json) ruleset="$github_role" ;;
      *)
        echo "unknown GitHub ruleset role: $github_role (want current|dev|main or a .json filename)" >&2
        exit 2
        ;;
    esac
    case "$ruleset" in
      */*|*\\*) echo "ruleset must be a file in ${here}: $ruleset" >&2; exit 2 ;;
    esac
    [ -f "${here}/${ruleset}" ] || { echo "ruleset not found: ${here}/${ruleset}" >&2; exit 2; }
    # GitHub Repository Rulesets REST API (server-side, cannot be bypassed by client flags):
    # https://docs.github.com/en/rest/repos/rules
    gh api -X POST "repos/${target}/rulesets" \
      --input "${here}/${ruleset}" \
      -H "Accept: application/vnd.github+json"
    echo "applied ${ruleset} to github.com/${target}" >&2
    ;;
  gitlab)
    # GitLab Project Push Rules API (server-side commit-message + secret validation):
    # https://docs.gitlab.com/api/projects/#edit-project-push-rule
    : "${GITLAB_TOKEN:?set GITLAB_TOKEN to a token with api scope}"
    base="${GITLAB_BASE_URL:-https://gitlab.com/api/v4}"
    rx="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["commit_message_regex"])' "${here}/gitlab-push-rules.json")"
    curl -fsS --request PUT "${base}/projects/${target}/push_rule" \
      --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
      --data-urlencode "commit_message_regex=${rx}" \
      --data "prevent_secrets=true" \
      --data "deny_delete_tag=true"
    echo "" >&2
    echo "applied fak push rules to gitlab project ${target}" >&2
    echo "NOTE: no-force-push + linear-history are GitLab PROTECTED-BRANCH settings, not push rules — set them on the 'main' protected branch separately." >&2
    ;;
  *) usage ;;
esac

# ---------------------------------------------------------------------------
# Terraform stub (GitHub) — keep the ruleset in IaC so it cannot drift silently.
# Save as forge-rulesets.tf, `terraform init && terraform apply`. The JSON above is
# the source of truth for the predicate set; this mirrors the current no-cutover
# template declaratively. For branch-regime cutover, create one resource for
# github-dev-ruleset.json and one for github-main-ruleset.json.
#
#   resource "github_repository_ruleset" "fak_trunk_laws" {
#     name        = "fak-trunk-laws"
#     repository  = "REPO"
#     target      = "branch"
#     enforcement = "active"
#     conditions { ref_name { include = ["refs/heads/main"], exclude = [] } }
#     rules {
#       deletion                = true
#       non_fast_forward        = true
#       required_linear_history = true
#       required_signatures     = true
#       required_status_checks {
#         strict_required_status_checks_policy = true
#         required_check { context = "ci" }
#       }
#     }
#   }
#
# Terraform stub (GitLab):
#   resource "gitlab_project_push_rules" "fak" {
#     project               = "PROJECT_ID"
#     commit_message_regex  = file("${path.module}/gitlab-push-rules.json")  # extract the field
#     prevent_secrets       = true
#     deny_delete_tag       = true
#   }
# ---------------------------------------------------------------------------
