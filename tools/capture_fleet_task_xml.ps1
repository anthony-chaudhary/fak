<#
capture_fleet_task_xml.ps1 - export the fleet's live Windows Scheduled Tasks to
version-controlled, SCRUBBED task XML under tools/scheduled-tasks/ (#3323).

Why this exists: most fleet loops are rebuildable from a versioned
tools/register_*.ps1, but some are not -- they launch a script that lives outside
the repo, or they are a one-shot campaign loop with no installer at all. Those
tasks are invisible to the repo: reimage the host and the loop is simply gone,
with nothing left that even remembers it existed. An exported task XML is the
fallback source of truth for that residual -- it records the schedule, the
principal shape, and the exact command line the task ran, so the loop can be
rebuilt (or at minimum reconstructed) rather than lost.

What an XML capture does NOT do: it does not vendor the script the task launches.
For a task whose action points outside the repo, restoring the XML restores a task
pointing at a file a fresh host does not have. That residual is recorded per-task
in internal/taskvc/inventory.go, and it is why an installer is always preferred
over a capture.

SCRUB: a raw Export-ScheduledTask carries the host's identity -- the account SID,
COMPUTERNAME\user in <Author>, and absolute home paths. None of that belongs in a
public forever-history tree. Every value is read from the LIVE environment and
replaced with a placeholder, then the result is re-scanned and the write is
REFUSED if any of them survived. The scrub is mechanical on purpose: hand-editing
an export is how a SID reaches the public tree.

  %LOCALAPPDATA% / %APPDATA% / %USERPROFILE% / %COMPUTERNAME%  Windows expands these
      itself when the task runs, so they are lossless -- no substitution needed.
  %FLEET_TASK_USER_SID%  MUST be substituted before re-registering. Principal policy
      (S4U vs InteractiveToken) is #3322's concern, not this capture's.
  %GCP_PROJECT%  a cloud project id, redacted; substitute before re-registering.

REFUSE vs REDACT (#5402): host identity is REDACTED because a faithful placeholder
exists -- Windows expands %USERPROFILE% itself, so the export stays restorable. A
credential, a routable address, a real host name or an email has no faithful
placeholder: rewriting it to %TOKEN% would produce an export that silently cannot be
restored AND would hide that a secret was ever on the command line. Those classes
therefore REFUSE the write. A refusal is recoverable (the operator inspects the task
and decides); a corrupted capture is not, which is also why every added needle is
anchored to keep its false-positive cost to a refused capture rather than a silently
mangled one. The refusal names the needle CLASS and the task only -- never the
matched value, because a scanner that echoes what it matched writes the secret to the
terminal and into the fleet log, turning the safety check into the leak.

Usage:
  .\tools\capture_fleet_task_xml.ps1                       # refresh every committed capture
  .\tools\capture_fleet_task_xml.ps1 -TaskName FleetFoo    # add/refresh one task
  .\tools\capture_fleet_task_xml.ps1 -Check                # drift check only, writes nothing
  .\tools\capture_fleet_task_xml.ps1 -SelfTest             # assert the scrub/refuse logic, no host access

Restore (after substituting the non-expanding placeholders above):
  Register-ScheduledTask -TaskName FleetStrandedRecovery -Xml (Get-Content -Raw tools\scheduled-tasks\FleetStrandedRecovery.xml)
#>
[CmdletBinding()]
param(
  # Tasks to capture. Default: refresh whatever is already committed, so a bare
  # run never silently widens the captured set.
  [string[]]$TaskName = @(),
  [string]$OutDir = (Join-Path $PSScriptRoot 'scheduled-tasks'),
  # Re-export and compare against the committed capture without writing.
  [switch]$Check,
  # Exercise the scrub + refusal logic against synthesized fixtures and the
  # committed corpus. Touches no live task and writes nothing.
  [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'

# ---- the scrub map -------------------------------------------------------------
# Ordered longest-prefix-first: LOCALAPPDATA must be replaced before USERPROFILE,
# and COMPUTERNAME\user before a bare COMPUTERNAME, or the shorter match wins and
# leaves a half-scrubbed path behind.
$sid = ([System.Security.Principal.WindowsIdentity]::GetCurrent()).User.Value
$scrub = [ordered]@{}
$scrub[$env:LOCALAPPDATA]                     = '%LOCALAPPDATA%'
$scrub[$env:APPDATA]                          = '%APPDATA%'
$scrub[$env:USERPROFILE]                      = '%USERPROFILE%'
$scrub["$env:COMPUTERNAME\$env:USERNAME"]     = 'fak-fleet'
$scrub[$env:COMPUTERNAME]                     = '%COMPUTERNAME%'
$scrub[$sid]                                  = '%FLEET_TASK_USER_SID%'

# The bare account name is deliberately still NOT in the REPLACE map above: it can be
# a short, generic word, and a blind replace would corrupt unrelated text. That is not
# a guess. Measured against the 13 committed captures on the host this was written on,
# a case-insensitive SUBSTRING replace of its account name would have rewritten 42
# sites in an export set an independent audit had already cleared as clean -- because a
# generic account name is a substring of <UserId>, UseUnifiedSchedulingEngine, a task
# name, and the SID placeholder itself. The home-path prefixes and the <Author> pair
# form above are what actually carry the account name, and they carry it unambiguously.
#
# #5402 narrowed that exemption rather than overturning it: it now covers the REPLACE
# side only. What changed the calculus is the mechanism, not the risk -- under a needle
# that REFUSES instead of rewriting there is no blind replace, so a false positive
# costs a refused capture the operator reads and adjudicates, not a silently mangled
# public artifact. A standalone occurrence (in an argument, a share path, a log target)
# survived the old re-scan entirely, so the account name is now a word-anchored REFUSAL
# needle (Get-AccountNamePattern below). The anchoring is load-bearing and was chosen by
# measurement on the same corpus: \b..\b matches 0 sites, while the underscore-tolerant
# (?<![A-Za-z0-9])..(?![A-Za-z0-9]) form matches 14. \b it is.

# Secret-shaped values that are not host identity. Narrow by construction -- each
# anchors on the assignment that introduces it, so it cannot eat neighbouring text.
$redactions = @(
  @{ Pattern = '(?<=PROJECT=)[a-z0-9][-a-z0-9]{4,}'; Replacement = '%GCP_PROJECT%'; Label = 'cloud project id' }
)

# ---- the refusal registry (#5402) ----------------------------------------------
# Classes with no faithful placeholder: matching one REFUSES the write. Evaluated
# AFTER the replaces and redactions above, which is deliberate and load-bearing:
#   * placeholders the scrub introduces (%USERPROFILE%, fak-fleet, %GCP_PROJECT%)
#     are never themselves refused;
#   * an account name that only ever appeared inside a home path or the <Author>
#     pair is already gone, so it cannot raise a spurious refusal;
#   * conversely the PROJECT= redaction consumes the project id before the
#     credential-assignment needle can see it -- intended, it is redacted, not hidden.
# Order within the list is specific-before-general so the reported class is the most
# informative one: an email is reported as an email, not as the host inside it.
#
# Each entry: Pattern, Label (the class named in the refusal -- never a value), and an
# optional Allow regex tested against the named group 'v' if the pattern has one, else
# against the whole match. A match that satisfies Allow is exempt.
#
# The token shapes mirror internal/ctxmmu/mmu.go's secretPattern rather than inventing
# a second vocabulary; the JWT, google-api-key and github_pat_ forms are the #5402
# additions. NOT added: a bare "long high-entropy hex/base64 run" needle. A 40-char
# lowercase hex run is a git commit sha, and a campaign-loop task pinned to a sha is
# ordinary -- that needle would refuse legitimate captures. The assignment-anchored
# entry below covers the same ground without eating shas.
$hostAllow = '(?i)^(?:localhost|schemas\.microsoft\.com|www\.w3\.org|(?:[a-z0-9-]+\.)*example\.(?:com|net|org|edu|invalid|test|lab|localhost))$'
# Final labels that are file extensions, not TLDs. 'com' is absent on purpose: it is
# the most common TLD in the world and denying it would blind the needle to most real
# hosts. A legacy foo.com executable name is the accepted false positive.
$notATld = '(?i)\.(?:exe|dll|bat|cmd|ps1|psm1|psd1|sh|py|go|js|ts|json|jsonl|xml|txt|log|md|yml|yaml|toml|ini|cfg|conf|lnk|url|zip|tar|gz|csv|db|sqlite|tmp|bak|old|new|lock|pid|sys|msi|vbs)$'
$credWords = 'password|passwd|secret|token|api[_-]?key|access[_-]?key|client[_-]?secret|bearer'

$refusals = [System.Collections.ArrayList]@()
# A private key block. Zero plausible false positive.
[void]$refusals.Add(@{ Label = 'a private-key block'
  Pattern = '-----BEGIN [A-Z ]*PRIVATE KEY-----' })
# Vendor-prefixed key shapes. Each is a registered prefix plus a length floor, so the
# only way to false-positive is to write a literal that already looks like a key.
[void]$refusals.Add(@{ Label = 'an API-key-shaped token'
  Pattern = '(?i)(?<![a-z0-9])(?:sk-[a-z0-9][a-z0-9-]{15,}|AKIA[0-9A-Z]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[A-Za-z0-9_-]{30,})' })
# A JWT, matched as the whole three-segment structure rather than the bare eyJ header
# -- "eyJ" alone appears inside any base64 that happens to encode a JSON object.
[void]$refusals.Add(@{ Label = 'a JSON Web Token'
  Pattern = '(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}' })
# Authorization: Bearer <token>, the one credential shape with no separator. The 16
# char floor is what keeps the English word "bearer" from tripping it.
[void]$refusals.Add(@{ Label = 'a bearer credential'
  Pattern = '(?i)(?<![a-z0-9])bearer[ \t:=]+(?<v>[A-Za-z0-9._~+/=-]{16,})(?![A-Za-z0-9])'
  Allow   = '(?i)^(?:%[a-z0-9_]+%|\$env:[a-z0-9_]+|\$\{?[a-z0-9_]+\}?)$' })
# The substitute for an entropy needle: a credential keyword and the assignment that
# introduces it. Could wrongly eat a long non-secret assigned to a credential-named
# switch (e.g. -token-file <path>); a literal placeholder or $env: reference is exempt
# so the export's own %FLEET_TOKEN%-style indirection does not wedge the capture.
[void]$refusals.Add(@{ Label = 'a credential assignment'
  Pattern = "(?i)(?<![a-z0-9])(?:$credWords)\b[ \t]*[:=][ \t]*(?<v>[^\s<>&]{8,})"
  Allow   = '(?i)^(?:%[a-z0-9_]+%|\$env:[a-z0-9_]+|\$\{?[a-z0-9_]+\}?)$' })
[void]$refusals.Add(@{ Label = 'a credential assignment'
  Pattern = "(?i)(?<![a-z0-9])--?(?:$credWords)\b[ \t]+(?<v>[^\s<>&-][^\s<>&]{7,})"
  Allow   = '(?i)^(?:%[a-z0-9_]+%|\$env:[a-z0-9_]+|\$\{?[a-z0-9_]+\}?)$' })
# An email address. Listed before the host needle so the more specific class wins.
# Could wrongly eat an at-sign-joined identifier such as a package@version spec whose
# version happens to be dotted and alphabetic-suffixed.
[void]$refusals.Add(@{ Label = 'an email address'
  Pattern = '(?i)(?<![a-z0-9._%+-])[a-z0-9._%+-]{1,64}@[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\.[a-z]{2,24}(?![a-z0-9.-])'
  Allow   = '(?i)@(?:[a-z0-9-]+\.)*example\.(?:com|net|org|edu|invalid|test|lab)$' })
# A routable IPv4 literal. The (?<![0-9A-Za-z.]) / (?![0-9A-Za-z.]) guards keep it
# from firing inside a longer dotted-numeric or alphanumeric run: a Windows build
# number (10.0.19041.1234) and a v-prefixed version (v1.2.3.4) both miss. It CAN
# still wrongly eat a BARE four-part version such as 1.2.3.4 -- accepted, because
# that costs a refusal the operator can read rather than a mangled capture.
[void]$refusals.Add(@{ Label = 'a routable IPv4 literal'
  Pattern = '(?<![0-9A-Za-z.])(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(?![0-9A-Za-z.])'
  Allow   = '^(?:0\.0\.0\.0|255\.255\.255\.255|127\.|169\.254\.|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)' })
# A dotted host name outside the placeholder vocabulary. Lowercase-only by design: a
# PascalCase dotted token is a .NET type or a file name (System.IO.FileInfo), never a
# host as an export renders it. The trailing (?![a-z0-9-]|\.[a-z0-9]) forces the match
# to consume the WHOLE dotted run, so a multi-dot file name cannot be refused via its
# prefix -- without it, fleet.tick.ps1 refuses as the "host" fleet.tick, which is the
# exact over-redaction this needle must not commit. It could still wrongly eat an
# all-lowercase reversed-domain identifier: a Java package name is deliberately shaped
# like an FQDN and is not distinguishable from one. Residual: a two-label domain whose
# final label is a real file extension (foo.zip) is exempted by $notATld, not refused.
[void]$refusals.Add(@{ Label = 'a host name outside the placeholder vocabulary'
  Pattern = '\b[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\.[a-z]{2,24}(?![a-z0-9-]|\.[a-z0-9])'
  Allow   = "$hostAllow|$notATld" })

# The bare account name -- see the exemption note above. Not in $refusals because the
# needle is a function of the account being scanned for, which lets -SelfTest witness
# it with a synthesized name on a host whose own account is a placeholder. Returns
# $null (needle disabled) for a placeholder/system account name -- the same set
# tools/check_secret_shapes.py exempts -- or for one too short to be distinguishable
# from noise, because refusing on a 1-2 character token would wedge every capture.
$placeholderAccounts = @('user','you','runner','public','default','all','username',
                         'administrator','guest','youruser','name','someone','system')
function Get-AccountNamePattern([string]$Account) {
  if ([string]::IsNullOrWhiteSpace($Account)) { return $null }
  if ($Account.Length -lt 3) { return $null }
  if ($placeholderAccounts -contains $Account.ToLowerInvariant()) { return $null }
  return '(?i)\b' + [regex]::Escape($Account) + '\b'
}

# Pure text transform: replaces, redactions, encoding + newline normalisation. No
# host access, so -SelfTest can drive the real ordering against a synthetic export.
function Invoke-TaskXmlScrub([string]$Text) {
  foreach ($k in $scrub.Keys) {
    if ([string]::IsNullOrEmpty($k)) { continue }
    $Text = $Text.Replace($k, $scrub[$k])
  }
  foreach ($r in $redactions) {
    $Text = [regex]::Replace($Text, $r.Pattern, $r.Replacement)
  }
  # Export-ScheduledTask hands back a UTF-16 declaration; we store UTF-8 bytes, so
  # the declaration has to agree or a downstream XML reader trips on the mismatch.
  $Text = $Text -replace '(?i)encoding="UTF-16"', 'encoding="UTF-8"'
  return ($Text -replace "`r`n", "`n").TrimEnd() + "`n"
}

# Pure scan: returns the CLASS of the first surviving needle, or $null when clean.
# It returns a class name and never a matched value -- the one regression that would
# make this hardening actively harmful is a scanner that echoes the secret it found.
function Get-ScrubRefusalReason([string]$Text, [string]$Account = $env:USERNAME) {
  # Literal host identity. Ordinal case-insensitive rather than -like, because a
  # -like needle carrying a [ or * would be read as a wildcard and silently miss.
  foreach ($needle in @($sid, $env:COMPUTERNAME, $env:USERPROFILE)) {
    if ([string]::IsNullOrEmpty($needle)) { continue }
    if ($Text.IndexOf($needle, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
      return 'a host identity value'
    }
  }
  $acct = Get-AccountNamePattern $Account
  if ($acct -and [regex]::IsMatch($Text, $acct)) { return 'the operator account name' }
  foreach ($r in $refusals) {
    foreach ($m in [regex]::Matches($Text, $r.Pattern)) {
      if ($r.Contains('Allow') -and $r.Allow) {
        $v = if ($m.Groups['v'].Success) { $m.Groups['v'].Value } else { $m.Value }
        $v = $v.Trim([char]34, [char]39)
        if ($v -match $r.Allow) { continue }
      }
      return $r.Label
    }
  }
  foreach ($r in $redactions) {
    if ([regex]::IsMatch($Text, $r.Pattern)) { return "a $($r.Label)" }
  }
  return $null
}

# The refusal text. Names the task and the class; deliberately carries no value.
function Get-ScrubRefusalMessage([string]$Name, [string]$Reason) {
  return "SCRUB_INCOMPLETE: '$Name' still contains $Reason after scrubbing; refusing to write (the matched value is deliberately not echoed)"
}

function Get-ScrubbedTaskXml([string]$Name) {
  $xml = Export-ScheduledTask -TaskName $Name
  if (-not $xml) { throw "Export-ScheduledTask returned nothing for '$Name'" }
  $text = Invoke-TaskXmlScrub ($xml -join "`n")

  # Fail closed: if any needle survived, refuse rather than write it. This is the
  # check that keeps a SID -- and now a credential, a routable address, a real host
  # name, an email address and a bare account name -- out of the forever history.
  $reason = Get-ScrubRefusalReason $text
  if ($reason) { throw (Get-ScrubRefusalMessage $Name $reason) }
  return $text
}

# ---- SelfTest (#5402) ----------------------------------------------------------
# Same shape as tools/reboot_advisor.ps1 -SelfTest: assert the decision logic before
# trusting it against the live host. Every fixture value is SYNTHESIZED here at run
# time or read from the environment by name -- no credential, address, host name or
# resolved operator value is stored in this file. For each needle it asserts BOTH
# halves: the shape is refused, and a lookalike-but-legitimate neighbour is not.
if ($SelfTest) {
  $fails = 0
  function Check([string]$name, [bool]$cond) {
    if ($cond) { Write-Output "  PASS  $name" }
    else       { Write-Output "  FAIL  $name"; $script:fails++ }
  }
  function New-FixtureXml([string]$ArgText, [string]$Desc = 'fleet loop fixture') {
    return @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.3" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>$Desc</Description><URI>\Fixture</URI></RegistrationInfo>
  <Principals><Principal id="Author"><UserId>$sid</UserId><LogonType>S4U</LogonType></Principal></Principals>
  <Actions Context="Author"><Exec><Command>powershell.exe</Command>
    <Arguments>-NoProfile -ExecutionPolicy Bypass -File "%LOCALAPPDATA%\Fleet\loop.ps1" $ArgText</Arguments>
  </Exec></Actions>
</Task>
"@
  }
  # Reason for a fixture, through the REAL pipeline (replaces + redactions, then scan).
  function Reason([string]$ArgText, [string]$Desc = 'fleet loop fixture') {
    return (Get-ScrubRefusalReason (Invoke-TaskXmlScrub (New-FixtureXml $ArgText $Desc)))
  }

  # --- prove the negative first: the carrier itself must be clean, or every REFUSE
  # --- assertion below is vacuous. Note the carrier already contains the XML
  # --- namespace host, powershell.exe, a .ps1 path and a %LOCALAPPDATA% placeholder.
  Check "bare fixture is clean (not vacuous)"        ($null -eq (Reason '-Once'))

  # --- credential shapes: REFUSE ---
  $pem  = ('-' * 5) + 'BEGIN ' + 'RSA' + ' PRIVATE KEY' + ('-' * 5)
  $skk  = 'sk' + '-' + ('a1b2c3d4' * 3)
  $ghp  = 'ghp' + '_' + ('A1b2C3d4' * 3)
  $xox  = 'xox' + 'b' + '-' + ('0a1b2c3d' * 2)
  $aiz  = 'AIza' + ('Zz9wYx8v' * 4)
  $akia = 'AKIA' + ('ABCD1234' * 2)
  $jwt  = 'eyJ' + ('abcdefgh' * 2) + '.' + ('ijklmnop' * 2) + '.' + ('qrstuvwx' * 2)
  $btok = ('t0k3nT0k3n' * 2)
  Check "PEM private key REFUSED"                    ((Reason "-Key $pem")  -eq 'a private-key block')
  Check "sk- key REFUSED"                            ((Reason "-K $skk")    -eq 'an API-key-shaped token')
  Check "github pat REFUSED"                         ((Reason "-K $ghp")    -eq 'an API-key-shaped token')
  Check "slack token REFUSED"                        ((Reason "-K $xox")    -eq 'an API-key-shaped token')
  Check "google api key REFUSED"                     ((Reason "-K $aiz")    -eq 'an API-key-shaped token')
  Check "aws access key id REFUSED"                  ((Reason "-K $akia")   -eq 'an API-key-shaped token')
  Check "JWT REFUSED"                                ((Reason "-K $jwt")    -eq 'a JSON Web Token')
  Check "bearer credential REFUSED"                  ((Reason "-H Authorization: Bearer $btok") -eq 'a bearer credential')
  Check "TOKEN= assignment REFUSED"                  ((Reason ('TOKEN=' + ('9f8e7d6c' * 2))) -eq 'a credential assignment')
  Check "--token flag REFUSED"                       ((Reason ('--token ' + ('9f8e7d6c' * 2))) -eq 'a credential assignment')
  Check "/password: assignment REFUSED"              ((Reason ('/password:' + ('Zq7x' * 3))) -eq 'a credential assignment')

  # --- credential lookalikes that must stay INTACT ---
  Check "placeholder token value allowed"            ($null -eq (Reason 'TOKEN=%FLEET_TOKEN%'))
  Check "env-ref token value allowed"                ($null -eq (Reason '--token $env:FLEET_TOKEN'))
  Check "--token-budget flag allowed"                ($null -eq (Reason '--token-budget 400000'))
  Check "the word tokenizer allowed"                 ($null -eq (Reason '-Mode tokenizer-warmup'))
  Check "40-char git sha allowed (no entropy needle)" ($null -eq (Reason ('--pin ' + ('0123456789abcdef' * 3).Substring(0, 40))))
  Check "short base64-looking flag allowed"          ($null -eq (Reason '-Profile QUJDRA'))
  Check "eyJ inside a word allowed (needs 3 segments)" ($null -eq (Reason '-Tag eyJqZmtsc2Rmag'))

  # --- IPv4: REFUSE routable, ALLOW loopback / link-local / RFC 5737 TEST-NET ---
  Check "routable IPv4 REFUSED"                      ((Reason ('-Host ' + ((10,20,30,40) -join '.'))) -eq 'a routable IPv4 literal')
  Check "loopback allowed"                           ($null -eq (Reason ('-Host ' + ((127,0,0,1) -join '.'))))
  Check "link-local allowed"                         ($null -eq (Reason ('-Host ' + ((169,254,10,20) -join '.'))))
  Check "TEST-NET-1 allowed"                         ($null -eq (Reason ('-Host ' + ((192,0,2,5) -join '.'))))
  Check "TEST-NET-2 allowed"                         ($null -eq (Reason ('-Host ' + ((198,51,100,5) -join '.'))))
  Check "TEST-NET-3 allowed"                         ($null -eq (Reason ('-Host ' + ((203,0,113,7) -join '.'))))
  Check "windows build number allowed"               ($null -eq (Reason ('-Build ' + ((10,0,19041,1234) -join '.'))))
  Check "v-prefixed version allowed"                 ($null -eq (Reason ('-Ver v' + ((1,2,3,4) -join '.'))))

  # --- host names: REFUSE real-shaped, ALLOW the placeholder vocabulary + filenames ---
  $badHost = ('svc', 'internal', 'invalid') -join '.'
  Check "internal FQDN REFUSED"                      ((Reason "-Host $badHost") -eq 'a host name outside the placeholder vocabulary')
  Check "two-label host REFUSED"                     ((Reason ('-Host ' + (('acmecorp', 'net') -join '.'))) -eq 'a host name outside the placeholder vocabulary')
  Check "example.com allowed"                        ($null -eq (Reason ('-Host ' + (('gpu-server', 'example', 'com') -join '.'))))
  Check "example.invalid allowed"                    ($null -eq (Reason ('-Host ' + (('a', 'example', 'invalid') -join '.'))))
  Check "localhost allowed"                          ($null -eq (Reason '-Host localhost'))
  Check "dotted filename allowed"                    ($null -eq (Reason '-File fleet.tick.ps1 -Also bench.node.json'))
  Check "exe name allowed"                           ($null -eq (Reason '-Exec conhost.exe -Then gcp-fleet-janitor.sh'))
  Check "PascalCase dotted type allowed"             ($null -eq (Reason '-Type System.IO.FileInfo'))
  Check "XML namespace host allowed"                 ($null -eq (Reason '-Once'))

  # --- email: REFUSE real-shaped, ALLOW the documentation domains ---
  Check "email REFUSED"                              ((Reason ('-Notify ' + 'ops@' + $badHost)) -eq 'an email address')
  Check "email beats host (specific class first)"    ((Reason ('-Notify ' + 'ops@' + $badHost)) -ne 'a host name outside the placeholder vocabulary')
  Check "example.com email allowed"                  ($null -eq (Reason ('-Notify ' + 'owner@' + (('example', 'com') -join '.'))))

  # --- the bare account name: the exemption this ticket had to re-decide ---
  # Driven with a SYNTHESIZED account name so the needle is witnessed on any host,
  # including one (like a CI runner) whose own account name is a placeholder.
  $acct = 'jonquil'
  function AcctReason([string]$Desc) {
    return (Get-ScrubRefusalReason (Invoke-TaskXmlScrub (New-FixtureXml '-Once' $Desc)) $acct)
  }
  Check "standalone account name REFUSED"            ((AcctReason "runs as $acct nightly") -eq 'the operator account name')
  Check "account name in a path segment REFUSED"     ((AcctReason "logs to D:\seats\$acct\out.log") -eq 'the operator account name')
  Check "account name as a substring allowed"        ($null -eq (AcctReason "runs as x${acct}y nightly"))
  Check "account name inside a word allowed"         ($null -eq (AcctReason "runs the ${acct}s loop"))
  Check "underscore-joined account name allowed"     ($null -eq (AcctReason "runs the ${acct}_loop stage"))
  Check "placeholder account name disables needle"   ($null -eq (Get-AccountNamePattern 'USER'))
  Check "short account name disables needle"         ($null -eq (Get-AccountNamePattern 'ab'))
  Check "real account name enables needle"           ($null -ne (Get-AccountNamePattern $acct))
  # Live-host forms: the pair and the home path must survive the REPLACE side.
  Check "COMPUTERNAME-account pair allowed"          ($null -eq (Reason '-Once' "runs as $env:COMPUTERNAME\$env:USERNAME nightly"))
  Check "home path allowed"                          ($null -eq (Reason ('-File "' + $env:USERPROFILE + '\loop.ps1"')))

  # --- the refusal must name the class and NOT the value ---
  $leakArg = '--token ' + ('9f8e7d6c' * 2)
  $msg = Get-ScrubRefusalMessage 'FixtureTask' (Reason $leakArg)
  Check "refusal is SCRUB_INCOMPLETE"                ($msg -like 'SCRUB_INCOMPLETE:*')
  Check "refusal names the class"                    ($msg -like '*credential assignment*')
  Check "refusal names the task"                     ($msg -like '*FixtureTask*')
  Check "refusal does NOT echo the matched value"    (-not $msg.Contains(('9f8e7d6c' * 2)))
  Check "refusal does NOT echo the account name"     ($env:USERNAME -and -not $msg.Contains($env:USERNAME))

  # --- ordering: the scrub runs BEFORE the scan, so a redacted value cannot refuse ---
  Check "PROJECT= id is redacted, not refused"       ($null -eq (Reason '-Env PROJECT=my-fleet-project-1234'))
  Check "SID is replaced, not refused"               ($null -eq (Reason '-Once'))

  # --- no false positive on the audited corpus already on trunk ---
  $corpus = @(Get-ChildItem -Path $OutDir -Filter '*.xml' -ErrorAction SilentlyContinue)
  Check "corpus is non-empty (not vacuous)"          ($corpus.Count -gt 0)
  foreach ($f in $corpus) {
    $why = Get-ScrubRefusalReason ([System.IO.File]::ReadAllText($f.FullName))
    Check ("committed capture clean: " + $f.BaseName) ($null -eq $why)
  }

  if ($fails -gt 0) { Write-Output "SELFTEST FAILED ($fails)"; exit 1 }
  Write-Output "SELFTEST OK"; exit 0
}

if (-not (Test-Path $OutDir)) {
  if ($Check) { Write-Error "no capture dir at $OutDir"; exit 1 }
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

# Default set = the already-committed captures. Explicit -TaskName adds to it, so
# growing the captured set is always a deliberate act.
$names = @($TaskName)
if (-not $names -or $names.Count -eq 0) {
  $names = @(Get-ChildItem -Path $OutDir -Filter '*.xml' -ErrorAction SilentlyContinue |
             ForEach-Object { $_.BaseName })
}
if (-not $names -or $names.Count -eq 0) {
  Write-Output "no tasks to capture (empty $OutDir and no -TaskName given)"
  exit 0
}

$drift = 0
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
foreach ($n in ($names | Sort-Object -Unique)) {
  $dest = Join-Path $OutDir "$n.xml"
  $live = Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue
  if (-not $live) {
    # A committed capture whose task is gone is a real signal, not an error: the
    # loop was retired and its row in internal/taskvc/inventory.go should go too.
    Write-Output "MISSING  $n (no live task; retire its capture + inventory row?)"
    $drift++
    continue
  }

  $text = Get-ScrubbedTaskXml -Name $n
  $prior = if (Test-Path $dest) { [System.IO.File]::ReadAllText($dest) } else { $null }

  if ($Check) {
    if ($null -eq $prior)      { Write-Output "UNCAPTURED  $n"; $drift++ }
    elseif ($prior -ne $text)  { Write-Output "DRIFT       $n (live task differs from the committed capture)"; $drift++ }
    else                       { Write-Output "ok          $n" }
    continue
  }

  if ($prior -eq $text) { Write-Output "unchanged   $n"; continue }
  [System.IO.File]::WriteAllText($dest, $text, $utf8NoBom)
  Write-Output $(if ($null -eq $prior) { "captured    $n" } else { "updated     $n" })
}

if ($Check -and $drift -gt 0) {
  Write-Error "$drift captured fleet task(s) drifted from the live host"
  exit 1
}
exit 0
