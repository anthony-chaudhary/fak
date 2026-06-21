<#
notify.ps1 — surface a relevant fleet action to the Windows notification center.

Sends a toast (lands in Action Center) and ALSO appends to a durable notify log
so there is a record even when no one is at the screen / the toast API is
unavailable (e.g. running headless under Task Scheduler).

  .\notify.ps1 -Title "Resumed session" -Message "f19dcbe1 (gem5 / C--work-fleet)"
  .\notify.ps1 -Title "..." -Message "..." -Level warn   # info|warn|err (affects log tag only)
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory)] [string]$Title,
  [Parameter(Mandatory)] [string]$Message,
  [ValidateSet('info','warn','err')] [string]$Level = 'info',
  [string]$AppId = 'Fleet.SessionWatchdog',
  [string]$LogDir = '',
  [string]$Launch = '',
  [string]$Key = '',
  [int]$MinIntervalMinutes = 0,
  [switch]$NoToast
)
$ErrorActionPreference = 'SilentlyContinue'
$stateRoot = if ($env:FLEET_STATE_DIR) {
  $env:FLEET_STATE_DIR
} elseif ($env:LOCALAPPDATA) {
  Join-Path $env:LOCALAPPDATA 'Fleet'
} else {
  Join-Path ([System.IO.Path]::GetTempPath()) 'Fleet'
}
if (-not $LogDir) { $LogDir = Join-Path $stateRoot 'watchdog' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

function Escape-Xml($s) {
  [System.Security.SecurityElement]::Escape([string]$s)
}

function Launch-Uri($value) {
  $raw = [string]$value
  if (-not $raw) {
    $raw = Join-Path (Join-Path $stateRoot 'registry') 'STATUS.txt'
  }
  if ($raw -match '^[A-Za-z][A-Za-z0-9+.-]+:') {
    return $raw
  }
  try {
    return ([System.Uri]::new([System.IO.Path]::GetFullPath($raw))).AbsoluteUri
  } catch {
    return $raw
  }
}

# Suppress keyed repeats before writing notifications.log; the log should be an
# operator signal, not another spam source.
if ($Key -and $MinIntervalMinutes -gt 0) {
  $throttlePath = Join-Path $LogDir '_notification_throttle.json'
  $state = @{}
  if (Test-Path $throttlePath) {
    try {
      (Get-Content $throttlePath -Raw | ConvertFrom-Json).PSObject.Properties |
        ForEach-Object { $state[$_.Name] = [string]$_.Value }
    } catch {}
  }
  $now = [DateTimeOffset]::UtcNow
  if ($state.ContainsKey($Key)) {
    try {
      $last = [DateTimeOffset]::Parse($state[$Key])
      if (($now - $last).TotalMinutes -lt $MinIntervalMinutes) {
        return
      }
    } catch {}
  }
  $state[$Key] = $now.ToString('o')
  try { ($state | ConvertTo-Json) | Set-Content -Path $throttlePath -Encoding UTF8 } catch {}
}

# durable record of every notification (so "relevant actions" are auditable)
$line = "{0}  [{1}]  {2} :: {3}" -f ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')), $Level.ToUpper(), $Title, $Message
Add-Content -Path (Join-Path $LogDir 'notifications.log') -Value $line

# register the AppUserModelId once so toasts show our name and persist in Action Center
$reg = "HKCU:\Software\Classes\AppUserModelId\$AppId"
if (-not (Test-Path $reg)) {
  New-Item -Path $reg -Force | Out-Null
  Set-ItemProperty -Path $reg -Name DisplayName -Value 'Fleet Session Watchdog'
  Set-ItemProperty -Path $reg -Name ShowInSettings -Value 0 -Type DWord
}

$shown = $false
try {
  [void][Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime]
  [void][Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime]
  $launchUri = Launch-Uri $Launch
  $launchXml = Escape-Xml $launchUri
  $xml = @"
<toast activationType="protocol" launch="$launchXml"><visual><binding template="ToastGeneric">
<text>$(Escape-Xml $Title)</text>
<text>$(Escape-Xml $Message)</text>
</binding></visual><actions><action content="Open status" activationType="protocol" arguments="$launchXml"/></actions></toast>
"@
  $doc = New-Object Windows.Data.Xml.Dom.XmlDocument
  $doc.LoadXml($xml)
  $toast = [Windows.UI.Notifications.ToastNotification]::new($doc)
  if (-not $NoToast) {
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($AppId).Show($toast)
    $shown = $true
  }
} catch { }

if (-not $shown -and -not $NoToast) {
  # fallback: tray balloon (legacy, but still pops on the desktop)
  try {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    $ni = New-Object System.Windows.Forms.NotifyIcon
    $ni.Icon = [System.Drawing.SystemIcons]::Information
    $ni.Visible = $true
    $ni.ShowBalloonTip(8000, $Title, $Message, 'Info')
    Start-Sleep -Milliseconds 250
    $ni.Dispose()
  } catch { }
}
