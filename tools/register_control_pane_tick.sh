#!/bin/sh
# Install/remove/status the recurring fleet control pane tick on POSIX hosts.
#
# Default install path:
#   1. user systemd timer, when systemctl --user is usable
#   2. a marked crontab block otherwise

set -eu

ACTION=install
TASK_NAME=FleetControlPaneTick
PYTHON=python
PANE=
INTERVAL_MIN=5
LIVE_RESUME=0
JSON_OUT=0
# MODE selects what the recurring tick runs:
#   control (default) — `fleet_control_pane.py tick`, the 5-min keep-alive
#   garden            — the default-on gardening bundle, ledger-wrapped + admit-gated
MODE=control

if [ "${1:-}" = "install" ] || [ "${1:-}" = "remove" ] || [ "${1:-}" = "status" ]; then
  ACTION=$1
  shift
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --action)
      ACTION=$2
      shift 2
      ;;
    --task-name)
      TASK_NAME=$2
      shift 2
      ;;
    --python)
      PYTHON=$2
      shift 2
      ;;
    --pane)
      PANE=$2
      shift 2
      ;;
    --interval-min)
      INTERVAL_MIN=$2
      shift 2
      ;;
    --live-resume)
      LIVE_RESUME=1
      shift
      ;;
    --mode)
      MODE=$2
      shift 2
      ;;
    --json)
      JSON_OUT=1
      shift
      ;;
    -h|--help)
      echo "usage: $0 [install|remove|status] [--task-name NAME] [--python PYTHON] [--pane PATH] [--interval-min N] [--live-resume] [--mode control|garden] [--json]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
repo_dir=$(CDPATH= cd "$script_dir/.." && pwd)
if [ -z "$PANE" ]; then
  PANE=$script_dir/fleet_control_pane.py
fi

# In garden mode the tick runs the gardening bundle, not the keep-alive pane, so it
# gets its own task/unit/runner names (the two ticks coexist) and a daily default
# cadence unless the caller overrode --interval-min.
case "$MODE" in
  control)
    reg_runner_name=control_pane_tick.sh
    unit_base=fleet-control-pane-tick
    ;;
  garden)
    if [ "$TASK_NAME" = "FleetControlPaneTick" ]; then
      TASK_NAME=FakGardenTick
    fi
    if [ "$INTERVAL_MIN" = "5" ]; then
      INTERVAL_MIN=1440
    fi
    reg_runner_name=garden_tick.sh
    unit_base=fak-garden-tick
    ;;
  *)
    echo "unknown --mode: $MODE (want control|garden)" >&2
    exit 2
    ;;
esac

reg_dir=$script_dir/_registry
runner=$reg_dir/$reg_runner_name
service_name=$unit_base.service
timer_name=$unit_base.timer
unit_dir=${XDG_CONFIG_HOME:-${HOME:-}/.config}/systemd/user
service_path=$unit_dir/$service_name
timer_path=$unit_dir/$timer_name
cron_begin="# fleet-control-pane-tick begin"
cron_end="# fleet-control-pane-tick end"

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

sh_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

systemd_available() {
  command -v systemctl >/dev/null 2>&1 && systemctl --user list-timers >/dev/null 2>&1
}

cron_installed() {
  command -v crontab >/dev/null 2>&1 && crontab -l 2>/dev/null | grep -F "$cron_begin" >/dev/null 2>&1
}

status_json() {
  mode=none
  state=missing
  installed=false
  detail=
  if systemd_available && [ -f "$timer_path" ]; then
    mode=systemd
    state=$(systemctl --user is-active "$timer_name" 2>/dev/null || true)
    enabled=$(systemctl --user is-enabled "$timer_name" 2>/dev/null || true)
    detail=$enabled
    if [ "$enabled" = "enabled" ]; then
      installed=true
    fi
  elif cron_installed; then
    mode=cron
    state=present
    installed=true
  fi
  printf '{"supported":true,"installed":%s,"task_name":"%s","mode":"%s","state":"%s","detail":"%s","runner":"%s"}\n' \
    "$installed" \
    "$(json_escape "$TASK_NAME")" \
    "$(json_escape "$mode")" \
    "$(json_escape "$state")" \
    "$(json_escape "$detail")" \
    "$(json_escape "$runner")"
}

status_text() {
  doc=$(status_json)
  if printf '%s' "$doc" | grep '"installed":true' >/dev/null 2>&1; then
    echo "$TASK_NAME installed: $doc"
  else
    echo "$TASK_NAME not installed: $doc"
  fi
}

remove_cron_block() {
  if ! command -v crontab >/dev/null 2>&1; then
    return 0
  fi
  tmp=$(mktemp)
  crontab -l 2>/dev/null | awk -v begin="$cron_begin" -v end="$cron_end" '
    $0 == begin { skip = 1; next }
    $0 == end { skip = 0; next }
    skip != 1 { print }
  ' > "$tmp" || true
  crontab "$tmp" 2>/dev/null || true
  rm -f "$tmp"
}

write_runner() {
  mkdir -p "$reg_dir"
  if [ "$MODE" = "garden" ]; then
    # The garden tick is ledger-wrapped and admit-gated: fak_loop_run.sh records
    # fire/admit/start/end in .fak/loops.jsonl and runs the bundle through the
    # governor (cadence floor / pause / FAK_GARDEN=off all take effect). Fail-open
    # to a bare `make garden` if the wrapper is absent, same contract as dispatch.
    wrapper=$script_dir/fak_loop_run.sh
    {
      echo "#!/bin/sh"
      echo "cd $(sh_quote "$repo_dir") || exit 1"
      echo "if [ -x $(sh_quote "$wrapper") ]; then"
      echo "  exec $(sh_quote "$wrapper") garden/default cron -- make garden"
      echo "else"
      echo "  exec make garden"
      echo "fi"
    } > "$runner"
    chmod +x "$runner"
    return
  fi
  live_arg=
  if [ "$LIVE_RESUME" = "1" ]; then
    live_arg=" --live-resume"
  fi
  {
    echo "#!/bin/sh"
    echo "cd $(sh_quote "$repo_dir") || exit 1"
    echo "exec $(sh_quote "$PYTHON") $(sh_quote "$PANE") tick$live_arg"
  } > "$runner"
  chmod +x "$runner"
}

install_systemd() {
  mkdir -p "$unit_dir"
  cat > "$service_path" <<EOF
[Unit]
Description=Fleet control pane tick

[Service]
Type=oneshot
WorkingDirectory=$repo_dir
ExecStart=/bin/sh "$runner"
EOF
  cat > "$timer_path" <<EOF
[Unit]
Description=Run fleet control pane tick every $INTERVAL_MIN minute(s)

[Timer]
OnBootSec=1min
OnUnitActiveSec=${INTERVAL_MIN}min
Unit=$service_name

[Install]
WantedBy=timers.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now "$timer_name"
  echo "installed $TASK_NAME via systemd user timer ($timer_name, every $INTERVAL_MIN min)"
  echo "runner: $runner"
}

install_cron() {
  if ! command -v crontab >/dev/null 2>&1; then
    echo "systemd user timers are unavailable and crontab was not found" >&2
    exit 1
  fi
  case "$INTERVAL_MIN" in
    ''|*[!0-9]*|0)
      echo "--interval-min must be a positive integer" >&2
      exit 2
      ;;
  esac
  if [ "$INTERVAL_MIN" -lt 60 ]; then
    cron_expr="*/$INTERVAL_MIN * * * *"
  elif [ "$INTERVAL_MIN" -ge 1440 ]; then
    # Daily (or longer) — fire once a day at an off-the-hour minute so a garden
    # tick doesn't pile onto the top of the hour with the 5-min control ticks.
    cron_expr="23 6 * * *"
  else
    # Hourly granularity for the 60..1439 band (cron can't express arbitrary
    # multi-hour intervals portably; round to the top of each hour).
    cron_expr="0 * * * *"
  fi
  old=$(mktemp)
  new=$(mktemp)
  crontab -l 2>/dev/null > "$old" || true
  awk -v begin="$cron_begin" -v end="$cron_end" '
    $0 == begin { skip = 1; next }
    $0 == end { skip = 0; next }
    skip != 1 { print }
  ' "$old" > "$new"
  {
    echo "$cron_begin"
    echo "$cron_expr $(sh_quote "$runner")"
    echo "$cron_end"
  } >> "$new"
  crontab "$new"
  rm -f "$old" "$new"
  echo "installed $TASK_NAME via crontab (every $INTERVAL_MIN min)"
  echo "runner: $runner"
}

case "$ACTION" in
  status)
    if [ "$JSON_OUT" = "1" ]; then
      status_json
    else
      status_text
    fi
    ;;
  remove)
    if command -v systemctl >/dev/null 2>&1; then
      systemctl --user disable --now "$timer_name" >/dev/null 2>&1 || true
      systemctl --user daemon-reload >/dev/null 2>&1 || true
    fi
    rm -f "$service_path" "$timer_path"
    remove_cron_block
    echo "removed $TASK_NAME"
    ;;
  install)
    case "$INTERVAL_MIN" in
      ''|*[!0-9]*|0)
        echo "--interval-min must be a positive integer" >&2
        exit 2
        ;;
    esac
    write_runner
    if systemd_available; then
      install_systemd
    else
      install_cron
    fi
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
