# Workshop shell helpers — auto-loaded by ~/.zshrc when entering this repo
# (triggered by direnv exporting WORKSHOP_DIR via .envrc).
#
# Commands:
#   ws-list                       list all runnable units in order
#   ws-current                    print the unit last marked/run (short form)
#   ws-next                       print the unit that follows current  (short form)
#   ws-set    <unit>              mark <unit> as current without running it
#   ws-run    <unit>              `go run` <unit> and mark it as current
#   ws-reset                      forget current state
#
# Unit short form (used for input, completion, and the state file):
#   exampleNN-stepM   for examples that have step1/, step2/, ... subdirs
#   exampleNN         for flat examples (single main.go at example root)
#
# Display form (used by the p10k segment, per Florin's preference):
#   exampleNN-slug/stepM   or   exampleNN-slug
#
# State is persisted in $WORKSHOP_DIR/.workshop-state (gitignored).
#
# A Powerlevel10k segment `prompt_workshop` is also defined here; register
# it in ~/.p10k.zsh's LEFT_PROMPT_ELEMENTS to show "current → next".

WORKSHOP_LOADED=1
WORKSHOP_STATE_FILE="${WORKSHOP_DIR}/.workshop-state"

# Enumerate runnable units (short form) in deterministic order.
# Alphabetical works because example numbers are zero-padded and step counts <10.
_ws_units() {
  emulate -L zsh
  setopt local_options null_glob
  local dir base num stepdir
  for dir in "${WORKSHOP_DIR}"/cmd/examples/example*/; do
    base="${dir%/}"; base="${base##*/}"     # e.g. example05-rag-motivation
    num="${base#example}"; num="${num%%-*}" # e.g. 05
    if [[ -n "${dir}"step*(/N) ]] 2>/dev/null; then :; fi
    local found_steps=0
    for stepdir in "${dir}"step*(/N); do
      found_steps=1
      printf 'example%s-%s\n' "${num}" "${stepdir:t}"
    done
    if (( ! found_steps )) && [[ -f "${dir}main.go" ]]; then
      printf 'example%s\n' "${num}"
    fi
  done
}

# Compact form used in the prompt segment.
#   example09-step1   ->  ex09-step1
#   example01         ->  ex01
_ws_compact() {
  local unit="$1"
  [[ -z "$unit" ]] && return 0
  print -- "${unit/example/ex}"
}

# Map a short-form unit to its full-slug display form.
#   example09-step1   ->  example09-retrieval-debug-step1
#   example01         ->  example01-vectors
_ws_display() {
  emulate -L zsh
  setopt local_options null_glob
  local unit="$1"
  [[ -z "$unit" ]] && return 0
  local num="${unit#example}"; num="${num%%-*}"
  local step=""
  [[ "$unit" == *-step<-> ]] && step="${unit##*-}"
  local -a matches=( "${WORKSHOP_DIR}"/cmd/examples/example${num}-*(/N) )
  (( ${#matches} == 0 )) && { print -- "$unit"; return 0; }
  local full="${matches[1]:t}"   # e.g. example09-retrieval-debug
  if [[ -n "$step" ]]; then
    print -- "${full}-${step}"
  else
    print -- "${full}"
  fi
}

# Resolve a short-form unit to its on-disk path under cmd/examples.
_ws_resolve() {
  emulate -L zsh
  setopt local_options null_glob
  local unit="$1"
  [[ -z "$unit" ]] && { print -u2 "usage: <unit> (e.g. example09-step1)"; return 2; }
  local num="${unit#example}"; num="${num%%-*}"
  local step=""
  [[ "$unit" == *-step<-> ]] && step="${unit##*-}"
  local -a matches=( "${WORKSHOP_DIR}"/cmd/examples/example${num}-*(/N) )
  if (( ${#matches} == 0 )); then
    print -u2 "unknown example: $unit"; return 1
  fi
  local example_dir="${matches[1]}"
  if [[ -n "$step" ]]; then
    [[ -d "${example_dir}/${step}" ]] || { print -u2 "unknown step: $unit"; return 1; }
    print -- "${example_dir}/${step}"
  else
    [[ -f "${example_dir}/main.go" ]] || { print -u2 "no main.go in: $unit"; return 1; }
    print -- "${example_dir}"
  fi
}

ws-list() {
  local u
  for u in ${(f)"$(_ws_units)"}; do
    _ws_display "$u"
  done
}

ws-current() {
  [[ -f "$WORKSHOP_STATE_FILE" ]] || return 0
  < "$WORKSHOP_STATE_FILE"
}

ws-next() {
  local cur="$(ws-current)"
  local -a all=( ${(f)"$(_ws_units)"} )
  if [[ -z "$cur" ]]; then
    print -- "${all[1]}"
    return
  fi
  local i=${all[(Ie)$cur]}
  (( i == 0 || i >= ${#all} )) && return
  print -- "${all[i+1]}"
}

ws-set() {
  _ws_resolve "$1" >/dev/null || return $?
  printf '%s\n' "$1" > "$WORKSHOP_STATE_FILE"
  print -- "current: $(_ws_display "$1")"
}

ws-run() {
  # NOTE: do NOT name a local 'path' — zsh ties 'path' to PATH (typeset -T),
  # so 'local path' empties PATH in this function and its subshells.
  local unit="$1" dir
  dir="$(_ws_resolve "$unit")" || return $?
  local rel="${dir#${WORKSHOP_DIR}/}"
  print -- "▶ go run ./${rel}"
  ( cd "$WORKSHOP_DIR" && go run "./${rel}" )
  local rc=$?
  printf '%s\n' "$unit" > "$WORKSHOP_STATE_FILE"
  return $rc
}

ws-reset() { rm -f "$WORKSHOP_STATE_FILE"; }

# Zsh completion: list all units as candidates for ws-run / ws-set.
_ws_complete() {
  local -a units=( ${(f)"$(_ws_units)"} )
  _describe -t units 'workshop unit' units
}
compdef _ws_complete ws-run ws-set 2>/dev/null

# Powerlevel10k segment. Renders nothing when outside the repo.
# Styling (POWERLEVEL9K_WORKSHOP_BACKGROUND/FOREGROUND) lives in ~/.p10k.zsh
# because p10k reads those at config time, not at render time.
# Uses compact form (exNN[-stepM]) to keep the band short.
prompt_workshop() {
  [[ -n "${WORKSHOP_DIR:-}" ]] || return
  local cur="$(ws-current)" nxt="$(ws-next)"
  local cur_c="$(_ws_compact "$cur")" nxt_c="$(_ws_compact "$nxt")"
  if [[ -z "$cur" ]]; then
    p10k segment -i '📚' -t "next: ${nxt_c:-—}"
  elif [[ -z "$nxt" ]]; then
    p10k segment -i '📚' -t "${cur_c} ✓"
  else
    p10k segment -i '📚' -t "${cur_c} → ${nxt_c}"
  fi
}
