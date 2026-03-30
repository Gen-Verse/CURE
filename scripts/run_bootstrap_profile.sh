#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ge 1 ]]; then
  export CURE_PROFILE="$1"
  export LOG_FILE="${LOG_FILE:-logs/$1_$(date +%Y%m%d_%H%M%S).log}"
fi

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_cure.sh"
