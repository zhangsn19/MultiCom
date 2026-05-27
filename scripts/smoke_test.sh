#!/usr/bin/env bash
set -euo pipefail

python -m py_compile multicom/*.py
python -m multicom.personas --help >/dev/null
python -m multicom.agent_rating --help >/dev/null
python -m multicom.oof_aggregation --help >/dev/null
python -m multicom.data_prep --help >/dev/null
echo "smoke test passed"

