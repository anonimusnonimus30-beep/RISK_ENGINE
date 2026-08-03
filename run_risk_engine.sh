#!/bin/bash

set -euo pipefail

cd /home/juanitotrader/Documents/RISK_ENGINE

source venv/bin/activate

python risk_engine.py
./sync_risk_state.sh
