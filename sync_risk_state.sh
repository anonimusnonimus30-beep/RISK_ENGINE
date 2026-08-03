#!/bin/bash

set -euo pipefail

SOURCE="/home/juanitotrader/Documents/RISK_ENGINE/output/risk_state.json"

SPY_DEST="/home/juanitotrader/Documents/SPY_CENTINEL/risk_engine/risk_state.json"
QQQ_DEST="/home/juanitotrader/qqq_sentinel/risk_engine/risk_state.json"

if [ ! -f "$SOURCE" ]; then
    echo "ERROR: no existe $SOURCE"
    exit 1
fi

cp "$SOURCE" "$SPY_DEST"
cp "$SOURCE" "$QQQ_DEST"

echo "Risk state sincronizado con SPY y QQQ."
