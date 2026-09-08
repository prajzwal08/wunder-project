#!/usr/bin/env bash
# Launch the viewer without needing conda/mamba activation.
cd "$(dirname "$0")"
exec "$HOME/miniforge3/envs/wunder/bin/streamlit" run app.py "$@"
