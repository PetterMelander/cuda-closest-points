rm -rf .venv
uv venv
uv pip install .
.venv/bin/stubgen -m closest_points_cuda -o .