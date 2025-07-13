rm -rf .venv
rm closest_points_cuda.pyi
uv venv
uv pip install .
.venv/bin/stubgen -m closest_points_cuda -o .