# Interactive Figure 2 Ribbon Plot (D3)

This folder contains an interactive D3 version of the Figure 2 ribbon plot.

## 1) Export data

From the repository root:

```bash
python scripts/export_fig2_ribbon_d3_data.py
```

This writes:

- `d3_ribbon_plot/data/fig2_ribbon_data.json`

## 2) Launch local server

From the repository root:

```bash
python -m http.server 8000
```

Open:

- `http://127.0.0.1:8000/d3_ribbon_plot/`

## Interactions

- Filter by theme
- Trace one country across periods
- Adjust minimum transition count
- Toggle topic labels
- Hover nodes/ribbons for details
