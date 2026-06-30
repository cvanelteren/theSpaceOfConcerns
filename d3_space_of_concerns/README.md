# Space of Concerns D3 Compare

Small force-directed D3 app for comparing an interactive layout with the paper's static space-of-concerns figure.

## Run

From repo root:

```bash
python scripts/export_space_of_concerns_graph_json.py
python d3_space_of_concerns/server.py --port 8000
```

Then open:

`http://127.0.0.1:8000/`

## Notes

- Graph data is exported to `d3_space_of_concerns/data/space_of_concerns_graph.json`.
- Saved layout coordinates are written to `d3_space_of_concerns/data/space_of_concerns_layout_saved.json`.
- Nodes are topics colored by theme.
- Links are MST edges plus top-5% strongest additional links.
- `Pin Dragged Nodes` keeps nodes fixed when drag ends.
- Click any node to toggle pin/unpin manually (pinned nodes use a dashed stroke).
- Use `Clear Pins` to release all fixed nodes.
- Turn on `Box Select` to drag-select nodes, then drag one selected node to move the group.
- Most controls now live in the right-side settings cards under the legend.
- Edge-repel `strength` and `radius` are adjustable in the settings card under the legend.
- `Layout` lets you switch between:
- `Network`: regular force-directed network.
- `Theme Bubble`: clusters by topic theme.
- `Ego Radial`: selected focus node at center, 1-hop neighbors on first ring, others on outer rings.
- `MST Shell`: shells based on graph distance along the MST backbone from the selected focus node.
- `Fit Inside Antarctica Shape` pulls and contains the layout within the Antarctica silhouette.
