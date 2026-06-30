# D3 Space Actor Interests

Static D3 app for the ATS concern-space graph with actor overlays at `RPA > 1`.

## Build data

From repo root:

```bash
python scripts/export_space_actor_interests_json.py
```

This writes:

- `d3_space_actor_interests/data/space_actor_interests.json`
- copied flag/logo assets into `d3_space_actor_interests/assets/flags/`
- copied Antarctica contour into `d3_space_actor_interests/assets/antarctica_contour.png`

## Serve locally

```bash
python -m http.server 8000
```

Then open:

- `http://localhost:8000/d3_space_actor_interests/`

## Notes

- The app uses the saved concern-space layout from `d3_space_of_concerns`.
- The overlay is static: no force simulation is rerun in the browser.
- Actor topics are those with `RPA > 1` in the exported panel.
