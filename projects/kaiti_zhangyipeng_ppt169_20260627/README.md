# kaiti_zhangyipeng

- Canvas format: ppt169
- Created: 20260627

## Directories

- `svg_output/`: raw SVG output
- `svg_final/`: finalized SVG output
- `images/`: runtime image pool; converter assets keep their original short filenames when possible
- `icons/`: project icon set — selected library icons copied in (via icon_sync.py) plus any custom icons you add; embedded from here at export
- `notes/`: speaker notes
- `templates/`: project templates
- `sources/`: source materials and normalized markdown
- `analysis/`: machine-extracted intermediate analysis (PPTX intake, image_analysis.csv) — the pipeline's canonical must-read source/asset facts
- `exports/`: main native pptx (timestamped); `_svg.pptx` sibling added when exported with `--svg-snapshot`
- `backup/<timestamp>/`: svg_output/ archive (always written in default-flow mode; safe to delete old timestamps)
