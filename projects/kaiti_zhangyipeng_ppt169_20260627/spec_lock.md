# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## mode
- mode: pyramid

## visual_style
- visual_style: editorial

## colors
- bg: #FAF8F4
- bg_secondary: #F1ECE3
- card_white: #FFFFFF
- primary: #1B3A5B
- accent: #C8A24B
- secondary_accent: #4A6B8A
- text: #2B2B2B
- text_secondary: #5A5A52
- text_tertiary: #8C8C82
- border: #D8D0C2
- success: #3C7A5A
- warning: #B24A38

## typography
- font_family: "Microsoft YaHei", Arial, sans-serif
- title_family: Cambria, SimSun, serif
- emphasis_family: Cambria, SimSun, serif
- code_family: Consolas, "Courier New", monospace
- body: 20
- hero_opacity_number: 120
- title: 34
- subtitle: 26
- annotation: 15
- cover_title: 60
- chapter_opener: 46
- hero_number: 38
- footnote: 12

## icons
- library: tabler-outline
- stroke_width: 2
- inventory: database, share, users, eye, scale, settings, refresh, file-text, math-function, robot, server, puzzle, target, adjustments, clock, flame, dice, cpu, checklist, calendar, bulb, stack

## images
- cuc_logo: images/cuc_logo.png | no-crop
- fig1_closed_loop: images/fig1_closed_loop.png | no-crop
- fig2_portrait: images/fig2_portrait.png | no-crop
- fig3_abm: images/fig3_abm.png | no-crop
- fig4_online: images/fig4_online.png | no-crop

## page_rhythm
- P01: anchor
- P02: anchor
- P03: breathing
- P04: dense
- P05: dense
- P06: dense
- P07: breathing
- P08: dense
- P09: breathing
- P10: dense
- P11: dense
- P12: dense
- P13: dense
- P14: dense
- P15: breathing
- P16: anchor

## page_charts
- P05: vertical_pillars
- P07: circular_stages
- P08: vertical_list
- P11: icon_grid
- P14: timeline

## forbidden
- Mixing icon libraries
- rgba()
- `<style>`, `class`, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<script>`, `<iframe>`, `<symbol>`+`<use>`
- `<g opacity>` (set opacity on each child element individually)
- HTML named entities in text — write as raw Unicode (`—`, `→`, `·`, `©`, NBSP); XML reserved chars `& < > " '` escaped as `&amp; &lt; &gt; &quot; &apos;`
- 术语「反演」「反推」「反向校准」——统一使用「校准」「参数校准」
