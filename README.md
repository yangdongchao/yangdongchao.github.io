# Dongchao Yang — personal website

Static research portfolio with an existing Jekyll blog, published through GitHub Pages.

## Editing

- `index.html`: homepage, background, and service.
- `research.html`: PhD research narrative.
- `publications.html` and `app.js`: searchable publications.
- `_research/research-summaries.json`: selected-work introductions.
- `_research/research-decks.json`: 14 presentations, 56 slides.

After changing research JSON, run `python3 _research/build_research.py`.
This updates the selected-work blocks and `presentations.html` while preserving
other homepage sections. The `_research` source folder is excluded by Jekyll's
underscore-directory convention.

Preview static pages with `python3 -m http.server 8765 --bind 127.0.0.1`.
Build the complete site, including the blog, with `bundle exec jekyll build`.

## Media and research sources

The pages cite original papers beside experimental comparisons and method diagrams.
Diagrams are explanatory schematics. LongCat descriptions identify team participation;
ReasoningCodec is part of UniAudio 2.0, not a separate publication.

Audio samples are from the author's public project pages:
- `media/heartmula.mp3`: https://heartmula.github.io/static/audios/heartmula/english/15.mp3
- `media/instructtts-calm.wav`: https://dongchaoyang.top/InstructTTS/demo/109.wav
- `media/instructtts-angry.wav`: https://dongchaoyang.top/InstructTTS/demo/101.wav

Presentation references: [Assertion–Evidence](https://www.assertion-evidence.com/),
[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), and
[Distill Feature Visualization](https://distill.pub/2017/feature-visualization/).
