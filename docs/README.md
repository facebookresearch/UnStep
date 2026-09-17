# UnStep project website

This self-contained static website is ready to publish at
https://facebookresearch.github.io/UnStep/. It does not import the inference code,
load models, or require a JavaScript build or external CDN.

## Publishing

In repository **Settings > Pages**, select **Deploy from a branch**, then
**main** and **/docs**. GitHub publishes the site after a push to `main`.
This initial repository setting requires a maintainer with Pages permissions.

## Editing

- `index.html`: authors, paper link, result tables, method explanation, citation.
- `examples.js`: selected appendix video IDs, categories, titles, and prompts.
- `site.js`: example navigation and playback controls.
- `style.css`: responsive layout, typography, and reduced-motion behavior.
- `media/`: selected compressed videos, posters, figures, font, and icons only.

The paper link is intentionally disabled and BibTeX intentionally empty. Add the
real URL and citation when available; do not substitute an unrelated paper.

The videos play at their original 16 FPS. Reported generation throughput is not
implemented by changing playback speed. Examples are selected for illustration,
not used to recompute aggregate benchmark scores. Figure and table values are
transcribed from the paper; no new measurements are made by the website.

Open `index.html` directly in a browser. For a remote devserver, serve this
directory with `python3 -m http.server 8000 --bind 127.0.0.1`, then forward port
8000 in the remote editor and visit `http://localhost:8000/`.

## Asset attribution

Research figures and generated examples are from the UnStep paper and appendix.
I2V inputs are the corresponding VBench-I2V reference crops; VBench's data terms
continue to apply. `media/provenance.json` records the selected sample IDs and
source-file hashes without private server paths.

Manrope is distributed under the SIL Open Font License; see
`media/licenses/manrope.txt`. Lucide icons are distributed under the ISC License;
see `media/licenses/lucide.txt`. All assets are self-hosted.
