# HyperFrames Composition Brief: DSPM

## Objective
Create a 15-second premium brag video for DSPM (Dynamic Semantic Patch Memory), a Python library for budgeted semantic memory over multi-turn LLM conversations.

## Output
- Composition directory: `brag-output-2026-10-28-124443/composition/`
- Rendered video: `brag-output-2026-10-28-124443/brag.mp4`
- Format: landscape - 1920x1080
- Duration: 15 seconds

## Source Material
- Project root: this local repository
- Primary files read: `README.md`, `pyproject.toml`, `src/dspm/memory.py`, `src/dspm/engine.py`, `src/dspm/config.py`
- Product name: DSPM (Dynamic Semantic Patch Memory); PyPI package `dspm-memory`
- Strongest product-specific moment: a webhook timeout changes from 30 to 10 seconds and the compressed revision story is represented as `10 (was 30)`.
- Real UI/data to recreate: conversation update, a typed `[DEC]` semantic patch, compression-stage labels, and qualified benchmark proof.
- Copy that must appear verbatim:
  - `[DEC] webhook timeout: 10 (was 30)`
  - `82.8% token reduction`
  - `100% critical patches selected into context`
  - `250-token budget`
  - `7 domains x 40 turns`
  - `pip install dspm-memory`
- Benchmark qualifier: the 100% figure is critical-patch selection into context, not a guarantee of fact recoverability; the 82.8% result is from the project's benchmark suite using 7 hand-authored 40-turn dialogues. The same model family performed extraction and judging, and the benchmark has not been independently replicated.
- Do not use any credentials, personal contact details, local machine paths, or unsupported marketing claims.

## Creative Direction
- Tone preset: polished
- Creative direction: premium dynamic motion graphic with smooth 3D camera movement, morphing shapes, and glow
- Angle: the revised fact is the hero. A change from 30 seconds to 10 seconds becomes a typed patch that keeps its revision history, then resolves into a compact benchmark receipt.
- Hook: two values collide; the stale `30` gives way to `10 (was 30)`.
- Outro: grounded benchmark evidence, followed by DSPM and the package install command.
- Avoid: generic SaaS language, unsupported claims, abstract filler, fast flashing text, and representing the unwired adaptive-per-type-budgeting stage as shipped.

## Visual Identity
- Background: `#0B1119` (chosen for video; the Python package has no website palette)
- Text: `#F1EBDD`
- Accent: `#E5A94B`
- Display and body font: system sans-serif; system monospace for code
- Visual references: `[CON]`/`[DEC]` patch notation, the timeout revision notation, and benchmark figures from the README.

## Storyboard
Use `brag-plan.md` as the creative contract:
1. The revision - 3.1s - conversation values transform from 30 to 10 seconds.
2. The patch - 3.2s - `[DEC] webhook timeout: 10 (was 30)` lands on one continuous morphing subject.
3. The compression path - 4.2s - six actual compression stages resolve into compact context.
4. The receipt - 4.5s - qualified 250-token benchmark, product name, and install command.

## Audio
- Role: restrained cinematic support
- Music: `assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3`
- Treatment: low-level bed with a short fade-in and fade-out; no voiceover
- Music cue guidance: bundled cue preset at 109.96 BPM; lock the pipeline resolve near 8.74s and the product/CTA lockup near 13.11s. Use nearby beat-grid points for small accents only.
- Audio-reactive treatment: subtle modulation of the existing warm glow/object presence; no visualizer graphics.
- SFX: one soft reveal accent and one quiet landing at most; use local CC0 assets and keep them below the music.
- Place all audio assets under `composition/assets/` with relative paths.

## HyperFrames Instructions
- The project is initialized as a standalone composition.
- Build one paused GSAP timeline registered under the root `data-composition-id`; make all motion seek-safe and deterministic.
- Make the dimensional camera travel and morphing shape serve the actual DSPM revision story; keep the shape continuous through its morph.
- Use only the bundled local assets. Keep copy legible, especially the benchmark caveat.
- Run `npx hyperframes check --snapshots`, inspect snapshots, then open a Studio preview. Render only after the user reviews and approves the preview.
