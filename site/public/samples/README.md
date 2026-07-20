# Sample ROIs for the live-diagnose box

These are real MVTec AD **test** images. The site's "Try it" section sends them to the
deployed `POST /diagnose` endpoint and shows Qwen-VL's real verdict. Filenames map to the
`SAMPLES` list in `src/data.ts`.

- `bottle-good.png`    — a clean bottle (Qwen returns defect-free, conf ~0.98)
- `bottle-broken.png`  — a bottle with a broken rim (Qwen returns structural:chip, conf ~0.95)
- `capsule-defect.png` — a capsule with surface contamination (Qwen returns
  structural:contamination, conf ~0.95)

Source: `bottle/test/{good,broken_large}/000.png` and `capsule/test/poke/000.png`, downscaled
to 512 px on the long edge for a snappy round-trip. MVTec AD is CC BY-NC-SA (non-commercial);
the credit line is shown in the UI.

To swap in different parts, drop the new images here, update `SAMPLES` in `src/data.ts`, and
verify the live verdict is clear before relying on it on stage. Prefer parts where the defect
is clearly visible, and use test images, not the `ground_truth` masks (those are black-and-white
blobs Qwen will correctly reject as "not a part").
