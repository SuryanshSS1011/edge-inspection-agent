# Sample ROIs for the live-diagnose box

Drop 3-4 real MVTec test images here. The site's "Try it" section sends them to the deployed
`POST /diagnose` endpoint and shows Qwen-VL's real verdict.

Expected filenames (edit `SAMPLES` in `src/data.ts` if you use different ones):

- `bottle-good.png`    — a clean bottle (Qwen should say defect-free)
- `bottle-broken.png`  — a bottle with visible broken glass
- `capsule-crack.png`  — a capsule with a visible crack

Guidance:
- Use actual MVTec **test** images (e.g. `bottle/test/good/*.png`,
  `bottle/test/broken_large/*.png`, `capsule/test/crack/*.png`), NOT the `ground_truth`
  masks (those are black-and-white blobs and Qwen will correctly say "not a part").
- Prefer images where the defect is clearly visible so the live verdict is punchy on stage.
- Cropping to a tight ROI matches what the real pipeline sends, but full test images work too.
- Keep it to a few images. MVTec is CC BY-NC-SA (non-commercial); a small credit line is
  already shown in the UI.
