# PostLens MVP

PostLens is a working prototype of an AI-assisted post-production review tool.

## What it does
- Upload multiple video clips.
- Samples footage and flags potential sharpness/focus and exposure issues.
- Optionally indexes visible objects using a pretrained Ultralytics YOLO model.
- Shows the exact evidence frame and timestamp for each alert.
- Lets an editor inspect the original clip before making a decision.

## Run

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Important MVP boundary
This prototype deliberately says **potential issue**, not **error**. Film continuity and creative choices can be intentional. The human editor remains the decision-maker.

## Next build steps
1. Add production metadata / shot-list ingestion.
2. Add scene and take grouping.
3. Add audio QC and transcription.
4. Add evidence-based continuity comparison between related shots.
5. Add semantic natural-language search across indexed footage.

## Stack
Streamlit + OpenCV + NumPy + optional Ultralytics YOLO.
