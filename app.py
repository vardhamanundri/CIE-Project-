import os, json, math, tempfile, subprocess, shutil
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title='PostLens', page_icon='🎬', layout='wide')

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

# Optional YOLO: the app still works without it, using the built-in visual/technical checks.
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


def fmt_time(sec):
    sec = float(sec)
    return f"{int(sec//60):02d}:{int(sec%60):02d}"


def sample_video(path, sample_every=2.0, max_frames=60):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frames / fps if fps else 0
    step = max(1, int(fps * sample_every))
    samples = []
    i = 0
    while i < frames and len(samples) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        samples.append({
            'time': i / fps,
            'brightness': brightness,
            'contrast': contrast,
            'blur_score': lap_var,
            'frame': frame,
        })
        i += step
    cap.release()
    return fps, duration, samples


def analyze_clip(path, yolo_model=None):
    fps, duration, samples = sample_video(path)
    issues = []
    if not samples:
        return {'duration': duration, 'samples': [], 'issues': issues, 'objects': {}}

    blur = [s['blur_score'] for s in samples]
    bright = [s['brightness'] for s in samples]
    median_blur = float(np.median(blur))
    median_bright = float(np.median(bright))

    # Conservative heuristics: report potential issues, never definitive failures.
    for s in samples:
        if median_blur > 20 and s['blur_score'] < max(8, median_blur * 0.28):
            issues.append({
                'type': 'Technical', 'severity': 'Medium', 'time': s['time'],
                'title': 'Possible focus / sharpness issue',
                'detail': f"Frame sharpness dropped to {s['blur_score']:.1f} vs project median {median_blur:.1f}."
            })
        if s['brightness'] < 28:
            issues.append({
                'type': 'Technical', 'severity': 'Low', 'time': s['time'],
                'title': 'Very dark frame detected',
                'detail': f"Average luminance is {s['brightness']:.1f}. Review exposure before delivery."
            })
        if s['brightness'] > 235:
            issues.append({
                'type': 'Technical', 'severity': 'Low', 'time': s['time'],
                'title': 'Very bright frame detected',
                'detail': f"Average luminance is {s['brightness']:.1f}. Review highlights/exposure."
            })

    # Collapse nearby duplicate alerts.
    compact = []
    for issue in sorted(issues, key=lambda x: x['time']):
        if compact and issue['type'] == compact[-1]['type'] and abs(issue['time'] - compact[-1]['time']) < 4:
            continue
        compact.append(issue)

    objects = {}
    if yolo_model is not None:
        # Analyze a small set of representative frames to keep the MVP responsive.
        for s in samples[::max(1, len(samples)//12)]:
            rgb = cv2.cvtColor(s['frame'], cv2.COLOR_BGR2RGB)
            try:
                result = yolo_model(rgb, verbose=False)[0]
                names = result.names
                for cls_id in result.boxes.cls.tolist() if result.boxes is not None else []:
                    label = names[int(cls_id)]
                    objects[label] = objects.get(label, 0) + 1
            except Exception:
                pass

    return {
        'duration': duration,
        'samples': [{k:v for k,v in s.items() if k != 'frame'} for s in samples],
        'issues': compact[:20],
        'objects': objects,
        'median_blur': median_blur,
        'median_brightness': median_bright,
    }


def frame_at(path, sec):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_yolo():
    if not YOLO_AVAILABLE:
        return None
    try:
        return YOLO('yolo26n.pt')
    except Exception:
        return None

st.markdown('''
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
.big-title {font-size: 3rem; font-weight: 800; margin-bottom: 0.2rem;}
.sub {font-size: 1.15rem; color: #777; margin-bottom: 1.5rem;}
.alert {padding: 14px 18px; border: 1px solid #ddd; border-radius: 12px; margin: 8px 0;}
</style>
''', unsafe_allow_html=True)

st.markdown('<div class="big-title">🎬 PostLens</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Your second pair of eyes in post-production.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header('Project')
    uploaded = st.file_uploader('Drop footage here', type=['mp4','mov','m4v','avi','mkv'], accept_multiple_files=True, max_upload_size=500)
    use_yolo = st.checkbox('Enable visual object analysis', value=False, disabled=not YOLO_AVAILABLE,
                           help='Uses a pretrained Ultralytics model to index visible objects. The model may download on first run.')
    st.caption('AI flags potential issues. The editor makes the final decision.')

if not uploaded:
    st.info('Upload one or more clips to start a PostLens review.')
    c1,c2,c3 = st.columns(3)
    c1.metric('What it does', 'Reviews footage')
    c2.metric('Output', 'Evidence + alerts')
    c3.metric('Human control', 'Always on')
    st.markdown('### MVP workflow')
    st.write('**Upload → Analyze → Review alerts → Inspect the exact frame → Decide.**')
    st.caption('The first prototype focuses on technical review and searchable visual content. Continuity is deliberately presented as a future/next-step capability, not an automatic truth.')
    st.stop()

# Persist uploads for this session.
session_dir = Path(tempfile.gettempdir()) / 'postlens_session'
session_dir.mkdir(exist_ok=True)
paths = []
for up in uploaded:
    p = session_dir / up.name
    p.write_bytes(up.getbuffer())
    paths.append(p)

if st.button('🔍 Analyze project', type='primary'):
    model = load_yolo() if use_yolo else None
    results = {}
    progress = st.progress(0)
    for i, p in enumerate(paths):
        results[p.name] = analyze_clip(p, model)
        progress.progress((i+1)/len(paths))
    st.session_state['results'] = results
    st.session_state['paths'] = {p.name: str(p) for p in paths}
    progress.empty()

results = st.session_state.get('results')
if not results:
    st.warning('Click **Analyze project** to process the uploaded footage.')
    st.stop()

# Summary metrics
all_issues = [(name, x) for name, r in results.items() for x in r['issues']]
total_duration = sum(r['duration'] for r in results.values())
objects = {}
for r in results.values():
    for k,v in r.get('objects',{}).items(): objects[k] = objects.get(k,0)+v

c1,c2,c3,c4 = st.columns(4)
c1.metric('Clips analyzed', len(results))
c2.metric('Total footage', fmt_time(total_duration))
c3.metric('Potential issues', len(all_issues))
c4.metric('Objects indexed', len(objects) if use_yolo else '—')

st.divider()

left, right = st.columns([1.2, 1])
with left:
    st.subheader('⚠️ Review queue')
    if not all_issues:
        st.success('No potential issues were detected by the current checks.')
    else:
        for idx, (name, issue) in enumerate(all_issues):
            with st.expander(f"{issue['title']} · {name} · {fmt_time(issue['time'])}"):
                st.write(issue['detail'])
                p = Path(st.session_state['paths'][name])
                fr = frame_at(p, issue['time'])
                if fr is not None:
                    st.image(fr, caption=f'{name} @ {fmt_time(issue["time"])}', use_container_width=True)
                st.caption('Potential issue — review the evidence before acting.')

with right:
    st.subheader('🔎 Search the footage')
    q = st.text_input('Search by detected object', placeholder='e.g. person, car, chair')
    if use_yolo and objects:
        matches = [k for k in objects if q.lower() in k.lower()] if q else sorted(objects, key=objects.get, reverse=True)
        if matches:
            for k in matches[:10]: st.write(f'**{k}** — detected in sampled frames ({objects[k]} observations)')
        else:
            st.info('No indexed object matches that search.')
    else:
        st.info('Enable visual object analysis and re-run the project to enable object search.')

st.subheader('🎞️ Clips')
for name, r in results.items():
    with st.expander(f'{name}  ·  {fmt_time(r["duration"])}  ·  {len(r["issues"])} potential issues'):
        st.write(f"Median sharpness: **{r.get('median_blur',0):.1f}** · Median brightness: **{r.get('median_brightness',0):.1f}**")
        st.video(st.session_state['paths'][name])

st.divider()
st.caption('PostLens MVP • AI-assisted review, not automated creative judgment.')
