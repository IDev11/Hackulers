"""Standalone detection -> association -> video workflow. Never loads possession GT."""
import argparse, hashlib, json, platform, time
from pathlib import Path
from datetime import datetime
from importlib.metadata import version
from dataclasses import asdict
import cv2
import numpy as np
import pandas as pd
from ball_candidate_tracker import associate, TrackerSettings
from resumable_detections import collect
from trajectory_reconstruction import reconstruct_selected_track

ROOT = Path(__file__).resolve().parents[1]
def resolve(value):
    p=Path(value).expanduser()
    return p if p.is_absolute() else ROOT/p

def load_geometry(path, stop):
    t=pd.read_csv(path)
    columns=[f'h{i}{j}' for i in range(3) for j in range(3)]
    if not set(['frame','valid',*columns]).issubset(t.columns):
        raise ValueError('Homography CSV needs frame, valid and h00 through h22')
    if not np.array_equal(t.frame.to_numpy(),np.arange(len(t))) or len(t)<stop:
        raise ValueError('Homography must be indexed by original video frames from zero')
    if not t.valid.isin([0,1,True,False]).all(): raise ValueError('valid must be numeric 0/1')
    return t[columns].to_numpy(float).reshape(-1,3,3),t.valid.to_numpy(bool)

def make_detector(video, weights, cfg):
    model = None
    ids = None

    def detect(start, stop):
        nonlocal model, ids
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(weights))
            names = model.names
            entries = names.items() if isinstance(names, dict) else enumerate(names)
            ids = [int(i) for i, name in entries if str(name).strip().casefold() == 'ball']
            if len(ids) != 1:
                raise ValueError(f'Expected exactly one ball class: {names}')
        records = []
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open video: {video}')
            if start and not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
                raise RuntimeError(f'Cannot seek to frame {start}')
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != start:
                raise RuntimeError(f'Video reader did not seek to frame {start}')
            for frame in range(start, stop):
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f'Cannot decode frame {frame}')
                t = time.perf_counter()
                result = model.predict(
                    image, classes=ids, conf=cfg['confidence'], imgsz=cfg['imgsz'],
                    device=cfg['device'], rect=False, iou=.7, max_det=100, verbose=False,
                )[0]
                boxes = result.boxes.cpu().numpy()
                records.append(dict(
                    frame=frame,
                    detections=[dict(xyxy=box.tolist(), confidence=float(confidence))
                                for box, confidence in zip(boxes.xyxy, boxes.conf)],
                    inference_ms=1000 * (time.perf_counter() - t),
                ))
                if (frame - start + 1) % 25 == 0:
                    print(f'Detected frames {start}-{frame} (batch ends at {stop - 1})', flush=True)
        finally:
            cap.release()
        return records

    return detect


def run(config_path, cached=None, render=True, *, reconstruct=None, resume_detections=None,
        chunk_frames=250):
    if cached is not None and resume_detections is not None:
        raise ValueError('Choose either a saved detection file or a resumable detection directory')
    cfg=json.loads(Path(config_path).read_text())
    video=resolve(cfg['video']); weights=resolve(cfg['weights']); geometry=resolve(cfg['homography'])
    start=int(cfg['start']);stop=start+int(cfg['frames'])
    if start<0 or stop<=start:raise ValueError('Invalid frame interval')
    cap=cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():raise ValueError(f'Cannot open video: {video}')
        fps=cap.get(cv2.CAP_PROP_FPS);width=int(cap.get(3));height=int(cap.get(4))
        if fps<=0 or stop>int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):raise ValueError('Video interval out of bounds')
    finally:cap.release()
    H,valid=load_geometry(geometry,stop)
    digest=None
    if cached:
        # Cache mode preserves its original source: it is NOT a new YOLO26 run.
        records=[json.loads(x) for x in Path(cached).read_text().splitlines() if x.strip()]
        if [r['frame'] for r in records]!=list(range(start,stop)):raise ValueError('Cache frame interval mismatch')
    else:
        digest=hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest!=cfg['expected_sha256']:raise ValueError('Checkpoint checksum differs from selected model')
        detect = make_detector(video, weights, cfg)
        if resume_detections is None:
            records = detect(start, stop)
        else:
            video_stat = video.stat()
            identity = dict(
                video=str(video.resolve()), video_bytes=video_stat.st_size,
                video_mtime_ns=video_stat.st_mtime_ns, fps=fps, width=width, height=height,
                checkpoint_sha256=digest, confidence=cfg['confidence'], imgsz=cfg['imgsz'],
                device=cfg['device'], ultralytics=version('ultralytics'), opencv=cv2.__version__,
                detector_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
            def report(first, end, saved):
                action = 'Reused' if saved else 'Saved'
                print(f'{action} frames {first}-{end - 1} ({end - start}/{stop - start})', flush=True)

            records = collect(resume_detections, start, stop, chunk_frames, identity, detect, report)
    out=resolve(cfg.get('output_root','outputs'))/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    settings=TrackerSettings()
    selected,audit=associate(records,H,valid,width,height,fps,settings)
    outputs = [('detections', records), ('selected', selected)]
    if reconstruct is not None:
        reconstructed = reconstruct_selected_track(selected, method=reconstruct)
        outputs.append(('reconstructed', reconstructed))
    for name,rows in outputs:
        with (out/f'{name}.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    pd.DataFrame(audit).to_csv(out/'candidate_decisions.csv',index=False)
    summary=dict(frames=len(records),raw_frames_with_candidates=sum(bool(r['detections']) for r in records),
        selected_observations=sum(s['status']=='observed' for s in selected),
        unknown=sum(s['status']=='unknown' for s in selected),accuracy_validated=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    manifest=dict(config=cfg,mode='cached_original_detections' if cached else 'fresh_detector',
        checkpoint_sha256=digest,cached_source=str(cached) if cached else None,
        cache_sha256=hashlib.sha256(Path(cached).read_bytes()).hexdigest() if cached else None,
        tracker_sha256=hashlib.sha256((ROOT/'src/ball_candidate_tracker.py').read_bytes()).hexdigest(),
        settings=asdict(settings),fps=fps,python=platform.python_version(),public_gt_read=False,
        homography_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest())
    if resume_detections is not None:
        manifest.update(mode='resumable_detector',
                        detection_chunks=str(Path(resume_detections).resolve()),
                        chunk_frames=chunk_frames)
    (out/'run_config.json').write_text(json.dumps(manifest,indent=2))
    if render:render_video(video,start,records,selected,out/'raw_vs_tracked.mp4',fps,width,height)
    print(json.dumps(summary,indent=2));print('Output:',out)
    return out

def render_video(video,start,records,selected,path,fps,width,height):
    import supervision as sv
    annotator=sv.BoxAnnotator(color=sv.Color.from_hex('#00FF80'),thickness=2)
    labels=sv.LabelAnnotator(color=sv.Color.from_hex('#00FF80'),text_scale=.5,text_thickness=1)
    def draw(image,items):
        if not items:return image
        ds=sv.Detections(xyxy=np.array([x['xyxy'] for x in items],np.float32),
            confidence=np.array([x['confidence'] for x in items]),class_id=np.zeros(len(items),int))
        return labels.annotate(scene=annotator.annotate(scene=image,detections=ds),detections=ds,
            labels=[f"ball {x['confidence']:.2f}" for x in items])
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width*2,height))
    try:
        if not cap.isOpened() or not writer.isOpened():raise RuntimeError('Cannot open video reader/writer')
        for r,s in zip(records,selected):
            ok,image=cap.read()
            if not ok:raise RuntimeError('Video ended early')
            left=draw(image.copy(),r['detections']);right=draw(image.copy(),[s] if s['status']=='observed' else [])
            for panel,title in [(left,'RAW YOLO'),(right,'TRACK: '+s['status'].upper())]:
                cv2.rectangle(panel,(0,0),(width,65),(0,0,0),-1)
                cv2.putText(panel,f"{title} | source frame {r['frame']}",(15,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            writer.write(np.hstack([left,right]))
    finally:cap.release();writer.release()
    check=cv2.VideoCapture(str(path))
    try:
        if int(check.get(cv2.CAP_PROP_FRAME_COUNT))!=len(records) or not check.read()[0]:raise RuntimeError('Output video verification failed')
    finally:check.release()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--no-render',action='store_true')
    detection = p.add_mutually_exclusive_group()
    detection.add_argument('--cached')
    detection.add_argument('--resume-detections', type=Path,
                           help='Directory for restartable detection chunks')
    p.add_argument('--chunk-frames', type=int, default=250,
                   help='Frames per saved chunk (default: 250)')
    p.add_argument('--reconstruct', nargs='?', const='linear', choices=('linear', 'hermite'),
                   help='Write reconstructed.jsonl (combine with --cached to reuse detections)')
    a = p.parse_args()
    run(a.config, a.cached, not a.no_render, reconstruct=a.reconstruct,
        resume_detections=a.resume_detections, chunk_frames=a.chunk_frames)
