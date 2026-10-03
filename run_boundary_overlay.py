"""Render experimental road edges alongside previously saved YOLO damage detections."""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from road_edge_estimator import RoadEdgeEstimator
from run_inference import annotate

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=BASE/'vid3.mp4')
    parser.add_argument('--detections', type=Path, default=BASE/'outputs/vid3_focused.detections.json')
    parser.add_argument('--output', type=Path, default=BASE/'outputs/vid3_estimated_boundary_boxes.mp4')
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix.lower() != '.mp4':
        parser.error('Choose a new .mp4 output filename.')
    records = json.loads(args.detections.read_text())
    metadata = json.loads(args.detections.with_name(args.detections.name.replace('.detections.json','.summary.json')).read_text())
    if Path(metadata['source']).resolve() != args.source.resolve():
        parser.error('Cached detections must belong to the selected source video.')
    names = {int(k):v for k,v in metadata['classes'].items()}
    cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width,height = int(cap.get(3)),int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or (width,height,total)!=(metadata['width'],metadata['height'],len(records)) or abs(fps-metadata['fps'])>.01:
        cap.release(); parser.error('Source properties do not match cached detections.')
    estimator = RoadEdgeEstimator(fps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    writer = cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():
        cap.release(); raise RuntimeError('Cannot create output video')
    coverage = Counter(); edge_records=[]; count=0
    try:
        for record in records:
            ok,frame = cap.read()
            if not ok: raise RuntimeError(f'Video ended at frame {count}')
            if record['frame']!=count: raise ValueError('Cache frame order mismatch')
            estimates = estimator.update(frame)
            boxes = [b for b in record['display_boxes'] if b['class_id'] in (0,1,2)]
            frame = annotate(frame,None,names,cv2,metadata.get('confidence_format','percent'),boxes)
            frame = estimator.draw(frame,estimates)
            writer.write(frame)
            for side,item in estimates.items(): coverage[f"{side}_{item['status']}"]+=1
            edge_records.append({'frame':count,'seconds':count/fps,'edges':estimates})
            if count in (0,300,600,900,1200):
                cv2.imwrite(str(args.output.with_name(args.output.stem+f'.frame{count}.jpg')),cv2.resize(frame,(1272,720)))
            count+=1
            if count%200==0: print(f'Rendered {count}/{total}',flush=True)
    finally:
        cap.release(); writer.release()
    summary = dict(source=str(args.source.resolve()),output=str(args.output.resolve()),
                   cached_yolo_detections=str(args.detections.resolve()),frames=count,fps=fps,
                   width=width,height=height,edge_coverage=dict(coverage),audio_preserved=False,
                   boundary_method='Experimental Hough line estimates, temporally smoothed; not YOLO',
                   boundary_display='Axis-aligned boxes enclosing each visible estimated edge with 0.5% frame-width padding',
                   maximum_hold_frames=estimator.max_gap,
                   limitations='Straight-road heuristic tuned to vid3; may follow curbs, lane marks, shadows or cracks incorrectly. Coverage is not accuracy.')
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
    args.output.with_suffix('.edges.json').write_text(json.dumps(edge_records))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
