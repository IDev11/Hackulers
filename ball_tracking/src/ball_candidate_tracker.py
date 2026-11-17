"""Experimental ball association. Outputs measured boxes only, never imputed coordinates.
Settings are engineering starting values, not fitted to competition labels.
"""
from dataclasses import dataclass, asdict
import numpy as np
import cv2

@dataclass(frozen=True)
class TrackerSettings:
    beam_size: int = 24
    confirm_observations: int = 3
    max_gap_seconds: float = .20
    gate_width_per_frame: float = .035
    max_area_ratio: float = 4.0
    switch_score_margin: float = .8
    outside_margin_width: float = .02
    outside_birth_penalty: float = 1.0
    score_decay: float = .90


def warp(points, matrix):
    points=np.asarray(points,dtype=float).reshape(-1,2)
    q=np.c_[points,np.ones(len(points))] @ matrix.T
    if not np.isfinite(q).all() or np.any(np.abs(q[:,2])<1e-9):
        raise ValueError('Unusable projection')
    return q[:,:2]/q[:,2,None]


def associate(records, matrices, valid, width, height, fps, settings=None):
    """Homography compensates planar camera motion only; aerial motion remains approximate.
    No Metrica ball positions, player labels, or possession ground truth enter association.
    """
    cfg=settings or TrackerSettings()
    assert width>0 and height>0 and fps>0
    assert cfg.confirm_observations>=2 and cfg.beam_size>=1
    max_gap=max(1,round(cfg.max_gap_seconds*fps))
    beam=[];anchor=None;previous=None;owner=None;serial=0
    output=[];audit=[]
    for record in records:
        frame=int(record['frame'])
        if previous is not None and frame<=previous:raise ValueError('Frames must strictly increase')
        discontinuity=previous is not None and frame!=previous+1
        previous=frame
        if discontinuity:beam=[];anchor=None;owner=None
        matrix=np.asarray(matrices[frame],dtype=float)
        geometry_ok=bool(valid[frame]) and bool(np.isfinite(matrix).all())
        candidates=[]
        try:
            if not geometry_ok:raise ValueError('Invalid homography')
            if anchor is None:anchor=matrix.copy()
            transform=anchor @ np.linalg.inv(matrix)
            polygon=warp([[0,0],[1,0],[1,1],[0,1]],matrix).astype(np.float32)
        except (ValueError,np.linalg.LinAlgError):
            beam=[];anchor=None;owner=None;geometry_ok=False
        for j,d in enumerate(record['detections']):
            log=dict(frame=frame,candidate_index=j,confidence=float(d['confidence']),
                     feasible_links=0,outside_pitch_margin=None,reason='invalid_box')
            audit.append(log)
            box=np.asarray(d['xyxy'],dtype=float)
            if box.shape!=(4,) or not np.isfinite(box).all() or not np.isfinite(log['confidence']):continue
            if not 0<=log['confidence']<=1 or not (box[2]>box[0] and box[3]>box[1]):continue
            if not geometry_ok:log['reason']='geometry_unavailable';continue
            center=(box[:2]+box[2:])/2
            try:
                corners=warp([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]],transform)
                position=warp([center],transform)[0]/width
                area=abs(cv2.contourArea(corners.astype(np.float32)))/(width*width)
                if not np.isfinite(position).all() or area<=1e-12:raise ValueError()
            except ValueError:log['reason']='projection_unavailable';continue
            outside=cv2.pointPolygonTest(polygon,tuple(map(float,center)),True)<-cfg.outside_margin_width*width
            log['outside_pitch_margin']=bool(outside)
            log['reason']='unconfirmed'
            candidates.append(dict(index=j,box=box,center=center,position=position,area=area,
                                   confidence=log['confidence'],outside=outside,log=log))
        branches=[]
        for h in beam:
            dt=frame-h['last_frame']
            if dt>max_gap:continue
            missed=dict(h);missed['score']=cfg.score_decay*h['score']-.5;missed['current']=None
            missed['streak']=0
            branches.append(missed)
            for c in candidates:
                ratio=max(c['area']/h['area'],h['area']/c['area'])
                predicted=h['position']+h['velocity']*dt
                residual=float(np.linalg.norm(c['position']-predicted))
                gate=cfg.gate_width_per_frame*np.sqrt(dt)
                if ratio>cfg.max_area_ratio or residual>gate:continue
                c['log']['feasible_links']+=1
                velocity=(c['position']-h['position'])/dt
                streak=h['streak']+1 if dt==1 else 1
                branches.append(dict(seed=h['seed'],position=c['position'],
                    velocity=.5*h['velocity']+.5*velocity,area=c['area'],last_frame=frame,
                    score=cfg.score_decay*h['score']+.5+c['confidence']-.5*(residual/gate)-.1*abs(np.log(ratio)),
                    streak=streak,confirmed=h['confirmed'] or streak>=cfg.confirm_observations,
                    current=c['index']))
        for c in candidates:
            serial+=1
            branches.append(dict(seed=serial,position=c['position'],velocity=np.zeros(2),area=c['area'],
                last_frame=frame,score=c['confidence']-(cfg.outside_birth_penalty if c['outside'] else 0),
                streak=1,confirmed=False,current=c['index']))
        # Keep alternatives, but avoid duplicate branches ending at the same candidate for one seed.
        branches.sort(key=lambda h:(-h['score'],h['seed']))
        unique={}
        for h in branches:
            key=(h['seed'],h['last_frame'],h['current'])
            if key not in unique:unique[key]=h
        beam=list(unique.values())[:cfg.beam_size]
        confirmed=[h for h in beam if h['confirmed']]
        incumbent=next((h for h in confirmed if h['seed']==owner),None)
        challenger=next((h for h in confirmed if h['current'] is not None),None)
        chosen=incumbent
        if challenger is not None and (incumbent is None or challenger['score']>incumbent['score']+cfg.switch_score_margin):
            chosen=challenger
        if chosen is not None:owner=chosen['seed']
        elif not any(h['seed']==owner for h in beam):owner=None
        selected=None if chosen is None else chosen['current']
        for c in candidates:
            c['log']['reason']=('selected_observation' if c['index']==selected else
                'supported_alternative' if c['log']['feasible_links'] else 'unconfirmed_birth')
        selected_candidate=next((c for c in candidates if c['index']==selected),None)
        output.append(dict(frame=frame,status='observed' if selected_candidate is not None else 'unknown',
            candidate_index=selected,track_id=owner if selected_candidate is not None else None,
            xyxy=selected_candidate['box'].tolist() if selected_candidate is not None else None,
            confidence=selected_candidate['confidence'] if selected_candidate is not None else None,
            geometry_available=geometry_ok))
    return output,audit

