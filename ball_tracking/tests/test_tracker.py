import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from ball_candidate_tracker import associate
class TrackerTests(unittest.TestCase):
    def test_causality_and_missing(self):
        H=np.repeat(np.eye(3)[None],20,axis=0);v=np.ones(20,bool)
        rows=[dict(frame=i,detections=[] if i in (5,6) else [dict(xyxy=[10+i,40,14+i,44],confidence=.9)]) for i in range(12)]
        full,_=associate(rows,H,v,100,100,25)
        prefix,_=associate(rows[:6],H,v,100,100,25)
        self.assertEqual(full[:6],prefix)
        self.assertTrue(all(full[i]['status']=='unknown' for i in (5,6)))
        for r,s in zip(rows,full):
            if s['status']=='observed':self.assertEqual(s['xyxy'],r['detections'][s['candidate_index']]['xyxy'])
    def test_invalid_geometry(self):
        H=np.repeat(np.eye(3)[None],5,axis=0);v=np.zeros(5,bool)
        rows=[dict(frame=i,detections=[dict(xyxy=[10,40,14,44],confidence=.9)]) for i in range(5)]
        out,_=associate(rows,H,v,100,100,25)
        self.assertTrue(all(x['status']=='unknown' for x in out))
if __name__=='__main__':unittest.main()
