"""Experimental straight-road edge estimates from image evidence, not YOLO."""
import cv2
import numpy as np


class RoadEdgeEstimator:
    def __init__(self, fps=30):
        self.state = {}
        self.max_gap = max(1, round(fps * 0.3))

    def update(self, frame):
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (960, round(height * 960 / width)))
        h, w = small.shape[:2]
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 130)
        mask = np.zeros_like(edges)
        polygon = np.array([(0, int(h*.18)), (int(w*.38), int(h*.10)),
                            (int(w*.62), int(h*.10)), (w-1, int(h*.18)),
                            (w-1, int(h*.91)), (0, int(h*.91))], np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
        segments = cv2.HoughLinesP(cv2.bitwise_and(edges, mask), 1, np.pi/180,
                                   35, minLineLength=65, maxLineGap=25)
        candidates = {'left': [], 'right': []}
        if segments is not None:
            for x1,y1,x2,y2 in segments[:,0]:
                x1,x2,y1,y2 = x1/w,x2/w,y1/h,y2/h
                if abs(y2-y1)<.035:
                    continue
                a=(x2-x1)/(y2-y1); b=x1-a*y1
                side='left' if a<0 else 'right'
                if not .5<abs(a)<4.5 or not .25<a*.02+b<.75:
                    continue
                if (side=='left' and (x1+x2)/2>.40) or (side=='right' and (x1+x2)/2<.60):
                    continue
                bottom=a*.85+b
                if (side=='left' and not -3.5<bottom<.05) or (side=='right' and not .95<bottom<4.5):
                    continue
                length=float(np.hypot(x2-x1,y2-y1))
                candidates[side].append((a,b,length))
        result={}
        for side, choices in candidates.items():
            best=None; best_score=0
            for a,b,length in choices:
                support=sum(l for aa,bb,l in choices if abs((a*.4+b)-(aa*.4+bb))<.035 and abs(a-aa)<.3)
                # Prefer the outer supported road edge over an inner lane stripe.
                outer=-(a*.65+b) if side=='left' else a*.65+b
                score=support + .12*outer
                if score>best_score:
                    best_score=score; best=np.array([a,b],float)
            previous=self.state.get(side)
            if best is not None:
                if previous is not None:
                    best=.18*best+.82*previous['line']
                state={'line':best,'age':0}
                self.state[side]=state
            elif previous is not None and previous['age']<self.max_gap:
                previous['age']+=1; state=previous
            else:
                self.state.pop(side,None)
                result[side]={'status':'unavailable'}
                continue
            a,b=map(float,state['line'])
            result[side]={'status':'held estimate' if state['age'] else 'estimate',
                          'line_x_equals_a_y_plus_b':[a,b], 'age_frames':state['age']}
        return result

    @staticmethod
    def draw(frame, estimates):
        h,w=frame.shape[:2]; thickness=max(3,round(w/500)); scale=max(.6,w/1900)
        for index,side in enumerate(('left','right')):
            item=estimates[side]; color=(255,210,0) if side=='left' else (70,230,100)
            if item['status']!='unavailable':
                a,b=item['line_x_equals_a_y_plus_b']
                p1=(round((a*.16+b)*w),round(.16*h)); p2=(round((a*.91+b)*w),round(.91*h))
                ok,p1,p2=cv2.clipLine((0,0,w,h),p1,p2)
                if ok:
                    # Enclose the visible estimated edge; no learned box is implied.
                    pad=max(6,round(w*.005))
                    top_left=(max(0,min(p1[0],p2[0])-pad),max(0,min(p1[1],p2[1])-pad))
                    bottom_right=(min(w-1,max(p1[0],p2[0])+pad),min(h-1,max(p1[1],p2[1])+pad))
                    cv2.rectangle(frame,top_left,bottom_right,color,thickness,cv2.LINE_AA)
            label=f"{side.title()} road edge: {item['status']}"
            (tw,th),base=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,scale,max(1,thickness//3))
            x=20; y=round(h*.04)+index*(th+base+24)
            cv2.rectangle(frame,(x-8,y-th-8),(x+tw+8,y+base+8),(20,20,20),-1)
            cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,max(1,thickness//3),cv2.LINE_AA)
        return frame
