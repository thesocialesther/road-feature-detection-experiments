"""Short-lived, class-aware display tracks; never changes raw model predictions."""


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


class BoxSmoother:
    def __init__(self, alpha=0.35, hold_frames=2):
        self.alpha = alpha
        self.hold_frames = hold_frames
        self.tracks = []
        self.next_id = 0

    def update(self, rows):
        candidates = []
        for ti, track in enumerate(self.tracks):
            for di, row in enumerate(rows):
                if int(track['row'][5]) == int(row[5]):
                    score = overlap(track['raw'], row)
                    if score >= 0.30:
                        candidates.append((score, ti, di))
        matched_tracks, matched_detections = set(), set()
        for _, ti, di in sorted(candidates, reverse=True):
            if ti in matched_tracks or di in matched_detections:
                continue
            track, row = self.tracks[ti], rows[di]
            # Smooth geometry only: confidence remains the actual latest model score.
            track['row'] = [self.alpha * row[i] + (1 - self.alpha) * track['row'][i]
                            for i in range(4)] + list(row[4:6])
            track['raw'] = list(row)
            track['missed'] = 0
            matched_tracks.add(ti)
            matched_detections.add(di)
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track['missed'] += 1
        self.tracks = [t for t in self.tracks if t['missed'] <= self.hold_frames]
        for di, row in enumerate(rows):
            if di not in matched_detections:
                self.tracks.append(dict(row=list(row), raw=list(row), missed=0, id=self.next_id))
                self.next_id += 1
        return [dict(xyxy=t['row'][:4], confidence=t['row'][4], class_id=int(t['row'][5]),
                     track_id=t['id'], held_frames=t['missed']) for t in self.tracks]
