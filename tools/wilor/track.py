"""Tracks the hands of videos with WiLoR (3D hand pose from single frames) and stores what it finds.

    .venv-wilor/bin/python tools/wilor/track.py VIDEO [VIDEO ...]

Runs in its own environment (PyTorch; `.venv-wilor`, see docs/ARCHITECTURE.md), apart from the backend's. For each video
it writes data/hands/<sha1 of the video file>.npz, which the backend picks up when it processes that video
(`rsl_gesture/hand_wilor.py`). At most one left and one right hand per frame. Which side a hand is and whether a
detection is a hand of the signer at all is checked against the body pose (MediaPipe Pose), see `assign`.

Stored per hand: `frame`, `box` (x0, y0, x1, y1, pixels), `score` (detector confidence, 0 where the detector lost the
hand and its box was taken from the frames around), `right` (1 right / 0 left, as the camera sees it), `points2d` (21 points in the MediaPipe hand order, pixels),
`points3d` (the same in metres, relative to the hand's own origin, x right, y down, z away from the camera) and
`camera` (where that origin is from the camera, metres, for the focal length `focal` in pixels), and the MANO hand
model's rotations as axis-angle: `orient` (the whole hand) and `pose` (15 finger joints).
WiLoR's weights and the MANO hand model it uses are licensed for non-commercial use.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "hands"
MIN_SCORE = 0.25
# The patch given to the hand model is this many times the detector's box (as in WiLoR's own pipeline).
PATCH_SCALE = 2.5
# A hand lost by the detector for at most this many frames is still estimated, in a box between the ones around it.
MAX_GAP = 15
# A spare detection this close (box sizes) to where the missing hand was is that hand.
REASSIGN_REACH = 0.6
# The body pose settles a detection's side when the detection lies this close to one wrist (shoulder widths) and this
# much farther from the other one.
SURE_NEAR = 0.45
SURE_MARGIN = 0.5
# A detection this far (shoulder widths) from both wrists of the body pose is not a hand of the signer.
BODY_REACH = 1.5
# A hand does not move farther than this many of its box sizes in one frame.
MAX_STEP = 1.5
# A left and a right hand whose points lie this close on average (of the hand's extent in the picture) are one hand
# read twice, when the pose has the other wrist elsewhere.
DOUBLE_NEAR = 0.25
# Pose landmarks: shoulders, and per hand kind (0 left, 1 right) the wrist, little finger and index finger knuckles.
SHOULDERS = (11, 12)
HAND_POINTS = {0: (15, 17, 19), 1: (16, 18, 20)}


def video_key(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pipeline():
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32, verbose=False)


def read_frames(video: Path):
    capture = cv2.VideoCapture(str(video))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield index, frame  # BGR, which both the detector and the hand model expect
        index += 1
    capture.release()


def centre(box: np.ndarray) -> np.ndarray:
    return (box[:2] + box[2:4]) / 2


def size(box: np.ndarray) -> float:
    return float(max(box[2] - box[0], box[3] - box[1]))


def body_side(row: np.ndarray, body: dict[int, np.ndarray], shoulder: float) -> tuple[bool, int | None]:
    """Whether the detection can be a hand of the body in the picture, and which side's when the pose leaves no doubt.
    `body`: per hand kind where the pose puts the wrist and the middle of the hand, pixels."""
    if len(body) < 2 or shoulder <= 0:
        return True, None
    reach = {kind: min(float(np.linalg.norm(centre(row) - point)) for point in body[kind]) / shoulder for kind in (0, 1)}
    near = min(reach, key=reach.get)
    if reach[near] > BODY_REACH:
        return False, None
    return True, near if reach[near] < SURE_NEAR and reach[1 - near] > reach[near] + SURE_MARGIN else None


def assign(detections: list[list[np.ndarray]], bodies: list[dict[int, np.ndarray]], shoulder: float) -> list[dict[int, np.ndarray]]:
    """One box per frame for the left hand (0) and the right (1), from the detector's boxes (x0, y0, x1, y1, score, right).

    The detector's own left/right is often wrong, and it takes things in the background for hands. So a detection far
    from both wrists of the body pose is dropped unless it follows a hand tracked so far, and one that lies at one
    wrist and clearly not at the other is that side's hand whatever the detector called it. Where the pose does not
    tell (hands together, an arm lost): a hand keeps the detection of its own kind nearest to where it was, unless that
    is farther than a hand moves in the time; two detections of one kind while the other hand is missing and one of
    them lies where the other hand was, or a single detection there and not where its own kind was, mean the detector
    mistook that hand's side, and it gets the box. The hand model then reads each box as the kind it was given here."""
    last: dict[int, tuple[int, np.ndarray]] = {}
    out = []
    for index, found in enumerate(detections):
        def gap(kind: int, row: np.ndarray) -> float:
            return float(np.linalg.norm(centre(row) - centre(last[kind][1])))

        def reachable(kind: int, row: np.ndarray) -> bool:
            return gap(kind, row) < MAX_STEP * size(last[kind][1]) * (index - last[kind][0])

        def followed(row: np.ndarray) -> bool:
            return any(kind in last and index - last[kind][0] <= 3 and gap(kind, row) < REASSIGN_REACH * size(last[kind][1]) for kind in (0, 1))

        sided = [(row, *body_side(row, bodies[index], shoulder)) for row in found]
        # The pose itself is wrong at times: a hand followed up to here stays a hand wherever the pose has the wrists.
        sided = [(row, side) for row, possible, side in sided if possible or followed(row)]
        # What the detector calls a left and a right hand at one wrist: the pose has lost the other arm, it tells nothing.
        for side in (0, 1):
            if len({int(row[5]) for row, own in sided if own == side}) > 1:
                sided = [(row, None if own == side else own) for row, own in sided]
        rows: list[tuple[np.ndarray, bool]] = []
        for row, side in sided:
            if side is not None:
                row = row.copy()
                row[5] = side
            rows.append((row, side is not None))

        chosen: dict[int, tuple[np.ndarray, bool]] = {}
        spare: list[np.ndarray] = []
        for kind in (0, 1):
            own = [(row, sure) for row, sure in rows if int(row[5]) == kind and (kind not in last or reachable(kind, row))]
            if not own:
                continue
            # The pose's word first, then nearness to where the hand was (or the detector's confidence at the start).
            own.sort(key=lambda item: (not item[1], gap(kind, item[0]) if kind in last else -item[0][4]))
            chosen[kind] = own[0]
            spare += [row for row, sure in own[1:] if not sure]
        # A lone detection where the other hand was, far from where its own kind was: the other hand, mislabelled.
        for kind in (0, 1):
            other = 1 - kind
            if kind in chosen and not chosen[kind][1] and other not in chosen and other in last:
                row = chosen[kind][0]
                near_other = gap(other, row) < REASSIGN_REACH * size(last[other][1])
                near_own = kind in last and gap(kind, row) < REASSIGN_REACH * size(last[kind][1])
                if near_other and not near_own:
                    del chosen[kind]
                    spare.append(row)
                    break
        for kind in (0, 1):
            if kind in chosen or kind not in last:
                continue
            near = [row for row in spare if gap(kind, row) < REASSIGN_REACH * size(last[kind][1])]
            if near:
                chosen[kind] = (min(near, key=lambda row: gap(kind, row)), False)
        for kind, (row, _) in chosen.items():
            last[kind] = (index, row)
        out.append({kind: row for kind, (row, _) in chosen.items()})
    return out


def gaps(tracks: list[dict[int, np.ndarray]]) -> dict[int, dict[int, np.ndarray]]:
    """Boxes for the frames where a hand was not found between two frames where it was, at most MAX_GAP apart: the
    detector loses a hand that the other one covers or that moves fast, while the hand model can still read it."""
    todo: dict[int, dict[int, np.ndarray]] = {}
    for kind in (0, 1):
        seen = [index for index, chosen in enumerate(tracks) if kind in chosen]
        for before, after in zip(seen, seen[1:]):
            if after - before < 2 or after - before - 1 > MAX_GAP:
                continue
            first, second = tracks[before][kind][:4], tracks[after][kind][:4]
            for index in range(before + 1, after):
                blend = (index - before) / (after - before)
                todo.setdefault(index, {})[kind] = first * (1 - blend) + second * blend
    return todo


def body_pose():
    import mediapipe as mp

    return mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)


def body_hands(pose, image: np.ndarray) -> tuple[dict[int, np.ndarray], float]:
    """Where the body pose puts the hands in a frame: per hand kind the wrist and the middle of the hand (pixels), and
    the shoulder width (0 when no body is found)."""
    result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if result.pose_landmarks is None:
        return {}, 0.0
    height, width = image.shape[:2]
    points = np.array([[mark.x * width, mark.y * height] for mark in result.pose_landmarks.landmark])
    hands = {kind: np.stack([points[marks[0]], points[list(marks)].mean(axis=0)]) for kind, marks in HAND_POINTS.items()}
    return hands, float(np.linalg.norm(points[SHOULDERS[0]] - points[SHOULDERS[1]]))


def estimate_hands(pipeline, image: np.ndarray, boxes: np.ndarray, kinds: np.ndarray) -> list[dict[str, np.ndarray]]:
    """WiLoR's hand for each box, as `predict_with_bboxes` of the WiLoR-mini pipeline gives it, but smoothing only the
    part of the picture each patch is cut from: the pipeline smooths the whole frame once per hand (against aliasing
    when the hand is shrunk to the model's 256 pixels), which on Full HD video is most of its running time."""
    from wilor_mini.utils import utils

    side = pipeline.IMAGE_SIZE
    height, width = image.shape[:2]
    img_size = np.array([width, height])
    centres = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
    sizes = (PATCH_SCALE * (boxes[:, 2:4] - boxes[:, 0:2])).max(axis=1)
    patches = []
    for centre_, size_, kind in zip(centres, sizes, kinds):
        sigma = (size_ / side / 2.0 - 1) / 2
        reach = size_ / 2 + (4 * sigma if sigma > 0.05 else 0) + 2
        x0, y0 = max(0, int(centre_[0] - reach)), max(0, int(centre_[1] - reach))
        x1, y1 = min(width, int(centre_[0] + reach) + 1), min(height, int(centre_[1] + reach) + 1)
        region = image[y0:y1, x0:x1]
        if size_ / side / 2.0 > 1.1:
            region = cv2.GaussianBlur(region, (0, 0), sigma)
        patch, _ = utils.generate_image_patch_cv2(region, centre_[0] - x0, centre_[1] - y0, size_, size_, side, side,
                                                  kind == 0, 1.0, 0, border_mode=cv2.BORDER_CONSTANT)
        patches.append(patch)
    batch = torch.from_numpy(np.stack(patches)).to(device=pipeline.device, dtype=pipeline.dtype)
    with torch.no_grad():
        output = {name: value.cpu().float().numpy() for name, value in pipeline.wilor_model(batch).items()}
    focal = pipeline.FOCAL_LENGTH / side * img_size.max()
    hands = []
    for i, kind in enumerate(kinds):
        hand = {name: value[[i]] for name, value in output.items()}
        camera = hand["pred_cam"]
        camera[:, 1] = (2 * kind - 1) * camera[:, 1]
        if kind == 0:  # estimated as a mirrored right hand
            hand["pred_keypoints_3d"][:, :, 0] = -hand["pred_keypoints_3d"][:, :, 0]
            hand["global_orient"] = np.concatenate((hand["global_orient"][:, :, 0:1], -hand["global_orient"][:, :, 1:3]), axis=-1)
            hand["hand_pose"] = np.concatenate((hand["hand_pose"][:, :, 0:1], -hand["hand_pose"][:, :, 1:3]), axis=-1)
        hand["pred_cam_t_full"] = utils.cam_crop_to_full(camera, centres[[i]], sizes[i], img_size[None], focal)
        hand["scaled_focal_length"] = focal
        hand["pred_keypoints_2d"] = utils.perspective_projection(hand["pred_keypoints_3d"], translation=hand["pred_cam_t_full"],
                                                                 focal_length=np.array([focal] * 2)[None], camera_center=img_size[None] / 2)
        hands.append(hand)
    return hands


def steady(bodies: list[dict[int, np.ndarray]]) -> list[dict[int, np.ndarray]]:
    """The pose's hands with each frame's replaced by the median of the five around it: the pose throws an arm
    somewhere else for a frame or two now and then."""
    out = []
    for index, body in enumerate(bodies):
        around = [other for other in bodies[max(0, index - 2):index + 3] if other]
        out.append({kind: np.median([other[kind] for other in around], axis=0) for kind in body} if body else {})
    return out


def twice(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether two hands' points in the picture are those of one hand."""
    extent = max(np.ptp(first, axis=0).max(), np.ptp(second, axis=0).max())
    return float(np.linalg.norm(first - second, axis=1).mean()) < DOUBLE_NEAR * extent


def track(pipeline, video: Path) -> dict[str, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    rows: dict[str, list] = {name: [] for name in ("frame", "box", "score", "right", "points2d", "points3d", "camera", "orient", "pose")}
    focal = 0.0

    def estimate(index: int, image: np.ndarray, boxes: np.ndarray, kinds: np.ndarray, scores: np.ndarray) -> None:
        nonlocal focal
        hands = estimate_hands(pipeline, image, boxes.astype(np.float32), kinds.astype(np.float32))
        for box, kind, score, predicted in zip(boxes, kinds, scores, hands):
            focal = float(predicted["scaled_focal_length"])
            rows["frame"].append(index)
            rows["box"].append(box)
            rows["score"].append(score)
            rows["right"].append(kind)
            rows["points2d"].append(predicted["pred_keypoints_2d"][0])
            rows["points3d"].append(predicted["pred_keypoints_3d"][0])
            rows["camera"].append(predicted["pred_cam_t_full"][0])
            rows["orient"].append(predicted["global_orient"][0].reshape(3))
            rows["pose"].append(predicted["hand_pose"][0].reshape(15, 3))

    detections, bodies, shoulders = [], [], []
    count = 0
    with body_pose() as pose:
        for index, image in read_frames(video):
            found = pipeline.hand_detector(image, conf=MIN_SCORE, verbose=False)[0].boxes.data.cpu().numpy().reshape(-1, 6)
            detections.append(list(found))
            hands, shoulder = body_hands(pose, image)
            bodies.append(hands)
            if shoulder:
                shoulders.append(shoulder)
            count = index + 1
    bodies = steady(bodies)
    shoulder = float(np.median(shoulders)) if shoulders else 0.0
    tracks = assign(detections, bodies, shoulder)
    todo = gaps(tracks)
    for index, image in read_frames(video):
        boxes, kinds, scores = [], [], []
        for kind, row in tracks[index].items():
            boxes.append(row[:4]); kinds.append(kind); scores.append(row[4])
        for kind, box in todo.get(index, {}).items():
            boxes.append(box); kinds.append(kind); scores.append(0.0)
        if boxes:
            estimate(index, image, np.array(boxes), np.array(kinds), np.array(scores))
            if len(boxes) == 2 and twice(rows["points2d"][-2], rows["points2d"][-1]):
                # The detector gave one hand as a left and a right one (`assign` leaves that open): the pose's side stays.
                sides = {body_side(box, bodies[index], shoulder)[1] for box in boxes} - {None}
                if len(sides) == 1:
                    drop = -2 if rows["right"][-2] != sides.pop() else -1
                    for values in rows.values():
                        del values[drop]
    shapes = {"box": (4,), "points2d": (21, 2), "points3d": (21, 3), "camera": (3,), "orient": (3,), "pose": (15, 3)}
    out = {name: np.array(values, dtype=np.int32 if name == "frame" else np.float32).reshape((len(values), *shapes.get(name, ())))
           for name, values in rows.items()}
    out.update(frames=np.int32(count), fps=np.float32(fps), width=np.int32(width), height=np.int32(height), focal=np.float32(focal))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    videos = [Path(arg) for arg in sys.argv[1:] if not arg.startswith("--")]
    redo = "--redo" in sys.argv
    pipeline = None
    for number, video in enumerate(videos, 1):
        target = OUT / f"{video_key(video)}.npz"
        if target.exists() and not redo:
            continue
        pipeline = pipeline or load_pipeline()
        result = track(pipeline, video)
        partial = target.with_suffix(".part.npz")
        np.savez_compressed(partial, **result)
        partial.rename(target)
        print(f"{number}/{len(videos)} {video.name}: {len(result['frame'])} hands in {int(result['frames'])} frames", flush=True)


if __name__ == "__main__":
    main()
