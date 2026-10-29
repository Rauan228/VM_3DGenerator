"""Face landmarks of the Tripo reference images (models/tripo/<key>_v3_restyle_generated_image_url.png), for
build_tripo_avatars.py: a handful of named points, the face oval, and all 478 points (eye and lip contours).

    .venv-linux/bin/python tools/ref_face_landmarks.py
"""

import json
from pathlib import Path

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
NAMED = {"eyeL": 473, "eyeR": 468, "mouthL": 291, "mouthR": 61, "mouthTop": 13, "mouthBottom": 14, "nose": 1, "chin": 152,
         "browL": 334, "browR": 105, "eyeLtop": 386, "eyeLbot": 374, "eyeRtop": 159, "eyeRbot": 145, "foreheadTop": 10,
         "faceL": 454, "faceR": 234}
OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150,
        136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
# The face is small in a full-body image: it is found on an enlarged crop of the top of the picture.
TOP_SHARE, ZOOM = 0.25, 3

out = {}
for key in ("male", "female"):
    image = cv2.imread(str(ROOT / "models" / "tripo" / f"{key}_v3_restyle_generated_image_url.png"))
    height, width = image.shape[:2]
    crop = cv2.resize(image[: int(height * TOP_SHARE)], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as mesh:
        result = mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    points = [(p.x * crop.shape[1] / ZOOM, p.y * crop.shape[0] / ZOOM) for p in result.multi_face_landmarks[0].landmark]
    out[key] = {"size": [width, height], "points": {k: points[i] for k, i in NAMED.items()}, "oval": [points[i] for i in OVAL], "all": points}
    print(key, len(points), "points")
(ROOT / "models" / "tripo" / "ref_landmarks.json").write_text(json.dumps(out, indent=1))
