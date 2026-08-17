"""
Этап 4. Анализ пальцев — самый критичный для РЖЯ.

Из 21 точки кисти MediaPipe вычисляем:
  joint_angles  — угол в каждом суставе каждого пальца (рад)
  finger_curl   — суммарный сгиб пальца, нормированный 0..1 (0 = прямой)
  palm_open     — раскрытие ладони 0..1 (среднее расстояние кончиков от ладони)
  palm_normal   — нормаль ладони (куда "смотрит" ладонь)
  palm_direction— направление от запястья к основанию среднего пальца
  spread        — разведение пальцев (углы между соседними)

Эти параметры — семантическое ядро жеста: два слова РЖЯ могут отличаться
только curl одного пальца или поворотом palm_normal на 90°.
Храним их отдельно от координат, чтобы поиск/распознавание шли по ним.
"""
from __future__ import annotations
import numpy as np

WRIST = 0
FINGERS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
TIPS = [4, 8, 12, 16, 20]
MCP = [5, 9, 13, 17]  # основания пальцев (без большого)

# максимальный суммарный сгиб пальца ~ 270° (3 сустава по ~90°)
_MAX_CURL = np.deg2rad(270)


def _angle(a, b, c) -> float:
    """Угол в точке b между ba и bc, рад. NaN-безопасно."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8 or np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
        return float("nan")
    cosv = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cosv))


def hand_features(hand: np.ndarray) -> dict | None:
    """hand: (21,3) нормализованные координаты одной кисти на одном кадре."""
    if hand is None or np.all(np.isnan(hand)):
        return None

    joint_angles, curls = {}, {}
    for name, chain in FINGERS.items():
        pts = [hand[WRIST]] + [hand[i] for i in chain]
        angles = [_angle(pts[i - 1], pts[i], pts[i + 1])
                  for i in range(1, len(pts) - 1)]
        joint_angles[name] = [round(a, 4) for a in angles]
        # сгиб = сумма (pi - угол): прямой сустав даёт 0
        bend = np.nansum([np.pi - a for a in angles if not np.isnan(a)])
        curls[name] = round(float(np.clip(bend / _MAX_CURL, 0, 1)), 4)

    # нормаль ладони: запястье, основание указательного, основание мизинца
    w, i_mcp, p_mcp = hand[WRIST], hand[5], hand[17]
    normal = np.cross(i_mcp - w, p_mcp - w)
    n = np.linalg.norm(normal)
    palm_normal = (normal / n).round(4).tolist() if n > 1e-8 else [0, 0, 0]

    direction = hand[9] - w
    dn = np.linalg.norm(direction)
    palm_direction = (direction / dn).round(4).tolist() if dn > 1e-8 else [0, 0, 0]

    # раскрытие: средняя дистанция кончиков до центра ладони / размер ладони
    palm_center = np.nanmean(hand[[0, 5, 9, 13, 17]], axis=0)
    palm_size = np.linalg.norm(hand[9] - w)
    tips_d = np.nanmean([np.linalg.norm(hand[t] - palm_center) for t in TIPS])
    palm_open = float(np.clip((tips_d / max(palm_size, 1e-8)) - 0.3, 0, 1.2) / 1.2)

    # разведение соседних пальцев (по MCP-векторам от запястья)
    spread = []
    for a_mcp, b_mcp in zip(MCP[:-1], MCP[1:]):
        spread.append(round(_angle(hand[a_mcp], w, hand[b_mcp]), 4))

    return {
        "joint_angles": joint_angles,
        "curl": curls,
        "palm_open": round(palm_open, 4),
        "palm_normal": palm_normal,
        "palm_direction": palm_direction,
        "spread": spread,
    }


def sequence_hand_features(hands: np.ndarray) -> list:
    """(T,21,3) -> список фич по кадрам (None где кисть потеряна)."""
    return [hand_features(hands[t]) for t in range(hands.shape[0])]


def hand_embedding_frame(feat: dict | None) -> np.ndarray:
    """
    Компактный вектор (15,) конфигурации кисти для одного кадра:
    5 curl + palm_open + 3 normal + 3 direction + 3 spread.
    Основа эмбеддингов для поиска похожих жестов (AI-уровень).
    """
    if feat is None:
        return np.zeros(15)
    return np.array(
        [feat["curl"][f] for f in FINGER_ORDER] +
        [feat["palm_open"]] + feat["palm_normal"] +
        feat["palm_direction"] +
        [0.0 if np.isnan(s) else s for s in feat["spread"]]
    )
