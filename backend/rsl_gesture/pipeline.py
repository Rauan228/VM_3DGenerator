from __future__ import annotations

from pathlib import Path

from rsl_gesture.compression import keyframe_compress_tracks
from rsl_gesture.expression import extract_expressions
from rsl_gesture.extractor import MediaPipeHolisticExtractor
from rsl_gesture.features import build_features
from rsl_gesture.filters import apply_filter, compare_filter_quality
from rsl_gesture.hand_cleanup import clean_hand_tracks
from rsl_gesture.hand_depth import add_hand_depth
from rsl_gesture.hand_pose_fix import fix_depth_flips, split_merged_hands
from rsl_gesture.models import CompressionInfo, GestureJson, GestureMetadata, NormalizationInfo
from rsl_gesture.normalization import frame_edges, normalize_tracks


class GesturePipeline:
    def __init__(self, extractor: MediaPipeHolisticExtractor | None = None):
        self.extractor = extractor or MediaPipeHolisticExtractor()

    def process_video(
        self,
        video_path: str | Path,
        word: str,
        filter_name: str = "kalman",
        compress: bool = True,
        epsilon: float = 0.003,
    ) -> GestureJson:
        raw_tracks, fps, duration, width, height = self.extractor.extract(video_path)
        hand_depth = add_hand_depth(video_path, raw_tracks, width, height)
        normalized_tracks = clean_hand_tracks(split_merged_hands(fix_depth_flips(normalize_tracks(raw_tracks, width, height))))
        filtered_tracks = apply_filter(normalized_tracks, filter_name)
        features = build_features(filtered_tracks)
        stored_tracks = keyframe_compress_tracks(filtered_tracks, epsilon=epsilon) if compress else filtered_tracks
        expression = extract_expressions(video_path, raw_tracks.get("face"), width, height)
        original_frame_count = len(filtered_tracks.get("pose", []))
        stored_frame_count = len(stored_tracks.get("pose", []))
        return GestureJson(
            word=word,
            duration=duration,
            fps=fps,
            metadata=GestureMetadata(
                source_video=str(video_path),
                filter_name=filter_name,
                video_width=width or None,
                video_height=height or None,
            ),
            normalization=NormalizationInfo(
                aspect_corrected=bool(width and height),
                hand_depth="world" if hand_depth else "image",
                frame=frame_edges(raw_tracks.get("pose", []), width, height),
            ),
            tracks=stored_tracks,
            features=features,
            compression=CompressionInfo(
                mode="keyframes" if compress else "none",
                epsilon=epsilon if compress else 0.0,
                original_frame_count=original_frame_count,
                stored_frame_count=stored_frame_count,
            ),
            expression=expression,
        )

    def compare_filters(self, video_path: str | Path, word: str) -> dict[str, object]:
        raw_tracks, fps, duration, width, height = self.extractor.extract(video_path)
        normalized_tracks = normalize_tracks(raw_tracks, width, height)
        reports = []
        for name in ("moving_average", "kalman", "savitzky_golay"):
            filtered = apply_filter(normalized_tracks, name)
            reports.append(compare_filter_quality(normalized_tracks["pose"], filtered["pose"], name).__dict__)
        return {"word": word, "fps": fps, "duration": duration, "reports": reports}
