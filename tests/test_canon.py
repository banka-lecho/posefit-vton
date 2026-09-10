import numpy as np
import pytest

from posefit.canon import (
    GUIDE_CHANNELS, U_RANGE, V_RANGE, affine_uv, box_uv, encode_uv, garment_box,
    garment_guide, mask_box, person_guide, scale_keypoints, skeleton_map, torso_frame,
)

H, W = 64, 48
SCORES = {n: 0.9 for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
                           "left_elbow", "left_wrist")}
# Торс в кадре: плечи на y=16, бёдра на y=40; левое плечо правее в кадре, как в COCO
# при взгляде на человека спереди.
KP = {"left_shoulder": (34, 16), "right_shoulder": (14, 16),
      "left_hip": (32, 40), "right_hip": (16, 40),
      "left_elbow": (40, 28), "left_wrist": (44, 40)}


def test_shoulders_and_hips_land_on_canonical_coordinates():
    frame = torso_frame(KP, SCORES, (W, H))
    uv = affine_uv(H, W, frame)
    # Пиксель под левым плечом: u=+0.5, v=0; под серединой бёдер: u=0, v=1.
    assert uv[0, 16, 34] == pytest.approx(0.5, abs=0.05)
    assert uv[1, 16, 34] == pytest.approx(0.0, abs=0.05)
    assert uv[0, 40, 24] == pytest.approx(0.0, abs=0.05)
    assert uv[1, 40, 24] == pytest.approx(1.0, abs=0.05)


def test_frame_follows_a_tilted_torso():
    # Тот же торс, повёрнутый на 30°: координаты в системе торса не меняются.
    angle = np.deg2rad(30)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centre = np.array([24.0, 28.0])
    tilted = {n: tuple(rot @ (np.array(p, dtype=float) - centre) + centre) for n, p in KP.items()}
    uv_straight = affine_uv(H, W, torso_frame(KP, SCORES, (W, H)))
    uv_tilted = affine_uv(H, W, torso_frame(tilted, SCORES, (W, H)))
    x, y = tilted["left_hip"]
    assert uv_tilted[0, int(y), int(x)] == pytest.approx(uv_straight[0, 40, 32], abs=0.1)
    assert uv_tilted[1, int(y), int(x)] == pytest.approx(uv_straight[1, 40, 32], abs=0.1)


def test_profile_view_has_no_frame():
    # Плечи в одной точке — система вырождена, нужен запасной вариант.
    profile = dict(KP, left_shoulder=(24, 16), right_shoulder=(24.5, 16))
    assert torso_frame(profile, SCORES, (W, H)) is None


def test_low_confidence_torso_has_no_frame():
    assert torso_frame(KP, dict(SCORES, left_hip=0.1), (W, H)) is None


def test_garment_box_prior_puts_collar_at_the_shoulders():
    uv = box_uv(H, W, (8, 4, 40, 44), "upper")
    assert uv[1, 4, 24] == pytest.approx(0.0, abs=0.05)          # верх выкладки — v=0
    assert uv[0, 4, 24] == pytest.approx(0.0, abs=0.05)          # середина — u=0
    assert uv[0, 4, 40] == pytest.approx(0.8, abs=0.05)          # правый край: 1.6/2
    assert uv[1, 43, 24] == pytest.approx(1.3, abs=0.05)         # подол — длина приора


def test_lower_garment_starts_near_the_hips():
    uv = box_uv(H, W, (8, 4, 40, 44), "lower")
    assert uv[1, 4, 24] == pytest.approx(0.9, abs=0.05)


def test_encoding_clips_and_scales_to_unit_range():
    uv = np.stack([np.full((H, W), 10.0, np.float32), np.full((H, W), -10.0, np.float32)])
    encoded = encode_uv(uv)
    assert encoded.min() >= -1.0 and encoded.max() <= 1.0
    assert encoded[0].max() == pytest.approx(1.0)
    assert encoded[1].min() == pytest.approx(-1.0)
    mid = encode_uv(np.stack([np.zeros((H, W), np.float32),
                              np.full((H, W), (V_RANGE[0] + V_RANGE[1]) / 2, np.float32)]))
    assert np.allclose(mid, 0.0)
    assert U_RANGE[0] == -U_RANGE[1]


def test_skeleton_draws_only_confident_limbs():
    drawn = skeleton_map(H, W, KP, SCORES)
    assert drawn.max() > 0.9                       # LINE_AA сглаживает край
    assert drawn[16, 24] > 0.5                       # линия между плечами
    assert drawn[40, 44] > 0.5                       # запястье
    nothing = skeleton_map(H, W, KP, {n: 0.0 for n in SCORES})
    assert nothing.max() == 0.0


def test_person_guide_falls_back_to_the_mask_box():
    mask = np.zeros((H, W), np.uint8)
    mask[10:50, 12:36] = 1
    guide = person_guide(H, W, {}, {}, "upper", fallback_mask=mask)
    assert guide.shape == (GUIDE_CHANNELS, H, W)
    # Верх маски — v=0 -> закодировано как (0 - 1) / 2 = -0.5.
    assert guide[1, 10, 24] == pytest.approx(-0.5, abs=0.05)
    assert guide[2].max() == 0.0                     # скелета без позы нет


def test_person_guide_uses_the_pose_when_present():
    guide = person_guide(H, W, KP, SCORES, "upper", fallback_mask=None)
    assert guide[0, 16, 34] == pytest.approx(0.25, abs=0.03)   # u=0.5 -> 0.25
    assert guide[2].max() > 0.5


def test_garment_guide_has_an_empty_skeleton_channel():
    guide = garment_guide(H, W, (8, 4, 40, 44), "dress")
    assert guide.shape == (GUIDE_CHANNELS, H, W)
    assert guide[2].max() == 0.0


def test_garment_box_prefers_parse_then_dark_pixels():
    from posefit.preprocess import ATR

    parse = np.zeros((H, W), np.uint8)
    parse[5:45, 10:38] = ATR["upper_clothes"]
    assert garment_box(parse, "upper", min_px=100) == (10.0, 5.0, 38.0, 45.0)

    white = np.full((H, W, 3), 255, np.uint8)
    white[20:30, 20:30] = 0
    assert garment_box(np.zeros((H, W), np.uint8), "upper", image=white, min_px=50) \
        == (20.0, 20.0, 30.0, 30.0)
    assert garment_box(np.zeros((H, W), np.uint8), "upper", image=None) is None


def test_mask_box_and_keypoint_scaling():
    assert mask_box(np.zeros((4, 4))) is None
    scaled = scale_keypoints({"nose": (768, 512)}, (768, 1024), (384, 512))
    assert scaled["nose"] == (384.0, 256.0)
