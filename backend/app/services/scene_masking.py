import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from backend.app.services.runtime import env_bool, env_float, resolve_torch_device

SAM2_MASK_CARS_DEFAULT = env_bool("GEOSPY_SAM2_MASK_CARS", False)
SAM2_MASK_TREES_DEFAULT = env_bool("GEOSPY_SAM2_MASK_TREES", False)
SAM2_MODEL_ID_DEFAULT = (
    str(os.getenv("GEOSPY_SAM2_MODEL_ID", "facebook/sam2-hiera-small")).strip()
    or "facebook/sam2-hiera-small"
)
SAM2_DEVICE_DEFAULT = str(os.getenv("GEOSPY_SAM2_DEVICE", "auto")).strip().lower() or "auto"
SAM2_CAR_DETECTION_THRESHOLD_DEFAULT = env_float(
    "GEOSPY_SAM2_CAR_DETECTION_THRESHOLD",
    0.45,
    minimum=0.05,
    maximum=0.99,
)
SAM2_TARGET_LABELS = {
    str(label).strip().lower()
    for label in str(os.getenv("GEOSPY_SAM2_TARGET_LABELS", "car,truck,bus")).split(",")
    if str(label).strip()
}

_SAM2_RUNTIME_CACHE: Dict[str, Any] = {}
_SAM2_RUNTIME_LOCK = threading.Lock()
_SAM2_INFERENCE_LOCK = threading.Lock()


def _import_cv_runtime():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Scene masking dependencies are missing. Install opencv-python-headless."
        ) from exc
    return cv2, np


def _load_sam2_vehicle_runtime():
    cache_key = f"{SAM2_MODEL_ID_DEFAULT}|{SAM2_DEVICE_DEFAULT}"
    cached = _SAM2_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _SAM2_RUNTIME_LOCK:
        cached = _SAM2_RUNTIME_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_V2_Weights,
                fasterrcnn_resnet50_fpn_v2,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2 car masking is unavailable. Install the official "
                "facebookresearch/sam2 package and ensure torchvision is installed locally."
            ) from exc
        device = resolve_torch_device(torch, SAM2_DEVICE_DEFAULT)
        detector_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        detector = fasterrcnn_resnet50_fpn_v2(weights=detector_weights)
        detector.to(device)
        detector.eval()
        predictor = SAM2ImagePredictor.from_pretrained(
            SAM2_MODEL_ID_DEFAULT,
            device=device,
        )
        predictor.model.to(device)
        predictor.model.eval()
        categories = [
            str(label).strip().lower()
            for label in list(detector_weights.meta.get("categories") or [])
        ]
        target_category_ids = {
            index
            for index, label in enumerate(categories)
            if label in SAM2_TARGET_LABELS
        }
        if not target_category_ids:
            raise RuntimeError(
                "Configured SAM2 target labels are not present in the local detector categories."
            )
        runtime = {
            "torch": torch,
            "predictor": predictor,
            "detector": detector,
            "device": device,
            "target_category_ids": target_category_ids,
        }
        _SAM2_RUNTIME_CACHE[cache_key] = runtime
        return runtime


def merge_sam2_mask_stats(
    stats: Dict[str, Any], update: Dict[str, Any], *, candidate_image: bool = False
) -> None:
    stats["sam2_enabled"] = bool(
        stats.get("sam2_enabled") or update.get("sam2_enabled")
    )
    stats["sam2_mask_cars"] = bool(
        stats.get("sam2_mask_cars") or update.get("sam2_mask_cars")
    )
    stats["sam2_mask_trees"] = bool(
        stats.get("sam2_mask_trees") or update.get("sam2_mask_trees")
    )
    stats["sam2_vehicle_boxes"] = int(stats.get("sam2_vehicle_boxes") or 0) + int(
        update.get("sam2_vehicle_boxes") or 0
    )
    stats["sam2_tree_boxes"] = int(stats.get("sam2_tree_boxes") or 0) + int(
        update.get("sam2_tree_boxes") or 0
    )
    stats["sam2_masked_pixels"] = int(stats.get("sam2_masked_pixels") or 0) + int(
        update.get("sam2_masked_pixels") or 0
    )
    if candidate_image and (
        int(update.get("sam2_vehicle_boxes") or 0) > 0
        or int(update.get("sam2_tree_boxes") or 0) > 0
        or int(update.get("sam2_masked_pixels") or 0) > 0
    ):
        stats["sam2_candidate_images_masked"] = int(
            stats.get("sam2_candidate_images_masked") or 0
        ) + 1
    if str(update.get("sam2_model_id") or "").strip():
        stats["sam2_model_id"] = str(update.get("sam2_model_id"))
    if str(update.get("sam2_device") or "").strip():
        stats["sam2_device"] = str(update.get("sam2_device"))


def build_tree_prompt_boxes(cv2, np, image_bgr) -> List[Any]:
    if image_bgr is None:
        return []
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    b_channel, g_channel, r_channel = cv2.split(image_bgr)
    h_channel, s_channel, v_channel = cv2.split(image_hsv)
    green_mask = (
        (h_channel >= 24)
        & (h_channel <= 96)
        & (s_channel >= 34)
        & (v_channel >= 28)
        & (g_channel >= (r_channel + 8))
        & (g_channel >= (b_channel + 6))
    )
    vegetation_mask = (green_mask.astype(np.uint8)) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    vegetation_mask = cv2.morphologyEx(
        vegetation_mask, cv2.MORPH_OPEN, kernel, iterations=1
    )
    vegetation_mask = cv2.morphologyEx(
        vegetation_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    contours, _ = cv2.findContours(
        vegetation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    height, width = image_bgr.shape[:2]
    image_area = max(1, height * width)
    min_area = max(700, int(image_area * 0.003))
    prompt_boxes: List[Tuple[float, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(min_area):
            continue
        x, y, box_w, box_h = cv2.boundingRect(contour)
        if box_w < 20 or box_h < 20:
            continue
        prompt_boxes.append(
            (
                area,
                np.asarray(
                    [x, y, min(width - 1, x + box_w), min(height - 1, y + box_h)],
                    dtype="float32",
                ),
            )
        )
    prompt_boxes.sort(key=lambda item: float(item[0]), reverse=True)
    return [box for _, box in prompt_boxes[:8]]


def build_sam2_scene_mask(
    *,
    image_bytes: Optional[bytes] = None,
    image_bgr=None,
    mask_cars: bool,
    mask_trees: bool,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    cv2, np = _import_cv_runtime()
    runtime = _load_sam2_vehicle_runtime()
    torch = runtime["torch"]
    if image_bgr is None:
        if not image_bytes:
            raise RuntimeError("Missing image payload for SAM2 masking.")
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Failed to decode image for SAM2 masking.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detector = runtime["detector"]
    predictor = runtime["predictor"]
    target_category_ids = runtime["target_category_ids"]
    image_tensor = None
    vehicle_boxes: List[Any] = []
    tree_boxes: List[Any] = []
    if mask_cars:
        image_tensor = (
            torch.from_numpy(image_rgb)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div(255.0)
            .to(runtime["device"])
        )
    if mask_trees:
        tree_boxes = build_tree_prompt_boxes(cv2, np, image_bgr)
    with _SAM2_INFERENCE_LOCK:
        if mask_cars and image_tensor is not None:
            with torch.inference_mode():
                detections = detector([image_tensor])[0]
            scores = detections["scores"].detach().cpu().numpy()
            labels = detections["labels"].detach().cpu().numpy()
            boxes = detections["boxes"].detach().cpu().numpy()
            vehicle_boxes = [
                box.astype("float32")
                for box, score, label in zip(boxes, scores, labels)
                if float(score) >= float(SAM2_CAR_DETECTION_THRESHOLD_DEFAULT)
                and int(label) in target_category_ids
            ]
        kept_boxes = [*vehicle_boxes, *tree_boxes]
        stats = {
            "sam2_enabled": bool(mask_cars or mask_trees),
            "sam2_mask_cars": bool(mask_cars),
            "sam2_mask_trees": bool(mask_trees),
            "sam2_model_id": SAM2_MODEL_ID_DEFAULT,
            "sam2_device": runtime["device"],
            "sam2_vehicle_boxes": len(vehicle_boxes),
            "sam2_tree_boxes": len(tree_boxes),
            "sam2_masked_pixels": 0,
        }
        if not kept_boxes:
            return None, stats
        predictor.set_image(image_rgb)
        union_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        for box in kept_boxes:
            with torch.inference_mode():
                masks, _, _ = predictor.predict(
                    box=box,
                    multimask_output=False,
                )
            if masks is None or len(masks) == 0:
                continue
            union_mask |= np.asarray(masks[0], dtype=bool)
        stats["sam2_masked_pixels"] = int(union_mask.sum())
        if not union_mask.any():
            return None, stats
        return union_mask, stats
