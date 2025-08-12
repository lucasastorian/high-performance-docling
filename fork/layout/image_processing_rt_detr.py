# optimized_rtdetr_image_processor.py
# CPU-only parity-preserving optimized fork of RTDetrImageProcessor
# Tested with transformers >= 4.40

import pathlib
from typing import Optional, Union, List, Tuple
import numpy as np

from transformers.image_processing_utils import BaseImageProcessor, get_size_dict
from transformers.image_utils import (
    AnnotationFormat,
    AnnotationType,
    ChannelDimension,
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_annotations,
    validate_preprocess_arguments,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from transformers.image_transforms import (
    PILImageResampling,
    resize as hf_resize,
)
from transformers.models.detr.image_processing_detr import (
    get_resize_output_image_size,
    get_image_size_for_max_height_width,
    corners_to_center_format,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging, is_torch_available

logger = logging.get_logger(__name__)

# Match HF original: only COCO_DETECTION supported
SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION,)


class OptimizedRTDetrImageProcessor(BaseImageProcessor):
    """
    CPU-only optimized fork with parity to HF's RTDetrImageProcessor.
    - Uses HF resize (PIL semantics) for geometry parity.
    - Batches NumPy rescale+normalize (BHWC) to reduce Python overhead.
    - Batch pads + builds masks in one pass.
    - Keeps annotation behavior equivalent to HF for boxes/area; masks padding is optional (commented).
    """

    model_input_names = ["pixel_values", "pixel_mask"]

    def __init__(
        self,
        format: Union[str, AnnotationFormat] = AnnotationFormat.COCO_DETECTION,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = False,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_annotations: bool = True,
        do_pad: bool = False,
        pad_size: Optional[dict[str, int]] = None,
        **kwargs,
    ) -> None:
        size = size if size is not None else {"height": 640, "width": 640}
        size = get_size_dict(size, default_to_square=False)

        if do_convert_annotations is None:
            do_convert_annotations = do_normalize

        super().__init__(**kwargs)
        self.format = AnnotationFormat(format)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_convert_annotations = do_convert_annotations
        self.image_mean = IMAGENET_DEFAULT_MEAN if image_mean is None else image_mean
        self.image_std = IMAGENET_DEFAULT_STD if image_std is None else image_std
        self.do_pad = do_pad
        self.pad_size = pad_size

    # ---------------- Annotation helpers (HF parity) ----------------

    @staticmethod
    def normalize_annotation(annotation: dict, image_size: tuple[int, int]) -> dict:
        """Same as HF normalize_annotation: boxes (xyxy abs) -> (cx,cy,w,h) relative."""
        image_height, image_width = image_size
        norm = {}
        for k, v in annotation.items():
            if k == "boxes":
                boxes = corners_to_center_format(v)
                boxes = boxes / np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
                norm[k] = boxes
            else:
                norm[k] = v
        return norm

    @staticmethod
    def prepare_coco_detection_annotation(
        image: np.ndarray,
        target: dict,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> dict:
        """Equivalent to HF prepare_coco_detection_annotation (no masks/keypoints to keep it lean)."""
        image_height, image_width = get_image_size(image, channel_dim=input_data_format)

        annotations = target["annotations"]
        annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

        classes = np.asarray([obj["category_id"] for obj in annotations], dtype=np.int64)
        area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
        iscrowd = np.asarray([obj.get("iscrowd", 0) for obj in annotations], dtype=np.int64)

        boxes = [obj["bbox"] for obj in annotations]
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
        boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        new_target = {
            "image_id": np.asarray([target["image_id"]], dtype=np.int64),
            "class_labels": classes[keep],
            "boxes": boxes[keep],
            "area": area[keep],
            "iscrowd": iscrowd[keep],
            "orig_size": np.asarray([int(image_height), int(image_width)], dtype=np.int64),
        }
        # If you use keypoints/masks, mirror HF here (omitted for inference parity + speed)
        return new_target

    def prepare_annotation(
        self,
        image: np.ndarray,
        target: dict,
        format: Optional[AnnotationFormat] = None,
        return_segmentation_masks: Optional[bool] = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> dict:
        fmt = self.format if format is None else AnnotationFormat(format)
        if fmt != AnnotationFormat.COCO_DETECTION:
            raise ValueError(f"Unsupported format {fmt}")
        return self.prepare_coco_detection_annotation(image, target, input_data_format=input_data_format)

    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """HF-parity resize selection + hf_resize call."""
        if "max_size" in kwargs:
            logger.warning_once(
                "The `max_size` parameter is deprecated. Use size['longest_edge'] instead."
            )
            max_size = kwargs.pop("max_size")
        else:
            max_size = None
        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        if "shortest_edge" in size and "longest_edge" in size:
            new_size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        elif "max_height" in size and "max_width" in size:
            new_size = get_image_size_for_max_height_width(
                image, size["max_height"], size["max_width"], input_data_format=input_data_format
            )
        elif "height" in size and "width" in size:
            new_size = (size["height"], size["width"])
        else:
            raise ValueError(f"Invalid size spec: {size.keys()}")
        return hf_resize(
            image,
            size=new_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def resize_annotation(
        self,
        annotation: dict,
        orig_size: tuple[int, int],
        size: tuple[int, int],
        threshold: float = 0.5,
        resample: PILImageResampling = PILImageResampling.NEAREST,
    ) -> dict:
        """Match HF resize_annotation for boxes/area (masks omitted by default for speed)."""
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, orig_size))
        ratio_h, ratio_w = ratios
        new_annotation = {"size": size}
        for key, value in annotation.items():
            if key == "boxes":
                boxes = value * np.asarray([ratio_w, ratio_h, ratio_w, ratio_h], dtype=np.float32)
                new_annotation["boxes"] = boxes
            elif key == "area":
                new_annotation["area"] = value * (ratio_w * ratio_h)
            elif key == "size":
                new_annotation["size"] = size
            else:
                new_annotation[key] = value
        return new_annotation

    # ---------------- Fast-path fused CPU ops ----------------

    @staticmethod
    def _ensure_hwc_float32(im: np.ndarray, input_data_format) -> np.ndarray:
        """Return HxWxC float32; expand channel if grayscale; no 1->3 upconvert."""
        arr = im
        if input_data_format == ChannelDimension.FIRST:
            if arr.ndim == 2:
                arr = arr[None, ...]
            arr = np.transpose(arr, (1, 2, 0))
        elif input_data_format == ChannelDimension.LAST:
            if arr.ndim == 2:
                arr = arr[..., None]
        elif input_data_format == ChannelDimension.NONE:
            arr = arr[..., None]
        else:
            if arr.ndim == 2:
                arr = arr[..., None]
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _hwc_to_chw(im_hwc: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.transpose(im_hwc, (2, 0, 1)))

    @staticmethod
    def _np_stack_float32(images: List[np.ndarray]) -> np.ndarray:
        return np.ascontiguousarray(np.stack(images, axis=0))

    @staticmethod
    def _fused_rescale_normalize_bhwc(
        x_bhwc: np.ndarray,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        mean: Union[List[float], Tuple[float, ...], float],
        std: Union[List[float], Tuple[float, ...], float],
    ) -> None:
        """In-place fused rescale + normalize on BHWC float32."""
        if do_rescale and rescale_factor != 1.0:
            np.multiply(x_bhwc, float(rescale_factor), out=x_bhwc)
        if do_normalize:
            mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, -1)
            std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, -1)
            if mean_arr.shape[-1] == 1 and x_bhwc.shape[-1] > 1:
                mean_arr = np.repeat(mean_arr, x_bhwc.shape[-1], axis=-1)
                std_arr = np.repeat(std_arr, x_bhwc.shape[-1], axis=-1)
            np.subtract(x_bhwc, mean_arr, out=x_bhwc)
            np.divide(x_bhwc, std_arr, out=x_bhwc)

    @staticmethod
    def _pad_batch_bhwc(
        images_hwc: List[np.ndarray],
        padded_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """Pad list of HWC float32 to BHWC float32 and build mask BHW (int64)."""
        B = len(images_hwc)
        Hpad, Wpad = padded_size
        C = images_hwc[0].shape[2]
        out = np.zeros((B, Hpad, Wpad, C), dtype=np.float32)
        mask = np.zeros((B, Hpad, Wpad), dtype=np.int64)
        orig_sizes: List[Tuple[int, int]] = []
        for i, im in enumerate(images_hwc):
            h, w, _ = im.shape
            out[i, :h, :w, :] = im
            mask[i, :h, :w] = 1
            orig_sizes.append((h, w))
        return out, mask, orig_sizes

    # ---------------- Main preprocess ----------------

    def preprocess(
        self,
        images: ImageInput,
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,
        return_segmentation_masks: Optional[bool] = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample=None,  # PILImageResampling
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        do_convert_annotations: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        format: Optional[Union[str, AnnotationFormat]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        pad_size: Optional[dict[str, int]] = None,
    ) -> BatchFeature:
        # ---- Resolve options exactly like HF ----
        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        size = get_size_dict(size=size, default_to_square=True)
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        do_convert_annotations = (
            self.do_convert_annotations if do_convert_annotations is None else do_convert_annotations
        )
        do_pad = self.do_pad if do_pad is None else do_pad
        pad_size = self.pad_size if pad_size is None else pad_size
        fmt = self.format if format is None else AnnotationFormat(format)

        imgs = make_list_of_images(images)
        if not valid_images(imgs):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if annotations is not None and isinstance(annotations, dict):
            annotations = [annotations]
        if annotations is not None and len(imgs) != len(annotations):
            raise ValueError(f"The number of images ({len(imgs)}) and annotations ({len(annotations)}) do not match.")

        if annotations is not None:
            validate_annotations(fmt, SUPPORTED_ANNOTATION_FORMATS, annotations)

        # Convert all to numpy once
        imgs_np = [to_numpy_array(im) for im in imgs]

        if do_rescale and is_scaled_image(imgs_np[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input images have pixel "
                "values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(imgs_np[0])

        # ---- Prepare annotations (COCO->DETR) ----
        if annotations is not None:
            prepared_images = []
            prepared_annotations = []
            for image, target in zip(imgs_np, annotations):
                target = self.prepare_annotation(
                    image,
                    target,
                    format=fmt,
                    return_segmentation_masks=return_segmentation_masks,
                    masks_path=masks_path,
                    input_data_format=input_data_format,
                )
                prepared_images.append(image)
                prepared_annotations.append(target)
            imgs_np = prepared_images
            annotations = prepared_annotations

        # ---- Resize with HF semantics ----
        if do_resize:
            if annotations is not None:
                resized_images, resized_annotations = [], []
                for image, target in zip(imgs_np, annotations):
                    orig_size = get_image_size(image, input_data_format)
                    resized_image = self.resize(
                        image, size=size, resample=resample, input_data_format=input_data_format
                    )
                    resized_annotation = self.resize_annotation(
                        target, orig_size, get_image_size(resized_image, input_data_format)
                    )
                    resized_images.append(resized_image)
                    resized_annotations.append(resized_annotation)
                imgs_np = resized_images
                annotations = resized_annotations
            else:
                imgs_np = [
                    self.resize(image, size=size, resample=resample, input_data_format=input_data_format)
                    for image in imgs_np
                ]

        # ---- Fused rescale+normalize (batched by equal H,W,C), keep HWC during math ----
        buckets: dict[tuple[int, int, int], list[int]] = {}
        hwc_list: List[np.ndarray] = [None] * len(imgs_np)  # type: ignore
        for i, im in enumerate(imgs_np):
            im_hwc = self._ensure_hwc_float32(im, input_data_format)
            hwc_list[i] = im_hwc
            key = (im_hwc.shape[0], im_hwc.shape[1], im_hwc.shape[2])
            buckets.setdefault(key, []).append(i)

        processed: List[np.ndarray] = [None] * len(hwc_list)  # type: ignore
        for _, idxs in buckets.items():
            group = [hwc_list[i] for i in idxs]
            x = self._np_stack_float32(group)  # BHWC
            self._fused_rescale_normalize_bhwc(
                x, do_rescale, float(rescale_factor), do_normalize, image_mean, image_std
            )
            for off, i in enumerate(idxs):
                processed[i] = x[off]
        imgs_cl = processed  # list of HWC float32 post-fused ops

        # ---- Convert annotations to normalized relative coords if requested ----
        if do_convert_annotations and annotations is not None:
            annotations = [
                self.normalize_annotation(anno, (im.shape[0], im.shape[1]))
                for anno, im in zip(annotations, imgs_cl)
            ]

        # ---- Pad (batched) if requested ----
        if do_pad:
            if pad_size is not None:
                padded_size = (int(pad_size["height"]), int(pad_size["width"]))
            else:
                hmax = max(im.shape[0] for im in imgs_cl)
                wmax = max(im.shape[1] for im in imgs_cl)
                padded_size = (hmax, wmax)

            pixel_values_cl, pixel_mask, orig_sizes = self._pad_batch_bhwc(imgs_cl, padded_size)

            # Update annotations to padded size (HF parity for boxes/area)
            if annotations is not None:
                padded_annotations = []
                out_h, out_w = padded_size
                for (in_h, in_w), anno in zip(orig_sizes, annotations):
                    new_anno = dict(anno)
                    if "boxes" in new_anno:
                        scale = np.asarray([in_w / out_w, in_h / out_h, in_w / out_w, in_h / out_h], dtype=np.float32)
                        new_anno["boxes"] = new_anno["boxes"] * scale
                    # Masks parity would require padding mask arrays too (omitted for speed)
                    new_anno["size"] = (out_h, out_w)
                    padded_annotations.append(new_anno)
                annotations = padded_annotations

            # Channel move to requested data_format (batched)
            if data_format == ChannelDimension.FIRST:
                pixel_values = np.ascontiguousarray(np.transpose(pixel_values_cl, (0, 3, 1, 2)))  # BCHW
            else:
                pixel_values = pixel_values_cl  # BHWC
            data = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
            encoded = BatchFeature(data=data, tensor_type=return_tensors)

        else:
            # No pad: return list of arrays in requested format (HF parity)
            if data_format == ChannelDimension.FIRST:
                pixel_values_list = [self._hwc_to_chw(im) for im in imgs_cl]  # list of CHW
            else:
                pixel_values_list = imgs_cl  # list of HWC
            encoded = BatchFeature(data={"pixel_values": pixel_values_list}, tensor_type=return_tensors)

        # Attach labels if present
        if annotations is not None:
            encoded["labels"] = [BatchFeature(anno, tensor_type=return_tensors) for anno in annotations]

        return encoded
