import dataclasses
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """确保图像是 (H, W, C) 格式的 uint8 数组。"""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # LeRobotDataset 可能会输出 (C, H, W) 格式，需要转换
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """
    将来自 Franka 数据集或环境的输入转换为模型所需的标准格式。
    """

    # 用于处理 PI0 和 PI0_FAST 模型之间的差异
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 从 RepackTransform 后的数据中解析图像
        # 你的数据有 'image_1' 和 'image_2'
        image_1 = _parse_image(data["image_1"])
        image_2 = _parse_image(data["image_2"])

        # 构建模型输入字典。字典的键是固定的，不能改变。
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": image_1,
                "left_wrist_0_rgb": image_2,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(image_1),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 动作只在训练时存在
        if "actions" in data:
            inputs["actions"] = data["actions"]
            print("actions shape in FrankaInputs:", inputs["actions"].shape)

        # 语言指令
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """
    将模型的输出转换回特定于环境的格式（仅用于推理）。
    """

    def __call__(self, data: dict) -> dict:
        # 模型输出的动作维度可能比实际需要的大（例如，被填充到16）
        # 我们的 Franka 机器人需要 8 维动作 (7 关节 + 1 夹爪)
        # 因此，我们只取前 8 维
        print("actions shape before FrankaOutputs:", data["actions"].shape)
        return {"actions": np.asarray(data["actions"][:, :8])}