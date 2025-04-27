from typing import List, Optional
from pydantic import BaseModel, Field


class BaseImageRequest(BaseModel):
    """包含输入图像的基本请求模型"""
    input_image: str = Field(..., description="Base64编码的输入图像数据")


class TaggerRequest(BaseImageRequest):
    """用于图像打标的请求模型"""
    pass # 只需要基础的 input_image


class StyleTransferRequest(BaseImageRequest):
    """用于单图风格迁移的请求模型"""
    positive_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    checkpoint_name: Optional[str] = None
    controlnet_name: Optional[str] = None
    lora_names: Optional[List[str]] = None
    seed: Optional[int] = None
    steps: Optional[int] = 20
    cfg: Optional[float] = 6.5
    sampler_name: Optional[str] = "euler"
    scheduler: Optional[str] = "normal"
    denoise: Optional[float] = 0.75
    lora_strength: Optional[float] = 1.0
    controlnet_strength: Optional[float] = 0.85
    controlnet_start_percent: Optional[float] = 0.0
    controlnet_end_percent: Optional[float] = 1.0
    output_prefix: Optional[str] = "StyleTransfer"
    use_tagger: Optional[bool] = False
    # 可以添加 workflow.py 中 create_style_transfer_workflow 支持的其他参数
    image_width: Optional[int] = 1024
    image_height: Optional[int] = 1024
    tagger_model: Optional[str] = "wd-v1-4-moat-tagger-v2"
    tagger_general_threshold: Optional[float] = 0.35
    tagger_character_threshold: Optional[float] = 0.85


class StyleTransferWithImageRequest(BaseImageRequest):
    """用于双图风格迁移（使用 IPAdapter）的请求模型"""
    style_image: str = Field(..., description="Base64编码的风格参考图像数据")
    style_prompt: str
    checkpoint_name: str
    controlnet_name: str
    # 可选参数及其默认值 (参考 workflow.py)
    positive_prompt: str = "masterpiece, best quality, high resolution, ultra-detailed, intricate details"
    negative_prompt: str = "(worst quality, low quality, normal quality:1.4), blurry, noisy, jpeg artifacts, deformed, disfigured"
    ipadapter_preset: str = "PLUS (high strength)"
    controlnet_strength: float = 0.3
    controlnet_start_percent: float = 0.0
    controlnet_end_percent: float = 0.5
    ipadapter_weight_style: float = 1.2
    ipadapter_weight_composition: float = 1.0
    ksampler1_seed: int = -1
    ksampler1_steps: int = 20
    ksampler1_cfg: float = 2.0
    ksampler1_sampler_name: str = "dpmpp_3m_sde_gpu"
    ksampler1_scheduler: str = "karras"
    ksampler1_end_at_step: int = 15
    ksampler2_seed: int = -1
    ksampler2_steps: int = 20
    ksampler2_cfg: float = 2.0
    ksampler2_sampler_name: str = "euler"
    ksampler2_scheduler: str = "normal"
    ksampler2_start_at_step: int = 15
    output_prefix: str = "StyleTransferWithImage"
    image_width: int = 1024
    image_height: int = 1024

class CombinedStyleTransferRequest(BaseImageRequest):
    """用于结合风格迁移的请求模型"""
    style_prompt: str
    checkpoint_name: str
    controlnet_name: str
    # 可选参数及其默认值 (参考 workflow.py)
    positive_prompt: str = "masterpiece, best quality, high resolution, ultra-detailed, intricate details"
    negative_prompt: str = "(worst quality, low quality, normal quality:1.4), blurry, noisy, jpeg artifacts, deformed, disfigured"
    ipadapter_preset: str = "PLUS (high strength)"
    ipadapter_weight_style: float = 1.2
    ipadapter_weight_composition: float = 1.0
    controlnet_strength: float = 0.8
    controlnet_start_percent: float = 0.0
    controlnet_end_percent: float = 0.6
    style_sampler_seed: int = -1
    style_sampler_steps: int = 5
    style_sampler_cfg: float = 1.5
    style_sampler_name: str = "euler"
    style_sampler_scheduler: str = "normal"
    main_sampler_seed: int = -1
    main_sampler_steps: int = 10
    main_sampler_cfg: float = 1.5
    main_sampler_name: str = "euler"
    main_sampler_scheduler: str = "normal"
    main_sampler_denoise: float = 0.84
    output_prefix: str = "CombinedStyleTransfer"
    image_width: int = 1024
    image_height: int = 1024
    image_scale_method: str = "nearest-exact"
    image_scale_crop: str = "disabled"
    use_tagger: bool = False
    tagger_model: str = "wd-v1-4-moat-tagger-v2"
    tagger_general_threshold: float = 0.35
    tagger_character_threshold: float = 0.85
    tagger_concat_delimiter: str = ", "
    tagger_concat_clean_whitespace: str = "true"

    
    
    
    
    

# 保留旧的 WorkflowRequest 以防万一，但建议逐步淘汰
class WorkflowRequest(BaseModel):
    input_image: Optional[str] = None  # base64编码的图片数据
    positive_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    checkpoint_name: Optional[str] = None
    controlnet_name: Optional[str] = None
    lora_names: Optional[List[str]] = None
    seed: Optional[int] = None
    steps: Optional[int] = 20
    cfg: Optional[float] = 6.5
    sampler_name: Optional[str] = "euler"
    scheduler: Optional[str] = "normal"
    denoise: Optional[float] = 0.75
    lora_strength: Optional[float] = 1.0
    controlnet_strength: Optional[float] = 0.85
    controlnet_start_percent: Optional[float] = 0
    controlnet_end_percent: Optional[float] = 1
    output_prefix: Optional[str] = "StyleTransfer"
    workflow_name: Optional[str] = None # 用于旧逻辑路由，新模型不再需要
    use_tagger: Optional[bool] = False
    # 添加 style_image 以兼容可能的旧请求，但新接口应使用专用模型
    style_image: Optional[str] = None
    style_prompt: Optional[str] = None
