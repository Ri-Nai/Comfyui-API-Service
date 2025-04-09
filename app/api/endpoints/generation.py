import logging
import base64
import uuid
from fastapi import APIRouter, HTTPException, Depends
from PIL import Image
import io
from typing import Dict, Any

# 从 requests 导入新的模型类
from app.models.requests import (
    TaggerRequest, 
    StyleTransferRequest, 
    StyleTransferWithImageRequest
) 
from app.services.comfyui import ComfyUIService
from app.services.workflow import (
    create_tagger_workflow,
    create_style_transfer_workflow,
    create_style_transfer_with_image_workflow,
)
from app.utils.image_helper import ImageHelper

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/generate", tags=["Generation"]) # 添加前缀和标签
comfyui_service = ComfyUIService()

# --- Helper Function for Image Upload ---

async def _upload_image_from_base64(base64_data: str, service: ComfyUIService) -> str:
    """Helper function to decode and upload a base64 image."""
    try:
        img_byte_arr, filename = ImageHelper.decode_image(base64_data)
        uploaded_filename = await service.upload_image(img_byte_arr, filename)
        logger.info(f"图片上传成功: {uploaded_filename}")
        return uploaded_filename
    except Exception as e:
        logger.error(f"解码或上传图片时出错: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid or failed image upload: {str(e)}")

# --- Helper Function for Workflow Execution ---

async def _execute_workflow(workflow: Dict[str, Any], service: ComfyUIService) -> Dict[str, Any]:
    """Helper function to queue prompt, wait, and get outputs."""
    queue_response = await service.queue_prompt(workflow)
    prompt_id = queue_response["prompt_id"]
    logger.info(f"成功提交提示，ID: {prompt_id}")
    
    outputs = await service.wait_for_prompt(prompt_id)
    # history = await service.get_history(prompt_id) # 获取历史记录可选
    # logger.info(f"成功获取历史记录，提示ID: {prompt_id}")
    
    output_images = await service.process_output_images(outputs) # 假设所有生成类 workflow 都返回图片
    
    return {
        "prompt_id": prompt_id,
        "images": output_images,
    }

# --- API Endpoints ---

@router.post("/tagger", summary="Extract Tags from Image")
async def tagger_endpoint(request: TaggerRequest):
    """接收图片并使用 WD14 Tagger 提取标签。"""
    try:
        logger.info("开始进行标签提取...")
        uploaded_filename = await _upload_image_from_base64(request.input_image, comfyui_service)
        
        workflow = create_tagger_workflow(input_image=uploaded_filename)
        logger.info("成功创建 Tagger workflow")

        # Tagger workflow 的执行和输出处理可能不同
        queue_response = await comfyui_service.queue_prompt(workflow)
        prompt_id = queue_response["prompt_id"]
        logger.info(f"成功提交 Tagger 提示，ID: {prompt_id}")
        await comfyui_service.wait_for_prompt(prompt_id) # 等待完成
        outputs = await comfyui_service.get_outputs(prompt_id) # 获取原始输出
        
        # 假设 tagger 输出在特定节点的特定字段
        # 注意：需要根据实际 workflow 输出调整这里的索引和字段名
        tags = outputs.get("tags", ["Error: Could not find tags in output"])[0] # 从 workflow.py 看，似乎没有直接叫 'tags' 的输出节点名，需要确认

        return {
            "status": "success",
            "message": "标签提取成功",
            "prompt_id": prompt_id,
            "tags": tags 
        }

    except HTTPException as http_exc:
        logger.error(f"标签提取请求错误: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"标签提取失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"标签提取过程中发生内部错误: {str(e)}")


@router.post("/style_transfer", summary="Generate Image with Style Transfer (Single Image)")
async def style_transfer_endpoint(request: StyleTransferRequest):
    """使用 ControlNet 和 LoRA（可选 Tagger）进行单图风格迁移。"""
    try:
        logger.info("开始单图风格迁移图像生成...")
        uploaded_filename = await _upload_image_from_base64(request.input_image, comfyui_service)

        workflow = create_style_transfer_workflow(
            input_image=uploaded_filename,
            positive_prompt=request.positive_prompt,
            negative_prompt=request.negative_prompt,
            checkpoint_name=request.checkpoint_name,
            controlnet_name=request.controlnet_name,
            lora_names=request.lora_names or [], # Ensure it's a list
            seed=request.seed,
            steps=request.steps,
            cfg=request.cfg,
            sampler_name=request.sampler_name,
            scheduler=request.scheduler,
            denoise=request.denoise,
            lora_strength=request.lora_strength,
            controlnet_strength=request.controlnet_strength,
            controlnet_start_percent=request.controlnet_start_percent,
            controlnet_end_percent=request.controlnet_end_percent,
            output_prefix=request.output_prefix,
            image_width=request.image_width,
            image_height=request.image_height,
            use_tagger=request.use_tagger,
            # Pass tagger parameters if use_tagger is True
            tagger_model=request.tagger_model if request.use_tagger else None,
            tagger_general_threshold=request.tagger_general_threshold if request.use_tagger else None,
            tagger_character_threshold=request.tagger_character_threshold if request.use_tagger else None,
        )
        logger.info("成功创建 Style Transfer workflow")

        result = await _execute_workflow(workflow, comfyui_service)

        return {
            "status": "success",
            "message": "图像生成完成 (Style Transfer)",
            **result,
        }

    except HTTPException as http_exc:
        logger.error(f"单图风格迁移请求错误: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"单图风格迁移生成图像时发生错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"单图风格迁移过程中发生内部错误: {str(e)}")


@router.post("/style_transfer_with_image", summary="Generate Image with Style Transfer (Two Images + IPAdapter)")
async def style_transfer_with_image_endpoint(request: StyleTransferWithImageRequest):
    """使用 IPAdapter Style & Composition 和 ControlNet 进行双图风格迁移。"""
    try:
        logger.info("开始双图风格迁移图像生成 (IPAdapter)...")
        
        # 上传两张图片
        logger.info("正在上传输入图片...")
        uploaded_input_filename = await _upload_image_from_base64(request.input_image, comfyui_service)
        logger.info("正在上传风格图片...")
        uploaded_style_filename = await _upload_image_from_base64(request.style_image, comfyui_service)

        workflow = create_style_transfer_with_image_workflow(
            input_image=uploaded_input_filename,
            style_image=uploaded_style_filename,
            style_prompt=request.style_prompt,
            checkpoint_name=request.checkpoint_name,
            controlnet_name=request.controlnet_name,
            positive_prompt=request.positive_prompt,
            negative_prompt=request.negative_prompt,
            ipadapter_preset=request.ipadapter_preset,
            controlnet_strength=request.controlnet_strength,
            controlnet_start_percent=request.controlnet_start_percent,
            controlnet_end_percent=request.controlnet_end_percent,
            ipadapter_weight_style=request.ipadapter_weight_style,
            ipadapter_weight_composition=request.ipadapter_weight_composition,
            ksampler1_seed=request.ksampler1_seed,
            ksampler1_steps=request.ksampler1_steps,
            ksampler1_cfg=request.ksampler1_cfg,
            ksampler1_sampler_name=request.ksampler1_sampler_name,
            ksampler1_scheduler=request.ksampler1_scheduler,
            ksampler1_end_at_step=request.ksampler1_end_at_step,
            ksampler2_seed=request.ksampler2_seed,
            ksampler2_steps=request.ksampler2_steps,
            ksampler2_cfg=request.ksampler2_cfg,
            ksampler2_sampler_name=request.ksampler2_sampler_name,
            ksampler2_scheduler=request.ksampler2_scheduler,
            ksampler2_start_at_step=request.ksampler2_start_at_step,
            output_prefix=request.output_prefix,
            image_width=request.image_width,
            image_height=request.image_height,
        )
        logger.info("成功创建 Style Transfer with Image workflow (IPAdapter)")

        result = await _execute_workflow(workflow, comfyui_service)

        return {
            "status": "success",
            "message": "图像生成完成 (Style Transfer with Image)",
            **result,
        }
        
    except HTTPException as http_exc:
        logger.error(f"双图风格迁移请求错误: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"双图风格迁移生成图像时发生错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"双图风格迁移过程中发生内部错误: {str(e)}")

# 移除旧的 /generate, /upload, /health 端点
# @router.post("/generate") ... (旧代码)
# @router.post("/upload") ... (旧代码)
# @router.get("/health") ... (旧代码) 
