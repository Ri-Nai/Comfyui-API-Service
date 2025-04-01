import logging
import base64
import uuid
from fastapi import APIRouter, HTTPException, File, UploadFile
from PIL import Image
import io

from app.models.requests import WorkflowRequest
from app.services.comfyui import ComfyUIService
from app.services.workflow import create_style_transfer_workflow

logger = logging.getLogger(__name__)
router = APIRouter()
comfyui_service = ComfyUIService()

@router.post("/generate")
async def generate_image(request: WorkflowRequest):
    try:
        logger.info("开始生成图像...")
        
        # 解码base64图片数据
        try:
            image_data = base64.b64decode(request.input_image)
            # 使用PIL验证图片数据
            image = Image.open(io.BytesIO(image_data))
            # 重新编码为PNG格式
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            filename = f"input_{uuid.uuid4()}.png"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

        # 上传图片到ComfyUI
        logger.info("正在上传图片到ComfyUI...")
        uploaded_filename = await comfyui_service.upload_image(img_byte_arr, filename)
        logger.info(f"图片上传成功: {uploaded_filename}")

        # 创建workflow
        workflow = create_style_transfer_workflow(
            input_image=uploaded_filename,
            positive_prompt=request.positive_prompt,
            negative_prompt=request.negative_prompt,
            checkpoint_name=request.checkpoint_name,
            controlnet_name=request.controlnet_name,
            lora_names=request.lora_names,
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
        )

        logger.info("成功创建workflow")

        # 发送提示到ComfyUI
        queue_response = await comfyui_service.queue_prompt(workflow)
        prompt_id = queue_response["prompt_id"]
        logger.info(f"成功提交提示，ID: {prompt_id}")

        # 等待生成完成
        await comfyui_service.wait_for_prompt(prompt_id)

        # 获取生成历史
        history = await comfyui_service.get_history(prompt_id)
        logger.info(f"成功获取历史记录，提示ID: {prompt_id}")

        # 处理输出
        output_images = await comfyui_service.process_output(prompt_id, history)

        return {
            "status": "success",
            "message": "图像生成完成",
            "prompt_id": prompt_id,
            "images": output_images,
        }

    except Exception as e:
        logger.error(f"生成图像时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """上传图片到ComfyUI"""
    try:
        # 读取文件内容
        file_content = await file.read()
        return await comfyui_service.upload_image(file_content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")

@router.get("/health")
async def health_check():
    return {"status": "healthy"} 
