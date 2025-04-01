import logging
from fastapi import FastAPI
from app.api.endpoints import generation

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(title="ComfyUI API Service")
    
    # 注册路由
    app.include_router(generation.router, prefix="/api/v1", tags=["generation"])
    
    return app 
