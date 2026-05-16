import asyncio
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.dao.database import init_db
from backend.routers.api import router
from backend.services.logger_service import logger
from backend.services.social_recsys import recsys




def create_app():
    # 定义lifespan函数（核心替换点）
    async def lifespan(app: FastAPI):
        # --- 启动逻辑（替代原@app.on_event("startup")）---
        logger.info("应用启动：加载推荐系统（含嵌入模型）、初始化数据库...")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: recsys.model)
        embedding_dim = recsys.model.get_sentence_embedding_dimension()
        logger.info(f"推荐系统已就绪，向量维度：{embedding_dim}")
        await init_db(embedding_dim=embedding_dim)
        yield  # 关键：yield之前是启动逻辑，之后是关闭逻辑，中间是应用运行阶段

        # --- 关闭逻辑（替代原@app.on_event("shutdown")）---
        logger.info("应用关闭：释放数据库连接、清理资源等")
        # 这里写你的关闭代码，比如：
        # await app.state.db.close()

    app = FastAPI(title="MOSS Platform", lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 包含路由
    app.include_router(router)

    # 健康检查接口需放在静态挂载之前，否则会被 "/" 挂载拦截
    @app.get("/api/v1/health")
    async def health_check():
        return {"status": "ok", "message": "FastAPI服务正常，React assets目录适配成功"}

    # 定义React静态文件目录的绝对路径
    FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "static")
    if not os.path.exists(FRONTEND_DIR):
        raise HTTPException(
            status_code=500, detail=f"React静态文件目录不存在：{FRONTEND_DIR}"
        )
    # 验证assets目录是否存在
    assets_dir = os.path.join(FRONTEND_DIR, "assets")
    if not os.path.exists(assets_dir):
        raise HTTPException(status_code=500, detail=f"React assets目录不存在：{assets_dir}")
    # 挂载React的static文件夹（匹配React引用的/static/前缀）
    # 核心配置：挂载整个frontend目录到根路径，启用html=True
    # html=True会自动：1. 识别assets下js/css的正确MIME类型 2. 处理SPA路由回退
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIR, html=True),  # html=True是关键
        name="frontend_static",
    )

    # 兜底SPA路由（备用，mount的html=True已处理，仅做双重保障）
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # 排除API路径，避免接口被SPA路由覆盖
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API接口不存在")

        # 检查请求的是否是真实存在的文件（如assets下的js/css）
        file_path = os.path.join(FRONTEND_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        # 非文件路径，返回index.html交给React路由处理
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
    return app

if __name__ == "__main__":
    import uvicorn
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    app = create_app()

    uvicorn.run(app, host="0.0.0.0", port=8000)
