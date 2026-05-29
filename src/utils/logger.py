import sys
from loguru import logger

def setup_logger():
    # 1. 移除 Loguru 默认的控制台处理器
    logger.remove()

    # 2. 配置标准控制台输出（开发时看，带炫酷颜色）
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        enqueue=True # 开启异步队列，保证线程安全
    )

    # 3. 配置生产级常规日志文件（每天滚动，保留10天）
    logger.add(
        "logs/app.log",
        rotation="00:00",          # 每天午夜 12 点创建新日志文件
        retention="10 days",       # 历史日志保留 10 天
        compression="zip",         # 旧日志自动压缩成 zip 释放空间
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        enqueue=True
    )

    # 4. 专门开辟一个 ERROR 级别日志，用于生产环境监控报警
    logger.add(
        "logs/error.log",
        rotation="100 MB",         # 按文件大小滚动
        retention="30 days",
        level="ERROR",
        backtrace=True,            # 打印详细反向追踪
        diagnose=True,             # 生产环境如怕泄露敏感变量可设为 False
        enqueue=True
    )

# 初始化
setup_logger()