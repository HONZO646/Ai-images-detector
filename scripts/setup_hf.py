#!/usr/bin/env python3
"""
Скрипт настройки HuggingFace для доступа к ImageNet-1k
"""
import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_hf_login():
    """Проверка аутентификации HuggingFace"""
    
    logger.info("=" * 60)
    logger.info("ПРОВЕРКА HUGGINGFACE АУТЕНТИФИКАЦИИ")
    logger.info("=" * 60)
    
    # Проверка токена
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    if token:
        logger.info("✅ HF_TOKEN найден в环境变量")
    else:
        logger.warning("⚠️ HF_TOKEN не найден")
        logger.info("\nДля получения токена:")
        logger.info("  1. Зайдите на https://huggingface.co/settings/tokens")
        logger.info("  2. Создайте новый токен (тип: Read)")
        logger.info("  3. Скопируйте токен")
        logger.info("\nЗатем выполните:")
        logger.info("  huggingface-cli login")
        logger.info("\nИли установите环境变量:")
        logger.info("  set HF_TOKEN=your_token_here  (Windows)")
        logger.info("  export HF_TOKEN=your_token_here  (Linux/Mac)")
    
    # Проверка доступа к ImageNet
    logger.info("\n" + "=" * 60)
    logger.info("ПРОВЕРКА ДОСТУПА К IMAGENET-1K")
    logger.info("=" * 60)
    logger.info("\nВажно: ImageNet-1k — gated dataset!")
    logger.info("Необходимо принять лицензию:")
    logger.info("  👉 https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    logger.info("\nНажмите кнопку 'Agree' или 'Accept Terms' на странице")
    
    # Попытка проверки доступа
    try:
        from huggingface_hub import HfApi, login
        api = HfApi()
        
        # Проверка dataset info
        try:
            info = api.dataset_info("ILSVRC/imagenet-1k")
            logger.info(f"\n✅ Dataset найден: {info.id}")
            logger.info(f"   Last modified: {info.last_modified}")
        except Exception as e:
            logger.error(f"\n❌ Нет доступа к ImageNet-1k: {e}")
            logger.info("   Убедитесь что:")
            logger.info("   1. Приняли лицензию на странице dataset")
            logger.info("   2. Выполнены вход через huggingface-cli login")
            
    except ImportError:
        logger.warning("⚠️ huggingface_hub не установлен")
        logger.info("  Установите: pip install huggingface_hub")


def main():
    check_hf_login()
    
    logger.info("\n" + "=" * 60)
    logger.info("ИНСТРУКЦИЯ ПО НАСТРОЙКЕ")
    logger.info("=" * 60)
    
    print("""
1️⃣  Примите лицензию ImageNet-1k:
    🔗 https://huggingface.co/datasets/ILSVRC/imagenet-1k
    
2️⃣  Получите HF токен:
    🔗 https://huggingface.co/settings/tokens
    
3️⃣  Войдите через CLI:
    huggingface-cli login
    
4️⃣  Или установите HF_TOKEN:
    Windows:  set HF_TOKEN=your_token
    Linux:    export HF_TOKEN=your_token
    
5️⃣  Запустите обучение:
    python train.py
""")


if __name__ == '__main__':
    main()
