import time
from typing import Optional
from urllib.parse import urlparse

import httpx

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BrowserLikeFetcher:
    """
    浏览器模拟 HTTP 抓取工具

    功能特性：
    - 浏览器 User-Agent / Accept 等常用头
    - 支持自动重试
    - 简单的按域名限频（避免短时间内过多请求触发风控）
    """

    # 简单的按域名节流：记录上次请求时间
    _last_request_ts: dict[str, float] = {}

    # 域名最小请求间隔（秒）
    _min_interval_seconds: float = 3.0

    # 默认请求超时时间
    _timeout_seconds: float = 30.0

    @classmethod
    def _get_default_headers(cls) -> dict[str, str]:
        """构造类似浏览器的默认请求头"""
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8,"
                "application/signed-exchange;v=b3;q=0.7"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Connection": "keep-alive",
        }

    @classmethod
    def _throttle(cls, url: str) -> None:
        """对同一域名做简单的最小间隔控制"""
        try:
            host = urlparse(url).netloc
        except Exception:
            return

        if not host:
            return

        now = time.time()
        last_ts = cls._last_request_ts.get(host)

        if last_ts is not None:
            delta = now - last_ts
            if delta < cls._min_interval_seconds:
                sleep_time = cls._min_interval_seconds - delta
                logger.info(
                    f"[BrowserLikeFetcher] 对域名 {host} 进行节流，"
                    f"睡眠 {sleep_time:.2f}s 以降低被风控风险"
                )
                time.sleep(max(0.0, sleep_time))

        cls._last_request_ts[host] = time.time()

    @classmethod
    def fetch(cls, url: str, *, timeout: Optional[float] = None, max_retries: int = 2, proxies: Optional[dict[str, str]] = None) -> str:
        """
        以浏览器风格抓取网页 HTML 文本

        Args:
            url: 目标 URL
            timeout: 超时时间（秒），默认 30s
            max_retries: 最大重试次数（总请求次数 = 1 + max_retries）
            proxies: 可选的代理配置，兼容 httpx 格式：
                e.g. {"http": "http://user:pass@proxy:port", "https": "..."}
        """
        if not url:
            raise ValueError("URL 不能为空")

        timeout_val = timeout or cls._timeout_seconds

        headers = cls._get_default_headers()

        attempt = 0
        last_err: Optional[Exception] = None

        while attempt <= max_retries:
            attempt += 1
            try:
                cls._throttle(url)

                logger.info(
                    f"[BrowserLikeFetcher] 开始抓取 | url={url} | attempt={attempt}/{1 + max_retries}"
                )

                with httpx.Client(headers=headers, timeout=timeout_val, follow_redirects=True) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    logger.info(f"[BrowserLikeFetcher] 抓取成功 | url={url} | status={resp.status_code}")
                    return resp.text

            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(
                    f"[BrowserLikeFetcher] 抓取失败 | url={url} | attempt={attempt} | err={e}"
                )
                # 失败后稍作等待再重试，避免频繁访问
                if attempt <= max_retries:
                    time.sleep(1.0)

        # 所有重试失败，抛出异常
        raise ValueError(f"Failed to fetch url after {1 + max_retries} attempts: {last_err}")


__all__ = ["BrowserLikeFetcher"]
