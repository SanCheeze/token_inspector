from typing import Any, Awaitable, Callable, Dict, Optional
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject
from cachetools import TTLCache


def throttled(rate: int, on_throttle: Callable or None = None):
    """Throttled decorator, must be above router/dispatcher decorator!!!

    Args:
        rate (int): Determines how long the user will have to wait (seconds) to use again.
        on_throttle (Callable or None, optional): Callback function if rate limit exceed. Defaults to None.
    """

    def decorator(func):
        setattr(func, "rate", rate)
        setattr(func, "on_throttle", on_throttle)
        return func

    return decorator


class ThrottlingMiddleware(BaseMiddleware):
    """
    Throttling middleware which is used as an anti-spam tool
    1. Initialize and attach to router/dispatcher
    2. Use @throttled(rate, on_throttle) above router decorator
    """

    def __init__(self):
        # self.cache = TTLCache(maxsize=10_000, ttl=self.delay)
        self.caches = dict()

    @staticmethod
    def _get_throttle_key(event: TelegramObject) -> Optional[int]:
        if hasattr(event, "from_user") and event.from_user:
            return event.from_user.id
        if hasattr(event, "chat") and event.chat:
            return event.chat.id
        return None

    async def __call__(
            self,
            handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
            event: TelegramObject,
            data: Dict[str, Any],
    ) -> Any:

        decorated_func = data["handler"].callback
        rate = getattr(decorated_func, "rate", None)
        on_throttle = getattr(decorated_func, "on_throttle", None)

        if rate and isinstance(rate, int) and rate > 0:
            if id(decorated_func) not in self.caches:  # Check if func TTL already in dict. If not - create it.
                self.caches[id(decorated_func)] = TTLCache(maxsize=10_000, ttl=rate)

            key = self._get_throttle_key(event)
            if key is None:
                return await handler(event, data)

            if key in self.caches[id(decorated_func)]:
                if callable(on_throttle):
                    return await on_throttle(event, data)
                else:
                    return
            else:
                self.caches[id(decorated_func)][key] = key
                return await handler(event, data)
        else:
            return await handler(event, data)
