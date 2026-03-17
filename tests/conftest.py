import sys
import types


playwright_module = types.ModuleType("playwright")
async_api_module = types.ModuleType("playwright.async_api")


async def _async_playwright():
    raise RuntimeError("playwright is stubbed for unit tests")


async_api_module.async_playwright = _async_playwright
async_api_module.Browser = object
async_api_module.BrowserContext = object
async_api_module.Page = object
async_api_module.Playwright = object

playwright_module.async_api = async_api_module

sys.modules.setdefault("playwright", playwright_module)
sys.modules.setdefault("playwright.async_api", async_api_module)


openai_module = types.ModuleType("openai")


class _OpenAIError(Exception):
    pass


class RateLimitError(_OpenAIError):
    pass


class BadRequestError(_OpenAIError):
    pass


class APIStatusError(_OpenAIError):
    def __init__(self, status_code=500, *args, **kwargs):
        super().__init__(*args)
        self.status_code = status_code


class AsyncOpenAI:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("openai.AsyncOpenAI must be monkeypatched in tests")


openai_module.RateLimitError = RateLimitError
openai_module.BadRequestError = BadRequestError
openai_module.APIStatusError = APIStatusError
openai_module.AsyncOpenAI = AsyncOpenAI

sys.modules.setdefault("openai", openai_module)
