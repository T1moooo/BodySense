"""OCR API routes for health report processing."""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from ...models.ocr import OCRResponse, OCRResult, TextExtractionResponse
from ...services.indicator_extractor import extract_indicators, get_overall_confidence
from ...services.ocr import extract_text

logger = logging.getLogger(__name__)

# 下面这一行就是使用 FastAPI 中的 APIRouter 方法，传入前缀和 tags，然后返回一个供后续接入路由的 router 对象
router = APIRouter(prefix="/api/ocr", tags=["ocr"])

# 定义最大文件大小的常量 10MB
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# 使用路由装饰符定义OCR提取端点,以及对应的响应模型 OCRResponse
@router.post("/extract", response_model=OCRResponse)
# 路由对应的处理函数 extract_ocr，接收一个上传文件参数 file，类型为 UploadFile，使用 File(...) 表示这是一个必填的文件上传参数
async def extract_ocr(file: UploadFile = File(...)):
    """
    Extract text and health indicators from an uploaded file.

    Accepts image (JPEG, PNG, WebP) or PDF files.
    Returns structured OCR results with extracted health indicators.
    """
    # Validate file type 定义的允许的文件类型集合， 使用 HTTPException 抛出异常，如果上传的文件类型不在允许的类型中
    allowed_types = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
            f"Allowed: {', '.join(allowed_types)}",
        )

    # 处理文件上传和 OCR 提取的逻辑，使用 try-except 块捕获异常，确保在出现错误时返回适当的 HTTP 响应
    try:
        # Read file content
        file_bytes = await file.read()
        # 进行文件内容的读取，如果文件为空，抛出 HTTP 400 错误
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        # 检查文件大小是否超过最大限制，如果超过，抛出 HTTP 413 错误
        if len(file_bytes) > _MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({len(file_bytes)} bytes). "
                    f"Maximum is {_MAX_FILE_SIZE} bytes."
                ),
            )

        # Extract text using OCR
        raw_text, confidence = extract_text(file_bytes, file.content_type)
        # 如果提取的文本为空，返回一个状态为 "completed" 的 OCRResponse 对象，包含空的 raw_text、空的 indicators 列表和低置信度
        if not raw_text.strip():
            return OCRResponse(
                status="completed",
                result=OCRResult(
                    raw_text="",
                    indicators=[],
                    confidence="low",
                ),
            )

        # Extract health indicators
        indicators = extract_indicators(raw_text)
        overall_confidence = get_overall_confidence(indicators)

        # Use the lower of OCR confidence and indicator confidence
        final_confidence = _min_confidence(
            _confidence_level(confidence),
            overall_confidence,
        )
        # 返回一个状态为 "completed" 的 OCRResponse 对象，包含提取的 raw_text、indicators 列表和最终的置信度
        return OCRResponse(
            status="completed",
            result=OCRResult(
                raw_text=raw_text,
                indicators=indicators,
                confidence=final_confidence,
            ),
        )
    # 捕获 HTTPException 异常并重新抛出，以便 FastAPI 可以处理它们并返回适当的 HTTP 响应
    except HTTPException:
        raise
    # 捕获其他异常，记录异常信息，并抛出 HTTP 500 错误，表示服务器内部错误
    except Exception as e:
        logger.exception("OCR extraction failed")
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}",
        )


@router.post("/extract-text", response_model=TextExtractionResponse)
async def extract_text_only(file: UploadFile = File(...)):
    """
    Extract raw text from an uploaded file without indicator parsing.

    Accepts image (JPEG, PNG, WebP) or PDF files.
    Returns the raw extracted text.
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
            f"Allowed: {', '.join(allowed_types)}",
        )

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        if len(file_bytes) > _MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large ({len(file_bytes)} bytes). "
                    f"Maximum is {_MAX_FILE_SIZE} bytes."
                ),
            )

        raw_text, _ = extract_text(file_bytes, file.content_type)

        # Count pages (approximate for PDFs)
        pages = raw_text.count("--- Page ") + 1 if "--- Page " in raw_text else 1

        return TextExtractionResponse(
            text=raw_text,
            pages=pages,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Text extraction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction failed: {str(e)}",
        )


def _confidence_level(score: float) -> str:
    """Convert numeric confidence score to level string."""
    if score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    return "low"


def _min_confidence(a: str, b: str) -> str:
    """Return the lower of two confidence levels."""
    order = {"high": 3, "medium": 2, "low": 1}
    return a if order.get(a, 0) <= order.get(b, 0) else b
