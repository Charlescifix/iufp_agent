import logging
import structlog
import sys
from typing import Any, Dict
from .config import settings

def setup_logging() -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if not settings.debug else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)

def log_function_call(logger: structlog.stdlib.BoundLogger, function_name: str, **kwargs: Any) -> None:
    logger.debug(
        "Function called",
        function=function_name,
        parameters={k: v for k, v in kwargs.items() if not k.startswith('_')}
    )

def log_function_result(logger: structlog.stdlib.BoundLogger, function_name: str, result: Any = None, error: Exception = None) -> None:
    if error:
        logger.error(
            "Function failed",
            function=function_name,
            error=str(error),
            error_type=type(error).__name__
        )
    else:
        logger.debug(
            "Function completed",
            function=function_name,
            result_type=type(result).__name__ if result is not None else None
        )

def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "WARNING") -> None:
    security_logger = get_logger("security")
    log_func = getattr(security_logger, severity.lower())
    log_func(
        "Security event",
        event_type=event_type,
        **details
    )