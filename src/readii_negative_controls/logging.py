from imgtools.logging import get_logger
import os
DEFAULT_LOGGING_LEVEL = os.environ.get('negctrls', None) or 'WARNING'
logger = get_logger('negctrls', DEFAULT_LOGGING_LEVEL)
