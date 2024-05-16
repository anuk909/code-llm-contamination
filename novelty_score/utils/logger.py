import logging


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d [%(filename)s:%(lineno)d] %(levelname)-s: %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
