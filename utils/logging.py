import time
from contextlib import ContextDecorator
import logging
from codetiming import Timer
from datetime import datetime
from humanfriendly import format_timespan
from functools import wraps
from rich.pretty import pretty_repr
from rich.console import Console
from rich.logging import RichHandler
import wandb
import hydra
from collections import defaultdict

logger = rich_logger = logging.getLogger()

rich_handler = RichHandler(
    rich_tracebacks=False,
    tracebacks_suppress=[hydra],
    console=Console(width=165),
    enable_link_path=False,
)
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[rich_handler],
)

def get_cur_time(timezone=None, t_format="%m-%d %H:%M:%S"):
    return datetime.fromtimestamp(int(time.time()), timezone).strftime(t_format)


def wandb_finish(result=None):
    if wandb.run is not None:
        wandb.summary.update(result or {})
        wandb.finish()

class timer(ContextDecorator):
    def __init__(self, name=None, log_func=logger.info):
        self.name = name
        self.log_func = log_func
        self.timer = Timer(name=name, logger=None)  # Disable internal logging

    def __enter__(self):
        self.timer.start()
        self.log_func(f"Started {self.name} at {get_cur_time()}")
        return self

    def __exit__(self, *exc):
        elapsed_time = self.timer.stop()
        formatted_time = format_timespan(elapsed_time)
        self.log_func(
            f"Finished {self.name} at {get_cur_time()}, running time = {formatted_time}."
        )
        return False

    def __call__(self, func):
        self.name = self.name or func.__name__

        @wraps(func)
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator
