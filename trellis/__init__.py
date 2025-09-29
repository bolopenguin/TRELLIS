from trellis import models
from trellis import modules
from trellis import pipelines
from trellis import representations
from trellis import utils

__version__ = "0.0.0"


def app():
    """Trellis"""

    from pipelime.cli import PipelimeApp

    trellis_app = PipelimeApp(
        "trellis.cli",
        app_version=__version__,
    )
    trellis_app()
