from . import models
from . import modules
from . import pipelines
from . import renderers
from . import representations
from . import utils

__version__ = "0.0.0"


def app():
    """Trellis"""

    from pipelime.cli import PipelimeApp

    trellis_app = PipelimeApp(
        "trellis.cli",
        app_version=__version__,
    )
    trellis_app()
