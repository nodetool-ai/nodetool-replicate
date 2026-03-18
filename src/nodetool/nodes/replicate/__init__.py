"""Replicate provider nodes.

Import submodules so `import nodetool.nodes.replicate` eagerly registers all
Replicate node classes. Graph loading resolves dynamic nodes from the top-level
namespace (`replicate.*`), so a fresh process must populate the global node
registry from this package import alone.
"""

import nodetool.nodes.replicate.audio.enhance  # noqa: F401
import nodetool.nodes.replicate.audio.generate  # noqa: F401
import nodetool.nodes.replicate.audio.separate  # noqa: F401
import nodetool.nodes.replicate.audio.transcribe  # noqa: F401
import nodetool.nodes.replicate.code_generation  # noqa: F401
import nodetool.nodes.replicate.cost_calculation  # noqa: F401
import nodetool.nodes.replicate.dynamic_schema  # noqa: F401
import nodetool.nodes.replicate.gencode  # noqa: F401
import nodetool.nodes.replicate.image.analyze  # noqa: F401
import nodetool.nodes.replicate.image.enhance  # noqa: F401
import nodetool.nodes.replicate.image.face  # noqa: F401
import nodetool.nodes.replicate.image.generate  # noqa: F401
import nodetool.nodes.replicate.image.ocr  # noqa: F401
import nodetool.nodes.replicate.image.process  # noqa: F401
import nodetool.nodes.replicate.image.upscale  # noqa: F401
import nodetool.nodes.replicate.prediction  # noqa: F401
import nodetool.nodes.replicate.replicate_node  # noqa: F401
import nodetool.nodes.replicate.text.generate  # noqa: F401
import nodetool.nodes.replicate.video.enhance  # noqa: F401
import nodetool.nodes.replicate.video.generate  # noqa: F401
