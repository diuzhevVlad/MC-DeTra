"""Minimal, scalable DeTra reproduction scaffold.

The package is intentionally split by responsibility:

- :mod:`detra_repro.data` reads Waymo frames and prepares training samples.
- :mod:`detra_repro.models` contains the BEV encoder, proposal head, and
  trajectory refinement transformer skeleton.
- :mod:`detra_repro.losses` contains shape-documented loss entry points.

The first target is an easy-to-overfit prototype on one Waymo segment. The
interfaces are shaped so that later replacements, such as a stronger voxel
encoder, HD-map tokens, or full DeTra losses, do not require rewriting the whole
pipeline.
"""

