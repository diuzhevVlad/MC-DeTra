"""Fair, comparable evaluation utilities for DeTra-repro.

This package separates the model forward pass (which only runs in the
``trajpred`` conda env) from the official Waymo detection metric op (which only
loads in the ``mtr`` conda env). The workflow is therefore decoupled:

1. ``dump_predictions`` (trajpred): run the model on a cache, write predictions
   and ground truth to disk.
2. ``eval_forecasting`` (trajpred or any env): compute Argoverse-2-style
   minADE/minFDE/MR/brier forecasting metrics at a fixed detection recall
   operating point. Pure numpy, no model.
3. ``waymo_ap`` (mtr): load the dump and run the official
   ``WODDetectionEvaluator`` for BEV (TYPE_2D) and 3D (TYPE_3D) AP/APH.

Joint detection+forecasting metrics (OccAP, TrajAP) are intentionally deferred.
"""
