from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """Dataset and sampling configuration.

    Attributes:
        history_sweeps: Number of lidar frames used as model input. The paper
            uses H=5, representing roughly 0.5 seconds of lidar history.
        future_steps: Number of future trajectory waypoints to predict. The
            paper uses T=10 at 2 Hz, representing 5 seconds. In this scaffold
            `T` includes future waypoints only in dataset targets, while model
            query time can include current time as index 0.
        frame_stride: Gap between target waypoints in raw Waymo frames. Waymo
            sensor frames are roughly 10 Hz; use stride=5 for 2 Hz futures.
        roi_m: Bird's-eye-view square region of interest in meters, encoded as
            (x_min, y_min, x_max, y_max) in ego/current-frame coordinates.
        voxel_size_m: BEV grid cell size in meters for the point scatter. The
            paper uses 0.1 m before a CNN downsamples to 0.4/0.8/1.6 m tokens.
            This is the paper default; for cheap local debugging override to
            0.4 m via the CLI/DataConfig (the API is unchanged).
        z_range_m: Vertical range filter for lidar points in current-frame
            coordinates.
        class_names: Waymo laser-label classes kept during preprocessing. DeTra
            reports vehicle metrics in the main tables, but Waymo 3D detection
            uses vehicle, pedestrian, and cyclist as the standard moving-agent
            classes. The scaffold keeps these three by default.
    """

    history_sweeps: int = 5
    future_steps: int = 10
    frame_stride: int = 5
    roi_m: tuple[float, float, float, float] = (-80.0, -80.0, 80.0, 80.0)
    voxel_size_m: float = 0.1
    z_range_m: tuple[float, float] = (-4.0, 8.0)
    class_names: tuple[str, ...] = ("VEHICLE", "PEDESTRIAN", "CYCLIST")


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration matching DeTra concepts with smaller defaults.

    Attributes:
        num_queries: Maximum number of object hypotheses N per scene. The paper
            uses N=600 for Waymo (default here) and N=400 for Argoverse 2; drop
            to 128/256 only for cheap first experiments.
        num_modes: Number of future hypotheses F per object. DeTra uses F=6.
        num_times: Number of query time slots including current time. For a
            10-step future, use num_times=11 if time index 0 is detection/current
            and indices 1..10 are future waypoints.
        hidden_dim: Query/token channel dimension d. The paper uses d=128.
        num_refinement_blocks: Number of iterative refinement blocks B. The
            paper uses B=3 (default here). Set to 1 only for cheap debugging;
            a single block cannot move boxes/trajectories far enough.
        point_feature_dim: Number of per-point features passed to the paper-like
            lidar encoder. This scaffold uses ``x,y,z,dt`` and intentionally
            ignores intensity.
        num_classes: Number of moving-agent classes. Waymo uses vehicle,
            pedestrian, cyclist for this reproduction.
        map_feature_dim: Number of cached vector-map feature channels.
        proposal_output_stride: Stride of proposal feature map relative to the
            raw point-scatter BEV grid. ``PaperLikeLidarEncoder`` returns a
            fused stride-4 map for proposals.
        deformable_points: Number of learned lidar sample points per query.
            DeTra's paper reports this as the lidar deformable attention
            hyperparameter.
        deformable_radius_m: Initial radius in meters used to scale learned
            offsets around each pose anchor.
        attention_type: Lidar cross-attention variant. ``deformable`` predicts
            learned offsets around each pose anchor. ``center`` samples the BEV
            feature exactly at the pose anchor and is useful as an ablation for
            debugging deformable attention.
        lidar_attention_anchor: Which pose anchors lidar attention uses.
            ``current`` samples every time slot around the current ``t=0`` pose,
            matching the original scaffold behavior. ``time`` samples each time
            slot around its own pose, allowing future queries to look along the
            current predicted trajectory in later refinement blocks.
        detach_pose_refinement: If true, stop gradients through pose anchors
            between refinement blocks. The paper does this ("we stop the
            gradients of the poses P(i) from flowing in between refinement
            transformer blocks, which we found to improve learning"), so the
            paper-faithful default is true. Set false only for ablations.
        map_k: Number of nearest map tokens attended by each query when map
            tokens are provided.
        use_query_velocity: Diagnostic option that allows training scripts to
            inject a per-query velocity embedding. Disabled by default because
            the paper-faithful path should infer motion from multi-sweep LiDAR,
            not from GT actor history.
        forecast_base_mode: How future waypoint means are parameterized.
            ``current`` predicts offsets from the detected current center,
            matching the original scaffold. ``constant_velocity`` predicts
            residuals on top of ``current + step * query_velocity`` and is a
            diagnostic path for testing whether direct large-offset regression
            is the forecasting bottleneck. ``learned_velocity`` predicts a
            per-query velocity from refined query features and rolls out
            residual forecasts around it; this is the fair non-oracle path.
        forecast_velocity_frame: Frame used by the learned velocity head.
            ``ego`` predicts ``vx,vy`` directly in current ego coordinates.
            ``object`` predicts longitudinal/lateral velocity in the predicted
            box frame and rotates it into ego coordinates for rollout/loss.
        use_history_occupancy: Diagnostic oracle input. When true, training can
            rasterize cached GT actor histories into BEV occupancy channels and
            feed them to the LiDAR encoder. Disabled by default because GT
            history is not available as paper-faithful inference input.
        history_occupancy_channels: Number of BEV history channels. The cache
            currently stores five history steps, oldest to current.
        dynamic_conv_experts: Number of convolution experts mixed by dynamic
            convolution in the BEV backbone.
        bev_channels: Number of BEV input channels for the old simple raster
            encoder. Kept for comparison with the paper-like point-scatter path.
    """

    num_queries: int = 600
    num_modes: int = 6
    num_times: int = 11
    hidden_dim: int = 128
    num_refinement_blocks: int = 3
    point_feature_dim: int = 4
    num_classes: int = 3
    map_feature_dim: int = 12
    map_encoder_type: str = "graph"
    map_graph_k: int = 8
    map_graph_layers: int = 2
    bev_channels: int = 4
    proposal_output_stride: int = 4
    deformable_points: int = 4
    deformable_radius_m: float = 2.0
    attention_type: str = "deformable"
    lidar_attention_anchor: str = "current"
    detach_pose_refinement: bool = True
    map_k: int = 4
    use_query_velocity: bool = False
    forecast_base_mode: str = "current"
    forecast_velocity_frame: str = "ego"
    use_history_occupancy: bool = False
    history_occupancy_channels: int = 5
    dynamic_conv_experts: int = 4
    predict_past: bool = False
    past_horizon: int = 0
