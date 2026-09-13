from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import numpy.typing as npt
import os
import json

import torch
from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.diffusiondrive.transfuser_te_config import TransfuserTEConfig
from navsim.agents.diffusiondrive.transfuser_te_features import BoundingBox2DIndex
from navsim.common.dataclasses import Annotations, Frame, Scene
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder


class TransfuserTETargetBuilder(AbstractTargetBuilder):
    """Output target builder for TransFuser."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        config: TransfuserTEConfig,
    ):
        """
        Initializes target builder.
        :param trajectory_sampling: trajectory sampling specification
        :param config: global config dataclass of TransFuser
        """
        self._trajectory_sampling = trajectory_sampling
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = torch.tensor(
            scene.get_future_trajectory(
                num_trajectory_frames=self._trajectory_sampling.num_poses
            ).poses
        )
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        agent_states, agent_labels = self._compute_agent_targets(annotations)
        bev_semantic_map = self._compute_bev_semantic_map(
            annotations, scene.map_api, ego_pose
        )
        traffic_elements = self._get_te(scene.frames[frame_idx])
        te_heat_map = self._compute_te_heat_map(traffic_elements)

        return {
            "trajectory": trajectory,
            "agent_states": agent_states,
            "agent_labels": agent_labels,
            "bev_semantic_map": bev_semantic_map,
            "te_heat_map": te_heat_map,
        }

    def _get_te(self, frame: Frame) -> List[Any]:
        """
        Extract traffic elements from Frame
        :param frame: frame dataclass
        :return: list of traffic elements
        """

        cam_path = frame.cameras.cam_f0.camera_path

        log_name, cam_name = cam_path.split("/")[0], cam_path.split("/")[2]

        te_dir = self._config.te_path

        te_path = os.path.join(te_dir, "trainval", log_name, cam_name.replace(".jpg", ".json"))

        assert os.path.exists(te_path)

        traffic_elements = []

        with open(te_path, "r") as f:
            traffic_elements = json.load(f)["traffic_elements"]

        return traffic_elements
    
    def _compute_te_heat_map(self, traffic_elements) -> npt.NDArray[np.bool_]:
        """
        Compute binary of traffic elements in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """

        def _xy_in_lidar(x: float, y: float, config: TransfuserTEConfig) -> bool:
            return (config.te_min_x <= x <= config.te_max_x) and (config.te_min_y <= y <= config.te_max_y)

        num_classes = self._config.te_num_classes  # 0-12 for TEs
        h, w = self._config.bev_semantic_frame
        raw_map = np.zeros((num_classes, h, w), dtype=np.float32)

        for te in traffic_elements:
            c3d = te.get("3d_center")
            if not c3d or len(c3d) < 2:
                continue

            # 1. Filter by LiDAR range
            x, y = float(c3d[0]), float(c3d[1])
            if not _xy_in_lidar(x, y, self._config):
                continue

            # 2. Get class label
            label = int(te.get("attribute"))
            if not 0 <= label < num_classes:
                continue

            # 3. Convert to pixel coordinates
            pixel_coords = self._coords_to_pixel(np.array([[x, y]]))
            pixel_x, pixel_y = pixel_coords[0, 0], pixel_coords[0, 1]

            # 4. Mark the exact pixel location if it's within bounds
            if 0 <= pixel_x < h and 0 <= pixel_y < w:
                raw_map[label, pixel_x, pixel_y] = 1.0

        return torch.from_numpy(raw_map)

    def _compute_agent_targets(
        self, annotations: Annotations
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float, config: TransfuserTEConfig) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (
                config.lidar_min_y <= y <= config.lidar_max_y
            )

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(
                    np.array(
                        [box_x, box_y, box_heading, box_length, box_width],
                        dtype=np.float32,
                    )
                )

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros(
            (max_agents, BoundingBox2DIndex.size()), dtype=np.float32
        )
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(
                agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1
            )
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(
                    map_api, ego_pose, layers
                )
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)
    
    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(
            self._config.bev_semantic_frame[::-1], dtype=np.uint8
        )
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(
                    map_object.polygon, ego_pose
                )
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(
            self._config.bev_semantic_frame[::-1], dtype=np.uint8
        )
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(
                    map_object.baseline_path.linestring, ego_pose
                )
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(
                    map_linestring_mask,
                    [points],
                    isClosed=False,
                    color=255,
                    thickness=2,
                )
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(
        self, annotations: Annotations, layers: TrackedObjectType
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(
            self._config.bev_semantic_frame[::-1], dtype=np.uint8
        )
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = (
                    box_value[3],
                    box_value[4],
                    box_value[5],
                )
                agent_box = OrientedBox(
                    StateSE2(x, y, heading), box_length, box_width, box_height
                )
                exterior = np.array(agent_box.geometry.exterior.coords).reshape(
                    (-1, 1, 2)
                )
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(
            geometry, [1, 0, 0, 1, xoff, yoff]
        )
        rotated_geometry = affinity.affine_transform(
            translated_geometry, [a, b, d, e, 0, 0]
        )

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)
