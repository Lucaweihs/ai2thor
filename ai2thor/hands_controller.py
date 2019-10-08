import math
from typing import List, Dict, Optional, Any

import numpy as np
import cv2
import random
from ai2thor.server import Event, MultiAgentEvent

MOVE_MAP = {
    0: dict(row=-1, col=0),
    90: dict(row=0, col=1),
    180: dict(row=1, col=0),
    270: dict(row=0, col=-1),
}

SMALL_TELEVISION_TEMPLATE_STRING = """
0 0 2 0 0
0 2 1 2 0
0 2 1 1 2 
0 2 1 2 0
0 0 2 0 0 
"""


class GridWorldController(object):
    def __init__(
        self,
        agent_reachable_pos: List[Dict[str, float]],
        rotation_to_object_reachable_pos: Dict[int, List[Dict[str, float]]],
        lifted_object_id: str,
        lifted_object_type: str,
        object_template_string: str,
        grid_size=0.25,
    ):
        self.agent_reachable_pos = agent_reachable_pos
        self.rotation_to_object_reachable_pos = {
            int(k): rotation_to_object_reachable_pos[k]
            for k in rotation_to_object_reachable_pos
        }
        self.grid_size = grid_size
        self.lifted_object_id = lifted_object_id
        self.lifted_object_type = lifted_object_type
        self.lifted_object: Optional[LiftedObject] = None
        self.lifted_object_template = None
        self.scene_name = None
        self.agents: List[Agent] = []

        self.lifted_object_template = self.parse_template_to_mask(
            object_template_string
        )
        self._build_grid_world(padding_units=max(3, *self.lifted_object_template.shape))

        self.last_event = None
        self.steps_taken = 0

    def start(self):
        pass

    def reset(self, scene_name):
        assert self.scene_name is None or scene_name == self.scene_name
        self.scene_name = scene_name
        self.last_event = None
        self.agent_count = 1
        self.agents = []
        self.steps_taken = 0

    def parse_template_to_mask(self, template):
        tv_tmpl = []
        for line in template.strip().split("\n"):
            row = map(lambda x: int(x.strip()), line.split())
            tv_tmpl.append(list(row))

        return np.array(tv_tmpl, dtype=np.uint8)

    def empty_mask(self):
        return np.zeros((self._nrows, self._ncols), dtype=np.bool)

    def _build_grid_world(self, padding_units):
        self._min_x = 2 ** 32
        self._max_x = -1 * 2 ** 32
        self._min_z = 2 ** 32
        self._max_z = -1 * 2 ** 32

        for point in self.agent_reachable_pos + sum(
            self.rotation_to_object_reachable_pos.values(), []
        ):
            if point["x"] < self._min_x:
                self._min_x = point["x"]

            if point["z"] < self._min_z:
                self._min_z = point["z"]

            if point["z"] > self._max_z:
                self._max_z = point["z"]

            if point["x"] > self._max_x:
                self._max_x = point["x"]

        # adding buffer of 6 (1.0 / grid_size) points to allow for the origin
        # of the object to be at the edge
        self._max_z += padding_units * self.grid_size
        self._max_x += padding_units * self.grid_size
        self._min_z -= padding_units * self.grid_size
        self._min_x -= padding_units * self.grid_size
        # print("max_x %s" % self._max_x)
        # print("max_z %s" % self._max_z)
        # print("min_x %s" % self._min_x)
        # print("min_z %s" % self._min_z)
        self._ncols = int((self._max_x - self._min_x) / self.grid_size) + 1
        self._nrows = int((self._max_z - self._min_z) / self.grid_size) + 1
        # print(self._x_size)
        # print(self._z_size)

        self.agent_reachable_positions_mask = self.empty_mask()
        self.rotation_to_object_reachable_position_masks = {
            rot: self.empty_mask() for rot in self.rotation_to_object_reachable_pos
        }

        # object_position = np.zeros((z_size, x_size), dtype=np.bool)
        # agent_positions = np.zeros((z_size, x_size), dtype=np.bool)

        for rot in self.rotation_to_object_reachable_pos:
            self._build_points_mask(
                self.rotation_to_object_reachable_pos[rot],
                self.rotation_to_object_reachable_position_masks[rot],
            )
        self._build_points_mask(
            self.agent_reachable_pos, self.agent_reachable_positions_mask
        )

    def _rowcol_to_xz(self, rowcol):
        row, col = rowcol
        x = (col * self.grid_size) + self._min_x
        z = (-row * self.grid_size) + self._max_z
        return x, z

    def _xz_to_rowcol(self, xz):
        x, z = xz
        row = round((self._max_z - z) / self.grid_size)
        col = round((x - self._min_x) / self.grid_size)
        return row, col

    def _build_points_mask(self, points, mask):
        for point in points:
            # print(point)
            row, col = self._xz_to_rowcol((point["x"], point["z"]))
            mask[row, col] = True

    def viz_mask(self, mask):
        viz_scale = 20
        viz_image = (
            np.ones(
                (self._nrows * viz_scale, self._ncols * viz_scale, 3), dtype=np.uint8
            )
            * 255
        )

        for point in np.argwhere(mask):
            cv2.circle(
                viz_image,
                (point[1] * viz_scale, point[0] * viz_scale),
                4,
                (255, 0, 0),
                -1,
            )

        cv2.imshow("aoeu", viz_image)
        cv2.waitKey(2000)

    def viz_world(self, wait_key=0):
        viz_scale = 20
        viz_image = (
            np.ones(
                (self._nrows * viz_scale, self._ncols * viz_scale, 3), dtype=np.uint8
            )
            * 255
        )

        for p in np.argwhere(self.agent_reachable_positions_mask):
            tl = (p[1] * viz_scale - viz_scale // 4, p[0] * viz_scale - viz_scale // 4)
            br = (p[1] * viz_scale + viz_scale // 4, p[0] * viz_scale + viz_scale // 4)

            # cv2.rectangle(viz_image, tl, br, (210, 210, 210), -2)
            cv2.rectangle(viz_image, tl, br, (210, 210, 210), -1)

        masks = [
            self.rotation_to_object_reachable_position_masks[rot]
            for rot in sorted(
                list(self.rotation_to_object_reachable_position_masks.keys())
            )
        ]
        for p in np.argwhere((np.stack(masks, axis=0)).any(0) != 0):
            color = np.array([0, 0, 0])
            for i, mask in enumerate(masks):
                if mask[p[0], p[1]] and i < 3:
                    color[i] = 255
                elif mask[p[0], p[1]]:
                    color = color // 2

            # offset = i + 1 + viz_scale // 4
            offset = 2 + viz_scale // 4
            tl = (p[1] * viz_scale - offset, p[0] * viz_scale - offset)
            br = (p[1] * viz_scale + offset, p[0] * viz_scale + offset)

            # cv2.rectangle(viz_image, tl, br, (210, 210, 210), -2)
            cv2.rectangle(viz_image, tl, br, tuple(int(i) for i in color), 2)

        agent_colors = [(0, 255, 0), (255, 0, 0)]
        for i, a in enumerate(self.agents):
            cv2.circle(
                viz_image,
                (a.col * viz_scale, a.row * viz_scale),
                4,
                agent_colors[i],
                -1,
            )
            dir = MOVE_MAP[a.rot]
            cv2.line(
                viz_image,
                (a.col * viz_scale, a.row * viz_scale),
                (
                    a.col * viz_scale + dir["col"] * viz_scale // 2,
                    a.row * viz_scale + dir["row"] * viz_scale // 2,
                ),
                agent_colors[i],
                2,
            )

        if self.lifted_object is not None:
            for p in np.argwhere(self.current_lifted_object_mask()):
                cv2.circle(
                    viz_image,
                    (p[1] * viz_scale, p[0] * viz_scale),
                    3,
                    (255, 0, 255),
                    -1,
                )

            for p in np.argwhere(self.current_interactable_mask()):
                cv2.circle(
                    viz_image,
                    (p[1] * viz_scale, p[0] * viz_scale),
                    2,
                    (180, 0, 180),
                    -1,
                )

        cv2.imshow("aoeu", viz_image)
        return str(chr(cv2.waitKey(wait_key) & 255))

    def Initialize(self, action):
        self.agent_count = action["agentCount"]
        for i in range(self.agent_count):
            self.agents.append(Agent(self, len(self.agents)))

        return (True, None)

    def GetReachablePositions(self, action):
        return (True, self.agent_reachable_pos)

    def RandomlyCreateLiftedFurniture(self, action):
        assert action["objectType"] == self.lifted_object_type

        # pick random reachable spot in object_reachable_pos
        # random.seed(0)
        for i in range(10):
            point = random.choice(
                # np.argwhere(self.rotation_to_object_reachable_position_masks[rotation])
                np.argwhere(self.agent_reachable_positions_mask)
            )
            possible_rotations = [
                rot
                for rot in self.rotation_to_object_reachable_position_masks
                if self.rotation_to_object_reachable_position_masks[rot][
                    point[0], point[1]
                ]
            ]
            if len(possible_rotations) == 0:
                continue
            rotation = random.choice(possible_rotations)

            self.lifted_object = LiftedObject(
                self, self.lifted_object_id, self.lifted_object_type
            )
            row, col = point
            self.lifted_object.row = row
            self.lifted_object.col = col
            self.lifted_object.rot = rotation

            current_state = self.empty_mask()
            object_mask = self.lifted_object_template == 1
            interactable_positions = self.lifted_object_template == 2
            rotations = int((360 - rotation) / 90)
            if rotations < 4:
                object_mask = np.rot90(object_mask, k=rotations)
                interactable_positions = np.rot90(interactable_positions, k=rotations)

            mask_buffer_row, mask_buffer_col = (
                object_mask.shape[0] // 2,
                object_mask.shape[1] // 2,
            )
            current_state[
                row - mask_buffer_row : row + mask_buffer_row + 1,
                col - mask_buffer_col : col + mask_buffer_col + 1,
            ] = interactable_positions
            current_state &= self.agent_reachable_positions_mask
            agent_points = random.sample(
                list(np.argwhere(current_state)), k=self.agent_count
            )

            # XXX need to retry if we can't put the agent in a location
            if len(agent_points) == self.agent_count:
                break

        if len(agent_points) != self.agent_count:
            raise Exception(
                "Couldn't create random start point for scene name %s" % self.scene_name
            )

        for i, agent in enumerate(self.agents):
            agent.row = agent_points[i][0]
            agent.col = agent_points[i][1]
            if random.random() < 0.5:
                if agent.row > self.lifted_object.row:
                    agent.rot = 0
                elif agent.row < self.lifted_object.row:
                    agent.rot = 180
                else:
                    agent.rot = random.choice([0, 180])
            else:
                if agent.col < self.lifted_object.col:
                    agent.rot = 90
                elif agent.col > self.lifted_object.col:
                    agent.rot = 270
                else:
                    agent.rot = random.choice([90, 270])

        return (True, self.lifted_object_id)

    def GetReachablePositionsForObject(self, action):
        return (True, self.rotation_to_object_reachable_pos)

    def MoveAhead(self, action):
        return self._move_agent(action, 0)

    def _move_object(self, obj, delta, valid_mask, skip_valid_check=False):
        if skip_valid_check or obj.is_valid_new_position(
            obj.row + delta["row"], obj.col + delta["col"], valid_mask
        ):
            obj.row += delta["row"]
            obj.col += delta["col"]
            return True
        else:
            return False

    def _move_agents_with_lifted(self, action, r):
        assert action["objectId"] == self.lifted_object_id

        agent = self.agents[action["agentId"]]
        delta = MOVE_MAP[int((agent.rot + r) % 360)]
        obj = self.lifted_object
        next_obj_z = obj.row + delta["row"]
        next_obj_x = obj.col + delta["col"]
        success = True
        if obj.is_valid_new_position(
            next_obj_z,
            next_obj_x,
            self.rotation_to_object_reachable_position_masks[
                int(self.lifted_object.rot)
            ],
        ):
            imask = self.current_interactable_mask(row=next_obj_z, col=next_obj_x)
            for a in self.agents:
                if not a.is_valid_new_position(
                    a.row + delta["row"],
                    a.col + delta["col"],
                    imask,
                    allow_agent_intersection=True,
                ):
                    success = False
                    break
        else:
            success = False

        if success:
            self._move_object(
                self.lifted_object,
                delta,
                self.rotation_to_object_reachable_position_masks[
                    int(self.lifted_object.rot)
                ],
            )
            for a in self.agents:
                self._move_object(a, delta, self.current_interactable_mask())

        return (success, None)

    def _move_lifted(self, action, r):
        assert action["objectId"] == self.lifted_object_id

        agent = self.agents[action["agentId"]]
        delta = MOVE_MAP[int((agent.rot + r) % 360)]
        obj = self.lifted_object
        next_obj_z = obj.row + delta["row"]
        next_obj_x = obj.col + delta["col"]
        success = True
        if obj.is_valid_new_position(
            next_obj_z,
            next_obj_x,
            self.rotation_to_object_reachable_position_masks[
                int(self.lifted_object.rot)
            ],
        ):
            imask = self.current_interactable_mask(row=next_obj_z, col=next_obj_x)
            for a in self.agents:
                if not a.is_valid_new_position(
                    a.row, a.col, imask, allow_agent_intersection=True
                ):
                    success = False
        else:
            success = False

        if success:
            self._move_object(
                self.lifted_object,
                delta,
                self.rotation_to_object_reachable_position_masks[
                    int(self.lifted_object.rot)
                ],
            )

        return (success, None)

    def _move_agent(self, action, r):
        agent = self.agents[action["agentId"]]
        delta = MOVE_MAP[int((agent.rot + r) % 360)]
        success = self._move_object(agent, delta, self.current_interactable_mask())
        return (success, None)

    def MoveLeft(self, action):
        return self._move_agent(action, -90)

    def MoveRight(self, action):
        return self._move_agent(action, 90)

    def MoveBack(self, action):
        return self._move_agent(action, 180)

    def RotateRight(self, action):
        agent = self.agents[action["agentId"]]
        agent.rot = (agent.rot + 90) % 360
        return (True, None)

    def RotateLeft(self, action):
        agent = self.agents[action["agentId"]]
        agent.rot = (agent.rot - 90) % 360
        return (True, None)

    def current_interactable_mask(self, row=None, col=None, rotation=None):
        if rotation is None:
            rotation = self.lifted_object.rot

        rotations = int((360 - rotation) / 90)
        interactable_mask = self.lifted_object_template == 2
        interactable_mask = np.rot90(interactable_mask, k=rotations)

        if col is None or row is None:
            row = self.lifted_object.row
            col = self.lifted_object.col

        mask_buffer_row, mask_buffer_col = (
            interactable_mask.shape[0] // 2,
            interactable_mask.shape[1] // 2,
        )
        current_state = self.empty_mask()
        current_state[
            row - mask_buffer_row : row + mask_buffer_row + 1,
            col - mask_buffer_col : col + mask_buffer_col + 1,
        ] = interactable_mask

        return current_state

    def current_lifted_object_mask(self):
        rotation = self.lifted_object.rot

        rotations = int((360 - rotation) / 90)
        object_mask = self.lifted_object_template == 1
        object_mask = np.rot90(object_mask, k=rotations)

        row = self.lifted_object.row
        col = self.lifted_object.col
        mask_buffer_row, mask_buffer_col = (
            object_mask.shape[0] // 2,
            object_mask.shape[1] // 2,
        )
        current_state = self.empty_mask()
        current_state[
            row - mask_buffer_row : row + mask_buffer_row + 1,
            col - mask_buffer_col : col + mask_buffer_col + 1,
        ] = object_mask

        return current_state

    def _rotate_lifted(self, new_rotation):
        imask = self.current_interactable_mask(rotation=new_rotation)
        for a in self.agents:
            if not imask[a.row, a.col]:
                return False

        self.lifted_object.rot = new_rotation
        return True

    def RandomlyCreateAndPlaceObjectOnFloor(self, action):
        object_mask = action["object_mask"]
        object_masks = [(k, np.rot90(object_mask, k=k)) for k in range(4)]

        positions = np.argwhere(self.agent_reachable_positions_mask)
        m = self.agent_reachable_positions_mask
        for i in np.random.permutation(positions.shape[0]):
            row, col = positions[i]
            random.shuffle(object_masks)
            for k, mask in object_masks:
                row_rad, col_rad = mask.shape[0] // 2, mask.shape[1] // 2
                reachable_subset = m[
                    row - row_rad : row + row_rad + 1, col - col_rad : col + col_rad + 1
                ]
                if (np.logical_and(reachable_subset, mask) == mask).all():
                    m[
                        row - row_rad : row + row_rad + 1,
                        col - col_rad : col + col_rad + 1,
                    ] &= np.logical_not(mask)
                    xz = self._rowcol_to_xz((row, col))
                    return (
                        True,
                        {
                            "position": {"x": xz[0], "y": math.nan, "z": xz[1]},
                            "row": row,
                            "col": col,
                            "rotation": 90 * k,
                        },
                    )
        return False, None

    def RotateLiftedObjectLeft(self, action):
        new_rotation = (self.lifted_object.rot - 90) % 360
        return (self._rotate_lifted(new_rotation), None)

    def RotateLiftedObjectRight(self, action):
        new_rotation = (self.lifted_object.rot + 90) % 360
        return (self._rotate_lifted(new_rotation), None)

    def MoveAgentsRightWithObject(self, action):
        return self._move_agents_with_lifted(action, 90)

    def MoveAgentsAheadWithObject(self, action):
        return self._move_agents_with_lifted(action, 0)

    def MoveAgentsBackWithObject(self, action):
        return self._move_agents_with_lifted(action, 180)

    def MoveAgentsLeftWithObject(self, action):
        return self._move_agents_with_lifted(action, -90)

    def MoveLiftedObjectRight(self, action):
        return self._move_lifted(action, 90)

    def MoveLiftedObjectAhead(self, action):
        return self._move_lifted(action, 0)

    def MoveLiftedObjectBack(self, action):
        return self._move_lifted(action, 180)

    def MoveLiftedObjectLeft(self, action):
        return self._move_lifted(action, -90)

    def step(self, action, raise_for_failure=False):
        self.steps_taken += 1
        # XXX should have invalid action
        # print("running method %s" % action)
        method = getattr(self, action["action"])
        success, result = method(action)
        events = []
        for a in self.agents:
            events.append(
                Event(
                    self._generate_metadata(
                        a, self.lifted_object, result, action, success
                    )
                )
            )

        self.last_event = MultiAgentEvent(
            action.get("agentId") if "agentId" in action else 0, events
        )
        return self.last_event

    def _generate_metadata(
        self,
        agent: "Agent",
        lifted_object: "GridObject",
        result: Any,
        action: str,
        success: bool,
    ):
        metadata = dict()
        metadata["agent"] = dict(position=agent.position, rotation=agent.rotation)
        metadata["objects"] = []
        if self.lifted_object:
            metadata["objects"].append(
                dict(
                    position=lifted_object.position,
                    rotation=lifted_object.rotation,
                    objectType=lifted_object.object_type,
                    objectId=lifted_object.object_id,
                )
            )
        metadata["actionReturn"] = result
        metadata["lastAction"] = action["action"]
        metadata["lastActionSuccess"] = success
        metadata["sceneName"] = self.scene_name
        metadata["screenHeight"] = 300
        metadata["screenWidth"] = 300
        metadata["colors"] = []

        return metadata


class GridObject(object):
    def __init__(self, controller):
        self.controller = controller
        self.col = 0
        self.row = 0
        self.rot = 0.0

    @property
    def position(self):
        cx = (self.col * self.controller.grid_size) + self.controller._min_x
        cz = (-self.row * self.controller.grid_size) + self.controller._max_z
        return dict(x=cx, y=1.0, z=cz)

    @property
    def rotation(self):
        return dict(x=0.0, y=self.rot, z=0.0)


class Agent(GridObject):
    def __init__(self, controller: GridWorldController, agent_id):
        super().__init__(controller)
        self.agent_id = agent_id

    def is_valid_new_position(
        self, new_row, new_col, mask, allow_agent_intersection=False
    ):
        mask &= self.controller.agent_reachable_positions_mask

        # mark spots occupied by agents as False
        if not allow_agent_intersection:
            for a in self.controller.agents:
                mask[a.row, a.col] = False

        return mask[new_row, new_col]


class LiftedObject(GridObject):
    def __init__(self, controller, object_id, object_type):
        super().__init__(controller)
        self.object_id = object_id
        self.object_type = object_type

    def is_valid_new_position(self, new_row, new_col, mask):
        return mask[new_row, new_col]


def run_demo(controller: GridWorldController):
    agent_id = 0
    trying_to_quit = False
    while True:
        c = controller.viz_world()

        key_to_action = {
            "w": "MoveAhead",
            "a": "MoveLeft",
            "s": "MoveBack",
            "d": "MoveRight",
            "z": "RotateLeft",
            "x": "RotateRight",
            "i": "MoveAgentsAheadWithObject",
            "j": "MoveAgentsLeftWithObject",
            "k": "MoveAgentsBackWithObject",
            "l": "MoveAgentsRightWithObject",
            "m": "RotateLiftedObjectLeft",
            ",": "RotateLiftedObjectRight",
            "t": "MoveLiftedObjectAhead",
            "f": "MoveLiftedObjectLeft",
            "g": "MoveLiftedObjectBack",
            "h": "MoveLiftedObjectRight",
        }

        if c in ["0", "1"]:
            trying_to_quit = False
            agent_id = int(c)
            print("Switched to agent {}".format(c))
        elif c == "q":
            print("Are you sure you wish to exit the demo? (y/n)")
            trying_to_quit = True
        elif trying_to_quit and c == "y":
            return
        elif c in key_to_action:
            trying_to_quit = False
            controller.step(
                {
                    "action": key_to_action[c],
                    "agentId": agent_id,
                    "objectId": "Television|1",
                }
            )
            print(
                "Taking action {}\nAction {}\n".format(
                    key_to_action[c],
                    "success"
                    if controller.last_event.metadata["lastActionSuccess"]
                    else "failure",
                )
            )
            print("Agent 0 position", controller.agents[0].position)
            print("Agent 1 position", controller.agents[1].position)
            print("Object position", controller.lifted_object.position)
            print("")
        else:
            trying_to_quit = False
            print('Invalid key "{}"'.format(c))

        controller.viz_world()
