import numpy as np
import cv2
import random
from ai2thor.server import Event, MultiAgentEvent

MOVE_MAP = {
    0: dict(z=1, x=0),
    90: dict(z=0, x=1),
    180: dict(z=-1, x=0),
    270: dict(z=0, x=-1),
}

TELEVISION_TEMPLATE = """
0 0 2 0 0
0 2 1 2 0
0 2 1 1 2 
0 2 1 2 0
0 0 2 0 0 
"""


class Controller(object):
    def __init__(
        self,
        agent_reachable_pos,
        object_reachable_pos,
        lifted_object_id,
        lifted_object_type,
        grid_size=0.25,
    ):
        self.agent_reachable_pos = agent_reachable_pos
        self.object_reachable_pos = object_reachable_pos
        self.grid_size = grid_size
        self._build_grid_world()
        self.lifted_object_id = lifted_object_id
        self.lifted_object_type = lifted_object_type
        self.lifted_object = None
        self.lifted_object_template = None
        self.scene_name = None
        self.agents = []
        self._parse_tv_tmpl()

    def start(self):
        pass

    def reset(self, scene_name):
        self.scene_name = scene_name
        self.agent_count = 1
        self.agents = []

    def _parse_tv_tmpl(self):
        tv_tmpl = []
        for line in TELEVISION_TEMPLATE.strip().split("\n"):
            row = map(lambda x: int(x.strip()), line.split())
            tv_tmpl.append(list(row))

        self.lifted_object_template = np.array(tv_tmpl, dtype=np.uint8)

    def empty_mask(self):
        return np.zeros((self._z_size, self._x_size), dtype=np.bool)

    def _build_grid_world(self):
        self._min_x = 2 ** 32
        self._max_x = -1 * 2 ** 32
        self._min_z = 2 ** 32
        self._max_z = -1 * 2 ** 32

        for point in self.agent_reachable_pos + self.object_reachable_pos:
            if point["x"] < self._min_x:
                self._min_x = point["x"]

            if point["z"] < self._min_z:
                self._min_z = point["z"]

            if point["z"] > self._max_z:
                self._max_z = point["z"]

            if point["x"] > self._max_x:
                self._max_x = point["x"]

        # adding buffer of 4 (1.0 / grid_size) points to allow for the origin
        # of the object to be at the edge
        self._max_z += 1.0
        self._max_x += 1.0
        self._min_z -= 1.0
        self._min_x -= 1.0
        print("max_x %s" % self._max_x)
        print("max_z %s" % self._max_z)
        print("min_x %s" % self._min_x)
        print("min_z %s" % self._min_z)
        self._x_size = int((self._max_x - self._min_x) / self.grid_size) + 1
        self._z_size = int((self._max_z - self._min_z) / self.grid_size) + 1
        print(self._x_size)
        print(self._z_size)

        self.agent_reachable_positions_mask = self.empty_mask()
        self.object_reachable_positions_mask = self.empty_mask()

        # object_position = np.zeros((z_size, x_size), dtype=np.bool)
        # agent_positions = np.zeros((z_size, x_size), dtype=np.bool)

        self._build_points_mask(
            self.object_reachable_pos, self.object_reachable_positions_mask
        )
        self._build_points_mask(
            self.agent_reachable_pos, self.agent_reachable_positions_mask
        )

    def _build_points_mask(self, points, mask):
        for point in points:
            # print(point)
            z = int((point["z"] - self._min_z) / self.grid_size)
            x = int((point["x"] - self._min_x) / self.grid_size)
            mask[z, x] = True

    def viz_mask(self, mask):
        viz_scale = 20
        viz_image = (
            np.ones(
                (self._z_size * viz_scale, self._x_size * viz_scale, 3), dtype=np.uint8
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
                (self._z_size * viz_scale, self._x_size * viz_scale, 3), dtype=np.uint8
            )
            * 255
        )

        agent_colors = [(0, 255, 0), (255, 0, 0)]
        for i, a in enumerate(self.agents):
            cv2.circle(
                viz_image, (a.x * viz_scale, a.z * viz_scale), 4, agent_colors[i], -1
            )

        for p in np.argwhere(self.current_lifted_object_mask()):
            cv2.circle(
                viz_image, (p[1] * viz_scale, p[0] * viz_scale), 4, (255, 0, 255), -1
            )

        cv2.imshow("aoeu", viz_image)
        cv2.waitKey(wait_key)

    def Initialize(self, action):
        self.agent_count = action["agentCount"]
        for i in range(self.agent_count):
            self.agents.append(Agent(self, len(self.agents)))

        return (True, None)

    def GetReachablePositions(self, action):
        return (True, self.agent_reachable_pos)

    def RandomlyCreateLiftedFurniture(self, action):
        # pick random reachable spot in object_reachable_pos
        # random.seed(0)
        point = random.choice(np.argwhere(self.agent_reachable_positions_mask))
        self.lifted_object = LiftedObject(
            self, self.lifted_object_id, self.lifted_object_type
        )
        z, x = point
        self.lifted_object.z = z
        self.lifted_object.x = x

        current_state = self.empty_mask()
        object_mask = self.lifted_object_template == 1
        interactable_positions = self.lifted_object_template == 2

        mask_buffer = object_mask.shape[0] // 2
        current_state[
            z - mask_buffer : z + mask_buffer + 1, x - mask_buffer : x + mask_buffer + 1
        ] = interactable_positions
        current_state &= self.agent_reachable_positions_mask
        print(type(np.argwhere(current_state)))
        agent_points = random.sample(
            list(np.argwhere(current_state)), k=self.agent_count
        )

        # XXX need to retry if we can't put the agent in a location
        assert len(agent_points) == self.agent_count
        for i, agent in enumerate(self.agents):
            agent.z = agent_points[i][0]
            agent.x = agent_points[i][1]

        return (True, self.lifted_object_id)

    def GetReachablePositionsForObject(self, action):
        return (True, self.object_reachable_pos)

    def MoveAhead(self, action):
        return self._move_agent(action, 0)

    def _move_object(self, obj, delta, valid_mask):

        if obj.is_valid_move(obj.z + delta["z"], obj.x + delta["x"], valid_mask):
            obj.z += delta["z"]
            obj.x += delta["x"]
        else:
            return False

    def _move_lifted(self, action, r):
        agent = self.agents[action["agentId"]]
        delta = MOVE_MAP[int((agent.rot + r) % 360)]
        obj = self.lifted_object
        next_obj_z = obj.z + delta["z"]
        next_obj_x = obj.x + delta["x"]
        success = True
        if obj.is_valid_move(
            next_obj_z, next_obj_x, self.object_reachable_positions_mask
        ):
            imask = self.current_interactable_mask(z=next_obj_z, x=next_obj_x)
            for a in self.agents:
                if not a.is_valid_move(a.z + delta["z"], a.x + delta["x"], imask):
                    success = False
        else:
            success = False

        if success:
            self._move_object(
                self.lifted_object, delta, self.object_reachable_positions_mask
            )
            for a in self.agents:
                self._move_object(a, delta, self.current_interactable_mask())

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

    def current_interactable_mask(self, x=None, z=None, rotation=None):
        if rotation is None:
            rotation = self.lifted_object.rot

        rotations = int((360 - rotation) / 90)
        interactable_mask = self.lifted_object_template == 2
        if rotations < 4:
            for i in range(rotations):
                interactable_mask = np.rot90(interactable_mask)

        if x is None or z is None:
            z = self.lifted_object.z
            x = self.lifted_object.x

        mask_buffer = interactable_mask.shape[0] // 2
        current_state = self.empty_mask()
        current_state[
            z - mask_buffer : z + mask_buffer + 1, x - mask_buffer : x + mask_buffer + 1
        ] = interactable_mask

        return current_state

    def current_lifted_object_mask(self):
        rotation = self.lifted_object.rot

        rotations = int((360 - rotation) / 90)
        object_mask = self.lifted_object_template == 1
        if rotations < 4:
            for i in range(rotations):
                object_mask = np.rot90(object_mask)

        z = self.lifted_object.z
        x = self.lifted_object.x
        mask_buffer = object_mask.shape[0] // 2
        current_state = self.empty_mask()
        current_state[
            z - mask_buffer : z + mask_buffer + 1, x - mask_buffer : x + mask_buffer + 1
        ] = object_mask

        return current_state

    def _rotate_lifted(self, new_rotation):
        imask = self.current_interactable_mask(rotation=new_rotation)
        for a in self.agents:
            if not imask[a.z, a.x]:
                return False

        self.lifted_object.rot = new_rotation
        return True

    def RotateLiftedObjectLeft(self, action):
        new_rotation = (self.lifted_object.rot - 90) % 360
        return (self._rotate_lifted(new_rotation), None)

    def RotateLiftedObjectRight(self, action):
        new_rotation = (self.lifted_object.rot + 90) % 360
        return (self._rotate_lifted(new_rotation), None)

    def MoveLiftedObjectRight(self, action):
        return self._move_lifted(action, 90)

    def MoveLiftedObjectAhead(self, action):
        return self._move_lifted(action, 0)

    def MoveLiftedObjectBack(self, action):
        return self._move_lifted(action, 180)

    def MoveLiftedObjectLeft(self, action):
        return self._move_lifted(action, -90)

    def step(self, action, raise_for_failure=False):
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

        return MultiAgentEvent(0, events)

    def _generate_metadata(self, agent, lifted_object, result, action, success):
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
        self.x = 0
        self.z = 0
        self.rot = 0.0

    @property
    def position(self):
        cx = (self.x * self.controller.grid_size) + self.controller._min_x
        cz = (self.z * self.controller.grid_size) + self.controller._min_z
        return dict(x=cx, y=1.0, z=cz)

    @property
    def rotation(self):
        return dict(x=0.0, y=self.rot, z=0.0)


class Agent(GridObject):
    def __init__(self, controller, agent_id):
        super().__init__(controller)
        self.agent_id = agent_id

    def is_valid_move(self, new_z, new_x, mask):
        mask &= self.controller.agent_reachable_positions_mask

        # mark spots occupied by agents as False
        for a in self.controller.agents:
            mask[a.z, a.x] = False

        return mask[new_z, new_x]


class LiftedObject(GridObject):
    def __init__(self, controller, object_id, object_type):
        super().__init__(controller)
        self.object_id = object_id
        self.object_type = object_type

    def is_valid_move(self, new_z, new_x, mask):
        return mask[new_z, new_x]
