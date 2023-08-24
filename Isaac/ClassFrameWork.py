import BaseTask
from isaacgym import gymapi
from numpy import random
import gymtorch
from Isaac.ObjetsEnvironnement.RoomManager import RoomManager
from Isaac.ObjetsEnvironnement.AlbertCube import AlbertCube
from Isaac.ObjetsEnvironnement.Room import Room
import numpy as np
from scipy.spatial.transform import Rotation


class AlbertEnvironment(BaseTask):

    # Common callbacks

    def create_sim(self):
        # create environments and actors
        self.gym = gymapi.acquire_gym()

        # configure sim params
        sim_params = self.set_sim_params()

        # create sim with these parameters
        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        # configure the ground plane
        self.configure_ground()

        #####################  ASSETS  #####################

        asset_albert, asset_room = self.prepare_assets()

        asset_options_base_cube = gym.assetOptions()
        asset_base_cube = gym.create_box(self.sim, width=0.5, height=0.5, depth=0.5,
                                         asset_options=asset_options_base_cube)

        asset_options_door = gym.assetOptions()
        asset_door = gym.create_box(self.sim, width=0.5, height=0.5, depth=0.5, asset_options=asset_options_door)

        asset_options_button = gym.assetOptions()
        asset_button = gym.create_box(self.sim, width=0.5, height=0.01, depth=0.5, asset_options=asset_options_button)

        #####################  ENVIRONMENT SETTING  #####################

        # set up the env grid
        self.num_envs = 64
        envs_per_row = 8
        env_spacing = 2.0
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        # cache some common handles for later use
        envs = []
        self.actor_handles = []

        # instanciate Room Manager
        self.room_manager_array = np.array([RoomManager() for i in range(self.num_envs)])
        self.albert_array = np.empty((self.num_envs,))

        # create and populate the environments
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            envs.append(env)

            pose_albert = gymapi.Transform()
            pose_albert.p = gymapi.Vec3(2.0, 3.0, 1.5)  # pose.r pour l'orientation
            # la position est relative
            actor_handle_albert = self.gym.create_actor(env, asset_albert, pose_albert, "Albert", i,
                                                        1)  # creation  d'un acteur à partir d'un asset

            self.actor_handles.append(actor_handle_albert)

            #pose_room = gymapi.Transform()
            #pose_room.p = gymapi.Vec3(0.0, 0.0, 0.0)  # pose.r pour l'orientation
            # la position est relative
            #actor_handle_room = self.gym.create_actor(env, asset_room, pose_room, "Room", i,
             #                                         1)  # creation  d'un acteur à partir d'un asset
            #self.actor_handles.append(actor_handle_room)

            self.build_basic_room(env,asset_base_cube,asset_door)
            pose_button=gymapi.Transform()
            pose_button.p=gymapi.Vec3(3,4,0.52)
            actor_handle_button=self.gym.create_actor(env,asset_button,pose_button,"button",i,1)
            self.actor_handles.append(actor_handle_button)

        # prepare simulation buffers and tensor storage - required to use tensor API
        self.gym.prepare_sim(self.sim)

        # Viewer Creation
        cam_props = gymapi.CameraProperties()
        self.viewer = self.gym.create_viewer(self.sim, cam_props)

        #####################  ACQUIRE TENSORS  #####################
        # forces de contact
        _net_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        net_contact_force_tensor = gymtorch.wrap_tensor(_net_contact_force_tensor)

        # pos,ori,vel,ang_vel des root ( a voir comment on fait si la compo de la room n'est pas full root
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor)

        for i in range(self.num_envs):  # on va mettre ici la création des objets
            self.albert_array[i] = AlbertCube(self.room_manager_array[i], i, 0, self.num_bodies)
            self.room_manager_array[i].add_room(Room(i,self.num_bodies))

        self.time_passed = [0 for i in range(self.num_envs)]
        self.time_episode = 10
        self.step = 0.01  # dt
        self.num_bodies = 60 # CHANGER CETTE VALEUR ########################### CHANGER ##############################
        self.curr_state = self.get_current_state()
        self.prev_state = self.get_previous_state()
        self.actions = None

    def build_basic_room(self, env, asset_base_cube, asset_door):  # construction de la structure de la chambre et stockage des blocs dans une liste
        x, y, l = 0, 0, 0
        depth = 6
        width = 11
        height = 3
        id=1
        for i in range(depth):
            for j in range(width):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(x + i, y + j, l)  # pose.r pour l'orientation
                name = "cube" + str(id)
                i+=1
                actor_handle_cube = self.gym.create_actor(env, asset_base_cube, pose, name, i, 1)
                self.actor_handles.append(actor_handle_cube)

                for z in range(height):  # MURS
                    if i == 0 or (j == 0 or j == 10):
                        if i == depth / 2 and (j == width - 1 or j == 0) and (z == 0):
                            if j == self.width - 1:
                                pose = gymapi.Transform()
                                pose.p = gymapi.Vec3(x + i, y + j, l + 1 + z)  # pose.r pour l'orientation
                                name = "door"
                                id+=1
                                actor_handle_door = self.gym.create_actor(env, asset_door, pose, name, i, 1)
                                self.actor_handles.append(actor_handle_door)
                        else:
                            pose = gymapi.Transform()
                            pose.p = gymapi.Vec3(x + i, y + j, l + 1 + z)  # pose.r pour l'orientation
                            name = "cube" + str(id)
                            id+=1
                            actor_handle_cube = self.gym.create_actor(env, asset_base_cube, pose, name, i, 1)
                            self.actor_handles.append(actor_handle_cube)

    def pre_physics_step(self, actions):
        # apply actions
        self.actions=actions # JE PEUX FAIRE CA ??? ############################ ISAAC ######################################
        for i in range(self.num_envs):
            self.albert_array[i].take_action(actions[3*i:3*(i+1)])


    def post_physics_step(self):#
        # compute observations,rewards,and resets
        self.compute_observations()
        self.compute_rewards()

        # trouver les ids à reset
        env_ids = []
        for id in range(self.num_envs):
            if (self.time_passed[id] >= self.time_episode or self.albert_array[id].has_fallen() or self.achieved_maze(id)):
                env_ids.append(id)
                self.time_passed[id] += self.step
        self.reset(env_ids)

    def reset(self, env_ids):
        # number of environments to reset
        num_resets = len(env_ids)
        for id in env_ids:
            room = self.room_manager_array[id].room_array[self.albert_array[id].actual_room]
            room.reset_room(self.root_tensor, self.albert_array[id])

            # generate random DOF positions and velocities
            pos = [np.random.uniform(1, 3),np.random.uniform(1, 5),0.75]

            ori_euler = [0,0,np.random.uniform(-np.pi, np.pi)]#ecrire le bon truc puis le mettre en quat ################################## CHANGE TO ISAAC #####################################
            ori = quat_from_euler(ori_euler)
            # rewrite root tensor
            self.root_tensor[id*self.num_bodies, :3] = pos
            self.root_tensor[id*self.num_bodies, 3:7] = ori

        self.refresh_actor_root_state_tensor(self.sim)

        self.compute_observations()
        for id in env_ids:
            self.time_passed[id] = 0

    def compute_observations(self):
        # refresh state tensor
        self.refresh_actor_root_state_tensor(self.sim)

        # ca c'est cartpole, ou changer obs_buf de dimensions ? jsp mais à trouver
        #self.obs_buf[:, 0] = self.dof_pos[:, 0]
        #self.obs_buf[:, 1] = self.dof_vel[:, 0]
        #self.obs_buf[:, 2] = self.dof_pos[:, 1]
        #self.obs_buf[:, 3] = self.dof_vel[:, 1]

        for i in range(self.num_envs):
            self.obs_buf[i]=self.albert_array[i].get_observation()

        self.update_state()

    def compute_rewards(self):
        # a revoir c'est casse couille : vidéo : à 47 min
        for i in range(self.num_envs):
            self.rew_buf[i]=self.compute_reward(i)

    def compute_reward(self,i):
        reward = 0
        contact = self.curr_state["contactPoints"] # regarder comment modifier la space du State courant
        if self.actions[3*i:3*(i+1)][2] == 1: # regarder comment passer action
            reward -= 0.05
        if (3 in contact or 4 in contact or 5 in contact):
            reward -= 0.1
        if (self.achieved_maze(i)):
            reward += 1
        if (self.button_distance(i) != 0):
            reward += 1
        if self.curr_state["CharacterPosition"][2] <= self.room_manager.room_array[0].global_coord[
            2]:  # a changer pr que ce soit qu'une fois ( quand il tombe )
            reward -= 0.5
        # compute done
        return reward


    def button_distance(self,i):# tout est à modifier ############################## CHANGE TO ISAAC #########################
        n = len(self.current_state[i]["buttonsState"])
        if self.prev_state[i] == None:
            return 0

        d = sum([np.abs(self.curr_state["buttonsState"][i][j] - self.prev_state["buttonsState"][i][j]) for j in range(n)])
        return d


    def achieved_maze(self,i):
        door_id = self.room_manager_array[i].room_array[self.albert_array[i].actual_room]

        character_pos = self.root_tensor[i*self.num_bodies]
        door_pos = self.root_tensor[i*self.num_bodies + door_id]
        dist = np.sqrt(sum([(character_pos[i] - door_pos[i]) ** (2) for i in range(2)]))
        return (dist < 0.5)  # pour l'instant 0.5 mais en vrai dépend de la dim de la sortie et du character

    def prepare_assets(self):

        # Asset 1 : Albert :
        asset_root_albert = "../../assets"  # a changer avec le dossier MJCF
        asset_file_albert = "Albert.xml"  # a changer avec le fichier MJCF
        asset_options_albert = gymapi.AssetOptions()
        asset_options_albert.fix_base_link = True  # a voir ce que c'est
        asset_options_albert.armature = 0.01  # a voir aussi

        asset_albert = self.gym.load_asset(self.sim, asset_root_albert, asset_file_albert, asset_options_albert)

        # Asset 2 : Room :
        asset_root_room = "../../assets"  # a changer avec le dossier MJCF
        asset_file_room = "Albert.xml"  # a changer avec le fichier MJCF
        asset_options_room = gymapi.AssetOptions()
        asset_options_room.fix_base_link = True  # a voir ce que c'est
        asset_options_room.armature = 0.01  # a voir aussi

        asset_room = self.gym.load_asset(self.sim, asset_root_room, asset_file_room, asset_options_room)

        return asset_albert, asset_room

    def configure_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0

        # create the ground plane
        self.gym.add_ground(self.sim, plane_params)

    def set_sim_params(self):
        # get default set of parameters
        sim_params = gymapi.SimParams()

        # set common parameters
        sim_params.dt = 0.01
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        # set PhysX-specific parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        # set Flex-specific parameters
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 20
        sim_params.flex.relaxation = 0.8
        sim_params.flex.warm_start = 0.5

        return sim_params

    def get_current_state(self):
        current_state = []
        for id in range(self.num_envs):
            current_state.append(self.albert_array[id].current_state)

        return current_state

    def get_previous_state(self):
        prev_state = []
        for id in range(self.num_envs):
            prev_state.append(self.albert_array[id].get_previous_state())

        return prev_state

    def update_state(self):
        self.curr_state = self.get_current_state()
        self.prev_state = self.get_previous_state()


def quat_from_euler(ori_euler):
    eu = Rotation.from_euler('zyx', ori_euler, degrees=False)
    quat = eu.as_quat()
    return quat