import BaseTask
from isaacgym import gymapi
from numpy import random
import gymtorch
from Isaac.ObjetsEnvironnement.RoomManager import RoomManager
from Isaac.ObjetsEnvironnement.AlbertCube import AlbertCube
from Isaac.ObjetsEnvironnement.Room import Room
import numpy as np


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
        num_envs = 64
        envs_per_row = 8
        env_spacing = 2.0
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        # cache some common handles for later use
        envs = []
        self.actor_handles = []

        # instanciate Room Manager
        self.room_manager_array = np.array([RoomManager() for i in range(num_envs)])
        self.albert_array = np.empty((num_envs,))

        # create and populate the environments
        for i in range(num_envs):
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
        root_tensor = gymtorch.wrap_tensor(_root_tensor)

        for i in range(num_envs):  # on va mettre ici la création des objets
            self.albert_array[i] = AlbertCube(self.room_manager_array[i], i, 0, num_bodies)
            self.room_manager_array[i].add_room(Room(i,num_bodies))

    def build_basic_room(self, env, asset_base_cube, asset_door):  # construction de la structure de la chambre et stockage des blocs dans une liste
        x, y, l = 0, 0, 0
        depth = 6
        width = 11
        height = 3
        for i in range(depth):
            for j in range(width):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(x + i, y + j, l)  # pose.r pour l'orientation
                name = "cube" + str(i * j)
                actor_handle_cube = self.gym.create_actor(env, asset_base_cube, pose, name, i, 1)
                self.actor_handles.append(actor_handle_cube)

                for z in range(height):  # MURS
                    if i == 0 or (j == 0 or j == 10):
                        if i == depth / 2 and (j == width - 1 or j == 0) and (z == 0):
                            if j == self.width - 1:
                                pose = gymapi.Transform()
                                pose.p = gymapi.Vec3(x + i, y + j, l + 1 + z)  # pose.r pour l'orientation
                                name = "door"
                                actor_handle_door = self.gym.create_actor(env, asset_door, pose, name, i, 1)
                                self.actor_handles.append(actor_handle_door)
                        else:
                            pose = gymapi.Transform()
                            pose.p = gymapi.Vec3(x + i, y + j, l + 1 + z)  # pose.r pour l'orientation
                            name = "cube" + str(i * j + z)
                            actor_handle_cube = self.gym.create_actor(env, asset_base_cube, pose, name, i, 1)
                            self.actor_handles.append(actor_handle_cube)

    def pre_physics_step(self, actions):
        # apply actions
        # prepare DOF force tensor
        forces = torch.zeros((num_envs, dofs_per_env), dtype=torch.float32, device=self.device)

        # scale actions and write to cart DOF slice
        forces[:, 0] = actions * self.max_push_force
        # apply the forces to all actors
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(forces))

    def post_physics_step(self):
        # compute observations,rewards,and resets
        ...

    def reset(self, env_ids):
        # number of environments to reset
        num_resets = len(env_ids)

        # generate random DOF positions and velocities
        p = 0.3 * (torch.rand((num_resets, dofs_per_env), device=self.device) - 0.5)
        v = 0.5 * (torch.rand((num_resets, dofs_per_env), device=self.device) - 0.5)

        # write new states to DOF state tensor
        self.dof_states[env_ids, 0] = p
        self.dof_states[env_ids, 1] = v

        # Apply the new DOF states for the selected envs, using env_ids as the actor index tensor
        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_states_desc, gymtorch.unwrap_tensor(env_ids),
                                              num_resets)

    def compute_observations(self):
        # refresh state tensor
        self.gym.refresh_dof_state_tensor(self.sim)

        # copy DOF states to observation tensor
        self.obs_buf[:, 0] = self.dof_pos[:, 0]
        self.obs_buf[:, 1] = self.dof_vel[:, 0]
        self.obs_buf[:, 2] = self.dof_pos[:, 1]
        self.obs_buf[:, 3] = self.dof_vel[:, 1]

    def compute_rewards(self):
        # a revoir c'est casse couille : vidéo : à 47 min
        self.rew_buf[:] = compute_reward(params)

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
