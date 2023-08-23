import gym
import mujoco.viewer
import numpy as np
import mujoco as mj
from gym.spaces import Box, MultiDiscrete, Tuple, Discrete, Dict, MultiBinary
from numpy.random import default_rng
import time
from MujocoSimu.ObjetsEnvironnement.AlbertCube import AlbertCube
from MujocoSimu.ObjetsEnvironnement.Room import Room
from MujocoSimu.ObjetsEnvironnement.RoomManager import RoomManager
from XmlConversionDirectory.xmlMerger import merge_mjcf_files


class AlbertEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, character=1):

        self.room_manager = RoomManager()

        room_manager_xml = "C:/Users/moyal/PycharmProjects/testEnviSim/xmlDirectory/Room2bis.xml"
        albert_xml = "C:/Users/moyal/PycharmProjects/testEnviSim/xmlDirectory/Actor.xml"
        # faire une fo qui construit l'objet room à partir du xml
        merge_mjcf_files(room_manager_xml, albert_xml, "AlbertEnvironment2")

        albert_environnement = "C:/Users/moyal/PycharmProjects/testEnviSim/xmlDirectory/AlbertEnvironment2.xml"
        # initialisation mujoco
        self.model = mj.MjModel.from_xml_path(albert_environnement)
        self.data = mj.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.room_manager.add_room(Room(self.model, name='room1'))
        self.character = AlbertCube(room_manager=self.room_manager, data=self.data, model=self.model)

        self.state_space = Dict(
            {
                "CharacterPosition": Box(low=np.array([0., 0., -5.]), high=np.array([10., 10., 10.])),
                "doorState": Discrete(2),
                "doorPosition": MultiDiscrete(np.array([3, 10])),  # a voir
                "buttonsState": MultiBinary(3),  # ici 3 bouttons
                "contactPoints": MultiDiscrete(np.array([6, 6, 6, 6, 6, 6]))
            }
        )
        self.curr_state = self.character.current_state
        self.prev_state = self.character.get_previous_state()

        # Observation_space : 21 * tuple(objType,distance) pour chaque Rayon
        self.observation_space = Box(
            low=np.concatenate((np.array([0 for _ in range(105)]), np.array([0 for _ in range(105)]))),
            high=np.concatenate([np.array([5 for _ in range(105)]), np.array([10 for _ in range(105)])]), shape=(210,))

        # Action_space : [rotate,move,jump] avec f possible seulement si z=zsol
        self.action_space = MultiDiscrete(np.array([3, 3, 2]))
        # [[rotGauche,rotDroite],[reculer,pas bouger,avancer],[sautArriere,saut,sautAvant]]

        # RNG :
        self.rng = default_rng()

        # Observation courante
        self.current_obs = None
        self.previous_obs = None

        # done
        self.time_episode = 10  # 10 secs
        self.time_passed = 0

        pass

    def step(self, action):
        # given current obs and action returns the next observation, the reward, done and optionally additional info
        self.character.take_action(action)
        # compute next obs
        self.current_obs = self.character.get_observation(self.viewer)
        self.update_state()

        # compute reward
        reward = 0
        contact = self.curr_state["contactPoints"]
        if action[2] == 1:
            reward -= 0.05
        if (3 in contact or 4 in contact or 5 in contact):
            reward -= 0.1
        if (self.achieved_maze()):
            reward += 1
        if (self.button_distance() != 0):
            reward += 1
        if self.curr_state["CharacterPosition"][2] <= self.room_manager.room_array[0].global_coord[
            2]:  # a changer pr que ce soit qu'une fois ( quand il tombe )
            reward -= 0.5
        # compute done

        self.time_passed += 1 / 240  # à suposer qu'un step corresponde à 1/240 eme de seconde
        done = False
        if (self.time_passed >= self.time_episode or self.character.has_fallen() or self.achieved_maze()):
            done = True

        return self.current_obs, reward, done, {}  # dictionnaire vide à la fin, pas important ici

    def reset(self):
        # position initiale : devante la première fenêtre
        # faudra changer ces params selon la taille d'albert et de la salle
        # car les positions correspondet a la position du centre de masse
        room = self.character.room_manager.room_array[self.character.actual_room]
        room.reset_room(self.model, self.character)

        x_alb = self.rng.uniform(1, 5)
        y_alb = self.rng.uniform(1, 3)
        z_alb = 0.75

        ori_euler = [0, 0, self.rng.uniform(-np.pi, np.pi)]

        self.character.reset_pos_ori([x_alb, y_alb, z_alb], ori_euler)

        # fait une requête pour avoir la première observation

        self.current_obs = self.character.get_observation(self.viewer)

        # le temps repart à 0
        self.time_passed = 0

        return self.current_obs

    def render(self):

        mj.mj_step(self.model, self.data)
        self.viewer.sync()
        time.sleep(1 / 240)

    def close(self):
        self.viewer.close()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def button_distance(self):
        n = len(self.character.current_state["buttonsState"])
        if self.prev_state == None:
            return 0

        d = sum([np.abs(self.curr_state["buttonsState"][i] - self.prev_state["buttonsState"][i]) for i in range(n)])
        return d

    def achieved_maze(self):
        char_pos = self.curr_state["CharacterPosition"]
        door_pos = self.curr_state["doorPosition"]
        dist = np.sqrt(sum([(char_pos[i] - door_pos[i]) ** (2) for i in range(2)]))
        return (dist < 0.5)  # pour l'instant 0.5 mais en vrai dépend de la dim de la sortie et du character

    def update_state(self):
        self.curr_state = self.character.current_state
        self.prev_state = self.character.get_previous_state()
