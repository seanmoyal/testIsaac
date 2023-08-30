from Isaac.ObjetsEnvironnement.Button import Button
from Isaac.ObjetsEnvironnement.Door import Door
from scipy.spatial.transform import Rotation
import torch

class Room:  # classe d'une chambre ( niveau )
    def __init__(self,env_id,num_bodies,num_envs):

        self.num_envs=num_envs

        self.global_coord = torch.zeros((num_envs,3))  # l=0.5 # Coordonées globales de la chambre
        self.buttons_array_tensor = torch.tensor([{} for _ in range(num_envs)])
        self.floor_array_tensor = None
        self.wall_array_tensor = None
        self.iblocks_array_tensor = None
        self.fences_array_tensor = None
        self.door_array_tensor = None
        self.depth = 6
        self.width = 11
        self.height = 3
        self.env_id=env_id ################ A CHANGER #############################
        self.num_bodies=num_bodies
        self.build_basic_room()

        # DEFINITION DU STATE :

    def build_basic_room(self):  # construction de la structure de la chambre et stockage des blocs dans une liste
        x, y, l = 0, 0, 0
        id = 1
        for i in range(self.depth):
            for j in range(self.width):
                torch.cat((self.floor_array_tensor,[[id+self.num_bodies*env_id] for env_id in range(self.num_envs)]),dim=1)
                id+=1

                for z in range(self.height):  # MURS
                    if i == 0 or (j == 0 or j == 10):
                        if i == self.depth / 2 and (j == self.width - 1 or j == 0) and (z == 0):
                            if j == self.width - 1:
                                self.door_array[:,0]=torch.tensor([id+self.num_bodies*env_id for env_id in range(self.num_envs)])
                                self.door_array[:,1]=Door(torch.tensor([id+self.num_bodies*env_id for env_id in range(self.num_envs)]))
                                id+=1
                        else:
                            torch.cat((self.wall_array_tensor,[[id+self.num_bodies*env_id] for env_id in range(self.num_envs)]),dim=1)
                            id+=1

        button_id_tensor = torch.tensor([id+self.num_bodies*env_id for env_id in (self.num_envs)])
        button_values = Button(button_id_tensor)
        for d, key, value in zip(self.buttons_array_tensor, button_id_tensor, button_values):
            d[key] = value


    def init_room(self, model, name='room1'): ################################ CHANGE TO ISAAC ####################################
        self.id = model.body(name).id
        id = self.id + 1
        while model.body(id).parentid[0] == self.id:
            body = model.body(id)
            name = body.name
            geom_id = body.geomadr[0]
            if "floor" in name:
                self.floor_array.append(id)
            elif "wall" in name:
                self.wall_array.append(id)
            elif "iblock" in name:
                self.iblocks_array.append(id)
            elif "door" in name:
                self.door_array = [id, Door(id)]
            elif "button" in name:
                self.buttons_array[id] = Button(id)
            elif "fence" in name:
                self.fences_array.append(id)
            id += 1

    def check_buttons_pushed(self, state_tensor):
        if not self.door_array[1].is_opened:
            a = False
            for button in self.buttons_array.values():
                if not button.is_pressed:
                    a = True
            if not a:
                self.door_array[1].open(state_tensor)

    def reset_room(self,state_tensor,character):  # dans la vidéo chaque simu se termine après 10s, on appelera cette fo après 10 s de simu
        # Evidemment elle est à compléter
        for id_button in self.buttons_array.keys():# j'ai viré le changement de rgb, inutile
            state_tensor[id_button][2]= state_tensor[id_button][2] + 1.9 * 0.02 # supposing that 0.02 is the size of the button
            # truc a changer au dessus, jsp pq mais ca reset pas à la bonne hauteur
            # p.changeVisualShape(id_button,-1,rgbaColor=[0,1,0,1])
            self.buttons_array[id_button].is_pressed = False
        character.reset_time()

        if self.door_array[1].is_opened:
            self.door_array[1].close(state_tensor)

    def translate(self, state_tensor, id,
                  translation):  # fonction de translation utilisiée dans changeglobal_coord(), translate un objet d'identifiant Id
        old_position = state_tensor[id][:3]

        new_position = [
            old_position[0] + translation[0],
            old_position[1] + translation[1],
            old_position[2] + translation[2]
        ]

        state_tensor[id][:3] = new_position

    def get_id_values_button_from_button_array(self):
        id_values_button_tensor = torch.tensor([[self.buttons_array_tensor[i].keys(),self.buttons_array_tensor[i].values()] for i in range(self.num_envs)])
        split_tensors = torch.split(id_values_button_tensor,split_size_or_sections = 1,dim=2)
        return split_tensors[0],split_tensors[1]

def quaternion_from_euler(euler):
    eu = Rotation.from_euler('xyz', euler, degrees=False)
    quaternion = eu.as_quat()
    return quaternion
