from Isaac.ObjetsEnvironnement.Cube import Cube
import torch
class Door(Cube):
    def __init__(self,id_tensor):
        self.is_opened=torch.full((id_tensor.numel(),),False)
        self.id_tensor=id_tensor

    def open(self,state_tensor):###################### A CHANGER SELON LA MANIERE DE TRAITER CA
        self.is_opened=True
        pos = state_tensor[self.id][:3]
        new_pos=[pos[0],pos[1],pos[2]+1]
        state_tensor[self.id][:3] = new_pos

    def close(self,state_tensor):################ pareil
        self.is_opened=False
        pos = state_tensor[self.id][:3]
        new_pos=[pos[0],pos[1],pos[2]-1]
        state_tensor[self.id][:3]=new_pos
