from Isaac.ObjetsEnvironnementTensor.CubeTensor import Cube
import torch
class Door(Cube):
    def __init__(self,id_tensor):
        self.is_opened=torch.full((id_tensor.numel(),),False)
        self.id_tensor=id_tensor

    def open(self,state_tensor,doors_to_open_tensor):###################### FINI #########################
        self.is_opened=torch.where(doors_to_open_tensor,True,self.is_opened)
        pos = state_tensor[self.id_tensor][:3]
        new_pos= pos+torch.tensor([0,0,1])
        state_tensor[self.id_tensor][:3] = torch.where(doors_to_open_tensor,new_pos,pos)

    def close(self,state_tensor,doors_to_close):################ FINI #############################
        self.is_opened=torch.where(doors_to_close,False,self.is_opened)
        pos = state_tensor[self.id_tensor][:3]
        new_pos= pos+torch.tensor([0,0,-1])
        state_tensor[self.id_tensor][:3] = torch.where(doors_to_close,new_pos,pos)
