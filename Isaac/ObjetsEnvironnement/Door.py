from Isaac.ObjetsEnvironnement.Cube import Cube

class Door(Cube):
    def __init__(self,id):
        self.is_opened=False
        self.id=id

    def open(self,state_tensor):
        self.is_opened=True
        pos = state_tensor[self.id][:3]
        new_pos=[pos[0],pos[1],pos[2]+1]
        state_tensor[self.id][:3] = new_pos

    def close(self,state_tensor):
        self.is_opened=False
        pos = state_tensor[self.id][:3]
        new_pos=[pos[0],pos[1],pos[2]-1]
        state_tensor[self.id][:3]=new_pos
