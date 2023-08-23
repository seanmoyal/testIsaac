from MujocoSimu.ObjetsEnvironnement.Cube import Cube

class Door(Cube):
    def __init__(self,id):
        self.is_opened=False
        self.id=id

    def open(self,model):
        self.is_opened=True
        pos= model.body(self.id).pos
        new_pos=[pos[0],pos[1],pos[2]+1]
        model.body(self.id).pos = new_pos

    def close(self,model):
        self.is_opened=False
        pos = model.body(self.id).pos
        new_pos=[pos[0],pos[1],pos[2]-1]
        model.body(self.id).pos=new_pos
