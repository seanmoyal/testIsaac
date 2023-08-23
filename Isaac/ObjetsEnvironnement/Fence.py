from Isaac.ObjetsEnvironnement.Cube import Cube
class Fence(Cube): # classe d'une barrière pour empêcher l'ia de passer

    def __init__(self,depth,height): #ajouter l'id
        self.depth = depth
        self.height = height
