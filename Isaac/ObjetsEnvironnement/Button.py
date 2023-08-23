from Isaac.ObjetsEnvironnement.Cube import Cube
from scipy.spatial.transform import Rotation
class Button(Cube): # Classe des bouttons : spots verts ou l'acteur doit passer pour ouvrir la porte

    def __init__(self,id):
        self.is_pressed = False # variable d'état du boutton : faux si l'acteur n'est pas passée dessus
        self.id=id


    def got_pressed(self,model): # fonction activée lorsque l'acteur passe sur le boutton
        model.geom(model.body(self.id).geomadr[0]).rgba=[0.7,0.7,0.7,1.0]

        model.body(self.id).pos[2]=model.body(self.id).pos[2]-1.9*model.geom(model.body(self.id).geomadr[0]).size[2]
        self.is_pressed=True



def quaternion_from_euler(euler):
    eu=Rotation.from_euler('xyz',euler,degrees=False)
    quaternion=eu.as_quat()
    return quaternion