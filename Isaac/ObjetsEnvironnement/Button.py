from Isaac.ObjetsEnvironnement.Cube import Cube
from scipy.spatial.transform import Rotation


class Button(Cube):  # Classe des bouttons : spots verts ou l'acteur doit passer pour ouvrir la porte

    def __init__(self, id):
        self.is_pressed = False  # variable d'état du boutton : faux si l'acteur n'est pas passée dessus
        self.id = id

    def got_pressed(self, state_tensor):  # I didn't put any color change

        state_tensor[self.id][2] = state_tensor[self.id][2] - 1.9 * 0.02  # supposing 0.02 is the button's size
        self.is_pressed = True


def quaternion_from_euler(euler):
    eu = Rotation.from_euler('xyz', euler, degrees=False)
    quaternion = eu.as_quat()
    return quaternion
