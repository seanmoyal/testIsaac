from Isaac.ObjetsEnvironnementTensor.CubeTensor import Cube
import torch

class Button(Cube):  # Classe des bouttons : spots verts ou l'acteur doit passer pour ouvrir la porte

    def __init__(self, id_tensor):
        self.is_pressed = torch.full((id_tensor.numel(),),False)  # variable d'état du boutton : faux si l'acteur n'est pas passée dessus
        self.id_tensor = id_tensor

    def got_pressed(self, state_tensor):  ################## A CHANGER

        state_tensor[self.id][2] = state_tensor[self.id][2] - 1.9 * 0.02  # supposing 0.02 is the button's size
        self.is_pressed = True

