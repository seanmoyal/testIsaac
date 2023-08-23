from MujocoSimu.ObjetsEnvironnement.Button import Button
from MujocoSimu.ObjetsEnvironnement.Door import Door
from scipy.spatial.transform import Rotation


class Room:  # classe d'une chambre ( niveau )
    def __init__(self, model, name='room1'):
        self.global_coord = model.body(name).pos  # l=0.5 # Coordonées globales de la chambre
        self.buttons_array = {}
        self.floor_array = []
        self.wall_array = []
        self.iblocks_array = []
        self.fences_array = []
        self.door_array = {}
        self.depth = 6
        self.width = 11
        self.height = 3
        self.init_room(model, name)

        # DEFINITION DU STATE :

    def init_room(self, model, name='room1'):
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

    def change_global_coord(self, data, x, y,
                            l):  # fonction qui translate une chambre d'un endroit à un autre ( translation de tout ce qu'il y a dedans aussi)
        old_global_coord = self.global_coord
        self.global_coord = [x, y, l]
        translation = [x - old_global_coord[0], y - old_global_coord[1], l - old_global_coord[2]]

        for cube_id in self.floor_array:
            self.translate(data, cube_id, translation)

        for cube_id in self.wall_array:
            self.translate(data, cube_id, translation)

        for button_id in self.buttons_array.keys():
            self.translate(data, button_id, translation)

        for fence_id in self.fences_array:
            self.translate(data, fence_id, translation)

        for iblock_id in self.iblocks_array:
            self.translate(data, iblock_id, translation)

        self.translate(data, self.door_array[0], translation)

    def check_buttons_pushed(self, model):
        if not self.door_array[1].is_opened:
            a = False
            for button in self.buttons_array.values():
                if not button.is_pressed:
                    a = True
            if not a:
                self.door_array[1].open(model)

    def reset_room(self, model,
                   character):  # dans la vidéo chaque simu se termine après 10s, on appelera cette fo après 10 s de simu
        # Evidemment elle est à compléter
        for id_button in self.buttons_array.keys():
            model.geom(model.body(id_button).geomadr[0]).rgba = [0, 1, 0, 1.0]
            model.body(self.id).pos[2] = model.body(id_button).pos[2] + 1.9 * \
                                         model.geom(model.body(id_button).geomadr[0]).size[2]
            # truc a changer au dessus, jsp pq mais ca reset pas à la bonne hauteur
            # p.changeVisualShape(id_button,-1,rgbaColor=[0,1,0,1])
            self.buttons_array[id_button].is_pressed = False
        character.reset_time()

        if self.door_array[1].is_opened:
            self.door_array[1].close(model)

    def translate(self, data, id,
                  translation):  # fonction de translation utilisiée dans changeglobal_coord(), translate un objet d'identifiant Id
        old_position = data.xpos[id]

        new_position = [
            old_position[0] + translation[0],
            old_position[1] + translation[1],
            old_position[2] + translation[2]
        ]

        data.xpos[id] = new_position


def quaternion_from_euler(euler):
    eu = Rotation.from_euler('xyz', euler, degrees=False)
    quaternion = eu.as_quat()
    return quaternion
