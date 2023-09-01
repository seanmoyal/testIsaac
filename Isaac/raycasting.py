import torch

def check_collision_AABB(to_check_tensor,Amin,Amax,Bmin,Bmax):
    condition = (Amin<=Bmax | Amax>=Bmin)
    result = torch.nn.functional.conv1d(condition.unsqueeze(0).float(), torch.ones(1, 3).float()).squeeze() == 3
    return result & to_check_tensor



def ray_collision(self,point_of_origin_tensor,end_pos_ray):
        room_tensor = self.room_manager.room_array[self.actual_room]
        distance = end_pos_ray - point_of_origin_tensor
        id_distance_collision_tensor=torch.full((self.num_envs,),[-1,10])

        to_check_tensor = torch.full((self.num_envs,),False)
        for i in range (100):
            ray_pos = point_of_origin_tensor + distance*i/100
            Amin_ray = ray_pos - 0.1
            Amax_ray = ray_pos + 0.1


            for button in room_tensor.buttons_array_tensor:
                button_pos = self.state_tensor[button.id_tensor]
                Amin_button = button_pos - torch.tensor([0.5,0.5,0.1])
                Amax_button = button_pos + torch.tensor([0.5,0.5,0.1])
                result = check_collision_AABB(to_check_tensor,Amin_button, Amax_button, Amin_ray, Amax_ray)
                id_distance_collision_tensor = torch.where(result,torch.tensor(torch.cat(button.id_tensor,[10*i/100],axis=1)),id_distance_collision_tensor)
                to_check_tensor = id_distance_collision_tensor[:,0]==-1


            for box_id in room_tensor.floor_array_tensor:
                box_pos = self.state_tensor[box_id][:3]
                Amin_box = box_pos - 0.5
                Amax_box = box_pos + 0.5
                result = check_collision_AABB(to_check_tensor,Amin_box,Amax_box,Amin_ray,Amax_ray)
                id_distance_collision_tensor = torch.where(result,torch.tensor(torch.cat(box_id,[10*i/100],axis=1)),id_distance_collision_tensor)
                to_check_tensor = id_distance_collision_tensor[:, 0] == -1

            for box_id in room_tensor.wall_array_tensor:
                box_pos = self.state_tensor[box_id][:3]
                Amin_box = box_pos - 0.5
                Amax_box = box_pos + 0.5
                result = check_collision_AABB(to_check_tensor,Amin_box,Amax_box,Amin_ray,Amax_ray)
                id_distance_collision_tensor = torch.where(result,torch.tensor(torch.cat(box_id,[10*i/100],axis=1)),id_distance_collision_tensor)
                to_check_tensor = id_distance_collision_tensor[:, 0] == -1

            # for door :
            door_pos = self.state_tensor[room_tensor.door_array_tensor[0]]
            Amin_box = door_pos - 0.5
            Amax_box = door_pos + 0.5
            result = check_collision_AABB(to_check_tensor,Amin_box, Amax_box, Amin_ray, Amax_ray)
            id_distance_collision_tensor = torch.where(result, torch.tensor(torch.cat(room_tensor.door_array_tensor[0], [10 * i / 100], axis=1)), id_distance_collision_tensor)
            to_check_tensor = id_distance_collision_tensor[:, 0] == -1

        return id_distance_collision_tensor




