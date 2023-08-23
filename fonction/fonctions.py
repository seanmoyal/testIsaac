import gym
import gymtorch


# returns true if
def check_touching_ground(sim,id_env,id,num_bodies_in_env):# on remplacera les bails d'id par des self.id, peut être que id sera global enft
    net_contact_force_data = gym.acquire_net_contact_force_tensor(sim)
    net_contact_force_tensor = gymtorch.wrap_tensor(net_contact_force_data)
    actor_collision = net_contact_force_tensor[id_env*num_bodies_in_env+id]
    if actor_collision[3]!=0:
        return True
    return False

def check_button_being_pushed(sim,id_env,Buttons,prev_buttons_state,num_bodies_in_env):# a changer pour l'id si button_id est global
    net_contact_force_data = gym.acquire_net_contact_force_tensor(sim)
    net_contact_force_tensor = gymtorch.wrap_tensor(net_contact_force_data)
    n = len(Buttons.keys())
    if prev_buttons_state==None:
        prev_buttons_state = []
        for button_id in Buttons.keys():
            actor_collision = net_contact_force_tensor[id_env * num_bodies_in_env + button_id]
            prev_buttons_state.append(actor_collision)

    else:
        i=0
        for button_id in Buttons.keys():
            actor_collision = net_contact_force_tensor[id_env * num_bodies_in_env + button_id]
            if prev_buttons_state!=actor_collision and not Buttons[button_id].is_pressed:
                Buttons[button_id].got_pressed()
            prev_buttons_state[i]=actor_collision
            i+=1
