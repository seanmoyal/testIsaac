from ClassFrameWork import AlbertEnvironment

env = AlbertEnvironment()
env.create_sim()
gym = env.gym
sim = env.sim
viewer = env.viewer
# Running the sim Sith Viewer Incorporation
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim);
    gym.draw_viewer(viewer, sim, True)

    # update tensors
    gym.refresh_actor_root_state(sim)
    gym.refresh_net_contact_force_tensor(sim)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

# exit
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
