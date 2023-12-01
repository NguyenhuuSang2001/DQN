from lib import * 

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        # state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, info, _ = env.step(action)
            # next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_reward = test_env(agent.model, env, agent.get_device())
                episode_rewards.append(episode_reward)
                print( str(episode) + ", " + str(episode_reward))
                # print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards

def test_env(model, env, device='cpu', vis=False):
    rewards = []

    for i in range(10):
        state, _ = env.reset()
        # state = env.reset()
        if vis: env.render()
        done = False
        total_reward = 0
        while not done:
            # print("state:", state)
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            act = model(state)
            ind_act = torch.argmax(act, dim=1).item()
            # print(ind_act.item())
            next_state, reward, done, info, _ = env.step(ind_act)
            # next_state, reward, done, info = env.step(ind_act)
            state = next_state
            if vis: env.render()
            total_reward += reward
        rewards.append(total_reward)
       
    r_avg = np.mean(rewards)

    return r_avg
