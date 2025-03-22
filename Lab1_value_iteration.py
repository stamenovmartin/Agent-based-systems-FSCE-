import gymnasium as gym
import numpy as np
from mdp import value_iteration, policy_iteration


def simulate_policy(env, policy, iterations):
    total_steps = 0
    total_reward = 0

    for _ in range(iterations):
        state, info = env.reset()
        steps = 0
        episode_reward = 0
        while True:
            action = np.argmax(policy[state])
            state, reward, terminated, truncated, info = env.step(action)
            steps += 1
            episode_reward += reward
            if terminated or truncated: break
        total_steps += steps
        total_reward += episode_reward

    avg_steps = total_steps / iterations
    avg_reward = total_reward / iterations
    return avg_steps, avg_reward


if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode='human')
    env = env.unwrapped

    discount_factors = [0.5, 0.7, 0.9]
    print("Задача 1 за value iteration \n")
    for gamma in discount_factors:
        policy_vi, V_vi = value_iteration(env=env, num_actions=env.action_space.n, num_states=env.observation_space.n,
                                          theta=0.00001, discount_factor=gamma)
        print(f"\n Discount Factor што го користам: {gamma}")
        state, info = env.reset()
        while True:
            env.render()
            action = np.argmax(policy_vi[state])
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.render()
                break

        avg_steps_50, avg_reward_50 = simulate_policy(env, policy_vi, 50)

        print(f"Просечни чекори за 50 итерации: {avg_steps_50:.2f}")
        print(f"Просечна награда за 50 итерации: {avg_reward_50:.2f}")

        avg_steps_100, avg_reward_100 = simulate_policy(env, policy_vi, 100)
        print(f"Просечни чекори за 100 итерации: {avg_steps_100:.2f}")
        print(f"Просечна награда за 100 итерации: {avg_reward_100:.2f}")

    print("Задача 2 за policy iteration \n")
    for gamma in discount_factors:
        policy_pi, V_pi = policy_iteration(
            env=env,
            num_actions=env.action_space.n,
            num_states=env.observation_space.n,
            discount_factor=gamma
        )
        print(f"\nDiscount Factor: {gamma}")
        state, info = env.reset()
        while True:
            env.render()
            action = np.argmax(policy_pi[state])
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.render()
                break

        # Тестирање на политиката во 50 епизоди
        avg_steps_50, avg_reward_50 = simulate_policy(env, policy_pi, 50)
        print(f"Просечни чекори за 50 итерации: {avg_steps_50:.2f}")
        print(f"Просечна награда за 50 итерации: {avg_reward_50:.2f}")

        # Тестирање на политиката во 100 епизоди
        avg_steps_100, avg_reward_100 = simulate_policy(env, policy_pi, 100)
        print(f"Просечни чекори за 100 итерации: {avg_steps_100:.2f}")
        print(f"Просечна награда за 100 итерации: {avg_reward_100:.2f}")

    env.close()
