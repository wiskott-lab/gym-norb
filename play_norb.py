import gym
import norb
import sys, termios, tty, os, time

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def convert_action(act_rep):
    if act_rep is 'a' or act_rep is 'A':
        return 0
    if act_rep is 'd' or act_rep is 'D':
        return 1
    if act_rep is 'w' or act_rep is 'W':
        return 2
    if act_rep is 's' or act_rep is 'S':
        return 3
    else:
        return None

if __name__ == '__main__':
    env = gym.make('Norb-v0')
    env.reset()
    
    total_reward = 0
    while True:
        env.render()
        action = convert_action(getch())
        if action is None:
            continue

        obs, reward, done, info = env.step(action)

        total_reward += reward
        print('reward: {}'.format(reward))
        if done:
            print('Environment done, total reward: {}'.format(total_reward))
            total_reward = 0
            env.reset()

    env.close()

