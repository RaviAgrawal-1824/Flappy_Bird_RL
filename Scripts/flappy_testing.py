import pickle
import time
import pygame
import numpy as np
import flappy_bird_gym
import matplotlib.pyplot as plt
import pandas as pd

with open('bird.pkl', 'rb') as f:
    Q_lookup = pickle.load(f)

env = flappy_bird_gym.make("FlappyBird-v0", normalize_obs=False, pipe_gap=150)
time_steps_list,score_list,reward_list=[],[],[]

def get_state(obs,state):
    hz_dis,vr_dis=0.0,0.0
    hz_dis=state[0]//4
    vr_dis=state[1]//3
    return(obs[2:]+(hz_dis,vr_dis))

k,n=0,0
for i in range(2000):

    done,n,reward,score=0,1,0,0
    new_pos=get_state((None,None,None,None,None,None),env.reset())
    if i%100==0:
        print('--'*50,'\n',i,'epoch')

    while(n<3700 and not done):
       
        # env.render()

        prev_pos=new_pos
        if (not prev_pos in Q_lookup):
            Q_lookup[prev_pos]={0:0.0,1:0.0}
            k=k+1
        
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        # time.sleep(1 / 30)  # FPS
        

        act=np.argmax(list(Q_lookup[prev_pos].values())) # conversion of dict.values() in list is important for correct greddy action selection.

        state,rew,done,info=env.step(act)  # taking a step in the environment
        rew=0
        if(info['score']!=score):
            rew=10
            score=info['score']
        if done == True:
            rew=-10

        reward=reward+rew
        
        new_pos=get_state(prev_pos,state)
        if (not new_pos in Q_lookup):
            Q_lookup[new_pos]={0:0.0,1:0.0}
            k=k+1

        n=n+1
    time_steps_list.append(n)
    score_list.append(score)
    reward_list.append(reward)

print('testing done')
# print(Q_lookup,'\n\n',score_list,'\n\n',reward_list,'\n\n',time_steps_list)
# print(Q_lookup,'\n\n',score_list,'\n\n',reward_list,'\n\n',time_steps_list)
# print('\n\n',score_list,'\n\n',reward_list,'\n\n',time_steps_list)
# print(Q_lookup,'\n\n')
# x = np.arange(len(reward_list))
time_plot,score_plot,rew_plot=[],[],[]
sum_time,sum_score, sum_rew=0,0,0
sum_time = pd.DataFrame({'data':time_steps_list})
sum_score = pd.DataFrame({'data':score_list})
sum_rew = pd.DataFrame({'data':reward_list})
time_plot = list(sum_time['data'].rolling(window=50).mean())
score_plot = list(sum_score['data'].rolling(window=50).mean())
rew_plot = list(sum_rew['data'].rolling(window=50).mean())

fig1=plt.figure()
plt.plot(reward_list,label='actual reward')
plt.plot(rew_plot,label='Average reward')
plt.xlabel('No. of Episodes')
plt.ylabel('Reward Function')
plt.title('[Testing] Q-learning on Flappy_bird_gym')
plt.legend(loc='upper left')
plt.show()

fig1=plt.figure()
plt.plot(score_list,label='Actual points')
plt.plot(score_plot,label='Average points')
plt.xlabel('No. of Episodes')
plt.ylabel('Points scored')
plt.title('[Testing] Q-learning on Flappy_bird_gym')
plt.legend(loc='upper left')
plt.show()

fig1=plt.figure()
plt.plot(time_steps_list,label='Actual time steps')
plt.plot(time_plot,label='Average time steps')
plt.xlabel('No. of Episodes')
plt.ylabel('Time steps alive')
plt.title('[Testing] Q-learning on Flappy_bird_gym')
plt.legend(loc='upper left')
plt.show()

env.close()