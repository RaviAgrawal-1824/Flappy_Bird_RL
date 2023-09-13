# 1.6423611111111112 one_max    1.75 horizontal     473
# 0.2673611111111111 one_min    0.25 horizontal     77
# -0.23046875 two_min   -0.25 vertical      -118
# 0.685546875 two_max   0.75 vertical       351
# considering 2.5 total difference and 5000 as state space count, 0.0005 as discretization accuracy
# effect of one flap lasts upto 6-8 observations as it rotates bird and resets acceleartion and velocity

# optimise reward, add score cases in it, increase epoch, increase state accuracy, increaes state length, looping Q table


import time
import pandas as pd
import pickle
import random
import pygame
import numpy as np
import flappy_bird_gym
import matplotlib.pyplot as plt

# env = flappy_bird_gym.make("FlappyBird-v0", normalize_obs=False)
env = flappy_bird_gym.make("FlappyBird-v0", normalize_obs=False, pipe_gap=150)

# one_min,one_max,two_min,two_max=2,-2,2,-2
obs=(None,None,None,None,None,None,None,None)
# obs = obs[2:] + env.reset()

    # for event in pygame.event.get():
            # if event.type == pygame.QUIT:
            #     pygame.quit()


time_steps_list,score_list,reward_list=[],[],[]
Q_lookup={}
# with open('bird_1.pkl', 'rb') as f:
#     Q_lookup = pickle.load(f)

gamma= 0.7
alpha=1/10  # step size can be reduced further
Lambda=0.9
epsilon=1

max_epoch,min_epoch=800000,0

def update(prev_pos,new_pos,action,imm_reward,Q_table):
    Qval_max = max(list(Q_table[new_pos].values()))
    change = (imm_reward+gamma*Qval_max) - Q_table[prev_pos][action]
    Q_table[prev_pos][action]=Q_table[prev_pos][action]+alpha*change

def get_state(obs,state):
    hz_dis,vr_dis=0.0,0.0
    hz_dis=state[0]//4
    vr_dis=state[1]//3
    return(obs[2:]+(hz_dis,vr_dis))
# def get_state(obs):
#     hz_dis,vr_dis=0.0,0.0
#     hz_dis=obs[0]//0.005
#     vr_dis=obs[1]//0.005
#     return((hz_dis,vr_dis))

k,n=0,0
for i in range(min_epoch,int(max_epoch*1.1)):

    done,n,reward,score=0,1,0,0
    new_pos=get_state((None,None,None,None,None,None),env.reset())
    # new_pos=get_state((None,None,None,None,None,None,None,None,None,None,None,None),env.reset())
    
    if i%1000==0:
        print('--'*50,'\n',i,'epoch')

    if(i>0 and i<max_epoch+1):
        epsilon=1-i/max_epoch
    # elif(i>max_epoch):
    #     epsilon=0

    # if(i>(max_epoch/3) and i<max_epoch*1.1 and i%1000==0):
    #     print(k)
    # if(i>(max_epoch/1.4) and i<max_epoch*1.1 and i%100==0):
    #     # print(k)
    #     print(Q_lookup)

    while(n<5000 and not done):
        # env.render()

        prev_pos=new_pos
        if (not prev_pos in Q_lookup):
            Q_lookup[prev_pos]={0:0.0,1:0.0}
            k=k+1

        prob_decider=random.uniform(0,1)  # mechanism for choosing action greedily or randomly controlled by epsilon.
        if(epsilon>=prob_decider):
            act=random.randint(0,1)
        else:
            act=np.argmax(list(Q_lookup[prev_pos].values())) # conversion of dict.values() in list is important for correct greddy action selection.
            # if(n<30):
            #     print(n,'iteration and action chosen greedily')

        state,rew,done,info=env.step(act)  # taking a step in the environment
        rew=0
        if(info['score']!=score):
            rew=1
            score=info['score']
            # print('&'*100,i,'epoch')

        if done == True:
            rew=-1

        reward=reward+rew
        
        new_pos=get_state(prev_pos,state)
        if (not new_pos in Q_lookup):
            Q_lookup[new_pos]={0:0.0,1:0.0}
            k=k+1

        update(prev_pos,new_pos,act,rew,Q_lookup)

        # if(not n>600 or done):
        # if(reward==120):
            # print('&'*100)
            # print("iteration",n,'and done is',done,'and reward is',reward)
        n=n+1
    time_steps_list.append(n)
    score_list.append(score)
    reward_list.append(reward)

print('training done')


k,n=0,0
for i in range(80000):

    done,n,reward,score=0,1,0,0
    new_pos=get_state((None,None,None,None,None,None),env.reset())
    if i%100==0:
        print('--'*50,'\n',i,'epoch')

    while(n<5000 and not done):
       
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

with open('bird.pkl', 'wb') as f:
    pickle.dump(Q_lookup, f)

time_plot,score_plot,rew_plot=[],[],[]
sum_time,sum_score, sum_rew=0,0,0
# for i in range(len(time_steps_list)):
#     # sum_rew=sum_rew + reward_list[i]
#     # sum_score=sum_score + score_list[i]
#     # sum_time=sum_time + time_steps_list[i]

    # time_plot.append(time_steps_list[i-100:i]/(100))
    # score_plot.append(sum_score/(i+1))
    # rew_plot.append(sum_rew/(i+1))

sum_time = pd.DataFrame({'data':time_steps_list})
sum_score = pd.DataFrame({'data':score_list})
sum_rew = pd.DataFrame({'data':reward_list})
time_plot = list(sum_time['data'].rolling(window=100).mean())
score_plot = list(sum_score['data'].rolling(window=100).mean())
rew_plot = list(sum_rew['data'].rolling(window=100).mean())

print('all done')

fig1=plt.figure()
plt.plot(reward_list,label='actual reward')
plt.plot(rew_plot,label='Average reward')
plt.xlabel('No. of Episodes')
plt.ylabel('Reward Function')
plt.title('[Training] Q-learning on Flappy_bird_gym')
plt.legend(loc='upper left')
plt.show()

fig1=plt.figure()
plt.plot(score_list,label='Actual points')
plt.plot(score_plot,label='Average points')
plt.xlabel('No. of Episodes')
plt.ylabel('Points scored')
plt.title('[Training] Q-learning on Flappy_bird_gym')
plt.legend(loc='upper left')
plt.show()

fig1=plt.figure()
plt.plot(time_steps_list,label='Actual time steps')
plt.plot(time_plot,label='Average time steps')
plt.xlabel('No. of Episodes')
plt.ylabel('Time steps alive')
plt.title('[Training] Q-learning on Flappy_bird_gym')
plt.legend(loc='upper left')
plt.show()

env.close()
