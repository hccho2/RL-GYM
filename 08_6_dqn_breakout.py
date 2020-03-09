"""
state diff & stack 방식
double dqn, dueling dqn 모두 적용.


https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import deque
from functools import partial
#from scipy.misc import imresize
import scipy
from skimage.transform import resize
from skimage.color import rgb2gray
import os,time,random
import pickle
new_HW = [84, 84] 
height_range=(31, 199)

def preprocessing(image, new_HW, height_range=(35, 195)):
    image = crop_image(image, height_range)  # (210, 160, 3)  --> (160, 160, 3)
    image = np.uint8(resize(rgb2gray(image), new_HW, mode='reflect') * 255)
    image = np.expand_dims(image, axis=2)  # (80, 80, 1)
    return image

def preprocessing2(image, new_HW, height_range=(35, 195)):
    image = crop_image(image, height_range)  # (210, 160, 3)  --> (160, 160, 3)
    image = resize(rgb2gray(image), new_HW, mode='reflect')
    image = np.expand_dims(image, axis=2)  # (80, 80, 1)
    return image
def crop_image(image, height_range=(35, 195)):
    """Crops top and bottom

    Args:
        image (3-D array): Numpy image (H, W, C)
        height_range (tuple): Height range between (min_height, max_height)
            will be kept

    Returns:
        image (3-D array): Numpy image (max_H - min_H, W, C)
    """
    h_beg, h_end = height_range
    return image[h_beg:h_end, ...]



class Agent(object):

    def __init__(self, input_dim, output_dim,h_size=512, logdir="logdir", checkpoint_dir="checkpoints"):
        """Agent class

        Args:
            input_dim (tuple): The input shape (H, W, C)
            output_dim (int): Number of actions
            logdir (str): Directory to save `summary`
            checkpoint_dir (str): Directory to save `model.ckpt`

        Notes:

            It has two methods.

                `choose_action(state)`
                    Will return an action given the state

                `train(state, action, reward)`
                    Will train on given `states`, `actions`, `rewards`

            Private methods has two underscore prefixes
        """
        self.input_dim = list(input_dim)   # [80, 80, 1]
        self.output_dim = output_dim  # 2
        self.learning_rate = 0.00025

        self.checkpoint_dir = checkpoint_dir
        self.h_size = h_size
        self.__build_network(self.input_dim, self.output_dim)


        if logdir is not None:
            self.__build_summary_op(logdir)
        else:
            self.summary_op = None

    def resotre(self,):
        if self.checkpoint_dir is not None:
            self.saver = tf.train.Saver()

            maybe_path = tf.train.latest_checkpoint(self.checkpoint_dir)
            if os.path.exists(self.checkpoint_dir) and maybe_path:
                print("Restored {}".format(maybe_path))
                sess = tf.get_default_session()
                self.saver.restore(sess, maybe_path)


                
    def __build_network(self,input_dim, output_dim):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.stateInput =  tf.placeholder(tf.float32, shape=[None, *input_dim], name='state')
        net = self.stateInput
        
        #init = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
        init = tf.variance_scaling_initializer(scale=2)  # He initialization
        net = tf.layers.conv2d(net,filters=32,kernel_size=8,strides=4,padding='valid',kernel_initializer=init,activation=tf.nn.relu)
        net = tf.layers.conv2d(net,filters=64,kernel_size=4,strides=2,padding='valid',kernel_initializer=init,activation=tf.nn.relu)
        net = tf.layers.conv2d(net,filters=64,kernel_size=3,strides=1,padding='valid',kernel_initializer=init,activation=tf.nn.relu)
        net = tf.layers.conv2d(net,filters=self.h_size,kernel_size=7,strides=1,padding='valid',kernel_initializer=init,activation=tf.nn.relu)

        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        
        self.streamAC,self.streamVC = tf.split(net,2,3)  # (N,1,1,512) --> (N,1,1,256), (N,1,1,256)
        
        self.streamAC = tf.layers.flatten(self.streamAC)
        self.Advantage = tf.layers.dense(self.streamAC,output_dim,use_bias=False,kernel_initializer=init)

        self.streamVC = tf.layers.flatten(self.streamVC)
        self.Value = tf.layers.dense(self.streamVC,1,use_bias=False,kernel_initializer=init)
 
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))  #(N,4)
        self.predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,output_dim,dtype=tf.float32)
        
        self.Q = tf.boolean_mask(self.Qout, self.actions_onehot)
        
        
        #self.td_error = tf.square(self.targetQ - self.Q)
        self.td_error = tf.losses.huber_loss(self.targetQ, self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.00025)
        self.updateModel = self.trainer.minimize(self.loss)


    def choose_action(self, S,random_mode = True):
        shape = S.shape  # (80, 80, 1)

        if len(shape) == 3:
            S = np.expand_dims(S, axis=0)

        np.testing.assert_equal(S.shape[1:], self.input_dim)

        sess = tf.get_default_session()
        predict,Qout = sess.run([self.predict,self.Qout],feed_dict={self.stateInput: S})
        action_prob = scipy.special.softmax(Qout[0])
        
        if random_mode and np.random.rand(1) < 0.01:   # dqn에서는 and로 해야 한다.
            #return np.random.choice(np.arange(self.output_dim), p=action_prob)
            return np.random.randint(self.output_dim)
        else: return predict[0]

    def save(self, global_step):
        sess = tf.get_default_session()
        path = os.path.join(self.checkpoint_dir, "model.ckpt")
        self.saver.save(sess, path,global_step=global_step) # self.global_step
        

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)     

def train():
    batch_size = 64 #How many experiences to use for each training step.
    h_size = 512
    gamma = .99 #Discount factor on the target Q-values
    startE = 1 #Starting chance of random action
    endE = 0.1 #Final chance of random action
    annealing_steps = 2000000. #How many steps of training to reduce startE to endE.
    tau = 1 #Rate to update target network toward primary network   1 --> 100% update
    REPLAY_MEMORY = 400000  # 1000000, 400000,
    
    pre_train_episodes = 300 # 이전까지는 random action을 취한다.  
    pre_train_steps = 50000  # 이전까지는 train하지 않고, replay memory에 쌓기만 한다.  강화학습(이웅원 책)에는 50,000
    
    update_freq = 4 #How often to perform a training step.
    TARGET_UPDATE_FREQUENCY = 10000  # 10000 ---> 40 episode 마다 train 하는 정도

    
        
    CHECK_POINT_DIR = './breakout-dqn'
    if tf.train.latest_checkpoint(CHECK_POINT_DIR):
        episode = int(tf.train.latest_checkpoint(CHECK_POINT_DIR).split('-')[-1])
    else:
        episode=0
    episode_reward_history = deque(maxlen = 100)
    s_time = time.time()
    
    
    

    env = gym.make("BreakoutDeterministic-v4")

    action_dim = 3 # 0,1,2,  --> env action으로는 1(Fire, stop), 2: right, 3: left
    action_offset = 1

    sess = tf.InteractiveSession()
    
    repeat = 4
    main_agent = Agent(new_HW + [repeat],output_dim=action_dim,h_size = h_size, logdir=None, checkpoint_dir=CHECK_POINT_DIR)
    target_agent = Agent(new_HW + [repeat],output_dim=action_dim,h_size = h_size, logdir=None, checkpoint_dir=None)


    trainables = tf.trainable_variables()
    
    targetOps = updateTargetGraph(trainables,tau)  # mainQN 값 --> targetQN에 update
    
    myBuffer = deque(maxlen=REPLAY_MEMORY)
    
    stepDrop = (startE - endE)/annealing_steps
    e = max(startE - episode*stepDrop*250,endE)
    
    #create lists to contain total rewards and steps per episode

    total_steps = 400*episode



    s_time = time.time()
    sess.run(tf.global_variables_initializer())
    main_agent.resotre()
    updateTarget(targetOps,sess)


    while True:
        episode += 1
        #Reset environment and get first new observation
        state = env.reset()  # (84,84,3)
        done = False
        episode_reward = 0  # 하나의 episode에 대한 reward
        for _ in range(random.randint(1, 30)):
            state, _, _, _ = env.step(1)  # 1: 정지
        
        old_s = preprocessing(state, new_HW, height_range)  # (84, 84, 1)
            
        history = np.concatenate((old_s, old_s, old_s, old_s,old_s,old_s),axis=2)  # (84, 84, 6)
        dead = False
        remain_lives = 5
        
        #The Q-Network
        while not done: #If the agent takes longer than 500 moves to reach either of the blocks, end the trial.
            #env.render()
            total_steps += 1 # episode와 상관없이 step counting
            if dead:
                dead = False
                for _ in range(random.randint(1, 10)):
                    old_s,_,_,_ = env.step(1) # 1: 정지
                
                old_s = preprocessing(old_s, new_HW, height_range)
                history = np.concatenate((old_s, old_s, old_s, old_s,old_s,old_s),axis=2)  # (84, 84, 6)

            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or episode < pre_train_episodes:
                action = np.random.randint(0,action_dim) + action_offset
            else:
                action = main_agent.choose_action((history[:,:,:4]-history[:,:,1:5])/255.,random_mode=True) + action_offset
                
            
            new_s, r, done, info = env.step(action)
            
            r = np.clip(r,-1.,1.)
            if remain_lives > info['ale.lives']:
                remain_lives = info['ale.lives']
                dead = True   
                r = -1  
            
            episode_reward += r
            new_s = preprocessing(new_s, new_HW, height_range)  # (210, 160, 3)
            history = np.append(new_s, history[:, :, :-1],axis=2)
            
            
            myBuffer.append((history,action-action_offset,r,dead)) # done이 아니고, dead여야 한다.
            old_s = new_s
            
            
            if total_steps > pre_train_steps and len(myBuffer) >= pre_train_steps:  # pre_train_steps = 10000 --> 초반에는 train없이 myBuffer만 채운다.
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    trainBatch = random.sample(myBuffer, batch_size) #Get a random batch of experiences.  (batch_size,5)
                    
                    
                    states_ = np.stack([x[0] for x in trainBatch])  # (64, 84, 84, 6)
                    actions_ = np.array([x[1] for x in trainBatch])
                    rewards_ = np.array([x[2] for x in trainBatch])
                    done_ = np.array([x[3] for x in trainBatch])
                    
                    states_ = (states_[:,:,:,:-1]-states_[:,:,:,1:]) /255.  # (64, 84, 84, 5)
                    next_states_ = states_[:,:,:,:-1] # (64, 84, 84, 4)
                    states_ = states_[:,:,:,1:] #  (64, 84, 84, 4)
                    
                    
                    # np.allclose(states_[:,:,:,:-1],next_states_[:,:,:,1:])  ---> True
                    
                    
                    #Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(main_agent.predict,feed_dict={main_agent.stateInput: next_states_})  # (batch_size,)
                    Q2 = sess.run(target_agent.Qout,feed_dict={target_agent.stateInput: next_states_})  # (batch_size, 3)
                    end_multiplier = -(done_ - 1)  # done이면 0, done이 아니면 1
                    doubleQ = Q2[range(batch_size),Q1]  #(batch_size,)
                    targetQ = rewards_ + (gamma*doubleQ * end_multiplier)  # reward + gamma*Q
                    #Update the network with our target values.
                    _ = sess.run(main_agent.updateModel, \
                        feed_dict={main_agent.stateInput: states_, main_agent.targetQ:targetQ, main_agent.actions: actions_})
                    
                if total_steps % TARGET_UPDATE_FREQUENCY == 0:
                    updateTarget(targetOps,sess) #Update the target network toward the primary network.
                

        episode_reward_history.append(episode_reward)  # episode 종료
        
        if episode % 10 ==0:
            print("Average reward for last 100 episode/total_step/epsilon {}/{}/{:.4f}: {:.3f} elapsed: {} sec".format(episode,total_steps,e, np.mean(episode_reward_history),int(time.time()-s_time)))

        if episode % 250 ==0:
            main_agent.save(episode)


def infer():
    CHECK_POINT_DIR = './breakout-dqn'
    
    sess = tf.InteractiveSession()
    action_dim = 3 # 0,1,2,  --> env action으로는 1(Fire, stop), 2: right, 3: left
    action_offset = 1
    agent = Agent(new_HW + [4],
                  output_dim=action_dim,
                  logdir=None,
                  checkpoint_dir=CHECK_POINT_DIR)
    
    
    sess.run(tf.global_variables_initializer())
    agent.resotre()
    
    env = gym.make("BreakoutDeterministic-v4")
    env._max_episode_steps = 10000
    state = env.reset()
    old_s = preprocessing(state, new_HW, height_range)  # (84, 84, 1)
    history = np.concatenate((old_s, old_s, old_s, old_s,old_s,old_s),axis=2)  # (84, 84, 6)
    dead = False
    random_episodes = 0
    total_reward = 0
    n_step=0
    remain_lives = 5
    episode_reward_history = deque(maxlen = 100)
    while random_episodes < 20:
        env.render()
        if dead:
            dead = False
            old_s,_,_,_ = env.step(1)
            old_s = preprocessing(old_s, new_HW, height_range)
            history = np.concatenate((old_s, old_s, old_s, old_s,old_s,old_s),axis=2)  # (84, 84, 6)
        
        action = agent.choose_action((history[:,:,:4]-history[:,:,1:5])/255.,random_mode=True) + action_offset
        
        new_s, r, done, info = env.step(action)
        new_s = preprocessing(new_s, new_HW, height_range)  # (210, 160, 3)
        history = np.append(new_s, history[:, :, :-1],axis=2)
        
        total_reward += r

        if remain_lives > info['ale.lives']:
            remain_lives = info['ale.lives']
            dead = True          
        n_step += 1
        
        
        if done:
            random_episodes += 1
            episode_reward_history.append(total_reward)
            print("Reward for this episode {} was: {}. n_step = {}, score: {}".format(random_episodes,total_reward,n_step,total_reward))
            state = env.reset()
            old_s = preprocessing(state, new_HW, height_range)  # (84, 84, 1)
            history = np.concatenate((old_s, old_s, old_s, old_s,old_s,old_s),axis=2)  # (84, 84, 6)         
            dead = False
            total_reward = 0
            n_step=0
            remain_lives = 5      
 
    print('average: {}, min: {}, max: {}'.format(np.mean(episode_reward_history),np.min(episode_reward_history),np.max(episode_reward_history)))

if __name__ == '__main__':

    #train()

    infer()
