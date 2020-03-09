"""
actor-critic(A2C)


https://github.com/Nimoab/DQN-Deepmind-NIPS-2013/tree/master/results   --- breakout 점수 체계
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import deque
from functools import partial
#from scipy.misc import imresize
from skimage.transform import resize
from skimage.color import rgb2gray
import os,time,random
import pickle
new_HW = [84, 84] 
height_range=(31, 199)

def preprocessing(image, new_HW, height_range=(35, 195)):
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

def discount_reward(r,gamma=0.99):
    '''Discounted reward를 구하기 위한 함수
    Args:
         r(np.array): reward 값이 저장된 array
    Returns:
        discounted_r(np.array): Discounted 된 reward가 저장된 array
    '''
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        if r[t] < 0: # life가 줄었을때 마다 return 초기화
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    discounted_r = discounted_r - discounted_r.mean()
    discounted_r = discounted_r / discounted_r.std()

    return discounted_r

class Agent(object):

    def __init__(self, input_dim, output_dim,h_size=512,action_offset=1, logdir="logdir", checkpoint_dir="checkpoints"):
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
        self.output_dim = output_dim  # 3
        self.learning_rate = 0.00025
        self.gamma = 0.99
        self.action_offset = action_offset

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
        self.global_step = tf.train.get_or_create_global_step()
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
        self.Policy = tf.clip_by_value(tf.layers.dense(self.streamAC,output_dim,use_bias=True,activation=tf.nn.softmax,kernel_initializer=init), 1e-10, 1.)
        self.predict = tf.argmax(self.Policy,1)
        
        self.streamVC = tf.layers.flatten(self.streamVC)
        self.Value = tf.layers.dense(self.streamVC,1,use_bias=False,activation=None,kernel_initializer=init)
        self.Value = tf.squeeze(self.Value)
        
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.action = tf.placeholder(shape=[None],dtype=tf.int32,name='action_input')
        self.actions_onehot = tf.one_hot(self.action,output_dim,dtype=tf.float32,name='action_onehot')
        self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage_input")
        self.reward = tf.placeholder(tf.float32, shape=[None], name="reward_input")
        
        policy_gain = tf.boolean_mask(self.Policy, self.actions_onehot)
        policy_gain = tf.log(policy_gain) * self.advantage
        policy_gain = tf.reduce_mean(policy_gain, name="policy_gain")

        entropy = - tf.reduce_sum(self.Policy * tf.log(self.Policy), 1)
        entropy = tf.reduce_mean(entropy)
        
        value_loss = tf.losses.mean_squared_error(self.Value, self.reward, scope="value_loss")
        
        # Becareful negative sign because we only can minimize
        # we want to maximize policy gain and entropy (for exploration)
        self.loss = - policy_gain  + value_loss - entropy * 0.02
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.00025)
        self.train_op = self.optimizer.minimize(self.loss,global_step=self.global_step)


    def choose_action(self, S,random_mode = True):
        shape = S.shape  # (80, 80, 1)

        if len(shape) == 3:
            S = np.expand_dims(S, axis=0)

        np.testing.assert_equal(S.shape[1:], self.input_dim)

        sess = tf.get_default_session()
        action_prob = sess.run(self.Policy,feed_dict={self.stateInput: S})
        action_prob = np.squeeze(action_prob)
        if random_mode or np.random.rand(1) < 0.03 :
            return np.random.choice(np.arange(self.output_dim) + self.action_offset, p=action_prob)
        else: return np.argmax(action_prob) + self.action_offset

    def save(self, global_step):
        sess = tf.get_default_session()
        path = os.path.join(self.checkpoint_dir, "model.ckpt")
        self.saver.save(sess, path,global_step=global_step) # self.global_step
        
    def train(self, S, A, R):
        S = np.array(S)
        A = np.array(A)
        R = np.array(R)
        np.testing.assert_equal(S.shape[1:], self.input_dim)
        assert len(A.shape) == 1, "A.shape = {}".format(A.shape)
        assert len(R.shape) == 1, "R.shape = {}".format(R.shape)

        sess = tf.get_default_session()
        R = discount_reward(R, gamma=self.gamma)

        A = A - self.action_offset  # env에 넣을 때는1,2,3 --> model에서는 0,1,2
        V = sess.run(self.Value, feed_dict={self.stateInput: S})
        
        # 3. Get Advantage values, A = R - V
        ADV = R - V        
        ADV = (ADV - np.mean(ADV)) / (np.std(ADV) + 1e-8)


        sess = tf.get_default_session()
        _, global_step_value = sess.run([self.train_op,self.global_step],
                                        feed_dict={self.stateInput: S,self.action: A,self.reward: R,self.advantage: ADV})       

def train():
    batch_size = 64 #How many experiences to use for each training step.
    h_size = 512
        
    CHECK_POINT_DIR = './breakout-a2c'
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
    agent = Agent(new_HW + [repeat],output_dim=action_dim,h_size = h_size, action_offset = action_offset, logdir=None, checkpoint_dir=CHECK_POINT_DIR)



    s_time = time.time()
    sess.run(tf.global_variables_initializer())
    agent.resotre()


    while True:
        episode += 1

        states = []
        actions = []
        rewards = []
    
        state = env.reset()
        for _ in range(random.randint(1, 30)):
            state, _, _, _ = env.step(1)  # 1: 정지
        done = False
        remain_lives = 5
        dead = False
        total_reward = 0
        step_counter = 0
    
        
        old_s = preprocessing(state, new_HW, height_range)
            
        history = np.concatenate((old_s, old_s, old_s, old_s),axis=2)
        
    
        while not done:
            #env.render()
            if dead:
                dead = False
                old_s,_,_,_ = env.step(1)
                for _ in range(random.randint(1, 10)):
                    old_s,_,_,_ = env.step(1) # 1: 정지
                old_s = preprocessing(old_s, new_HW, height_range)
                history = np.concatenate((old_s, old_s, old_s, old_s),axis=2)
            action = agent.choose_action(history)
            step_counter += 1
            new_s, r, done, info = env.step(action)
            
            r = np.clip(r,-1.,1.)
            if remain_lives > info['ale.lives']:
                remain_lives = info['ale.lives']
                dead = True   
                r = -1     
            
            total_reward += r
    
            states.append(history)
            actions.append(action)
            rewards.append(r)
    
            new_s = preprocessing(new_s, new_HW, height_range)  # (210, 160, 3)
            
            history = np.append(new_s-old_s, history[:, :, :3],axis=2)
            old_s = new_s
            
            if done:  # done까지 기다리면, episode의 길이가 길어진다.
                step_counter = 0
                # Agent expects numpy array
                
                agent.train(states, actions, rewards)
    

                

        episode_reward_history.append(total_reward)  # episode 종료
        
        if episode % 10 ==0:
            print("Average reward for last 100 episode: {} / {:.3f} elapsed: {} sec".format(episode,np.mean(episode_reward_history),int(time.time()-s_time)))

        if episode % 500 ==0:
            agent.save(episode)


def infer():
    CHECK_POINT_DIR = './breakout-a2c'
    
    sess = tf.InteractiveSession()
    action_dim = 3
    action_offset = 1
    agent = Agent(new_HW + [4],
                  output_dim=action_dim,action_offset = action_offset,
                  logdir=None,
                  checkpoint_dir=CHECK_POINT_DIR)
    
    
    agent.resotre()
    
    
    env = gym.make("BreakoutDeterministic-v4")
    env._max_episode_steps = 10000
    env = gym.wrappers.Monitor(env, './movie/', force=True)
    state = env.reset()


    episode_reward_history = deque(maxlen = 100)
    random_episodes = 0
    total_reward = 0
    n_step=0
    remain_lives = 5
    dead = True
    while random_episodes < 20:
        env.render()
        if dead:
            env.step(1)  #action 1을 하며, 다시 시작한다. 이게 없으도 몇 step 경과하면 다시 시작한다.
            dead = False
            old_s,_,_,_ = env.step(1)
            old_s = preprocessing(old_s, new_HW, height_range)
            history = np.concatenate((old_s, old_s, old_s, old_s),axis=2)
        
        action = agent.choose_action(history,random_mode=False)  # 2 또는 3  random_mode=False가 더 좋다.
        
        new_s, r, done, info = env.step(action)
        new_s = preprocessing(new_s, new_HW, height_range)  # (210, 160, 3)

        
        total_reward += r

        if remain_lives > info['ale.lives']:
            remain_lives = info['ale.lives']
            dead = True          
        n_step += 1
        
        
        history = np.append(new_s-old_s, history[:, :, :3],axis=2)
        old_s = new_s
        
        if done:
            random_episodes += 1
            episode_reward_history.append(total_reward)
            print("Reward for this episode {} was: {}. n_step = {}, score: {}".format(random_episodes,total_reward,n_step,total_reward))
            return total_reward
            total_reward = 0
            state = env.reset()
            n_step=0
            remain_lives = 5
    
    print('average: {}, min: {}, max: {}'.format(np.mean(episode_reward_history),np.min(episode_reward_history),np.max(episode_reward_history)))
    
if __name__ == '__main__':

    train()

    #infer()


#     for i in range(100):
#         tf.reset_default_graph()
#         print('try: ', i)
#         r = infer()
#         if r > 500:
#             break

