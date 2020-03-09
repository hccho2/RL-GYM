

'''
ddpg는 continuous action 모델

1. github jcwleo(주찬성), 
2. PG is all you need, 
3. MorvanZhou(https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)

ranibow: https://github.com/Curt-Park/rainbow-is-all-you-need
PG is all you need(MrSyee): https://github.com/MrSyee/pg-is-all-you-need   (PPO, DDPG 등)



https://github.com/jinPrelude/Americano_lab/tree/master/DDPG_Pendulum

https://github.com/openai/gym/wiki/Leaderboard#pendulum-v0  --> leaderboard --> https://github.com/msinto93/DDPG

TRPO: https://reinforcement-learning-kr.github.io/2018/06/24/5_trpo/



PER: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
https://github.com/rlcode/per   <---- 여기 코드도 위 blog의 SumTree Class 사용.



'''


import tensorflow as tf
import numpy as np
import gym
import time,os,random
from collections import deque

#####################  hyper parameters  ####################


MAX_EP_STEPS = 200
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.005      # soft replacement
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

RENDER = False
ENV_NAME = 'Pendulum-v0'


###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.replay_buffer = deque(maxlen = MEMORY_CAPACITY)  # state, next_state, action, reward
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_next = tf.placeholder(tf.float32, [None, s_dim], 's_next')   # next state
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))  # shadow variable에 접근

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation(shadow variable이 생성된다)
        a_ = self._build_a(self.S_next, reuse=True, custom_getter=ema_getter)   # replaced target parameters(shadow variable로 network 만듬)
        q_ = self._build_c(self.S_next, a_, reuse=True, custom_getter=ema_getter)
        
        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def train(self):

        trainBatch = random.sample(self.replay_buffer, BATCH_SIZE)
        states = np.stack([x[0] for x in trainBatch])  # (N,3)
        actions = np.array([x[1] for x in trainBatch])
        rewards = np.array([x[2] for x in trainBatch])
        next_states = np.stack([x[3] for x in trainBatch])

        self.sess.run(self.atrain, {self.S: states})
        self.sess.run(self.ctrain, {self.S: states, self.a: actions, self.R: rewards, self.S_next: next_states})

    def store_transition(self, s, a, r, s_):
        self.replay_buffer.append((s, a.flatten(), r.flatten(), s_))

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 100, activation=tf.nn.relu, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 30, activation=tf.nn.relu, name='l2', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            net_s =  tf.layers.dense(s, 100, activation=tf.nn.relu, name='Ls1', trainable=trainable)
            net_s =  tf.layers.dense(net_s, 30, activation=None, name='ls2', trainable=trainable)
            net_a  = tf.layers.dense(a, 30, activation=None,use_bias=False, name='la1', trainable=trainable)
            net = tf.nn.relu(net_s + net_a)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
###############################  training  ####################################
def train():
    env = gym.make(ENV_NAME)
    env = env.unwrapped  # max step 200이  풀린다.
    env.seed(1)
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    
    ddpg = DDPG(a_dim, s_dim, a_bound)
    

    s_time = time.time()
    episode_reward_history = deque(maxlen = 100)
    
    saver=tf.train.Saver()
    checkpoint_dir = "./ddpg-model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    save_path = os.path.join(checkpoint_dir, "model.ckpt")
    
    if tf.train.latest_checkpoint(checkpoint_dir):
        n_episode = int(tf.train.latest_checkpoint(checkpoint_dir).split('-')[-1])
    else:
        n_episode=0
    
    exploration_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(a_dim))
    noise_scaling = 0.1 * (a_bound*2)
    best_avg_reward = -180
    
    while True:
        n_episode += 1
        s = env.reset()
        exploration_noise.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()
    
            # Add exploration noise
            a = ddpg.choose_action(s)
            #a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
            noise = exploration_noise() * noise_scaling
            a = np.clip(a + noise, -2, 2)
            s_, r, done, info = env.step(a)
    
            ddpg.store_transition(s, a, r / 10, s_)
    
            if len(ddpg.replay_buffer) > BATCH_SIZE*10:
                ddpg.train()
    
            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                episode_reward_history.append(ep_reward)
                
                if n_episode% 10 ==0:
                    print('Episode:', n_episode, ' 100 avg. Reward: %.2f' % np.mean(episode_reward_history), 'noise: %.2f' % noise, )
                
                if best_avg_reward * 0.99 <= np.mean(episode_reward_history):
                    best_avg_reward = np.mean(episode_reward_history)
                    saver.save(ddpg.sess,save_path,global_step=n_episode)
                    print('Checkpoint Saved to {}/{}/{}. elapsed = {}'.format(save_path, n_episode,best_avg_reward ,time.time()-s_time))
                
                #if ep_reward > -300:RENDER = True
                break
    
    print('Running time: ', time.time() - s_time)
    
    
    
def test():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)
    env._max_episode_steps = 1000
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    
    ddpg = DDPG(a_dim, s_dim, a_bound)  

    saver=tf.train.Saver()
    checkpoint_dir = "./ddpg-model"
    save_path = os.path.join(checkpoint_dir, "model.ckpt")


    maybe_path = tf.train.latest_checkpoint(checkpoint_dir)
    if os.path.exists(checkpoint_dir) and maybe_path:
        print("Restored {}".format(maybe_path))

        saver.restore(ddpg.sess, maybe_path)

    else:
        print("No model is found")
        exit()


    env = gym.wrappers.Monitor(env, './movie/', force=True)
    state = env.reset()


    episode_reward_history = deque(maxlen = 100)
    random_episodes = 0
    total_reward = 0
    n_step=0
    state = env.reset()

    while random_episodes < 1:
        env.render()
        
        action = ddpg.choose_action(state)
        #action = np.clip(np.random.normal(action, var), -2, 2)    # add randomness to action selection for exploration
        state, reward, done, info = env.step(action)
        

        total_reward += reward
        n_step += 1
        if n_step% 50 ==0:
            print(n_step, total_reward, action)
        if n_step > 1000: done=True   # error가 나지만, 이렇게 멈추어야 mp4파일이 만들어짐
        if done:
            
            print("Reward for this episode {} was: {}. n_step = {}".format(random_episodes,total_reward,n_step))
            total_reward = 0
            state = env.reset()
            random_episodes += 1
            total_reward = 0
            n_step = 0

    
if __name__ == '__main__':
    #train()
    test()

