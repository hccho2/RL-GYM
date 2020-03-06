"""
breakout은  state diff 방식 : train_state_diff - infer1()    보다 state stack 방식 : train_state_stack - infer2()  으로 해야 잘된다.


https://github.com/jcwleo/Reinforcement_Learning/blob/master/Breakout/Breakout_PolicyGradient.py
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

new_HW = [84, 84] 
height_range=(31, 199)

def plot_image(image):
    """Plot an image

    If an image is a grayscale image,
    plot in `gray` cmap.
    Otherwise, regular RGB plot.

    Args:
        image (2-D or 3-D array): (H, W) or (H, W, C)
    """
    image = np.squeeze(image)
    shape = image.shape

    if len(shape) == 2:
        plt.imshow(image, cmap="gray")

    else:
        plt.imshow(image)

    plt.show()


def pipeline(image, new_HW, height_range=(35, 195)):
    """Returns a preprocessed image

    (1) Crop image (top and bottom)
    (2) Remove background & grayscale
    (3) Reszie to smaller image

    Args:
        image (3-D array): (H, W, C)
        new_HW (tuple): New image size (height, width)
        height_range (tuple): Height range (H_begin, H_end) else cropped
        bg (tuple): Background RGB Color (R, G, B)

    Returns:
        image (3-D array): (H, W, 1)
    """
#     image = crop_image(image, height_range)  # (210, 160, 3)  --> (160, 160, 3)
#     image = rgb2gray(image)  # plt.imshow(image,cmap='gray') (160,160)
#     image =  resize(image, new_HW, mode='constant')
#     image = np.expand_dims(image, axis=2)  # (80, 80, 1)
#     return image


    image = crop_image(image, height_range)  # (210, 160, 3)  --> (160, 160, 3)
    image = resize(rgb2gray(image), new_HW, mode='reflect')
    image = np.expand_dims(image, axis=2)  # (80, 80, 1)
    return image

def resize_image(image, new_HW):
    """Returns a resized image

    Args:
        image (3-D array): Numpy array (H, W, C)
        new_HW (tuple): Target size (height, width)

    Returns:
        image (3-D array): Resized image (height, width, C)
    """
    #return imresize(image, new_HW, interp="nearest")
    return resize(image, new_HW,order=0,preserve_range=True).astype(np.uint8)   # bg (109, 118,  43)  ---> (108,117,42)로 바뀐다.


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

    def __init__(self, input_dim, output_dim, logdir="logdir", checkpoint_dir="checkpoints"):
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
        self.gamma = 0.99
        self.entropy_coefficient = 0.02
        self.RMSPropdecay = 0.99
        self.learning_rate = 0.00025

        self.checkpoint_dir = checkpoint_dir
        self.__build_network(self.input_dim, self.output_dim)


        if logdir is not None:
            self.__build_summary_op(logdir)
        else:
            self.summary_op = None

        if checkpoint_dir is not None:
            self.saver = tf.train.Saver()

            maybe_path = tf.train.latest_checkpoint(self.checkpoint_dir)
            if os.path.exists(self.checkpoint_dir) and maybe_path:
                print("Restored {}".format(maybe_path))
                sess = tf.get_default_session()
                self.saver.restore(sess, maybe_path)

            else:
                print("No model is found")
                os.makedirs(checkpoint_dir, exist_ok=True)
                sess = tf.get_default_session()
                init = tf.global_variables_initializer()
                sess.run(init)
                


    def __build_network(self, input_dim, output_dim):
        # input_dim: [80, 80, 1], output_dim: 2
        self.global_step = tf.train.get_or_create_global_step()

        self.X = tf.placeholder(tf.float32, shape=[None, *input_dim], name='state')
        self.action = tf.placeholder(tf.uint8, shape=[None], name="action")
        action_onehot = tf.one_hot(self.action, output_dim, name="action_onehot")
        self.reward = tf.placeholder(tf.float32, shape=[None], name="reward")


        f1 = tf.get_variable("f1", shape=[1, 1, 4, 1], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        f2 = tf.get_variable("f2", shape=[8, 8, 1, 16], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        f3 = tf.get_variable("f3", shape=[4, 4, 16, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        w1 = tf.get_variable("w1", shape=[9 * 9 * 32, 256], initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable("w2", shape=[256, output_dim], initializer=tf.contrib.layers.xavier_initializer())


        c1 = tf.nn.relu(tf.nn.conv2d(self.X, f1, strides=[1, 1, 1, 1], padding="VALID"))  # (?, 84, 84, 4) -> (?, 84, 84, 1)
        c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 4, 4, 1], padding="VALID"))  # -> (?, 20, 20, 16)
        c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1, 2, 2, 1], padding="VALID"))  # --> (?, 9, 9, 32)

        l1 = tf.reshape(c3, [-1, w1.get_shape().as_list()[0]])  # --> (?, 2592)
        l2 = tf.nn.relu(tf.matmul(l1, w1))  # --> (?, 256)


        self.action_prob = tf.clip_by_value(tf.nn.softmax(tf.matmul(l2, w2), name="action_prob"), 1e-10, 1.)

        log_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)
        log_action_prob = tf.log(log_action_prob)

        entropy = - self.action_prob * tf.log(self.action_prob)
        self.entropy = tf.reduce_sum(entropy, axis=1)

        loss = -log_action_prob * self.reward - self.entropy * self.entropy_coefficient
        self.loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.train_op = optimizer.minimize(loss,global_step=self.global_step)


    def __build_summary_op(self, logdir):
        tf.summary.histogram("Action Probability Histogram", self.action_prob)
        tf.summary.histogram("Entropy", self.entropy)
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Mean Reward", tf.reduce_mean(self.reward))

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    def choose_action(self, S,random_mode = True):
        shape = S.shape  # (80, 80, 1)

        if len(shape) == 3:
            S = np.expand_dims(S, axis=0)

        np.testing.assert_equal(S.shape[1:], self.input_dim)

        sess = tf.get_default_session()
        action_prob = sess.run(self.action_prob,feed_dict={self.X: S})
        action_prob = np.squeeze(action_prob)
        if random_mode or np.random.rand(1) < 0.03 :
            return np.random.choice(np.arange(self.output_dim) + 1, p=action_prob)
        else: return np.argmax(action_prob) + 1
    def train(self, S, A, R):
        S = np.array(S)
        A = np.array(A)
        R = np.array(R)
        np.testing.assert_equal(S.shape[1:], self.input_dim)
        assert len(A.shape) == 1, "A.shape = {}".format(A.shape)
        assert len(R.shape) == 1, "R.shape = {}".format(R.shape)

        R = discount_reward(R, gamma=self.gamma)

        A = A - 1  # env에 넣을 때는1,2,3 --> model에서는 0,1,2

        sess = tf.get_default_session()
        if self.summary_op is not None:
            _, summary_op, global_step_value = sess.run([self.train_op,
                                                         self.summary_op,
                                                         self.global_step],
                                                        feed_dict={self.X: S,
                                                                   self.action: A,
                                                                   self.reward: R})
        else:
            _, global_step_value = sess.run([self.train_op,self.global_step],
                                                        feed_dict={self.X: S,
                                                                   self.action: A,
                                                                   self.reward: R})       
        if self.summary_op is not None:
            self.summary_writer.add_summary(summary_op, global_step_value)

    def save(self, global_step):
        sess = tf.get_default_session()
        path = os.path.join(self.checkpoint_dir, "model.ckpt")
        self.saver.save(sess, path,global_step=global_step) # self.global_step
        


def discount_reward_bak(rewards, gamma=0.99):
    """Returns discounted rewards

    Args:
        rewards (1-D array): Reward array
        gamma (float): Discounted rate

    Returns:
        discounted_rewards: same shape as `rewards`

    Notes:
        In Pong, when the reward can be {-1, 0, 1}.

        However, when the reward is either -1 or 1,
        it means the game has been reset.

        Therefore, it's necessaray to reset `running_add` to 0
        whenever the reward is nonzero
    """
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if rewards[t] < 0:  # dead가 발생하면,
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r

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
def run_episode(env, agent):
    """Runs one episode and returns a total reward

    Args:
        env (gym.env): Gym Environment
        agent (Agent): Agent Player
        pipeline (function): Preprocessing function.
            processed_image = pipeline(image)

    Returns:
        total_reward (int): Total reward earned in an episode.
    """
    states = []
    actions = []
    rewards = []

    old_s = env.reset()

    done = False
    remain_lives = 5
    dead = False
    total_reward = 0
    step_counter = 0

    
    for _ in range(random.randint(1,30)):
        old_s,_,_,_ = env.step(1)  
    old_s = pipeline(old_s, new_HW, height_range)
    
    new_s,_,_,_ = env.step(1)
    new_s = pipeline(new_s, new_HW, height_range)
    state_diff =  new_s - old_s
    old_s = new_s
    while not done:
        #env.render()
        if dead:
            env.step(1)
            dead = False
        action = agent.choose_action(state_diff)
        step_counter += 1
        new_s, r, done, info = env.step(action)
        
        r = np.clip(r,-1.,1.)
        if remain_lives > info['ale.lives']:
            remain_lives = info['ale.lives']
            dead = True   
            r = -1     
        
        total_reward += r

        states.append(state_diff)  #(80,80,1) ---> list로 쌓기 때문에, (N,80,80,1)이 된다.
        actions.append(action)
        rewards.append(r)

        new_s = pipeline(new_s, new_HW, height_range)  # (210, 160, 3)
        state_diff = new_s - old_s  # (80, 80, 1)
        old_s = new_s



        if step_counter >= 20 or done:  # done까지 기다리면, episode의 길이가 길어진다.
            step_counter = 0
            # Agent expects numpy array
            
            agent.train(states, actions, rewards)

            states, actions, rewards = [], [], []

    return total_reward
def run_episode2(env, agent):
    # state stack 방식
    """Runs one episode and returns a total reward

    Args:
        env (gym.env): Gym Environment
        agent (Agent): Agent Player
        pipeline (function): Preprocessing function.
            processed_image = pipeline(image)

    Returns:
        total_reward (int): Total reward earned in an episode.
    """
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

    
    old_s = pipeline(state, new_HW, height_range)
        
    history = np.concatenate((old_s, old_s, old_s, old_s),axis=2)
    

    while not done:
        #env.render()
        if dead:
            dead = False
            old_s,_,_,_ = env.step(1)
            for _ in range(random.randint(1, 10)):
                old_s,_,_,_ = env.step(1) # 1: 정지
            old_s = pipeline(old_s, new_HW, height_range)
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

        new_s = pipeline(new_s, new_HW, height_range)  # (210, 160, 3)
        
        history = np.append(new_s-old_s, history[:, :, :3],axis=2)
        old_s = new_s
        
        if done:  # done까지 기다리면, episode의 길이가 길어진다.
            step_counter = 0
            # Agent expects numpy array
            
            agent.train(states, actions, rewards)

            states, actions, rewards = [], [], []

    return total_reward

def train_state_diff():
    
    CHECK_POINT_DIR = './breakout'
    #LOG_DIR = './breakout/logs'
    if tf.train.latest_checkpoint(CHECK_POINT_DIR):
        episode = int(tf.train.latest_checkpoint(CHECK_POINT_DIR).split('-')[-1])
    else:
        episode=0
    episode_reward_history = deque(maxlen = 100)
    s_time = time.time()
    
    
    

    env = gym.make("BreakoutDeterministic-v4")
    #env = gym.wrappers.Monitor(env, "monitor", force=True)
    action_dim = 3 # 0,1,2,  --> env action으로는 1(Fire, stop), 2: right, 3: left

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    

    
    repeat = 1
    agent = Agent(new_HW + [repeat],
                  output_dim=action_dim,
                  logdir=None,
                  checkpoint_dir=CHECK_POINT_DIR)



    

    while True:
        episode += 1
        episode_reward = run_episode(env, agent)
        episode_reward_history.append(episode_reward)
        if episode % 10 ==0:
            print("Average reward for last 100 episode {}: {:.3f} elapsed: {} sec".format(episode, np.mean(episode_reward_history),int(time.time()-s_time)))


        if episode % 1000 == 0:
            agent.save(episode)

            

def train_state_stack():
    
    CHECK_POINT_DIR = './breakout2'
    if tf.train.latest_checkpoint(CHECK_POINT_DIR):
        episode = int(tf.train.latest_checkpoint(CHECK_POINT_DIR).split('-')[-1])
    else:
        episode=0
    episode_reward_history = deque(maxlen = 100)
    s_time = time.time()
    
    
    

    env = gym.make("BreakoutDeterministic-v4")
    #env = gym.wrappers.Monitor(env, "monitor", force=True)
    action_dim = 3 # 0,1,2,  --> env action으로는 1(Fire, stop), 2: right, 3: left

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    

    
    repeat = 4
    agent = Agent(new_HW + [repeat],
                  output_dim=action_dim,
                  logdir=None,
                  checkpoint_dir=CHECK_POINT_DIR)



    

    while True:
        episode += 1
        episode_reward = run_episode2(env, agent)
        episode_reward_history.append(episode_reward)
        if episode % 10 ==0:
            print("Average reward for last 100 episode {}: {:.3f} elapsed: {} sec".format(episode, np.mean(episode_reward_history),int(time.time()-s_time)))


        if episode % 500 == 0:
            agent.save(episode)

def infer1():
    CHECK_POINT_DIR = './breakout'
    
    sess = tf.InteractiveSession()
    action_dim = 3
    agent = Agent(new_HW + [1],
                  output_dim=action_dim,
                  logdir=None,
                  checkpoint_dir=CHECK_POINT_DIR)
    
    
    
    
    env = gym.make("BreakoutDeterministic-v4")
    old_s = env.reset()
    old_s = pipeline(old_s, new_HW, height_range)
    state_diff = old_s
    random_episodes = 0
    total_reward = 0
    n_step=0
    remain_lives = 5
    dead = True
    while random_episodes < 5:
        env.render()
        if dead:
            action = 1  # action 1을 하며, 다시 시작한다. 이게 없으도 몇 step 경과하면 다시 시작한다.
            dead = False
        else: action = agent.choose_action(state_diff)  # 2 또는 3
        
        new_s, r, done, info = env.step(action)
        new_s = pipeline(new_s, new_HW, height_range)  # (210, 160, 3)
        state_diff = new_s - old_s  # (80, 80, 1)
        old_s = new_s
        
        total_reward += r

        if remain_lives > info['ale.lives']:
            remain_lives = info['ale.lives']
            dead = True          
        n_step += 1
        if done:
            random_episodes += 1
            print("Reward for this episode {} was: {}. n_step = {}, score: {}".format(random_episodes,total_reward,n_step,total_reward))
            total_reward = 0
            old_s = env.reset()
            old_s = pipeline(old_s, new_HW, height_range)
            state_diff = old_s
            n_step=0


def infer2():
    CHECK_POINT_DIR = './breakout2'
    #CHECK_POINT_DIR = './breakout2/new-try1'
    
    sess = tf.InteractiveSession()
    action_dim = 3
    agent = Agent(new_HW + [4],
                  output_dim=action_dim,
                  logdir=None,
                  checkpoint_dir=CHECK_POINT_DIR)
    
    
    
    
    env = gym.make("BreakoutDeterministic-v4")
    env._max_episode_steps = 10000
    #env = gym.wrappers.Monitor(env, './movie/', force=True)
    state = env.reset()
#     old_s = pipeline(state, new_HW, height_range)
#     history = np.concatenate((old_s, old_s, old_s, old_s),axis=2)


    random_episodes = 0
    total_reward = 0
    n_step=0
    remain_lives = 5
    dead = True
    episode_reward_history = deque(maxlen = 100)
    while random_episodes < 20:
        env.render()
        if dead:
            env.step(1)  #action 1을 하며, 다시 시작한다. 이게 없으도 몇 step 경과하면 다시 시작한다.
            dead = False
            old_s,_,_,_ = env.step(1)
            old_s = pipeline(old_s, new_HW, height_range)
            history = np.concatenate((old_s, old_s, old_s, old_s),axis=2)
        
        action = agent.choose_action(history,random_mode=False)  # 2 또는 3  random_mode=False가 더 좋다.
        
        new_s, r, done, info = env.step(action)
        new_s = pipeline(new_s, new_HW, height_range)  # (210, 160, 3)

        
        total_reward += r

        if remain_lives > info['ale.lives']:
            remain_lives = info['ale.lives']
            dead = True          
        n_step += 1
        
        
        history = np.append(new_s-old_s, history[:, :, :3],axis=2)
        old_s = new_s
        
#         if n_step > 5000:
#             done=True  # gym.wrappers.Monitor가 작동하는 중에는 강제로 done시키면 error발생.
        if done:
            random_episodes += 1
            episode_reward_history.append(total_reward)
            print("Reward for this episode {} was: {}. n_step = {}, score: {}".format(random_episodes,total_reward,n_step,total_reward))
            total_reward = 0
            state = env.reset()
            n_step=0
            remain_lives = 5

    print('average: {}, min: {}, max: {}'.format(np.mean(episode_reward_history),np.min(episode_reward_history),np.max(episode_reward_history)))
            
if __name__ == '__main__':
    #train_state_diff()
    #train_state_stack()
    #infer1()
    infer2()
