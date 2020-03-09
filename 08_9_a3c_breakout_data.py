"""
Simple Asynchronous Methods for Deep Reinforcement Learning (A3C)

local Agent가 gradient를 update하는 방식이 이니, global network를 train한다.


"""
import tensorflow as tf
import numpy as np
import threading
import gym
import os,random,time
from skimage.transform import resize
from skimage.color import rgb2gray
import skimage
from collections import deque
tf.reset_default_graph()
n_episode = 0
episode_reward_history = deque(maxlen = 100)
new_HW = [84, 84] 
height_range=(31, 199)
def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation

    Args:
        from_scope (str): The name of scope to copy from
            It should be "global"
        to_scope (str): The name of scope to copy to
            It should be "thread-{}"

    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


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


class A3CNetwork(object):

    def __init__(self, name, input_shape, output_dim,h_size=512, logdir=None):
        """A3C Network tensors and operations are defined here

        Args:
            name (str): The name of scope
            input_shape (list): The shape of input image [H, W, C]
            output_dim (int): Number of actions
            logdir (str, optional): directory to save summaries

        Notes:
            You should be familiar with Policy Gradients.
            The only difference between vanilla PG and A3C is that there is
            an operation to apply gradients manually
        """
        self.h_size = h_size 
        with tf.variable_scope(name):
            #The network recieves a frame from the game, flattened into an array.
            #It then resizes it and processes it through four convolutional layers.
            self.stateInput =  tf.placeholder(tf.float32, shape=[None, *input_shape], name='state')
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
            self.train_op = self.optimizer.minimize(self.loss)





class Agent(threading.Thread):

    def __init__(self, session, env, coord, name, global_network, input_shape, output_dim,h_size,action_offset, logdir=None):
        """Agent worker thread

        Args:
            session (tf.Session): Tensorflow session needs to be shared
            env (gym.Env): Gym environment (BreakoutDeterministic-v4)
            coord (tf.train.Coordinator): Tensorflow Queue Coordinator
            name (str): Name of this worker
            global_network (A3CNetwork): Global network that needs to be updated
            input_shape (list): Required for local A3CNetwork, [H, W, C]
            output_dim (int): Number of actions
            logdir (str, optional): If logdir is given, will write summary

        Methods:
            print(reward): prints episode rewards
            play_episode(): a single episode logic is stored in here
            run(): override threading.Thread.run
            choose_action(state)
            train(states, actions, rewards)
        """
        super(Agent, self).__init__()
        self.local = A3CNetwork(name, input_shape, output_dim,h_size, logdir)
        self.global_to_local = copy_src_to_dst("global", name)
        self.global_network = global_network

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        self.coord = coord
        self.name = name
        self.logdir = logdir
        
        self.action_offset =  action_offset

    def print(self, reward):
        global n_episode
        message = "Agent(name = {}, episode  = {},  reward = {})".format(self.name,n_episode, reward)
        print(message)

    def play_episode(self):
        global n_episode
        
        n_episode = n_episode + 1
        
        self.sess.run(self.global_to_local)

        states = []
        actions = []
        rewards = []
    
        state = self.env.reset()
        for _ in range(random.randint(1, 30)):
            state, _, _, _ = self.env.step(1)  # 1: 정지
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
                old_s,_,_,_ = self.env.step(1)
                for _ in range(random.randint(1, 10)):
                    old_s,_,_,_ = self.env.step(1) # 1: 정지
                old_s = preprocessing(old_s, new_HW, height_range)
                history = np.concatenate((old_s, old_s, old_s, old_s),axis=2)
            action = self.choose_action(history)
            step_counter += 1
            new_s, r, done, info = self.env.step(action)
            
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
                self.train(states, actions, rewards)

        #self.print(total_reward)
        episode_reward_history.append(total_reward)
        if n_episode % 10 ==0:
            message = "Agent(name = {}, episode  = {},  100 average reward = {})".format(self.name,n_episode, np.mean(episode_reward_history))
            print(message)

    def run(self):
        while not self.coord.should_stop():
            self.play_episode()

    def choose_action(self, S,random_mode = True):
        shape = S.shape  # (80, 80, 1)

        if len(shape) == 3:
            S = np.expand_dims(S, axis=0)

        np.testing.assert_equal(S.shape[1:], self.input_shape)


        action_prob = self.sess.run(self.local.Policy,feed_dict={self.local.stateInput: S})
        action_prob = np.squeeze(action_prob)
        if random_mode or np.random.rand(1) < 0.03 :
            return np.random.choice(np.arange(self.output_dim) + self.action_offset, p=action_prob)
        else: return np.argmax(action_prob) + self.action_offset

    def train(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions) - self.action_offset
        rewards = np.array(rewards)

        feed = {
            self.local.stateInput: states
        }

        values = self.sess.run(self.local.Value, feed)

        rewards = discount_reward(rewards, gamma=0.99)

        advantage = rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        feed = {
            self.global_network.stateInput: states,
            self.global_network.action: actions,
            self.global_network.reward: rewards,
            self.global_network.advantage: advantage
        }

        self.sess.run(self.global_network.train_op, feed)


def train():
    s_time = time.time()
    global n_episode
    try:
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        coord = tf.train.Coordinator()

        checkpoint_dir = "./breakout-a3c-data"
        save_path = os.path.join(checkpoint_dir, "model.ckpt")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print("Directory {} was created".format(checkpoint_dir))

        n_threads = 4
        h_size = 512
        input_shape = [84, 84, 4]
        output_dim = 3  # {0, 1, 2}
        action_offset = 1
        global_network = A3CNetwork(name="global",input_shape=input_shape,output_dim=output_dim,h_size = h_size)

        thread_list = []
        env_list = []

        for id in range(n_threads):
            env = gym.make("BreakoutDeterministic-v4")

            single_agent = Agent(env=env,
                                 session=sess,
                                 coord=coord,
                                 name="thread_{}".format(id),
                                 global_network=global_network,
                                 input_shape=input_shape,
                                 output_dim=output_dim,h_size=h_size,action_offset=action_offset)
            thread_list.append(single_agent)
            env_list.append(env)

        init = tf.global_variables_initializer()
        sess.run(init)        
        if tf.train.get_checkpoint_state(os.path.dirname(save_path)):
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
            saver = tf.train.Saver(var_list=var_list)
            maybe_path = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess,maybe_path )
            print("Model restored to global: ", maybe_path)
            
            n_episode = int(tf.train.latest_checkpoint(checkpoint_dir).split('-')[-1])

        else:
            print("No model is found")

        for t in thread_list:
            t.start()

#         print("Ctrl + C to close")


        while True:
            time.sleep(60*30)
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
            saver = tf.train.Saver(var_list=var_list)
            saver.save(sess, save_path, global_step = n_episode)
            print('Checkpoint Saved to {}/{}. elapsed = {}'.format(save_path, n_episode,time.time()-s_time))

    except KeyboardInterrupt:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
        saver = tf.train.Saver(var_list=var_list)
        saver.save(sess, save_path)
        print('Checkpoint Saved to {}'.format(save_path))

        print("Closing threads")
        coord.request_stop()
        coord.join(thread_list)

        print("Closing environments")
        for env in env_list:
            env.close()

        sess.close()

def test():
    checkpoint_dir = "./breakout-a3c-data"
    save_path = os.path.join(checkpoint_dir, "model.ckpt")
    coord = tf.train.Coordinator()
    
    sess = tf.InteractiveSession()
    input_shape = [84, 84, 4]
    output_dim = 3  # {1, 2, 3}
    action_offset = 1
    h_size = 512
    global_network = A3CNetwork(name="global",input_shape=input_shape,output_dim=output_dim,h_size = h_size)
    
    
    env = gym.make("BreakoutDeterministic-v4")
    
    agent = Agent(env=env,session=sess,coord=coord,
                         name="test-agent",
                         global_network=global_network,
                         input_shape=input_shape,
                         output_dim=output_dim,h_size=h_size,action_offset=action_offset)
    
    
    if tf.train.get_checkpoint_state(os.path.dirname(save_path)):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
        saver = tf.train.Saver(var_list=var_list)
        
        maybe_path = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, maybe_path)
        print("Model restored to global: ", maybe_path)
        
        sess.run(agent.global_to_local)
    else:
        print('No Model Found!')
        exit()
    

    state = env.reset()
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
            total_reward = 0
            state = env.reset()
            n_step=0
            remain_lives = 5
    
    print('average: {}, min: {}, max: {}'.format(np.mean(episode_reward_history),np.min(episode_reward_history),np.max(episode_reward_history)))
    
if __name__ == '__main__':
    #train()
    test()