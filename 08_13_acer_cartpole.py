

'''
https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch  ----> acer 없음.


http://www.secmem.org/blog/2019/06/17/ACER/

'''

import multiprocessing
import multiprocessing.connection
import time
import gym
import tensorflow as tf
import numpy as np
import os, random
from collections import deque,namedtuple

INPUT_DIM = 4
OUTPUT_DIM = 2


Transition = namedtuple('Transition', ('pi_logit', 'pi', 'q'))


# remove last step
def strip(var, nenvs, nsteps, flat = False):
    # var: [ nenvs*(nsteps+1), last_dim]
    last_dim = var.get_shape()[-1].value
    vars = batch_to_seq(var, nenvs, nsteps + 1, flat)  # var: (nenvs,last_dim)  ---> list of length nsteps+1
    # vars: [(nenvs,last_dim), (nenvs,last_dim), .... ]  <----- nsteps+1 길이
    return seq_to_batch(vars[:-1],last_dim, flat)


# 2D tensor일 때, flat=True, 3D tensor이면 False
def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

def seq_to_batch(h,last_dim=None, flat = False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert(len(shape) > 1)
        return tf.reshape(tf.concat(axis=1, values=h), [-1, last_dim])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])
def check_shape(ts,shapes):
    i = 0
    for (t,shape) in zip(ts,shapes):
        assert t.get_shape().as_list()==shape, "id " + str(i) + " shape " + str(t.get_shape()) + str(shape)
        i += 1
# For ACER
def get_by_index(x, idx):
    assert(len(x.get_shape()) == 2)
    assert(len(idx.get_shape()) == 1)
    idx_flattened = tf.range(0, tf.shape(x)[0]) * tf.shape(x)[1] + idx
    y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                  idx_flattened)  # use flattened indices
    return y


def cat_entropy_softmax(p0):
    return - tf.reduce_sum(p0 * tf.log(p0 + 1e-6), axis = 1)


def gradient_add(g1, g2, param):
    print([g1, g2, param.name])
    assert (not (g1 is None and g2 is None)), param.name
    if g1 is None:
        return g2
    elif g2 is None:
        return g1
    else:
        return g1 + g2

def q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma):
    """
    Calculates q_retrace targets

    :param R: Rewards   (nenvs*nsteps,)
    :param D: Dones     (nenvs*nsteps,)
    :param q_i: Q values for actions taken    (nenvs*nsteps,)
    :param v: V values                        (nenvs*(nsteps+1),)
    :param rho_i: Importance weight for each action  (nenvs*nsteps,)
    :return: Q_retrace values
    """
    rho_bar = batch_to_seq(tf.minimum(1.0, rho_i), nenvs, nsteps, True)  # rho_i: (nenvs*nsteps,) --> [(nenvs,) (nenvs,), ...] <--nsteps개
    rs = batch_to_seq(R, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    ds = batch_to_seq(D, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    q_is = batch_to_seq(q_i, nenvs, nsteps, True)
    vs = batch_to_seq(v, nenvs, nsteps + 1, True) # v (nenvs*nsteps,) --> vs: [(nenvs,) (nenvs,), ...] <--(nsteps+1)개
    v_final = vs[-1]
    qret = v_final
    qrets = []
    for i in range(nsteps - 1, -1, -1):  #(nsteps-1, nsteps-2, ..., 0)
        check_shape([qret, ds[i], rs[i], rho_bar[i], q_is[i], vs[i]], [[nenvs]] * 6)
        qret = rs[i] + gamma * qret * (1.0 - ds[i])
        qrets.append(qret)
        qret = (rho_bar[i] * (qret - q_is[i])) + vs[i]
    qrets = qrets[::-1]   # 역으로 배영
    qret = seq_to_batch(qrets,None, flat=True)
    return qret

class Game(object):
    def __init__(self, seed: int):
        self.env = gym.make('CartPole-v1')  # BreakoutNoFrameskip-v4    "BreakoutDeterministic-v4"
        self.env.seed(seed)
        self.max_epLength = 500
    def step(self, action):
        self.next_state, reward, done, _ = self.env.step(action)
        
        if done and self.total_reward < self.max_epLength-1:
            reward = -100
        
        self.total_reward += reward
        
        if done:
            episode_info = {"reward": self.total_reward}
            self.reset()
        else:
            episode_info = None


        return self.next_state, reward, done, episode_info
        
    def reset(self):
        self.total_reward = 0
        
        self.next_state = self.env.reset()
        return self.next_state

def worker_process(remote: multiprocessing.connection.Connection, seed: int):

                                                                             
    game = Game(seed)

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError

class Worker(object):

    child: multiprocessing.connection.Connection
    process: multiprocessing.Process
    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed))
        self.process.start()

class ACER(object):

    def __init__(self,nenvs,nsteps, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.input_dim = INPUT_DIM
        self.output_dim = OUTPUT_DIM
        alpha = 0.99  # polyak average: exponential weight
        gamma = 0.99  # discount rate
        c = 10.
        delta = 1
        ent_coef = 0.01
        q_coef = 0.5
        trust_region = True
        max_grad_norm = 10.
        learning_rate = 0.01
        rprop_alpha = 0.99
        rprop_epsilon = 1e-05



        self.obs = tf.placeholder(shape=(None,4), name="obs", dtype=np.float32)

        self.A = tf.placeholder(tf.int32, [None]) # actions
        self.D = tf.placeholder(tf.float32, [None]) # dones
        self.R = tf.placeholder(tf.float32, [None]) # rewards, not returns
        self.MU = tf.placeholder(tf.float32, [None, self.output_dim]) # mu's     policy probability  train_model_p
        eps = 1e-6
        


        with tf.variable_scope('acer_model', reuse=tf.AUTO_REUSE):
            self.policy = self.build_policy()
        
        params = tf.trainable_variables('acer_model')
        '''
        print("Params {}".format(len(params)))
        for var in params:
            print(var)
        '''


        # create polyak averaged model
        ema = tf.train.ExponentialMovingAverage(alpha)  # alpha=0.99
        ema_apply_op = ema.apply(params)

        def custom_getter(getter, *args, **kwargs):
            v = ema.average(getter(*args, **kwargs))
            #print(v.name)
            return v
        
        with tf.variable_scope("acer_model", custom_getter=custom_getter, reuse=True):
            self.polyak_policy = self.build_policy()
            
        self.action = ACER._sample(self.policy.pi_logit)
        
        
        ################
        v = tf.reduce_sum(self.policy.pi * self.policy.q, axis = -1)
        f, f_pol, q = map(lambda var: strip(var, nenvs, nsteps), [self.policy.pi, self.polyak_policy.pi, self.policy.q])  # nenvs*nsteps x 2
        # Get pi and q values for actions taken
        f_i = get_by_index(f, self.A)  # (nenvs*nsteps,)
        q_i = get_by_index(q, self.A)  # (nenvs*nsteps,)

        # Compute ratios for importance truncation
        rho = f / (self.MU + eps)  # (nenvs*nseps,2)
        rho_i = get_by_index(rho, self.A)  # (nenvs*nseps,)

        # Calculate Q_retrace targets
        qret = q_retrace(self.R, self.D, q_i, v, rho_i, nenvs, nsteps, gamma)   # qret: ( nenvs*nsteps , )

        # Calculate losses
        # 1, Entropy
        # entropy = tf.reduce_mean(strip(train_model.pd.entropy(), nenvs, nsteps))
        entropy = tf.reduce_mean(cat_entropy_softmax(f))  # entropy: scalar     

        # 2. Policy Graident loss, with truncated importance sampling & bias correction
        v = strip(v, nenvs, nsteps, True)   # ( nenvs*nsteps , )

        # Truncated importance sampling
        adv = qret - v
        logf = tf.log(f_i + eps)
        gain_f = logf * tf.stop_gradient(adv * tf.minimum(c, rho_i))  # [nenvs * nsteps]
        loss_f = -tf.reduce_mean(gain_f)

        # 3. Bias correction for the truncation
        adv_bc = (q - tf.reshape(v, [nenvs * nsteps, 1]))  # [nenvs * nsteps, nact]
        logf_bc = tf.log(f + eps) # / (f_old + eps)
        gain_bc = tf.reduce_sum(logf_bc * tf.stop_gradient(adv_bc * tf.nn.relu(1.0 - (c / (rho + eps))) * f), axis = 1) #IMP: This is sum, as expectation wrt f
        loss_bc= -tf.reduce_mean(gain_bc)

        loss_policy = loss_f + loss_bc

        # Value/Q function loss, and explained variance
        loss_q = tf.reduce_mean(tf.square(tf.stop_gradient(qret) - q_i)*0.5)

        # Net loss
        loss = loss_policy + q_coef * loss_q - ent_coef * entropy  # q_coef = 0.5, ent_coef=0.01

        if trust_region:
            # tf.gradients는 list를 return한다.
            # 여기서  nsteps * nenvs을 곱했다, 아래에서 다시 나누었다. 안정성을 위해???    그리고, -1을 곱했다. 아래에서 다시 -1을 곱한다.
            g = tf.gradients(- (loss_policy - ent_coef * entropy) * nsteps * nenvs, f) #[(nenvs * nsteps, nact)] <----1개짜리 list
            # k = tf.gradients(KL(f_pol || f), f)
            k = - f_pol / (f + eps) #[nenvs * nsteps, nact] # Directly computed gradient of KL divergence wrt f
            k_dot_g = tf.reduce_sum(k * g, axis=-1)
            adj = tf.maximum(0.0, (tf.reduce_sum(k * g, axis=-1) - delta) / (tf.reduce_sum(tf.square(k), axis=-1) + eps)) #[nenvs * nsteps]

            g = g - tf.reshape(adj, [nenvs * nsteps, 1]) * k  # g를 새로운 값으로 대체
            
            # 위에서 곱한 (nenvs*nsteps)를 나누었다.
            grads_f = -g/(nenvs*nsteps) # These are turst region adjusted gradients wrt f ie statistics of policy pi
            grads_policy = tf.gradients(f, params, grads_f)  # chain rule
            
            # value loss에 대한 미분
            grads_q = tf.gradients(loss_q * q_coef, params)
            
            # 2가지 gradient 더하기. None이 있기 때문에, gradient_add에서 잘 처리 해 주어야 한다.
            grads = [gradient_add(g1, g2, param) for (g1, g2, param) in zip(grads_policy, grads_q, params)]

        else:
            grads = tf.gradients(loss, params)


        if max_grad_norm is not None:
            grads, norm_grads = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=rprop_alpha, epsilon=rprop_epsilon)
        _opt_op = trainer.apply_gradients(grads)

        # so when you call _train, you first do the gradient step, then you apply ema
        with tf.control_dependencies([_opt_op]):
            self.train_op = tf.group(ema_apply_op)

        
        
        self.xx = loss

        pass



    def build_policy(self):
        net = tf.layers.dense(self.obs, 32, activation=tf.nn.relu)
        latent = tf.layers.dense(net, 32,activation=tf.nn.relu)
        
        pi_logits = tf.layers.dense(latent, self.output_dim, activation=None, name="pi_logits")
        pi = tf.nn.softmax(pi_logits)
        
        
        q = tf.layers.dense(latent, self.output_dim, activation=None, name="critic")
        return Transition(pi_logits, pi, q)


    def step(self, session, obs: np.ndarray):
        # actions, qs, mus
        return session.run([self.action, self.policy.pi], feed_dict={self.obs: obs})


    @staticmethod
    def _sample(logits: tf.Tensor):
        #  Gumbel Softmax Trick               
        uniform = tf.random_uniform(tf.shape(logits))
        return tf.argmax(logits - tf.log(-tf.log(uniform)),
                         axis=-1,
                         name="action")


    def train(self, sess, states_, actions_,rewards_, mus_, dones_):
        
        states_ = states_.reshape(-1,INPUT_DIM)  # (nenvs,nsteps+1,INPUT_DIM) --> 
        mus_ = mus_.reshape(-1,OUTPUT_DIM)        # (nenvs,nsteps,OUTPUT_DIM) --> 
        actions_ = actions_.reshape(-1)          # (nenvs,nsteps)  --> flatten
        rewards_ = rewards_.reshape(-1)          # (nenvs,nsteps)  --> flatten
        dones_ = dones_.reshape(-1)          # (nenvs,nsteps)  --> flatten
        
        
        feed_dict = {self.obs: states_, self.A: actions_, self.D: dones_, self.R: rewards_, self.MU: mus_}
        

        
        sess.run(self.train_op, feed_dict = feed_dict)  ##############  MAIN TRAIN

        
      
    def resotre(self,sess):
        if self.checkpoint_dir is not None:
            self.saver = tf.train.Saver()

            maybe_path = tf.train.latest_checkpoint(self.checkpoint_dir)
            if os.path.exists(self.checkpoint_dir) and maybe_path:
                print("Restored {}".format(maybe_path))
                self.saver.restore(sess, maybe_path)
        else:
            print('No Model Found!!!')

    def save(self,sess, global_step):
        path = os.path.join(self.checkpoint_dir, "model.ckpt")
        self.saver.save(sess, path,global_step=global_step) # self.global_step
        
        
    
class Agent(object):
    def __init__(self):

        self.gamma = 0.99
        self.lamda = 0.95

        buffer_size = 15000
        self.replay_start = 5000   # train을 위한, experience_replay의 최소 크기.
        self.replay_ratio= 4 # on-policy로 data생성 --> train 후, replay-buffer에 있는 data로 몇번 train할 것인가.


        self.n_workers = 4
        self.worker_steps = 20  # 각 episode를 worker_steps만큼씩 만들어 data를 생성할 것인지 결정.
        self.batch_size = self.n_workers * self.worker_steps


        self.workers = [Worker(47 + i) for i in range(self.n_workers)]

        self.obs = np.zeros((self.n_workers, 4), dtype=np.float32)  # 모든 worker의 state를 모아 놓는 저장 소.
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()


        CHECK_POINT_DIR = './cartpole-acer'
        if not os.path.exists(CHECK_POINT_DIR):
            os.makedirs(CHECK_POINT_DIR)
        if tf.train.latest_checkpoint(CHECK_POINT_DIR):
            self.n_update = int(tf.train.latest_checkpoint(CHECK_POINT_DIR).split('-')[-1])
        else:
            self.n_update=0



        self.acer_model = ACER(self.n_workers,self.worker_steps,checkpoint_dir=CHECK_POINT_DIR)   ###### ACER
 
        np.random.seed(7)
        tf.set_random_seed(7)                                                  
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
         
        self.acer_model.resotre(self.session)
        
        self.experience_replay = deque(maxlen=buffer_size)

    def sample(self):
        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        obs = np.zeros((self.n_workers, self.worker_steps+1, INPUT_DIM), dtype=np.float32)
        mus = np.zeros((self.n_workers, self.worker_steps, OUTPUT_DIM), dtype=np.float32)
        episode_infos = []

        for t in range(self.worker_steps):

            obs[:, t] = self.obs  # value만 복사. obs와 self.obs는 별개.

            # model로 부터, actions, mus(확률)
            actions[:, t], mus[:, t, :] = self.acer_model.step(self.session, self.obs) 

            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w, t]))

            for w, worker in enumerate(self.workers):

                # env로 부터, next_states, rewards, dones, info                                                                   
                self.obs[w], rewards[w, t], dones[w, t], info = worker.child.recv()  # self.obs는 위에서 obs[:,t]에 넣었으니, self.obs를 new_state로 update

                if info:
                    episode_infos.append(info)
        
        obs[:,-1] = self.obs  # 마지막 state를 하나 더. 
        
        samples = []
        for o,a,r,m,d in zip(obs,actions,rewards,mus,dones):
            samples.append([o,a,r,m,d])

        return samples, episode_infos



    def train(self, on_policy_flag):
        # smaples를 reaply memory에 넣는다.
        # samples로 on policy train을 한번 하고,  off policy train을  self.replay_ratio번 한다.

        if on_policy_flag:
            samples, sample_episode_info = self.sample()  # episode 생성

            self.experience_replay.extend(samples)  # nenv개의 data를 쌓는다.
            
        else:
            samples = random.sample(self.experience_replay, self.n_workers)
            sample_episode_info = None
            
        states_, actions_,rewards_, mus_, dones_ = self.convert_to_batch(samples)
        
        
        self.acer_model.train(self.session, states_, actions_,rewards_, mus_, dones_)
            
        return sample_episode_info
    
    def convert_to_batch(self,samples):
        
        states_ = np.stack([x[0] for x in samples])
        actions_ = np.stack([x[1] for x in samples])
        rewards_ = np.stack([x[2] for x in samples])
        mus_ = np.stack([x[3] for x in samples])
        dones_ = np.stack([x[4] for x in samples])
        
        return states_, actions_,rewards_, mus_, dones_


    def run_training_loop(self):
        episode_info = deque(maxlen=100)

        best_reward = 450
        s_time = time.time()
        while True:
            self.n_update += 1
            sample_episode_info = self.train(on_policy_flag=True) # on policy로 한번 train
            episode_info.extend(sample_episode_info)
            
            if self.replay_ratio > 0 and len(self.experience_replay) > self.replay_start:
                n = np.random.poisson(self.replay_ratio)
                for _ in range(n):
                    self.train(on_policy_flag=False)  # no simulation steps in this
            
            
        
            if self.n_update % 100 ==0:
                reward_mean = Agent._get_mean_episode_info(episode_info)
                print(self.n_update, "100 mean reward {:.3f} / elapsed: {} sec".format(reward_mean,int(time.time()-s_time)))
                 
                if reward_mean >= best_reward+1:
                    #self.acer_model.save(self.session,global_step = self.n_update)
                    best_reward = reward_mean

            if best_reward > 490:
                self.acer_model.save(self.session,global_step = self.n_update)  
                break               
 
            if self.n_update % 5000 ==0:
                self.acer_model.save(self.session,global_step = self.n_update)
                
    @staticmethod
    def _get_mean_episode_info(episode_info):
        if len(episode_info) > 0:
            return np.mean([info["reward"] for info in episode_info])
        else:
            return np.nan

    def destroy(self):
        for worker in self.workers:
            worker.child.send(("close", None))







def train():
    agent = Agent()
    agent.run_training_loop()
    agent.destroy()



def infer():
    CHECK_POINT_DIR = './cartpole-acer'
    
    sess = tf.InteractiveSession()

    acer_model = ACER(1,1,checkpoint_dir=CHECK_POINT_DIR)
    acer_model.resotre(sess)
    
    
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 500

    state = env.reset()

    for i in range(1):
        state = env.reset()
        reward_sum = 0
        while True:
    
            env.render()
            action = sess.run(acer_model.action,feed_dict={acer_model.obs: [state]})[0]

            state, reward, done, _ = env.step(action)
            reward_sum += reward

            if done or reward_sum >= 500:
                print("Try: {}, Total score: {}".format(i,reward_sum))
                break  














if __name__ == "__main__":
    #train()
    infer()

    print('Done')

