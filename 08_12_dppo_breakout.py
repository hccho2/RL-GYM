'''
http://blog.varunajayasiri.com/ml/ppo.html?fbclid=IwAR1tsCK7rUT-JioH4z2kXYu_MmesO9ZKiY_J14uGMU1gRVUnNCELm6Z1Z3g

원본에 
- reward clip
- message 출력 주기 설정

추가:
- BreakoutDeterministic-v4로 변경,
- Game도 skip 있다는 가정으로 변경.  life는 1개.

ppo2.py: 아래 2개 조건을 하나 추가하고, 다시 하나 추가해보니, 각각 잘 됨.
- dead 추가 life 5개로 변경.
- reward -1

ppo3.py
- state: int --> float로 변환


ppo4.py
- sample_model, trainer model ---> 1개로...
- summary 제거

ppo5.py
- action dim, action_offset 반영
- 불필요한 op 제거.


ppo7.py
- coding 구조 수정.


ppo8.py
- preprocessing 방벗 ---> 내가 하던 방식으로....
- save, restore 추가
- 코드 정리

   0 184 -4.50 98.833
  10 223 -4.11 111.213
  20 204 -3.72 127.840
  30 208 -3.36 137.500
  40 220 -3.11 144.180
  50 219 -2.11 179.190
  60 229 -1.09 213.000
  70 222 -0.58 230.650
  80 216 -0.11 247.820
  90 185 0.34 260.960
 100 187 1.27 291.450
 110 247 2.05 319.320
 120 231 3.21 360.520
 130 240 3.81 380.920
 140 227 4.32 394.780
 150 238 4.82 411.440
 160 236 4.94 413.910
 170 225 4.87 413.010
 180 233 4.83 413.050
 190 225 4.67 408.160
 200 242 4.80 409.500
 210 237 5.25 423.700
 220 222 5.75 440.700
 230 246 6.04 446.720
 240 230 6.68 470.510
 250 228 6.75 475.180
 260 221 7.03 481.640
 270 229 7.20 486.930
 280 233 7.83 505.110
 290 237 7.72 500.030
 300 232 7.74 496.090
 310 231 7.71 495.790
 320 228 7.82 495.770
 330 235 8.11 506.240
 340 168 8.20 506.550

10 100 mean reard / episode-length: / -3.988 / 115.146 elapsed: 52 sec
20 100 mean reard / episode-length: / -3.680 / 127.170 elapsed: 105 sec
30 100 mean reard / episode-length: / -3.240 / 142.090 elapsed: 155 sec
40 100 mean reard / episode-length: / -3.020 / 152.460 elapsed: 203 sec
50 100 mean reard / episode-length: / -2.930 / 157.570 elapsed: 251 sec
60 100 mean reard / episode-length: / -2.740 / 165.640 elapsed: 299 sec
70 100 mean reard / episode-length: / -2.830 / 161.120 elapsed: 348 sec
80 100 mean reard / episode-length: / -2.910 / 159.420 elapsed: 399 sec
90 100 mean reard / episode-length: / -2.930 / 160.040 elapsed: 449 sec
100 100 mean reard / episode-length: / -2.800 / 166.910 elapsed: 497 sec
110 100 mean reard / episode-length: / -2.370 / 185.050 elapsed: 544 sec
120 100 mean reard / episode-length: / -2.660 / 171.780 elapsed: 593 sec
130 100 mean reard / episode-length: / -2.480 / 176.040 elapsed: 641 sec
140 100 mean reard / episode-length: / -2.410 / 179.690 elapsed: 689 sec
150 100 mean reard / episode-length: / -2.270 / 188.630 elapsed: 736 sec
160 100 mean reard / episode-length: / -1.760 / 207.610 elapsed: 782 sec
170 100 mean reard / episode-length: / -1.330 / 222.410 elapsed: 827 sec
180 100 mean reard / episode-length: / -1.010 / 228.210 elapsed: 873 sec
190 100 mean reard / episode-length: / -0.640 / 236.240 elapsed: 919 sec
200 100 mean reard / episode-length: / -0.760 / 225.820 elapsed: 966 sec
210 100 mean reard / episode-length: / -1.060 / 213.390 elapsed: 1014 sec
220 100 mean reard / episode-length: / -0.990 / 221.100 elapsed: 1060 sec
230 100 mean reard / episode-length: / -0.590 / 241.240 elapsed: 1105 sec
240 100 mean reard / episode-length: / 0.150 / 279.800 elapsed: 1150 sec
250 100 mean reard / episode-length: / 1.040 / 322.930 elapsed: 1194 sec
260 100 mean reard / episode-length: / 1.870 / 364.270 elapsed: 1237 sec
270 100 mean reard / episode-length: / 2.530 / 397.810 elapsed: 1281 sec
280 100 mean reard / episode-length: / 3.060 / 420.580 elapsed: 1325 sec
290 100 mean reard / episode-length: / 3.160 / 422.730 elapsed: 1368 sec
300 100 mean reard / episode-length: / 3.540 / 438.610 elapsed: 1411 sec
310 100 mean reard / episode-length: / 3.700 / 445.390 elapsed: 1455 sec
320 100 mean reard / episode-length: / 4.050 / 459.430 elapsed: 1498 sec
330 100 mean reard / episode-length: / 4.250 / 466.730 elapsed: 1542 sec
340 100 mean reard / episode-length: / 4.450 / 474.310 elapsed: 1585 sec
350 100 mean reard / episode-length: / 4.680 / 482.050 elapsed: 1628 sec
360 100 mean reard / episode-length: / 4.970 / 492.380 elapsed: 1671 sec
370 100 mean reard / episode-length: / 4.820 / 486.040 elapsed: 1714 sec
380 100 mean reard / episode-length: / 4.950 / 489.230 elapsed: 1757 sec
390 100 mean reard / episode-length: / 4.850 / 485.550 elapsed: 1800 sec
400 100 mean reard / episode-length: / 4.820 / 484.530 elapsed: 1843 sec
410 100 mean reard / episode-length: / 4.820 / 485.380 elapsed: 1888 sec
420 100 mean reard / episode-length: / 5.100 / 493.670 elapsed: 1935 sec
430 100 mean reard / episode-length: / 5.140 / 493.340 elapsed: 1981 sec
440 100 mean reard / episode-length: / 5.390 / 502.850 elapsed: 2028 sec
450 100 mean reard / episode-length: / 5.580 / 506.960 elapsed: 2074 sec
460 100 mean reard / episode-length: / 5.560 / 506.670 elapsed: 2121 sec
470 100 mean reard / episode-length: / 5.630 / 511.380 elapsed: 2168 sec
480 100 mean reard / episode-length: / 5.450 / 504.830 elapsed: 2214 sec
490 100 mean reard / episode-length: / 5.440 / 503.010 elapsed: 2261 sec
500 100 mean reard / episode-length: / 5.230 / 494.880 elapsed: 2308 sec
510 100 mean reard / episode-length: / 4.920 / 479.170 elapsed: 2354 sec
520 100 mean reard / episode-length: / 4.830 / 474.700 elapsed: 2399 sec
530 100 mean reard / episode-length: / 4.730 / 470.690 elapsed: 2444 sec


10 100 mean reard / episode-length: / -0.162 / 244.162 elapsed: 138 sec
20 100 mean reard / episode-length: / 0.025 / 253.000 elapsed: 276 sec
30 100 mean reard / episode-length: / 0.400 / 264.760 elapsed: 413 sec
40 100 mean reard / episode-length: / 0.770 / 279.960 elapsed: 551 sec
50 100 mean reard / episode-length: / 1.390 / 303.470 elapsed: 688 sec
60 100 mean reard / episode-length: / 2.130 / 331.980 elapsed: 823 sec
70 100 mean reard / episode-length: / 2.700 / 353.550 elapsed: 956 sec
80 100 mean reard / episode-length: / 3.270 / 373.440 elapsed: 1092 sec
90 100 mean reard / episode-length: / 3.760 / 392.710 elapsed: 1232 sec
100 100 mean reard / episode-length: / 4.080 / 405.300 elapsed: 1368 sec
110 100 mean reard / episode-length: / 4.600 / 426.560 elapsed: 1500 sec
120 100 mean reard / episode-length: / 5.060 / 443.900 elapsed: 1633 sec
130 100 mean reard / episode-length: / 5.480 / 459.010 elapsed: 1762 sec
140 100 mean reard / episode-length: / 5.870 / 473.280 elapsed: 1891 sec
150 100 mean reard / episode-length: / 5.940 / 473.990 elapsed: 2019 sec
160 100 mean reard / episode-length: / 6.440 / 490.220 elapsed: 2150 sec
170 100 mean reard / episode-length: / 6.300 / 483.440 elapsed: 2280 sec
180 100 mean reard / episode-length: / 6.560 / 489.230 elapsed: 2407 sec
190 100 mean reard / episode-length: / 6.680 / 488.940 elapsed: 2538 sec
200 100 mean reard / episode-length: / 6.360 / 473.510 elapsed: 2667 sec
210 100 mean reard / episode-length: / 6.690 / 483.870 elapsed: 2795 sec
220 100 mean reard / episode-length: / 7.080 / 498.610 elapsed: 2922 sec
230 100 mean reard / episode-length: / 6.910 / 497.040 elapsed: 3054 sec
240 100 mean reard / episode-length: / 7.380 / 516.340 elapsed: 3185 sec
250 100 mean reard / episode-length: / 7.850 / 533.250 elapsed: 3317 sec
260 100 mean reard / episode-length: / 7.930 / 535.090 elapsed: 3450 sec
270 100 mean reard / episode-length: / 7.950 / 534.360 elapsed: 3582 sec
280 100 mean reard / episode-length: / 8.640 / 554.040 elapsed: 3711 sec

'''


import io
from collections import deque
from pathlib import Path
from typing import Dict, List, Union

import cv2
import multiprocessing
import multiprocessing.connection
import time
import gym
import tensorflow as tf
import numpy as np
from matplotlib import pyplot

import os,random
from skimage.transform import resize
from skimage.color import rgb2gray
ACTION_OFFSET = 1
ACTION_DIM = 3

class Orthogonal(object):

    def __init__(self, scale=1.):
        self.scale = scale

    def __call__(self, shape, dtype=None, partition_info=None):
 
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (self.scale * q[:shape[0], :shape[1]]).astype(np.float32)
 
    def get_config(self):
        return {
            'scale': self.scale
        }

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


class Game(object):
    def __init__(self, seed: int):
        self.env = gym.make("BreakoutDeterministic-v4")  # BreakoutNoFrameskip-v4    "BreakoutDeterministic-v4"
        self.env.seed(seed)

    def step(self, action):

            self.dead = False
            self.step_counter += 1
            new_s, r, done, info = self.env.step(action)
            
            r = np.clip(r,-1.,1.)
            if self.remain_lives > info['ale.lives']:
                self.remain_lives = info['ale.lives']
                self.dead = True   
                r = -1     
            
            self.total_reward += r
    
    
            new_s = preprocessing(new_s, new_HW, height_range)  # (210, 160, 3)
            self.next_state = np.append(new_s-self.old_s, self.next_state[:, :, :3],axis=2)
            self.old_s = new_s

            if done:
                episode_info = {"reward": self.total_reward, "length": self.step_counter}
                self.reset()
            else:
                episode_info = None

            if self.dead:
                old_s,_,_,_ = self.env.step(1)
                for _ in range(random.randint(1, 10)):
                    self.old_s,_,_,_ = self.env.step(1) # 1: 정지
                self.old_s = preprocessing(self.old_s, new_HW, height_range)
                self.next_state = np.concatenate((self.old_s, self.old_s, self.old_s, self.old_s),axis=2)
                
            
                     
            return self.next_state, r, self.dead, episode_info


    def reset(self):

        state = self.env.reset()
        for _ in range(random.randint(1, 30)):
            state, _, _, _ = self.env.step(1)  # 1: 정지
        
        self.remain_lives = 5
        self.total_reward = 0
        self.step_counter = 0

        self.old_s = preprocessing(state, new_HW, height_range)
            
        self.next_state = np.concatenate((self.old_s, self.old_s, self.old_s, self.old_s),axis=2)
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


class PPONet(object):

    def __init__(self, *, reuse: bool,checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir

        self.obs = tf.placeholder(shape=(None, 84, 84, 4), name="obs", dtype=np.float32)
        obs_float = self.obs

        with tf.variable_scope("model", reuse=reuse):

            self.h = PPONet._cnn(obs_float)

            self.pi_logits = PPONet._create_policy_network(self.h, ACTION_DIM)
            self.prob = tf.nn.softmax(self.pi_logits)

            self.value = PPONet._create_value_network(self.h)


            self.params = tf.trainable_variables()

        self.action = PPONet._sample(self.pi_logits)

        self.neg_log_pi = self.neg_log_prob(self.action, "neg_log_pi_old")

        self.policy_entropy = PPONet._get_policy_entropy(self.pi_logits)



        # train op


        self.sampled_action = tf.placeholder(dtype=tf.int32, shape=[None], name="sampled_action")

        self.sampled_return = tf.placeholder(dtype=tf.float32, shape=[None], name="sampled_return")

        self.sampled_normalized_advantage = tf.placeholder(dtype=tf.float32, shape=[None],
                                                           name="sampled_normalized_advantage")


        self.sampled_neg_log_pi = tf.placeholder(dtype=tf.float32, shape=[None], name="sampled_neg_log_pi")


        self.sampled_value = tf.placeholder(dtype=tf.float32, shape=[None], name="sampled_value")  # old policy와 대비되는 old value


        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")

        self.clip_range = tf.placeholder(dtype=tf.float32, shape=[], name="clip_range")

        neg_log_pi = self.neg_log_prob(self.sampled_action, "neg_log_pi")


        ratio = tf.exp(self.sampled_neg_log_pi - neg_log_pi, name="ratio")


        clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range, name="clipped_ratio")
        self.policy_reward = tf.reduce_mean(tf.minimum(ratio * self.sampled_normalized_advantage,
                                                       clipped_ratio * self.sampled_normalized_advantage),name="policy_reward")


        self.entropy_bonus = tf.reduce_mean(self.policy_entropy, name="entropy_bonus")


        # old value와 많이 벗어날 수 없는 예측닶.
        clipped_value = tf.add(self.sampled_value,
                               tf.clip_by_value(self.value - self.sampled_value, -self.clip_range, self.clip_range),
                               name="clipped_value")
        self.vf_loss = tf.multiply(0.5,
                                   tf.reduce_mean(tf.maximum(tf.square(self.value - self.sampled_return),
                                                             tf.square(clipped_value - self.sampled_return))),
                                   name="vf_loss")

        self.loss = -(self.policy_reward - 0.5 * self.vf_loss + 0.01 * self.entropy_bonus)


        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.params), 0.5)


        adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-5)
        grads_and_vars = list(zip(grads, self.params))
        self.train_op = adam.apply_gradients(grads_and_vars, name="apply_gradients")


        self.approx_kl_divergence = .5 * tf.reduce_mean(tf.square(neg_log_pi - self.sampled_neg_log_pi))
        self.clip_fraction = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.clip_range)))



    @staticmethod
    def _get_policy_entropy(logits: tf.Tensor):

        a = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        exp_a = tf.exp(a)
        z = tf.reduce_sum(exp_a, axis=-1, keepdims=True)
        p = exp_a / z

        return tf.reduce_sum(p * (tf.log(z) - a), axis=-1)

    def neg_log_prob(self, action: tf.Tensor, name: str) -> tf.Tensor:

        one_hot_actions = tf.one_hot(action, ACTION_DIM)
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.pi_logits,
            labels=one_hot_actions,
            dim=-1,
            name=name)


    @staticmethod
    def _sample(logits: tf.Tensor):
                                        
        uniform = tf.random_uniform(tf.shape(logits))
        return tf.argmax(logits - tf.log(-tf.log(uniform)),
                         axis=-1,
                         name="action")

    @staticmethod
    def _cnn(scaled_images: tf.Tensor):


        h1 = tf.layers.conv2d(scaled_images,
                              name="conv1",
                              filters=32,
                              kernel_size=8,
                              kernel_initializer=Orthogonal(scale=np.sqrt(2)),
                              strides=4,
                              padding="valid",
                              activation=tf.nn.relu)

        h2 = tf.layers.conv2d(h1,
                              name="conv2",
                              filters=64,
                              kernel_size=4,
                              kernel_initializer=Orthogonal(scale=np.sqrt(2)),
                              strides=2,
                              padding="valid",
                              activation=tf.nn.relu)

        h3 = tf.layers.conv2d(h2,
                              name="conv3",
                              filters=64,
                              kernel_size=3,
                              kernel_initializer=Orthogonal(scale=np.sqrt(2)),
                              strides=1,
                              padding="valid",
                              activation=tf.nn.relu)

        nh = np.prod([v.value for v in h3.get_shape()[1:]])
        flat = tf.reshape(h3, [-1, nh])


        h = tf.layers.dense(flat, 512,
                            activation=tf.nn.relu,
                            kernel_initializer=Orthogonal(scale=np.sqrt(2)),
                            name="hidden")

        return h

    @staticmethod
    def _create_policy_network(h: tf.Tensor, n: int) -> tf.Tensor:
                             
        return tf.layers.dense(h, n,
                               activation=None,
                               kernel_initializer=Orthogonal(scale=0.01),
                               name="logits")

    @staticmethod
    def _create_value_network(h: tf.Tensor) -> tf.Tensor:
        value = tf.layers.dense(h, 1,
                                activation=None,
                                kernel_initializer=Orthogonal(),
                                name="value")
        return value[:, 0]

    def step(self, session: tf.Session, obs: np.ndarray) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        return session.run([self.action, self.value, self.neg_log_pi],
                           feed_dict={self.obs: obs})

    def get_value(self, session: tf.Session, obs: np.ndarray) -> tf.Tensor:

        return session.run(self.value,
                           feed_dict={self.obs: obs})

    def train(self, session: tf.Session, samples: Dict[str, np.ndarray], learning_rate: float, clip_range: float):

        feed_dict = {self.obs: samples['obs'],
                     self.sampled_action: samples['actions'],
                     self.sampled_return: samples['values'] + samples['advantages'],
                     self.sampled_normalized_advantage: PPONet._normalize(samples['advantages']),
                     self.sampled_value: samples['values'],
                     self.sampled_neg_log_pi: samples['neg_log_pis'],
                     self.learning_rate: learning_rate,
                     self.clip_range: clip_range}


        return session.run(self.train_op, feed_dict=feed_dict)
    @staticmethod
    def _normalize(adv: np.ndarray):
        return (adv - adv.mean()) / (adv.std() + 1e-8)                      
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
        
    def choose_action(self, S,random_mode = True):
        shape = S.shape  # (80, 80, 1)

        if len(shape) == 3:
            S = np.expand_dims(S, axis=0)

        sess = tf.get_default_session()
        action_prob = sess.run(self.prob,feed_dict={self.obs: S})
        action_prob = np.squeeze(action_prob)
        if random_mode or np.random.rand(1) < 0.03 :
            return np.random.choice(np.arange(ACTION_DIM) + ACTION_OFFSET, p=action_prob)
        else: return np.argmax(action_prob) + ACTION_OFFSET
        
class Agent(object):
    def __init__(self):

        self.gamma = 0.99
        self.lamda = 0.95

        self.updates = 40000

        self.epochs = 4  # batch_size = n_workers * worker_steps  크기의 data를 몇번 train할 것인가?

        self.n_workers = 2


        self.worker_steps = 128

        self.n_mini_batch = 4  # n_workers * worker_steps 크기의 data를 몇번 나누어 train 할 것인가?
        self.batch_size = self.n_workers * self.worker_steps

        self.mini_batch_size = self.batch_size // self.n_mini_batch
        assert (self.batch_size % self.n_mini_batch == 0)




        self.workers = [Worker(47 + i) for i in range(self.n_workers)]

        self.obs = np.zeros((self.n_workers, 84, 84, 4), dtype=np.float32)
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()


        CHECK_POINT_DIR = './breakout-dppo'
        if not os.path.exists(CHECK_POINT_DIR):
            os.makedirs(CHECK_POINT_DIR)
        if tf.train.latest_checkpoint(CHECK_POINT_DIR):
            self.n_update = int(tf.train.latest_checkpoint(CHECK_POINT_DIR).split('-')[-1])
        else:
            self.n_update=0



        self.ppo_model = PPONet(reuse=False,checkpoint_dir=CHECK_POINT_DIR)

        np.random.seed(7)
        tf.set_random_seed(7)                                                  
        self.session = tf.Session()
        init_op = tf.global_variables_initializer()
        self.session.run(init_op)
        
        self.ppo_model.resotre(self.session)
        
        

    def sample(self) -> (Dict[str, np.ndarray], List):
        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        obs = np.zeros((self.n_workers, self.worker_steps, 84, 84, 4), dtype=np.float32)
        neg_log_pis = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        values = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        episode_infos = []

        for t in range(self.worker_steps):

            obs[:, t] = self.obs

            actions[:, t], values[:, t], neg_log_pis[:, t] = self.ppo_model.step(self.session, self.obs)

            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w, t]+ACTION_OFFSET))

            for w, worker in enumerate(self.workers):

                                                                                                               
                self.obs[w], rewards[w, t], dones[w, t], info = worker.child.recv()

                if info:
                    episode_infos.append(info)
                    
        advantages = self._calc_advantages(dones, rewards, values)
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values,
            'neg_log_pis': neg_log_pis,
            'advantages': advantages
        }

        samples_flat = {}
        for k, v in samples.items():
            samples_flat[k] = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])

        return samples_flat, episode_infos

    def _calc_advantages(self, dones: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:


        advantages = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0
        last_value = self.ppo_model.get_value(self.session, self.obs)

        for t in reversed(range(self.worker_steps)):

            mask = 1.0 - dones[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            delta = rewards[:, t] + self.gamma * last_value - values[:, t]

            last_advantage = delta + self.gamma * self.lamda * last_advantage

            advantages[:, t] = last_advantage

            last_value = values[:, t]

        return advantages

    def train(self, samples: Dict[str, np.ndarray], learning_rate: float, clip_range: float):

        indexes = np.arange(self.batch_size)
        train_info = []

        for _ in range(self.epochs):

            np.random.shuffle(indexes)

            for start in range(0, self.batch_size, self.mini_batch_size):

                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                
                              
                self.ppo_model.train(session=self.session,
                                     learning_rate=learning_rate,
                                     clip_range=clip_range,
                                     samples=mini_batch)



    def run_training_loop(self):
        best_reward = 90
        episode_info = deque(maxlen=100)

        best_reward = 0
        s_time = time.time()
        while True:
            self.n_update += 1
            progress = min(1.0*self.n_update / self.updates, 0.95)
            learning_rate = 2.5e-4 * (1 - progress)
            clip_range = 0.1 * (1 - progress)

            samples, sample_episode_info = self.sample()

            self.train(samples, learning_rate, clip_range)

            episode_info.extend(sample_episode_info)


            if self.n_update % 10 ==0:
                reward_mean, length_mean = Agent._get_mean_episode_info(episode_info)
                print(self.n_update, "100 mean reard / episode-length: / {:.3f} / {:.3f} elapsed: {} sec".format(reward_mean,length_mean,int(time.time()-s_time)))
                
                if reward_mean >= best_reward+1:
                    self.ppo_model.save(self.session,global_step = self.n_update)
                    best_reward = reward_mean
                

            if self.n_update % 500 ==0:
                self.ppo_model.save(self.session,global_step = self.n_update)
    @staticmethod
    def _get_mean_episode_info(episode_info):
        if len(episode_info) > 0:
            return (np.mean([info["reward"] for info in episode_info]),
                    np.mean([info["length"] for info in episode_info]))
        else:
            return np.nan, np.nan

    def destroy(self):
        for worker in self.workers:
            worker.child.send(("close", None))

def train():
    agent = Agent()
    agent.run_training_loop()
    agent.destroy()

def infer():
    CHECK_POINT_DIR = './breakout-dppo'
    
    sess = tf.InteractiveSession()

    ppo_model = PPONet(reuse=False,checkpoint_dir=CHECK_POINT_DIR)
    ppo_model.resotre(sess)
    
    
    env = gym.make("BreakoutDeterministic-v4")
    env._max_episode_steps = 10000
    env = gym.wrappers.Monitor(env, './movie/', force=True,video_callable=lambda count: count % 1 == 0)
    state = env.reset()

    episode_reward_history = deque(maxlen = 100)
    random_episodes = 0
    total_reward = 0
    n_step=0
    remain_lives = 5
    dead = True
    while random_episodes < 1000:
        #env.render()
        if dead:
            env.step(1)  #action 1을 하며, 다시 시작한다. 이게 없으도 몇 step 경과하면 다시 시작한다.
            dead = False
            old_s,_,_,_ = env.step(1)
            old_s = preprocessing(old_s, new_HW, height_range)
            history = np.concatenate((old_s, old_s, old_s, old_s),axis=2)
        
        action = ppo_model.choose_action(history,random_mode = True)
        #print('step: {}, action: {}'.format(n_step, action))
        
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
            if total_reward==864:
                exit()
            total_reward = 0
            state = env.reset()
            n_step=0
            remain_lives = 5
    
    print('average: {}, min: {}, max: {}'.format(np.mean(episode_reward_history),np.min(episode_reward_history),np.max(episode_reward_history)))
    
    
    
    

if __name__ == "__main__":
    train()
    #infer()





