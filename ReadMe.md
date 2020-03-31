# Reinfocement Learning: Breakout을 400점 넘게 train.
* Tensorflow 1.14
* DQN
* Vanilla Policy Gradient
* Actor-Critic
* A3C
* DDPG
* ACER
--------------------------------------
### Breakout에서 주의할 점들
- life가 5개 이므로, done/dead를 잘 구분하자. (dead는 life를 하나 잃는 상황)
- done/dead 구분은 구현과장에서 bug를 만들어 내기 쉽다.


--------------------------------------
Breakout Train Graph: Episode-Reward
<p align="center"><img src="gif/breakout-train-graph.png" />  </p>


| Model | # of episode | 20회 test. min/max/avg |
|---|:---:|:---:|
| `PG` | 90500 | 328 / 736 / 419 |
| `DQN(double,dueling) ` | 14000 | 194 / 420 / 329 (Boltzmann), 102 / 421 / 311(ϵ-greedy) |
| `A2C(MC)` |40000 | 402 / 459 / 432 |
| `A3C(grad,MC)`  | 40023| 376 / 431 / 413|
| `A3C(data,MC)`  | 35283| 263 / 371 / 284 |
| `A3C(grad,30step)` | 41742| 340 / 428 / 386|

표 설명: 모델별 Breakout 결과 비교(argmax방식에 확률 3%로 action확률에 기반한 randomness 적용): action 선택에 약간의 randomness가 있다. 그렇지 않으면 episode마다 결과가 달라지지 않는다. randomness를 주는 정도에 따라서도 결과가 달라진다.


--------------------------------------
### Policy Gradient-REINFORCE(BreakoutDeterministic-v4)

Policy Gradient - REINFORCE - BreakoutDeterministic-v4  - reward(750)
<p align="center"><img src="gif/PG.gif" />  </p>

- [Code](https://github.com/hccho2/RL-GYM/blob/master/08_5_softmax_pg_breakout.py)
- [Pretrained Model](https://github.com/hccho2/RL-GYM/tree/master/breakout2)
- env = gym.make("BreakoutDeterministic-v4")
- env가 만들어 주는 state는 (210,160,3)형태의 uint8 data이다. Preprocessing을 통해, (80,80,1) 또는 (84,84,1) float data로 변환 
- Network입력 data:
	* 현재 state와 직전 state의 차이 값을 network의 입력값으로 사용: (80,80,1)
	* 현재 state와 직전 state의 차이 값들을 4개 쌓은 후, 입력값으로 사용: (80,80,4)
	* train해 보면, 후자가 더 잘 된다. 코드에서 train_state_stack(), infer2()
- RMSprop보다 Adam을 사용하는 것이 좋다.
- learning rate: 0.001은 잘 되지 않는다. 0.00025가 잘 된다.
- image crop: Pong과 동일하게 불필요한 부분은 잘라내는게 좋다.
- reward discount: dead가 발생하면 reward -1을 부여한다. reward discount할 때, Pong은 reward가 발생하는 시점을 기준으로 reward reset.  Breakout에서도 dead를 기준으로 실질적인 done처리를 해야한다. 
- Neural Netowk 크기: convolution layer를 2단으로 하는 것보다 3단으로 하는 것이 좋다.
- No-Operation: Breakout은 게임은 공이 코너쪽으로 가는 패튼으로 시작한다. 같은 패튼의 data가 많이 들어가는 것을 방지하기 위해, random하게 최대 30번의 action을 1(`FIRE')로 고정한다. dead상태일 때 reward를 -1로 한다면, No-Operation을 설정하지 않아도 효과가 있지만, overfitting을 방지하기 위해서는 해주는 것이 좋다. dead 상태에서 시작할 때는 최대 10번 No-Operation.
- 위 표에 있는 것처럼, 20회 평균 점수는 419점이다. 위의 play하는 gif는 특히 잘 된 것을 하나 뽑았다(750점).
--------------------------------------
### Advantage Actor Critic(A2C)
A2C - Advantage Actor Critic - BreakoutDeterministic-v4 - reward(792)
<p align="center"><img src="gif/A2C.gif" />  </p>

- [Code](https://github.com/hccho2/RL-GYM/blob/master/08_7_a2c_breakout.py)
- [Pretrained Model](https://github.com/hccho2/RL-GYM/tree/master/breakout-a2c)
- Actior-Critic 모델은 MC방식과 TD방식으로 구현할 수 있는데, 구현된 코드는 MC방식이다.
  * MC방식: episode를 done이 될때까지 생성한 후, train
  * TD방식: episode의 각 step마다 train
- Actor-Critic은 Vanilla Policy Gradient(REINFORCE)에 비해 train 속도가 훨씬 빠르다.

--------------------------------------
### DQN
- [Code](https://github.com/hccho2/RL-GYM/blob/master/08_6_dqn_breakout.py)
- [Pretrained Model](https://github.com/hccho2/RL-GYM/tree/master/breakout-dqn)
- env가 return하는 done이 아닌, life가 줄어드는 dead를 기준으로 DONE 처리를 해야 한다.
- epsilon-greedy, No-Operation을 적용하기 때문에, train할 때 달성되는 reward보다 test reward가 높게 나온다( 20점 -> 320점. 그 이유는 reward clipping. 그리고, train할 때는 exploration으로 인해 life를 더 쉽게 잃기 때문이다. exploration에서도 완전히 random한 action을 취한다. 반면, PG에서는 확률에 기반한 random이다.).
- Huber loss 사용
- replay momory에는 정수값으로 state 정보를 저장해야 메모리 관리가 효율적이다.

--------------------------------------
### Asynchronous Advantage Actor Critic(A3C)
- [Code](https://github.com/hccho2/RL-GYM/blob/master/08_8_a3c_breakout.py): MC, gradient 방식
- [Pretrained Model](https://github.com/hccho2/RL-GYM/tree/master/breakout-a3c)
- gradient 방식과 data방식으로 적용할 수 있다.
  * gradient방식: 각 local agent가 gradient를 계산 한 후, global network의 weight를 update한다.
  * data방식: 각 local agent는 episode를 생성하기만 하고, 생성된 data를 global network가 직접 자신을 train한다. [Code](https://github.com/hccho2/RL-GYM/blob/master/08_9_a3c_breakout_data.py), [Pretrained Model](https://github.com/hccho2/RL-GYM/tree/master/breakout-a3c-data)
- agent의 독립적인 episode 생성을 위해서는 MC방식 보다 n-step이 더 좋다. MC방식은 episode가 done이 될 때까지 진행 후, train을 수행한다.
- A2C와 동일하게, MC방식 또는 n-step 방식이 있을 수 있다.
   * MC방식: episode를 done이 될때까지 생성 후, train.
   * n-step방식: episode의 길이가 n에 도달하면 train을 수행한다. n-step 방식이 MC방식보다 초반에는 빠르게 train되지만, 후반으로 갈 수록 MC가 더 빨리 train된다. [Code](https://github.com/hccho2/RL-GYM/blob/master/08_8_a3c_breakout_max_step.py), [Pretrained Model](https://github.com/hccho2/RL-GYM/tree/master/breakout-a3c-max-step)
- 일반적으로 A3C에서는 thread의 갯수가 많을수록 train 속도가 좋다. Breakout에서는 thread 갯수가 미치는 영향이 작다. train 속도은 빨라지지만, episode 개수 대비로 보면 많은 차이가 없다.

--------------------------------------
### DDPG
DDPG - Pendulum-v0 - 1000 step reward: -116.21 (gif가 멈추어 있는 것처럼 보이지만, 균형 상태를 이루고 있는 것임)
<p align="center"><img src="gif/DDPG.gif" />  </p>

- DDPG는 continous action을 다룰 수 있는 모델.
- gym 게임 중, continous action 환경인 Pendulum-v0에 적용
- [Code](https://github.com/hccho2/RL-GYM/blob/master/08_10_ddpg_pendulum.py)
- [Pretrained Model](https://github.com/hccho2/RL-GYM/tree/master/ddpg-model)

--------------------------------------
--------------------------------------
### OpenAI GYM Tips
```
# force=True --> 디렉토리에 남아 있던 파일을 지우고 mp4파일을 생성한다.
# video_callable=lambda count: count % 1 == 0 ---> 모든 episode를 mp4로 만든다.
env = gym.wrappers.Monitor(env, './movie/', force=True,video_callable=lambda count: count % 1 == 0)
```


---------------------------------
Reference
- <https://github.com/hunkim/ReinforcementZeroToAll>{:target="_blank"}
- [직접 정리한 자료](https://drive.google.com/open?id=16olGwVvk_smtgopmuUtouOf1ad1RGpIf "title" target="_blank")
- [ACER blog](https://hccho2.github.io/2020/03/27/ACER-analysis.html target="_blank)
