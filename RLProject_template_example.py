# 참고 문헌
# https://gymnasium.farama.org/introduction/create_custom_env/
########################################################
# 1. 기본 Package 불러오기
import stable_baselines3
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
########################################################

# 추가 package 사용 시 불러올 위치 (필요 시 사용하세요.)

########################################################
# 2. 사용자 정의 환경 만들 위치
# 주의사항: class 이름을 변경하지 마세요.
class CustomizedEnv(gym.Env):
    def __init__(self):
        MAP = [
            [' ', ' ', 'D', ' ', ' '],
            [' ', 'S', ' ', 'S', ' '],
            [' ', ' ', ' ', ' ', ' '],
            [' ', 'S', ' ', 'S', ' '],
        ]  # D는 물건을 가져다 놓을 depot, S는 물류가 들어오는 Station
        action_name = {0: "LEFT", 1: "UP", 2: "RIGHT", 3: "DOWN"}

        self.map = np.array(MAP)
        self.lot_max = 3
        self.height, self.width = self.map.shape
        self.depot = None
        self.stations = []

        for c in range(self.width):
            for r in range(self.height):
                if self.map[r, c] == 'D':
                    self.depot = (r, c)
                elif self.map[r, c] == 'S':
                    self.stations.append((r, c))

        self.arrvial_rate = 0.06 # per time unit
        self.c_loss = 0.1 # loss cost per lot
        self.reward_per_success = 10.0

        # 필수: action_space, observation_space 정의
        self.action_space = gym.spaces.Discrete(4) # LEFT, UP, RIGHT, DOWN
        self.observation_space = gym.spaces.MultiDiscrete([self.height, self.width, 2] + [self.lot_max+1]*len(self.stations))
        # location y, x, load or not, lot per stations

        self.state = None # state를 관리할 instance 변수

    def arrival(self):
        """
        :return number of newly arrived lots per station
        Assume poisson distribution with arrival rate
        """
        return self.np_random.poisson(self.arrvial_rate, len(self.stations))

    def get_cost(self, lot_per_stations):
        """
        :return holding cost
        Compute cost given the current number of lots
        Assume coefficient*x^2
        """
        cost = 0.0
        coeff = 0.01
        for x in lot_per_stations:
            cost += coeff*(x**2)
        return cost

    def _get_obs(self):
        # TO-DO: self.state를 통해 obs 변수 생성
        obs = np.array(self.state)
        return obs
    def _get_info(self):
        # info 변수 사용 시 변경 진행
        info = {}
        return info

    def step(self, action):
        # To-do: 현재 state에 대해 action이 들어왔을 때 transition을 표현하여 다음 observation, reward, 종료 여부 등을 return
        row, col, load, *lot_per_stations = self.state
        prev_row, prev_col = row, col
        reward = -self.get_cost(lot_per_stations)
        # print("Holding costs: ", reward)
        terminated = False
        info = {}
        truncated = False # 필요할 시 수정

        # New lot arrival
        arrv = self.arrival()
        # print("New arrivals: ", arrv)
        lot_per_stations = np.array(lot_per_stations) + arrv

        # Move
        if action == 0: # LEFT
            col -= 1
        elif action == 1: # UP
            row -= 1
        elif action == 2: # RIGHT
            col += 1
        elif action == 3: # DOWN
            row += 1
        else:
            raise Exception('bad action {}'.format(action))

        c_loc = (row, col)
        blocked = False
        if col < 0 or col >= self.width or row < 0 or row >= self.height: # out of bounds, cannot move
            blocked = True
        elif c_loc in self.stations:
            if lot_per_stations[self.stations.index(c_loc)] > 0:
                if load: # blocked at a station
                    blocked = True
                else: # Take a lot
                    load = 1
                    lot_per_stations[self.stations.index(c_loc)] -= 1
        elif c_loc == self.depot: # into a goal cell
            if load:
                load = 0
                reward += self.reward_per_success
                # print("Success move")
                info['success'] = True

        # Set max lot_per_stations
        for i in range(len(lot_per_stations)):
            if lot_per_stations[i] > self.lot_max:
                reward -= (lot_per_stations[i] - self.lot_max) * self.c_loss # Compute loss cost
                lot_per_stations[i] = self.lot_max
        # lot_per_stations = [(self.lot_max if x > self.lot_max else x) for x in lot_per_stations]

        if blocked: # cannot move
            self.state = (prev_row, prev_col, load,) + tuple(lot_per_stations)
            return self._get_obs(), reward, terminated, truncated, info
        else:
            self.state = c_loc + (load,) + tuple(lot_per_stations)
            return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TO-DO: 각 episode에서 reset할 내용 있으면 작성
        self.state = self.depot + (0,) * (len(self.stations)+1)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # 필요 시 추가 작성하세요.
    # def close(self):
    #     super().close()
    #
    # def render(self):
    #     super().render()
########################################################
# [변경하지 마세요!] 정의된 CustomizedEnv 확인 함수
def checkEnv(test_env):
    print("Action space: ", test_env.action_space)
    print("Observation space: ", test_env.observation_space)
    print("=" * 50)
    print("Random agent")
    obs = test_env.reset()
    for i in range(500): # 500 step 동안 확인
        dones = False
        action = test_env.action_space.sample()
        n_obs, reward, done, truncated, info = test_env.step(action)
        print(f"Current observation: {obs}, Current action: {action}, Reward: {reward}, Next observation: {n_obs}, Terminated: {done}")
        obs = n_obs
    print("="*50)
########################################################
# 3. 사용자 정의 환경 정의
env = CustomizedEnv() # 초기화 함수에 따라 환경 생성이 달라지면 적절하게 변경하세요.
########################################################
# 4. 사용자 정의 환경 테스트 # 환경 검토 이후 학습 시에는 주석 처리해도 됩니다.
try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
########################################################
# 5. 강화학습 알고리즘(에이전트) 생성
env.reset() # 초기화 재진행
# 기본 설정 PPO - 'MlpPolicy'
ppo_model = PPO('MlpPolicy', env, verbose=1)
# ppo_model = PPO('MultiInputPolicy', env, verbose=1) # observation_space가 Dict인 경우
########################################################
# 6. 랜덤 에이전트에 대한 성능 평가 # 학습 시에는 주석 처리해도 됩니다.
# A. infinite horizon model인 경우
test_steps = 5000
obs, info = env.reset()
cumul_reward = 0
for i in range(5000):
    action = env.action_space.sample()
    n_obs, reward, done, truncated, info = env.step(action)
    cumul_reward += reward
    obs = n_obs
print("[랜덤 에이전트 성능] 총 보상합:", cumul_reward)
# # B. episodic task인 경우
# mean_reward, std_reward = evaluate_policy(ppo_model, env, n_eval_episodes=5)
# print("[랜덤 에이전트 성능] 에피소드 당 평균 보상:", mean_reward, "에피소드 당 보상의 표준편차: ", std_reward)
########################################################
# 7. 학습
ppo_model.learn(total_timesteps=int(1.0e5))
########################################################
# 8. 학습 결과 평가
# A. infinite horizon model인 경우
test_steps = 5000
obs, info = env.reset()
cumul_reward = 0
for i in range(5000):
    action, _states = ppo_model.predict(obs)
    n_obs, reward, done, truncated, info = env.step(action)
    cumul_reward += reward
    obs = n_obs
print("[학습 에이전트 성능] 총 보상합:", cumul_reward)
# # B. episodic task인 경우
# mean_reward, std_reward = evaluate_policy(ppo_model, env, n_eval_episodes=5)
# print("[학습 에이전트 성능] 에피소드 당 평균 보상:", mean_reward, "에피소드 당 보상의 표준편차: ", std_reward)
########################################################