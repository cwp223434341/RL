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
        # 필수: action_space, observation_space 정의
        # 예시: 각각 Discrete, Box 형태의 action_space와 observation_space 정의
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,2), dtype=np.float32)

        self.state = None # state를 관리할 instance 변수

    def _get_obs(self):
        # TO-DO: self.state를 통해 obs 변수 생성
        obs = self.state
        return obs
    def _get_info(self):
        # info 변수 사용 시 변경 진행
        info = {}
        return info

    def step(self, action):
        # To-do: 현재 state에 대해 action이 들어왔을 때 transition을 표현하여 다음 observation, reward, 종료 여부 등을 return

        # 기본 정의
        n_obs = self._get_obs()
        info = self._get_info()
        truncated = False # 필요할 시 수정

        return n_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TO-DO: 각 episode에서 reset할 내용 있으면 작성
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