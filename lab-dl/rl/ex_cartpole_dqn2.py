import gym
import numpy as np
from collections import deque
from tensorflow import keras


class QNetwork:
    def __init__(self, state_size, action_size):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(16, activation='relu', input_dim=state_size))
        self.model.add(keras.layers.Dense(16, activation='relu'))
        self.model.add(keras.layers.Dense(16, activation='relu'))
        self.model.add(keras.layers.Dense(action_size, activation='linear'))
        self.model.compile(loss=keras.losses.Huber(),
                           optimizer=keras.optimizers.Adam(lr=0.0001))


class Memory:
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)

    def __len__(self):  # 재현 메모리 사이즈
        return len(self.buffer)

    def add(self, exprience):  # 경험 추가
        self.buffer.append(exprience)

    def sample(self, batch_size):  # 배치 사이즈 만큼 경험을 랜덤하게 취득
        idx = np.random.choice(range(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[i] for i in idx]


if __name__ == '__main__':
    n_episodes = 500  # 학습할 에피소드 수
    max_steps = 200  # 1 에피소드에서 최대 스텝 수
    gamma = 0.99  # discount rate
    warmup = 10  # 초기화할 때 조작하지 않는 스텝 수
    e_start = 1.0  # epsilon 시작값
    e_stop = 0.01  # epsilon 최종값
    e_decay_rate = 0.001  # epsilon decay rate
    memory_size = 10_000  # 재현 메모리 사이즈
    batch_size = 32  # 배치 사이즈

    env = gym.make('CartPole-v1')  # 환경 생성
    state_size = env.observation_space.shape[0]  # 상태 수
    action_size = env.action_space.n   # 행동 수

    online_dqn = QNetwork(state_size, action_size)  # Online DQN 생성
    target_dqn = QNetwork(state_size, action_size)  # Target DQN 생성
    memory = Memory(memory_size)  # 재현 메모리 생성

    # 학습 개시
    # 환경 초기화
    state = env.reset()  # state: [x, v, theta, w]
    state = state.reshape((1, -1))  # np.reshape(state, [1, state_size])

    # 에피소드 수 만큼 에피소드 반복
    total_step = 0  # 총 스텝 수
    success_count = 0  # 성공 수
    for episode in range(1, n_episodes + 1):
        step = 0  # 스텝 수

        # Target DQN 갱신
        target_dqn.model.set_weights(online_dqn.model.get_weights())

        # 1 에피소드 루프
        for _ in range(1, max_steps + 1):
            step += 1
            total_step += 1

            # ε를 감소시킴
            epsilon = e_stop + (e_start - e_stop) * np.exp(-e_decay_rate * total_step)

            if epsilon > np.random.rand():  # 랜덤하게 생동 선택
                action = env.action_space.sample()
            else:  # 행동 가치 함수에 따른 행동 선택
                action = np.argmax(online_dqn.model.predict(state)[0])

            # 행동에 맞추어 상태와 보상을 얻음
            next_state, _, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            if done:  # 에피소드 완료 시
                # 보상 지정
                if step >= 190:
                    success_count += 1
                    reward = 1
                else:
                    success_count = 0
                    reward = 0
                # 다음 상태에 상태 없음을 대입
                next_state = np.zeros(state.shape)
                # 경험 추가
                if step > warmup:
                    memory.add((state, action, reward, next_state))

            else:  # 에피소드 미완료 시
                reward = 0  # 보상 지정

                # 경험 추가
                if step > warmup:
                    memory.add((state, action, reward, next_state))

                # 상태에 다음 상태 대입
                state = next_state

            # 행동 평가 함수 갱신
            if len(memory) >= batch_size:
                # 신경망의 입력과 출력 준비
                inputs = np.zeros((batch_size, 4))  # 입력(상태)
                targets = np.zeros((batch_size, 2))  # 출력(행동별 가치)

                # 배치 사이즈 만큼 경험을 랜덤하게 선택
                minibatch = memory.sample(batch_size)

                # 뉴럴 네트워크 입력과 출력 생성
                for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
                    # 입력 상태 지정
                    inputs[i] = state_b

                    # 선택한 행동의 가치 계산
                    if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                        target = reward_b + gamma * np.amax(target_dqn.model.predict(next_state_b)[0])
                    else:
                        target = reward_b

                    # 출력에 행동 별 가치를 지정
                    targets[i] = online_dqn.model.predict(state_b)
                    targets[i][action_b] = target  # 선택한 행동의 가치

                # 행동 가치 함수 갱신
                online_dqn.model.fit(inputs, targets, epochs=1, verbose=0)

            if done:  # 에피소드 완료 시 에피소드 루프 종료
                break

        # 에피소드 완료 시 로그 표시
        print('에피소드: {}, 스텝 수: {}, epsilon: {:.3f}'.format(episode, step, epsilon))

        # 5회 연송 성공으로 학습 완료
        if success_count >= 5:
            break

        # 환경 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])
