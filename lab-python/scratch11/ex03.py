from collections import Counter

import numpy as np


def train_test_split(X, y, test_size):
    """
    X: numpy.ndarray. n x m
    y: numpy.ndarray. 원소의 개수가 n개인 1차원 배열
    len(X) == len(y) 가정.
    test_size: 0.0 ~ 1.0
    """
    length = len(X)
    # 인덱스를 저장하는 배열
    indices = np.array([i for i in range(length)])
    print('shuffle 전:', indices)
    # 인덱스를 임의로 섞음
    np.random.shuffle(indices)
    print('shuffle 후:', indices)
    # Train set의 개수
    cut = int(length * (1 - test_size))  # 소수점 버림
    X_train = X[indices[:cut]]  # Train set points
    y_train = y[indices[:cut]]  # Train set labels
    X_test = X[indices[cut:]]  # Test set points
    y_test = y[indices[cut:]]  # Test set labels
    return X_train, X_test, y_train, y_test


class MyScaler:
    def fit(self, X):
        """X의 각 특성(컬럼)들의 평균과 표준 편차를 저장"""
        # 컬럼별로 평균을 계산해서 저장
        self.feature_means = np.mean(X, axis=0)  # axis=0: 컬럼별 계산, axis=1: row별 계산
        # 컬럼별로 표준 편차를 계산해서 저장
        self.feature_stds = np.std(X, axis=0)
        print(self.feature_means)
        print(self.feature_stds)

    def transform(self, X):
        # X와 같은 크기를 갖는 배열을 생성.
        # 행(row)과 열(column)의 개수가 같은 배열을 생성.
        dim = X.shape
        transformed = np.empty(dim)
        for row in range(dim[0]):  # row 개수만큼 반복
            for col in range(dim[1]):  # column 개수만큼 반복
                # x_new = (x - mean) / std
                transformed[row, col] = (X[row, col] -
                                         self.feature_means[col]) / self.feature_stds[col]
        return transformed


class MyKnnClassifier:
    def __init__(self, n_neighbors=5):  # 객체 생성
        """최근접 이웃으로 선택할 개수를 저장함."""
        self.k = n_neighbors

    def fit(self, X_train, y_label):  # 모델 훈련
        """레이블을 가지고 있는 데이터(point)를 저장함."""
        self.points = X_train
        self.labels = y_label

    def predict(self, X_test):  # 예측
        """테스트 세트 X_test의 각 점들마다,
        1) 학습 세트에 있는 모든 점들과의 거리를 계산.
        2) 계산된 거리들 중에서 가장 짧은 거리 k개를 선택.
        3) k개 선택된 레이블들 중에서 가장 많은 것(다수결)을 예측값으로 함."""
        predicts = []  # 예측값들을 저장할 리스트
        for test_pt in X_test:  # 테스트 세트에 있는 점들의 개수만큼 반복
            # 학습 세트의 점들과의 거리를 계산
            distances = self.distance(self.points, test_pt)
            print(test_pt)
            print(distances)
            # 다수결로 예측값 결정
            winner = self.majority_vote(distances)
            # 예측값을 리스트에 저장
            predicts.append(winner)

        return np.array(predicts)  # 예측값들의 배열을 리턴

    def distance(self, X, y):
        """점(벡터) y와 점(벡터)들 X 사이의 거리들의 배열을 리턴 """
        return np.sqrt(np.sum((X - y) ** 2, axis=1))

    def majority_vote(self, distances):
        # 거리 순서로 정렬된 인덱스를 찾는다.
        indices_by_distance = np.argsort(distances)
        print(indices_by_distance)
        # 가장 가까운 k개 이웃의 레이블을 찾는다.
        k_nearest_neighbor = []
        # for i in range(self.k):
        #     idx = indices_by_distance[i]
        #     k_nearest_neighbor.append(self.labels[idx])
        for i in indices_by_distance[0:self.k]:
            k_nearest_neighbor.append(self.labels[i])
        print(k_nearest_neighbor)
        # 가장 많은 득표를 얻은 레이블을 찾는다.
        vote_counts = Counter(k_nearest_neighbor)
        print(vote_counts)
        # most_common(n): 가장 많은 빈도수 순위 n까지의 리스트를 리턴
        # 빈도수가 동률일 수도 있기 때문에 리스트를 리턴
        print(vote_counts.most_common(1))
        print(vote_counts.most_common(1)[0])
        winner, winner_count = vote_counts.most_common(1)[0]
        return winner


if __name__ == '__main__':
    np.random.seed(1210)
    X = np.random.randint(10, size=(10, 2))  # Points
    print(X)
    y = np.array(['a', 'b', 'a', 'b', 'a'] * 2)  # labels
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)

    scaler = MyScaler()  # 객체 생성
    scaler.fit(X_train)  # 객체가 가지고 있는 메소드 호출
    X_train_scaled = scaler.transform(X_train)
    print(X_train_scaled)
    X_test_scaled = scaler.transform(X_test)
    print(X_test_scaled)

    knn = MyKnnClassifier(n_neighbors=3)  # k-NN 분류기 객체 생성 - 생성자 호출
    print('k =', knn.k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    print(y_pred)
    print(y_test == y_pred)  # 정답과 예측값 비교

