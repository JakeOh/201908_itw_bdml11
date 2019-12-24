"""
MNIST 숫자 손글씨 데이터 신경망 구현
"""
import pickle

from dataset.mnist import load_mnist


def init_network():
    """가중치 행렬들(W1, W2, W3, b1, b2, b3)을 생성"""
    # 교재의 저자가 만든 가중치 행렬(sample_weight.pkl)을 읽어 옴.
    with open('sample_weight.pkl', mode='rb') as file:
        network = pickle.load(file)
    print(network.keys())
    # W1, W2, W3, b1, b2, b3 shape 확인
    return network


def predict():
    """신경망에서 사용되는 가중치 행렬들과 테스트 데이터를 파라미터로 전달받아서,
    테스트 데이터의 예측값(배열)을 리턴."""
    pass


def accuracy():
    """테스트 데이터 레이블과 테스트 데이터 예측값을 파라미터로 전달받아서,
    정확도(accuracy) = (정답 개수)/(테스트 데이터 개수) 를 리턴."""
    pass


if __name__ == '__main__':
    network = init_network()
    (X_train, y_train), (X_test, y_test) = load_mnist(파라미터들 설정)
    y_pred = predict(network, X_test)
    acc = accuracy(y_test, y_pred)








