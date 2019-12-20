import graphviz
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz

# 데이터 준비
iris = load_iris()
X = iris.data  # 데이터(샘플 2차원 리스트(행렬))
y = iris.target  # 타겟(품종)

# 의사 결정 나무(알고리즘 구현) 객체 생성
decision_tree = DecisionTreeClassifier()
# 데이터를 모델에 fitting, 학습
decision_tree.fit(X, y)
# 예측 - 과제??

# 의사 결정 나무 텍스트 출력
text_result = export_text(decision_tree, iris.feature_names)
print(text_result)

# 의사 결정 나무 그래프 출력
graph_data = export_graphviz(decision_tree,
                             feature_names=iris.feature_names,
                             class_names=iris.target_names,
                             filled=True,
                             special_characters=True)  # 그래프를 그릴 수 있는 데이터
graph = graphviz.Source(graph_data)  # 그래프 객체
graph.render('iris')  # 그래프 객체를 파일(pdf)로 작성




