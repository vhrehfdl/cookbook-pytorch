### Overall Architecture

해당 Cookbook의 함수 구성은 다음과 같이 되어 있습니다.
- load_data : csv 형태의 데이터를 불러온다.   
train 데이터에서 split해 validation 데이터를 생성한다.

- pre_processing : 데이터 전처리하는 부분.  
vocab을 생성하거나 batch size로 데이터를 나누는 작업을 수행한다.

- build_model : 모델을 생성.

- train : 생성한 모델을 사용해 학습한다.

- evalutate : 학습한 모델을 사용한 평가.



### Category

Classification task에서는 데이터의 클래스가 2개인 binary class와 2개 이상인 multi class로 분류했습니다.
- self_trained : Pre-trained 임베딩을 사용하지 않고 자체 임베딩을 훈련해 분류하는 모델.  
load_data, evalutae 부분을 상황에 맞게 변경하면 됨.

- pre_trained : torchtext에서 제공하는 word2vec, glove, fasttext 사전 훈련된 파일을 사용하는 모델.  
load_data, evalutae 부분을 상황에 맞게 변경하면 됨.

- ELMo : Allen NLP에서 제공하는 ELMo를 사용한 모델.  
config, load_data, evaluate 부분만 상황에 맞게 변경하면 됨.

- BERT : Allen NLP에서 제공하는 BERT를 사용하는 모델.  
config, load_data, evaluate 부분만 상황에 맞게 변경하면 됨.

