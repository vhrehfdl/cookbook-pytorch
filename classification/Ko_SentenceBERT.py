import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
from PyKomoran import *
from sentence_transformers import SentenceTransformer, util
import numpy as np


embedder = SentenceTransformer('distiluse-base-multilingual-cased')

train_corpus = ['주문취소에 대한 카드승인취소가 아직 안되어 있어서요 언제쯤 승인 취소가 되는지요.',
          '주문취소해주세요.',
          '위 주문번호로 두건을 구매했고 배송료도 각각 지불했는데 같은 상자에 담겨왔네요   배송비 두건중 한건은 환불해 주셔야 할것같습니다.',
          '묶음배송되었습니다. 배송비 각각 냈는데요 1회 환불해주세요.',
          '주문번호 입니다.  마구리 25지름 2개 세트중에 1개가 조여지지않고 흘러내려요. 교환물품 준비해놔야되나요.',
          '구매결제건신용카드 전표 발급 및 출력',
          '송장 번호가 등록이 되어있는데 아직도 배송 정보가 없으며 배송 실시가 안됨.',
          '가입되어 있는 아이디 탈퇴처리 해주세요.',
          '판매자가 택배정보를 등록했다고 나오는데 아직까지 택배확인이 안됩니다.']

corpus_embeddings = embedder.encode(train_corpus)

# Query sentences:
test_corpus = ['송장번호만 업데이트 되고  실제로 제품이 배송되지는 않고 있어요     언제 받을 수 있나요   빨리 보내주세요',
           '운송장은 나왔는데 배송조회도 안되고 그때문에 취소도 안되고 있습니다 어찌 된 영문인지 판매했던 페이지는 들어갈수도 없고 판매자랑은 연락도 안되고',
           '택배확인이 안되요.']

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 5
for query in test_corpus:
    query_embedding = embedder.encode(query)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()

    #We use np.argpartition, to only partially sort the top_k results
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx in top_results[0:top_k]:
        print(train_corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))