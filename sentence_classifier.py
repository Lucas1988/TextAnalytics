import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier

model = SentenceTransformer('distiluse-base-multilingual-cased')
sentences = pd.read_excel('sentences_supervised.xlsx', sheet_name='Sheet1')
sentences_set = sentences['Sentences']
sentences_set = [re.sub('[\n\r]', ' ', sentence) for sentence in sentences_set]	

# Convert sentences to sentence_embeddings
print('Creating embeddings...')
sentence_embeddings = model.encode(sentences_set)
labels = sentences['Labels']

# Split data in train and test set
embeddings_train, embeddings_test, y_train, y_test = train_test_split(
	sentence_embeddings, labels, test_size=0.2, random_state=1000)

# Running classifier (logistic regression)
print('Running classifier...')
classifier = LogisticRegression()
classifier.fit(embeddings_train, y_train)
score = classifier.score(embeddings_test, y_test)
#predictions = classifier.predict(embeddings_test)
print('Accuracy:', score)
#print(predictions)

# Running classifier (SVM)
print('Running classifier...')
classifier = svm.SVC()
classifier.fit(embeddings_train, y_train)
score = classifier.score(embeddings_test, y_test)
print('Accuracy:', score)

# Running classifier (multi-layer perceptron)
print('Running classifier...')
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
	hidden_layer_sizes=(5000, 50), random_state=1000)
classifier.fit(embeddings_train, y_train)
score = classifier.score(embeddings_test, y_test)
print('Accuracy:', score)

