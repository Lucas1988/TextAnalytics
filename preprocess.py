import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from itertools import groupby

joined_dataframe_loaded = pd.read_pickle('joined_dataframe.pkl')
labels_supervised = joined_dataframe_loaded['Labels']
note_callcenter_agent = joined_dataframe_loaded['Notitie Call Center Agent']

# Convert sentences to sentence_embeddings
print('Creating embeddings...')
note_callcenter_agent = list(map(str, note_callcenter_agent))
model = SentenceTransformer('distiluse-base-multilingual-cased')
sentence_embeddings = model.encode(note_callcenter_agent)
sentence_embeddings = np.array(sentence_embeddings)

# Join sentence embeddings and customer id in 1 dataframe
training_data = pd.DataFrame(data=sentence_embeddings[0:,0:])
training_data['Ticket_Doorgestuurd_Naar'] = joined_dataframe_loaded['Ticket_Doorgestuurd_Naar'].astype('category').cat.codes
#ohe_2 = pd.get_dummies(joined_dataframe_loaded['Turfje 3'])
#training_data = pd.concat([training_data,ohe_2], axis=1)
#training_data['Aangemaakt door'] = joined_dataframe_loaded['Aangemaakt door'].astype('category').cat.codes
print(training_data)
print(training_data.dtypes)
print(list(joined_dataframe_loaded.columns.values))
#print(list(training_data['Pakketten']))
labels = joined_dataframe_loaded['Labels']

# Split data in train and test set
X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.05, random_state=1000)

# Running classifier (multi-layer perceptron)
print('Running classifier...')
classifier = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(400,), random_state=1000, max_iter=200000)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print('Accuracy:', score)

predictions = classifier.predict(X_test)
print(predictions)

frequencies = [[key, len(list(group))] for key, group in groupby(sorted(predictions))]
print(frequencies)

