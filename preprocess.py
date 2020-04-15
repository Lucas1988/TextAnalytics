import pandas as pd
import numpy as np
import sklearn
import seaborn as sn
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from itertools import groupby
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

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
print(training_data)
#print(training_data.dtypes)
#print(list(joined_dataframe_loaded.columns.values))
labels = joined_dataframe_loaded['Labels']

# Running classifier (multi-layer perceptron)
print('Running classifier...')
classifier = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(400,), random_state=1000, max_iter=200000)

'''
# Split data in train and test set
X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.2, random_state=1000)

classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print('Accuracy:', score)

# Show list of predictions
predictions = classifier.predict(X_test)
print(predictions)

'''

# Perform cross validation and show result
#print(sorted(sklearn.metrics.SCORERS.keys()))
cv_results = cross_validate(classifier, training_data, labels, cv=5, scoring='accuracy')
print(cv_results)
print('Average accuracy: ', np.mean(cv_results['test_score']))

# Show cross validation predictions
y_pred = cross_val_predict(classifier, training_data, labels, cv=5)
print(y_pred)

#for i in range(len(labels)):
#	print('Prediction:', y_pred[i], 'Label:', labels[i])

# Show frequencies for all the predictions
print('Frequency counts for ground truth')
frequencies_ground_truth = [[key, len(list(group))] for key, group in groupby(sorted(labels))]
print(frequencies_ground_truth)
print('Frequency counts for predictions')
frequencies_predictions = [[key, len(list(group))] for key, group in groupby(sorted(y_pred))]
print(frequencies_predictions)

# Display confusion matrix of cross validation results
confusion_matrix = confusion_matrix(labels, y_pred, normalize='all')
#print(confusion_matrix)
confusion_matrix_df = pd.DataFrame(confusion_matrix)
sn.set(font_scale=1)
sn.heatmap(confusion_matrix_df)
plt.show()
