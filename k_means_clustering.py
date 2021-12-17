import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
import re
from operator import itemgetter

model = SentenceTransformer('distiluse-base-multilingual-cased')
sentences = pd.read_excel('sentences.xlsx', sheet_name='sentences')
sentences = sentences['Oorzaak_Storing_Monteur']
sentences = [x for x in sentences if str(x) != 'nan']
sentences = [re.sub('[\n\r]', ' ', sentence) for sentence in sentences]	
sentences = [re.sub('(.*)Source:.*', '\1', sentence) for sentence in sentences]	

# Convert sentences to sentence_embeddings
print('Creating embeddings...')
sentence_embeddings = model.encode(sentences)

# Train the k-means model
print('Training k-means...')
classifier = MiniBatchKMeans(n_clusters=15, random_state=1, max_iter=100)
classifier.fit(sentence_embeddings)

# Display the results
print('Predicting clusters...')
predictions = classifier.predict(sentence_embeddings)

results = []
for i in range(len(sentences)):
	results.append([sentences[i], predictions[i]])

sorted_list = sorted(results, key=itemgetter(1))
for item in sorted_list:
	print(item)