import os
import sys
import re
import csv
import matplotlib.pyplot as plt
from wordcloud import WordCloud

tickets = open('Tickets.csv')
tickets = csv.reader(tickets)
tickets = list(tickets)
ticket_onderwerp = [ticket[13] for ticket in tickets]
ticket_onderwerp = [x for x in ticket_onderwerp if str(x) != 'NULL']
for i in range(len(ticket_onderwerp)):
	ticket_onderwerp[i] = re.sub('\n.*', '', ticket_onderwerp[i], flags=re.DOTALL)
	ticket_onderwerp[i] = re.sub(',', ' ', ticket_onderwerp[i])
ticket_onderwerp = filter(None, ticket_onderwerp)

ticket_onderwerp = '\n'.join(ticket_onderwerp)
print(ticket_onderwerp)

wordcloud = WordCloud(width = 1000, height = 500).generate(ticket_onderwerp)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("wordcloud.png", bbox_inches='tight')
plt.close()
