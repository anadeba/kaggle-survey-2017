import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt


schema = pd.read_csv("X:\\Hackathon\\kaggle-survey-2017\\schema.csv")
schema_freeform = schema[schema['Column'].str.contains("FreeForm")]

schema_freeform_asked_all = schema_freeform[schema_freeform['Asked'] == 'All']
schema_freeform_asked_codingworker = schema_freeform[schema_freeform['Asked'] == 'CodingWorker']
schema_freeform_asked_codingworkernc = schema_freeform[schema_freeform['Asked'] == 'CodingWorker-NC']
schema_freeform_asked_learners = schema_freeform[schema_freeform['Asked'] == 'Learners']
schema_freeform_asked_nonswitcher = schema_freeform[schema_freeform['Asked'] == 'Non-switcher']
schema_freeform_asked_onlinelearners = schema_freeform[schema_freeform['Asked'] == 'OnlineLearners']
schema_freeform_asked_worker = schema_freeform[schema_freeform['Asked'].isin(['Worker1','Worker'])]

data_freeform = pd.read_csv("X:\\Hackathon\\kaggle-survey-2017\\freeformResponses.csv", dtype = object)

### List of questions asked to all
schema_freeform_asked_all[['Column','Question']]

### Text Clustering Model

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        #print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


def extract_topics(document, n_topics):
    vect = CountVectorizer(stop_words='english',lowercase= True)
    dtm = vect.fit_transform(document.dropna())
    #kmeans = KMeans(n_clusters= number_of_clusters, random_state=0).fit(dtm)
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(dtm)
    print_top_words(lda, vect.get_feature_names(), 3)
    #return (kmeans.labels_)


def calculate_response(col,lst):
    pct = []
    x = pd.DataFrame(data_freeform[col].dropna().str.lower())
    for word in lst:
        total_response = len(x[col])
        p = (len(x[x[col].str.contains(word)])/total_response)*100
        pct.append(p)
    return pd.DataFrame({'Percentage':pct, col:lst}, columns = [col,'Percentage']), total_response

def draw_chart(df):
    plt.bar(np.arange(len(df[0])), df[0]['Percentage'], align='center', alpha=0.5)
    plt.xticks(np.arange(len(df[0])), df[0][list(df[0])[0]])
    plt.ylabel('Percentage')
    plt.title(list(df[0])[0])
    plt.show()

# extract_topics(data_freeform['GenderFreeForm'], 3)
# calculate_response('MLToolNextYearFreeForm',['keras','tensorflow'])