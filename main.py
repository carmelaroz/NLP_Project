import spacy
import os
import json
import enchant
import string
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from itertools import cycle
import numpy as np
from collections import Counter
from heapq import nlargest



nlp = spacy.load("en_core_web_md")

def average_similarity_score(similarity_scores):
    total = sum(similarity_scores.values())
    average = total / len(similarity_scores)
    return average

def lemmatize_and_count(text):
    # Process the text
    doc = nlp(text)
    
    # Lemmatize each token in the text and collect their lemmas
    lemmas = [token.lemma_ if token.lemma_ != "-PRON-" else token.text for token in doc if token.text.isalpha()]
    
    # Count the frequency of each lemma
    lemma_counts = Counter(lemmas)
    sorted_lemma_counts = dict(sorted(lemma_counts.items(), key=lambda item: item[1], reverse=True))

    return sorted_lemma_counts


def text_summarization(text, num_sentences=3):
    # Tokenize the text into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Create a list of words for each sentence
    sentence_tokens = [nlp(sentence.lower()) for sentence in sentences ]

    # Calculate word frequencies
    word_freq = Counter()
    for sentence in sentence_tokens:
        for token in sentence:
            if not token.is_stop and not token.is_punct:
                word_freq[token.text] += 1
    
    # Calculate sentence scores based on word frequencies
    sentence_scores = {sentence: sum(word_freq[word.text] for word in sentence) / len(sentence)
                       for sentence in sentence_tokens}

    # Select top-ranked sentences for the summary
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    # Join the selected sentences to form the summary
    summary = ' '.join(str(summary_sentences))
    
    return summary

def count_pos_tags(pos_tag_dict):
    # Initialize a dictionary to store counts of POS tags
    pos_tag_counts = {}
    
    # Iterate through the values (POS tags) in the input dictionary
    for pos_tag in pos_tag_dict.values():
        # Increment the count for this POS tag
        if pos_tag in pos_tag_counts:
            pos_tag_counts[pos_tag] += 1
        else:
            pos_tag_counts[pos_tag] = 1
    
     # Sort the dictionary by counts in descending order
    sorted_pos_tag_counts = dict(sorted(pos_tag_counts.items(), key=lambda item: item[1], reverse=True))

    return pos_tag_counts


def count_non_real_words(song_lyrics):
    # Initialize the enchant dictionary
    d = enchant.Dict("en_US")

    # Split the lyrics into words
    words = song_lyrics.split()

    # Initialize a list to store non-real words
    non_real_words = []

    # Remove punctuation from each word and check if it's a real word
    for word in words:
        # Remove punctuation from the word
        clean_word = word.translate(str.maketrans('', '', string.punctuation))
        
        # Check if the cleaned word is a real word
        if clean_word and not d.check(clean_word):
            non_real_words.append(word)

    # Return the count of non-real words and the list of non-real words
    return len(non_real_words), non_real_words

def analyzeSong(song):
    lyrics = song.get_lyrics()
    doc = nlp(lyrics)

    analysis_results = {
        "pos_tags": {},
        "pos_tags_freq":{},
        "word_frequency": {},
        "named_entities": {},
        "dependency_parse": {},
        "text_classification": None,  
        "text_similarity": None,
        "text_similarity_average":{},
        "word_vectors": {},  # Placeholder for word vectors result  # Placeholder for text similarity result
        "text_lemmatization": {},
        "text_summarization": {},
        "non_real_words_freq": {}
    }
    
    # Tokenization, POS tagging, Dependency Parsing
    for token in doc:
        analysis_results["pos_tags"][token.text] = token.pos_
        word_to_search = token.text
        exclude_chars = set(string.punctuation) | {'\n'} | {"," , ":", " ", "'", "(", ")", "/", "[", "]", "{", "}", "'re", "!", "-", "n't", "'s", "watts", "'m", "", "\n\n"}
        if word_to_search in analysis_results["word_frequency"] :
            if word_to_search not in exclude_chars and not word_to_search.startswith("'"):
                analysis_results["word_frequency"][token.text] += 1
        else:
            analysis_results["word_frequency"][token.text] = 1
        analysis_results["word_frequency"] = dict(sorted(analysis_results["word_frequency"].items(), key=lambda item: item[1], reverse=True))
        # Dependency Parsing
        analysis_results["dependency_parse"][token.text] = {
            "dep": token.dep_,
            "head": token.head.text
        }

    analysis_results["pos_tags_freq"] = count_pos_tags(dict(analysis_results["pos_tags"].items()))
    
    # Perform text classification (placeholder: rule-based sentiment analysis)
    sentiment_score = sum(1 for token in doc if token.text.lower() in ["good", "great", "excellent"]) \
                    - sum(1 for token in doc if token.text.lower() in ["bad", "poor", "terrible"])
    
    if sentiment_score > 0:
        analysis_results["text_classification"] = "Positive"
    elif sentiment_score < 0:
        analysis_results["text_classification"] = "Negative"
    else:
        analysis_results["text_classification"] = "Neutral"

    # Calculate text similarity
    similarity_scores = {}
    for other_song in song.get_album().get_songs():
        if other_song.get_name() != song.get_name():
            similarity_score = doc.similarity(nlp(other_song.get_lyrics()))
            similarity_scores[other_song.get_name()] = similarity_score
        analysis_results["text_similarity"] = similarity_scores 

    analysis_results["text_similarity_average"] = average_similarity_score(similarity_scores)

    # Named Entity Recognition (NER)
    for ent in doc.ents:
        analysis_results["named_entities"][ent.text] = ent.label_

    analysis_results["text_lemmatization"] = lemmatize_and_count(lyrics)

    analysis_results["text_summarization"] = text_summarization(lyrics)

    analysis_results["non_real_words_freq"] = count_non_real_words(lyrics)

    return analysis_results

def load_analysis(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    else:
        return -1

def save_analysis(analysis, filename):
    with open(filename, 'w') as file:
        json.dump(analysis, file)

def parse_json_to_dict(json_data):
    return json.loads(json_data)

def album_pos_tags(album):
    """
    Count the POS tags for all songs in the given album.
    Returns a dictionary where keys are POS tags and values are their frequencies.
    """
    pos_tags_freq = {}
    for song in album.get_songs():
        song_pos_tags = song.get_analysis()["pos_tags_freq"]
        for tag, count in song_pos_tags.items():
            pos_tags_freq[tag] = pos_tags_freq.get(tag, 0) + count
    # return pos_tags_freq
    return dict(sorted(pos_tags_freq.items(), key=lambda item: item[1], reverse=True))


def album_text_similarity_average(album):
    songs = album.get_songs()
    num_songs = len(songs)
    total_similarity_score = sum(song.get_analysis()["text_similarity_average"] for song in songs)
    average_similarity_score = total_similarity_score / num_songs
    
    return average_similarity_score

def album_word_freq(album):
    songs = album.get_songs()
    total_word_freq = Counter()
    for song in songs:
        song_word_freq = song.get_analysis()["word_frequency"]
        total_word_freq.update(song_word_freq)
    words = dict(sorted(total_word_freq.items(), key=lambda item: item[1], reverse=True))
    
    return words 

def album_named_entities(album):
    songs = album.get_songs()
    entity_type_freq = Counter()
    for song in songs:
        song_named_entities = song.get_analysis()["named_entities"]
        for entity_type in song_named_entities.values():
            entity_type_freq[entity_type] += 1
    
    return entity_type_freq

def album_dependency_parses(album):
    songs = album.get_songs()
    combined_dependency_parses = {}
    for song in songs:
        song_dependency_parses = song.get_analysis()["dependency_parse"]
        for word, parse_info in song_dependency_parses.items():
            if word in combined_dependency_parses:
                combined_dependency_parses[word].append(parse_info)
            else:
                combined_dependency_parses[word] = [parse_info]
    
    return combined_dependency_parses

def album_text_classification(album):
    """
    Count clssifications of all the songs in the given album.
    Calculates total calssification score and returns a dictionary with the result and 
    """
    songs = album.get_songs()
    classification_counter = Counter()
    
    for song in songs:
        classification = song.get_analysis()["text_classification"]
        if classification == "Positive":
            classification_counter[classification] += 1
        elif classification == "Neutral":
            classification_counter[classification] += 1
        elif classification == "Negative":
            classification_counter[classification] += 1
        
    # Calculate the classification score
    total_songs = len(songs)
    classification_score = classification_counter["Positive"] - classification_counter["Negative"]
    classification_counter = dict(classification_counter)
    
    classification_counter["classification_score"] = classification_score
    return classification_counter

def album_text_lemmatization(album):
    lemma_counts = {}

    songs = album.get_songs()

    for song in songs:
        song_lemmas = song.get_analysis()["text_lemmatization"]
        for lemma, count in song_lemmas.items():
            if lemma in lemma_counts:
                lemma_counts[lemma] += count
            else:
                lemma_counts[lemma] = count

    return dict(sorted(lemma_counts.items(), key=lambda item: item[1], reverse=True))

def album_text_summarization(album, num_sentences=3):
    # Initialize an empty string to store the combined summary
    combined_summary = ""

    songs = album.get_songs()

    for song in songs:

        lyrics = song.get_lyrics()

        # Generate a summary for the current song
        song_summary = text_summarization(lyrics, num_sentences=num_sentences)

        combined_summary += f"\n\nSummary for '{song.get_name()}':\n{song_summary}"

    return combined_summary

def read_lyrics(album):
    lyrics = []
    for song in album.get_songs():
                lyrics.append(song.get_lyrics())
    return lyrics

# Function to preprocess lyrics
def preprocess_lyrics(lyrics):
    df = pd.DataFrame(lyrics, columns=[  "Lyrics"])
    return df

# Function to train topic model
def train_topic_model(lyrics, n_topics=1):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(lyrics["Lyrics"])
    
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(X)
    
    return lda_model, vectorizer

# Function to display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print()

# Function to analyze topic model results
def analyze_topic_model(model, vectorizer, lyrics):
    word_topics = model.components_
    word_names = vectorizer.get_feature_names_out()

    topics = []
    for topic_idx, topic in enumerate(word_topics):
        topic_words = [word_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics.append((topic_idx + 1, topic_words))

    return topics

def album_topic_modeling(album):
    
    michael_jackson_lyrics = read_lyrics(album)

    # Preprocess lyrics
    michael_jackson_df = preprocess_lyrics(michael_jackson_lyrics)

    # Train topic model
    lda_model, vectorizer = train_topic_model(michael_jackson_df)

    # Display topics
    no_top_words = 10
    display_topics(lda_model, vectorizer.get_feature_names_out(), no_top_words)

    # Analyze topic model results
    topics = analyze_topic_model(lda_model, vectorizer, michael_jackson_df)
    return topics

def album_non_real_words_freq(album):

    songs = album.get_songs()
    
    total_word_freq = [0,[]]

    for song in songs:
        song_non_real_word_freq = song.get_analysis()["non_real_words_freq"]
        total_word_freq[0] += song_non_real_word_freq[0]
        total_word_freq[1].append(song_non_real_word_freq[1])
    
    return total_word_freq 

def analyzeAlbum(album):
    analysis_results = {
        "pos_tags_freq": {},
        "word_frequency": {},
        "named_entities": {},
        "dependency_parse": {},
        "text_classification": None,  
        "text_similarity_average":{},
        "text_lemmatization_freq": {},
        "text_summarization": {},
        "topic_modeling": {},
        "non_real_words_freq": {}
    }
    analysis_results["pos_tags_freq"] = album_pos_tags(album)
    analysis_results["word_frequency"] = album_word_freq(album)
    analysis_results["named_entities"] = album_named_entities(album)
    analysis_results["dependency_parse"] = album_dependency_parses(album)
    analysis_results["text_classification"] = album_text_classification(album)
    analysis_results["text_similarity_average"] = album_text_similarity_average(album)
    analysis_results["text_lemmatization_freq"] = album_text_lemmatization(album)
    analysis_results["text_summarization"] = album_text_summarization(album)
    analysis_results["topic_modeling"] = album_topic_modeling(album)
    analysis_results["non_real_words_freq"] = album_non_real_words_freq(album)
    return analysis_results

class Song:
    def __init__(self, name, release_date, lyrics, album):
        self.name = name
        self.release_date = release_date
        self.lyrics = lyrics
        self.album = album
        self.analysis = []

    def get_name(self):
        return self.name

    def get_release_date(self):
        return self.release_date

    def get_other_artists(self):
        return self.other_artists
    
    def get_lyrics(self):
        return self.lyrics
    
    def get_album(self):
        return self.album
    
    def get_analysis(self):
        return self.analysis
    
    def update_analysis(self):
        self.analysis = analyzeSong(self)
        save_analysis(self.get_analysis(), 'songs analysis/'+self.get_name()+'-analysis.json')
        print(self.get_name()+" analysis has been updated.")

class Album:
    def __init__(self, name, release_date):
        self.name = name
        self.release_date = release_date
        self.songs = []
        self.songs = self.create_songs()
        self.analyze_songs()
        self.analysis = []
        self.analyze_album()

    def create_songs(self):
        songs = []
        current_directory = "C:/Users/Asus/Computer Science/Semester E/NLP/spacyProject/Michael_Jackson"
        current_directory += '/'+self.name
        for file_name in os.listdir(current_directory):
            if file_name.lower().endswith('.txt'):
                song_name = os.path.splitext(file_name)[0]
                with open(current_directory + '/' + song_name + '.txt', "r", encoding="utf-8") as file:
                    song_lyrics = file.read()
                songs.append(Song(song_name, self.release_date, song_lyrics, self))
        return songs
    
    def analyze_songs(self):
        for song in self.get_songs():
            res = load_analysis('songs analysis/'+song.get_name()+'-analysis.json')
            if res != -1:
                song.analysis = res
                print("-\""+song.get_name()+"\" analysis has been loaded succesfully")
            else:
                song.analysis = analyzeSong(song)
                save_analysis(song.get_analysis(), 'songs analysis/'+song.get_name()+'-analysis.json')
                print("-\""+song.get_name()+"\" has been analyzed succesfully")
        print()
        print("-All the songs of \"" + self.get_name() + "\" album has been analyzed/loaded succesfully ")
        print()

    def analyze_album(self):
        
        res = load_analysis('albums analysis/'+self.get_name()+'-analysis.json')
        if res != -1:
            self.analysis = res
            print("-\""+self.get_name()+"\" album analysis has been loaded succesfully")
            print()
        else:
            self.analysis = analyzeAlbum(self)
            save_analysis(self.get_analysis(), 'albums analysis/'+self.get_name()+'-analysis.json')
            print("-\""+self.get_name()+"\"  album  has been analyzed succesfully")
            print()

    def get_name(self):
        return self.name

    def get_release_date(self):
        return self.release_date
    
    def get_num_release_date(self):

        year, month, day = self.get_release_date().split('-')
        
        year_int = int(year)
        month_int = int(month)
        day_int = int(day)
       
        release_date_int = year_int * 10000 + month_int * 100 + day_int
        return release_date_int

    def get_songs(self):
        return self.songs
    
    def print_songs(self):
        for song in self.songs:
            print ('-' + song.get_name() + ' - ' + song.get_release_date())
            print("lyrics: "+ song.get_lyrics())
        print()

    def get_analysis(self):
        return self.analysis
    
    def update_analysis(self):
        self.analysis = analyzeAlbum(self)
        save_analysis(self.get_analysis(), 'albums analysis/'+self.get_name()+'-analysis.json')
        print("\""+self.get_name()+"\" album analysis has been updated.")


albums_info = {
    # Michael Jackson's Solo Albums
    "Got to Be There": "1972-01-24",
    "Ben": "1972-08-04",
    "Music & Me": "1973-04-13",
    "Forever, Michael": "1975-01-16",
    "Off the Wall": "1979-08-10",
    "Thriller": "1982-11-30",
    "Bad": "1987-08-31",
    "Dangerous": "1991-11-26",
    "HIStory- Past, Present and Future, Book I": "1995-06-20",
    "Invincible": "2001-10-30",

    # Albums with The Jackson 5
    "Diana Ross Presents The Jackson 5": "1969-12-12",
    "ABC": "1970-05-08",
    "Third Album": "1970-09-08"
}

albums = []
for album_name, release_date in albums_info.items():
    album = Album(album_name, release_date)
    albums.append(album)
print("-All the albums has been analyzed/loaded succesfully ")
print()

class Event:
    def __init__(self, description, date):
        self.description = description
        self.date = date

    def get_description(self):
        return self.description
    
    def get_date(self):
        return self.date
    
    def get_num_date(self):
        year, month, day = self.get_date().split('-')
        
        year_int = int(year)
        month_int = int(month)
        day_int = int(day)
        
        date_int = year_int * 10000 + month_int * 100 + day_int
        return date_int
    
    def print_event(self):
        print (self.get_date()+ ": "+self.get_description())

# Dictionary of Michael Jackson's events with dates formatted as "YYYY-MM-DD"
michael_jackson_events = [
    #  Event("Michael performs with the Jackson 5", "1960-01-01"),
    #  Event("The Jackson 5 signs with Motown Records", "1969-01-01"),
     Event("Release of Michael's first solo album 'Got to Be There'", "1971-01-01"),
     Event("Michael stars in the film 'The Wiz'", "1978-01-01"),
     Event("Release of 'Off the Wall', Michael's breakthrough solo album", "1979-01-01"),
     Event("Release of 'Thriller', the best-selling album of all time", "1982-01-01"),
     Event("Michael suffers burns during the filming of a Pepsi commercial", "1984-01-01"),
     Event("Release of 'Bad', Michael's album", "1987-01-01"),
     Event("Michael purchases the Neverland Ranch", "1988-01-01"),
     Event("Release of 'Dangerous', Michael's album", "1991-01-01"),
     Event("Michael is accused of child sexual abuse for the first time", "1993-01-01"),
     Event("Michael marries Lisa Marie Presley", "1994-01-01"),
     Event("Michael divorces Lisa Marie Presley and marries Debbie Rowe", "1996-01-01"),
     Event("Birth of Michael's first child, Michael Joseph 'Prince' Jackson Jr.", "1997-01-01"),
     Event("Michael divorces Debbie Rowe", "1999-01-01"),
     Event("Release of 'Invincible', Michael's album", "2001-01-01"),
     Event("Birth of Michael's second child, Paris-Michael Katherine Jackson", "2002-01-01"),
     Event("Michael faces child sexual abuse charges; trial begins", "2003-01-01"),
     Event("Michael is acquitted of child sexual abuse charges", "2005-01-01")
]

def compare_dicts(dict_before, dict_after):
    comparison = {}
    for key in dict_before:
        if key in dict_after:
            if dict_before[key] != 0:
                percentage_change = ((dict_after[key] - dict_before[key]) / dict_before[key]) * 100
            else:
                percentage_change = dict_after[key] * 100  
            comparison[key] = {
                "before": dict_before[key],
                "after": dict_after[key],
                "change": dict_after[key] - dict_before[key],  #
                "percentage_change": percentage_change
            }
        else:
            # Key not present in dict_after
            comparison[key] = {
                "before": dict_before[key],
                "after": None,
                "change": None,
                "percentage_change": None
            }
    # Keys present in dict_after but not in dict_before
    for key in dict_after:
        if key not in dict_before:
            comparison[key] = {
                "before": None,
                "after": dict_after[key],
                "change": None,
                "percentage_change": None
            }
    return comparison

def compare_albums_internal(event, albums_before, albums_after):
    comparison_results = {}
    
    for album_before in albums_before:
        for album_after in albums_after:
            # Check if the album analysis exists
            if album_before.get_analysis() and album_after.get_analysis():
                analysis_before = album_before.get_analysis()
                analysis_after = album_after.get_analysis()

                # Compare pos_tags_freq
                comparison_results["pos_tags_freq"] = compare_dicts(analysis_before["pos_tags_freq"], analysis_after["pos_tags_freq"])

                # Compare word_frequency
                comparison_results["word_frequency"] = compare_dicts(analysis_before["word_frequency"], analysis_after["word_frequency"])

                # Compare named_entities
                comparison_results["named_entities"] = compare_dicts(analysis_before["named_entities"], analysis_after["named_entities"])

                # Compare text_classification
                comparison_results["text_classification"] = compare_dicts(analysis_before["text_classification"], analysis_after["text_classification"]),
                       
            else:
                # Analysis data not available for one or both albums
                comparison_results["pos_tags_freq"] = None
                comparison_results["word_frequency"] = None
                comparison_results["named_entities"] = None
                comparison_results["text_classification"] = None

    return comparison_results

def compare_albums(events, albums):
    comparison_results = {}

    for event in events:
        event_date = event.get_num_date()
        event_name = event.get_description()

        # Albums before and after the event
        albums_before = [album for album in albums if album.get_num_release_date() < event_date]
        albums_after = [album for album in albums if album.get_num_release_date() > event_date]

        if albums_before and albums_after:
            comparison_results[event_name] = {
                "short_term": compare_albums_internal(event, [albums_before[-1]], [albums_after[0]]),
                "long_term": compare_albums_internal(event, albums_before, albums_after)
            }
        else:
            # Either albums_before or albums_after is empty
            comparison_results[event_name] = {
                "short_term": None,
                "long_term": None
            }

    return comparison_results

def load_albums(albums_info):
    albums = []
    for name, release_date in albums_info.items():
        album = Album(name, release_date)
        albums.append(album) 
    return albums

# Function to plot album analysis
def plot_album_analysis(albums):
    
    parameters = ["non_real_words_freq", "word_frequency", "named_entities", "text_classification", "text_similarity_average", "text_lemmatization_freq"]

    fig, axes = plt.subplots(nrows=len(parameters), ncols=1, figsize=(12, 10), sharex=True)  

    # Define markers and line styles to cycle through
    markers = cycle(['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'd'])
    line_styles = cycle(['-', '--', '-.', ':', '-', '--'])

    # Iterate over each parameter
    for idx, param in enumerate(parameters):
        ax = axes[idx]
        years = []
        keys = []
        values = []

        # Collect data for the parameter from all albums
        for album in albums:
            album_analysis = album.get_analysis()
            if album_analysis and param in album_analysis:
                year = int(album.get_release_date().split("-")[0])  
                analysis_data = album_analysis[param]
                if isinstance(analysis_data, dict):
                    if param == "text_classification":
                        for key, value in analysis_data.items():
                            keys.append(key)
                            values.append(value)
                            years.append(year)
                        
                    else:
                        average_value = np.mean(list(analysis_data.values()))
                        keys.append('Average ' + param)  
                        values.append(average_value)
                        years.append(year)    

                elif param == "non_real_words_freq":
                    keys.append(param)
                    values.append(analysis_data[0])
                    years.append(year)
                else:
                    # Handle the case when analysis_data is not a dictionary
                    keys.append(param)
                    values.append(analysis_data)
                    years.append(year)
                    

        # Sort the data points by year
        sorted_indices = sorted(range(len(years)), key=lambda k: years[k])
        sorted_years = [years[i] for i in sorted_indices]
        sorted_keys = [keys[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]

        # Plot the parameter values with different markers for each key
        unique_keys = sorted(set(sorted_keys))  # Ensure consistent ordering of keys
        num_keys = len(unique_keys)
        marker = next(markers)  # Get the next marker from the cycle
        line_style = next(line_styles)  # Get the next line style from the cycle
        for key in unique_keys:
            key_values = [value for k, value in zip(sorted_keys, sorted_values) if k == key]
            key_years = [year for k, year in zip(sorted_keys, sorted_years) if k == key]
            ax.plot(key_years, key_values, label=key, marker=marker, linestyle=line_style)

        if param == 'text_classification':
            
            annotations = {
                'negative': (0, 45, 'Negative', 'tab:red'),
                'positive': (0, 30, 'Positive', 'tab:green'),
                'neutral': (0, 15, 'Neutral', 'tab:orange'),
                'classification score': (0, 0, 'Classification score', 'tab:blue')
            }
            for category, (x, y, label, color) in annotations.items():
                ax.annotate(label, xy=(1, 0), xycoords='axes fraction',
                            xytext=(x, y), textcoords='offset points',
                            color=color, fontsize=8, ha='left', va='center')

        ax.set_title(param)
        ax.set_ylabel("Value")
        ax.grid(True)

    # Set the main title of the graph
    plt.suptitle("Michael Jackson's Writing Across Career", fontsize=16)
    plt.xlabel("Year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.6)  
    plt.show()

def write_word_freqs_to_file(albums):
    with open("word_freqs.txt", "w", encoding="utf-8") as file:
        for album in albums:
            album_name = album.get_name()
            analysis = album.get_analysis()

            if analysis and "word_frequency" in analysis:
                file.write(f"Album: {album_name}\n")
                word_freqs = analysis["word_frequency"]
                first_10_items = {k: word_freqs[k] for k in list(word_freqs.keys())[:10]}
                for word, freq in first_10_items.items():
                    file.write(f"{word}: {freq}\n")
                file.write("\n")

def write_topic_modeling_to_file(albums):
    with open("topic_modeling.txt", "w", encoding="utf-8") as file:
        for album in albums:
            album_name = album.get_name()
            analysis = album.get_analysis()

            if analysis and "topic_modeling" in analysis:
                file.write(f"Album: {album_name}\n")
                topic_modeling = analysis["topic_modeling"]
                
                for topic in topic_modeling:
                    file.write(f"Topic {topic[0]}: {topic[1]}")
                file.write("\n")


write_word_freqs_to_file(albums)

write_topic_modeling_to_file(albums)

plot_album_analysis(albums)