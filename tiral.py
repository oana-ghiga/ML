import os
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

stopwords = nltk.corpus.stopwords.words("english")
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()

main_directory = os.path.join(os.getcwd(), 'lingspam_public')
data = []
labels = []
accuracies = []

for subdir in ['bare', 'lemm', 'lemm_stop', 'stop']:
    subdir_path = os.path.join(main_directory, subdir)
    for i in range(1, 11):
        folder_path = os.path.join(subdir_path, f'part{i}')
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                text = file.read()
                label = 1 if file_name.startswith('spmsg') else 0
                data.append(text)
                labels.append(label)

# Create a DataFrame to store data and labels
df = pd.DataFrame({'Text': data, 'Label': labels})

# Function to tokenize text
def tokenize(text):
    tokens = re.split("\W+", text)
    return tokens

# Function to preprocess text
def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return text

# Function to tokenize text
def tokenize(text):
    tokens = re.split("\W+", text)# W+ means all capital, small alphabets and integers 0-9
    return tokens

# Function to remove stop words
def remove_stopwords(token):
    text = [word for word in token if word not in stopwords]# to remove all stopwords
    return text

# Function to stem words
def stemming(text):
    stem_text = [ps.stem(word) for word in text]
    return stem_text

# Function to lemmatize
def lemmatizer(text):
    lem_text = [wn.lemmatize(word) for word in text]
    return lem_text

def preprocess(text):
    text_no_punct = remove_punctuation(text)
    tokens = tokenize(text_no_punct)
    stemmed_tokens = stemming(tokens)
    lemmatized_tokens = lemmatizer(stemmed_tokens)

    return lemmatized_tokens

# Apply preprocessing to the 'Text' column
df['Text'] = df['Text'].apply(preprocess)

# Calculate priors
priors = df['Label'].value_counts(normalize=True).to_dict()

# Calculate conditional probabilities
def calculate_conditional_probs(data):
    conditional_probs = defaultdict(lambda: [0, 0])  # Initialize with smoothing
    word_counts = defaultdict(lambda: [0, 0])

    for _, row in data.iterrows():
        words = row['Text']
        label = row['Label']
        for word in words:
            word_counts[word][label] += 1

    total_spam = data['Label'].sum()
    total_ham = len(data) - total_spam
    vocab_size = len(word_counts)

    for word, counts in word_counts.items():
        conditional_probs[word][0] = (counts[0] + 1) / (total_ham + vocab_size)
        conditional_probs[word][1] = (counts[1] + 1) / (total_spam + vocab_size)

    return conditional_probs

# Calculate conditional probabilities
conditional_probs = calculate_conditional_probs(df)

# Function to predict labels
def predict(tokens, priors, conditional_probs):
    scores = {label: np.log(priors[label]) for label in priors}
    for token in tokens:
        if token in conditional_probs:
            for label, prob in enumerate(conditional_probs[token]):
                scores[label] += np.log(prob)
    return max(scores, key=scores.get)

# Iterate through each row and perform LOOCV
for index, _ in df.iterrows():
    # Exclude the current row for testing
    train_data = df.drop(index)
    test_data = df.loc[[index]]

    # Make predictions on the test set
    priors = train_data['Label'].value_counts(normalize=True).to_dict()
    test_data['Predicted'] = test_data['Text'].apply(lambda x: predict(x, priors, conditional_probs))
    print(test_data[['Label', 'Predicted']])
    print("Done")

    # Calculate accuracy for this iteration
    accuracy = (test_data['Label'] == test_data['Predicted']).mean()
    accuracies.append(accuracy)

# Calculate average accuracy
avg_accuracy = np.mean(accuracies)

# Print average accuracy
print(f"Average Accuracy using LOOCV: {avg_accuracy}")

# Plot histogram of accuracies
plt.figure(figsize=(8, 6))
plt.hist(accuracies, bins=[-0.5, 0.5, 1.5], color='skyblue', edgecolor='black')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Distribution of Accuracies (LOOCV)')
plt.xticks([0, 1])
plt.grid(True)
plt.show()
