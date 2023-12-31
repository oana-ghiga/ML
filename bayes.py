import os
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Function to preprocess text
def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    # Add additional preprocessing steps here (e.g., remove stopwords, lemmatization)
    return text

# Apply preprocessing to the 'Text' column
df['Text'] = df['Text'].apply(preprocess)

# Calculate priors
priors = df['Label'].value_counts(normalize=True).to_dict()

# Calculate conditional probabilities
def calculate_conditional_probs(data):
    vocab = set(' '.join(data['Text']).split())
    conditional_probs = {word: {0: 0, 1: 0} for word in vocab}
    for _, row in data.iterrows():
        words = row['Text'].split()
        label = row['Label']
        for word in words:
            conditional_probs[word][label] += 1

    total_spam = data['Label'].sum()
    total_ham = len(data) - total_spam
    for word, counts in conditional_probs.items():
        conditional_probs[word][0] = (conditional_probs[word][0] + 1) / (total_ham + len(vocab))
        conditional_probs[word][1] = (conditional_probs[word][1] + 1) / (total_spam + len(vocab))

    return conditional_probs

conditional_probs = calculate_conditional_probs(df)

# Function to predict labels
def predict(text, priors, conditional_probs):
    text = preprocess(text)
    words = text.split()
    scores = {label: np.log(priors[label]) for label in priors}
    for word in words:
        if word in conditional_probs:
            for label in priors:
                scores[label] += np.log(conditional_probs[word][label])
    return max(scores, key=scores.get)

# Make a copy of the test data to avoid SettingWithCopyWarning
test_data = df[df.index % 10 == 0].copy()

# Apply predictions to the copied test data
test_data['Predicted'] = test_data['Text'].apply(lambda x: predict(x, priors, conditional_probs))

# Calculate accuracy
accuracy = (test_data['Label'] == test_data['Predicted']).mean()
print(f"Accuracy of Naive Bayes Classifier: {accuracy}")

# LOOCV
for index, _ in df.iterrows():
    # Exclude the current row for testing
    train_data = df.drop(index)
    test_data = df.loc[[index]].copy()

    # Calculate priors and conditional probabilities
    priors = train_data['Label'].value_counts(normalize=True).to_dict()
    conditional_probs = calculate_conditional_probs(train_data)

    # Make predictions on the test set
    test_data['Predicted'] = test_data['Text'].apply(lambda x: predict(x, priors, conditional_probs))

    # Calculate accuracy for this iteration
    accuracy = (test_data['Label'] == test_data['Predicted']).mean()
    accuracies.append(accuracy)

# Calculate average accuracy
avg_accuracy = np.mean(accuracies)

# Print average accuracy
print(f"Average Accuracy using LOOCV: {avg_accuracy}")

# Plot histogram of accuracies
plt.figure(figsize=(8, 6))
plt.hist(accuracies, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Distribution of Accuracies (LOOCV)')
plt.grid(True)
plt.show()
