# Importing all the necessary libraries

import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
import os
import chardet         #Use to identify the correct character encoding, in short it gives direction
import pandas as pd
nltk.download('cmudict')
import string
import re
from nltk.corpus import stopwords
import syllables 

# Analysis and cleaning of StopWord and MasterDictionary 


# Mentioning the path of our  master dictionary folder and stopwords folder

master_dict_folder = r"C:\Users\91880\Desktop\Task\MasterDictionary"
stopwords_folder = r"C:\Users\91880\Desktop\Task\StopWords"


# Creating a word dictionary so that later i can select only positive and negative words 
def create_word_dictionary(master_dict_folder, stopwords_folder):
    word_dictionary = {'positive': set(), 'negative': set()}

    # Load stop words
    stopwords_library = set()
    encodings_to_try = ['utf-8', 'latin-1', 'utf-16']                    #Types of encoding we will use to analyze text


#For loop because we have multiple files in the folder
    for filename in os.listdir(stopwords_folder):
        file_path = os.path.join(stopwords_folder, filename)
        with open(file_path, 'rb') as file:
            raw_data = file.read()                                        #all the words in the folder files will stored in this variable  
            detected_encoding = chardet.detect(raw_data)['encoding']      #Here chardet will identify which encoding will be required in the text  

            for encoding in encodings_to_try:
                try:
                    stopwords = raw_data.decode(encoding).splitlines()
                    stopwords_library.update(stopwords)
                    break  # Break the loop if decoding is successful
                except UnicodeDecodeError:
                    continue  # Continue to the next encoding if decoding fails

    # Process positive words file
    positive_file_path = os.path.join(master_dict_folder, 'positive-words.txt')
    with open(positive_file_path, 'rb') as file:
        raw_data = file.read()
        detected_encoding = chardet.detect(raw_data)['encoding']

        for encoding in encodings_to_try:
            try:
                positive_words = raw_data.decode(encoding).splitlines()
                word_dictionary['positive'].update(positive_word for positive_word in positive_words    
                                                   if positive_word not in stopwords_library)
                break                                                       # Break the loop if decoding is successful
            except UnicodeDecodeError:
                continue                                                    # Continue to the next encoding if decoding fails


    # Process negative words file
    negative_file_path = os.path.join(master_dict_folder, 'negative-words.txt')
    with open(negative_file_path, 'rb') as file:
        raw_data = file.read()
        detected_encoding = chardet.detect(raw_data)['encoding']

        for encoding in encodings_to_try:
            try:
                negative_words = raw_data.decode(encoding).splitlines()
                word_dictionary['negative'].update(negative_word for negative_word in negative_words
                                                   if negative_word not in stopwords_library)
                break  # Break the loop if decoding is successful
            except UnicodeDecodeError:
                continue  # Continue to the next encoding if decoding fails

    return word_dictionary

# Here we have a set of word dictionary which contains positive and negative words

word_dictionary = create_word_dictionary(master_dict_folder, stopwords_folder)
positive_words = word_dictionary['positive']
negative_words = word_dictionary['negative']

# Analysis on our text file



# Input file from which we will extract the URL's 
url_file = r"C:\Users\91880\Desktop\Task\Input\Input.xlsx"

# Load the URL Excel file
df_urls = pd.read_excel(url_file)

# Convert the 'URL_ID' column to string type in both DataFrames
df_urls['URL_ID'] = df_urls['URL_ID'].astype(str)

# Create an empty DataFrame to store the data
df_output = pd.DataFrame(columns=['URL_ID','URL', 'POSITIVE SCORE','NEGATIVE SCORE','POLARITY SCORE','SUBJECTIVITY SCORE','AVG SENTENCE LENGTH','PERCENTAGE OF COMPLEX WORDS','FOG INDEX','AVG NUMBER OF WORDS PER SENTENCE','COMPLEX WORD COUNT','WORD COUNT','SYLLABLE PER WORD','PERSONAL PRONOUNS','AVG WORD LENGTH'])



# Path of our .txt files on which we have to perform text analysis

input_folder = r"C:\Users\91880\Desktop\Task\TextFiles"

# Iterate over the input files
for filename in os.listdir(input_folder):
    # Get the URL_ID from the input file name (without the extension)
    url_id = int(os.path.splitext(filename)[0])

    # Construct the input file path
    file_path = os.path.join(input_folder, filename)

    # Read the input file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Word Count for text 
    words = nltk.word_tokenize(text)
    words = text.split()
    word_count = len(words)


    # Count the number of sentences
    sentences = nltk.sent_tokenize(text)
    sentence_count = len(sentences)

      # Tokenizing the input files
    tokens = word_tokenize(text)    


    # Step 1: Sentimental Analysis: Calculate the positive and negative counts
    positive_score = sum(word in positive_words for word in text.split())
    negative_score = sum(word in negative_words for word in text.split())

    # Get the URL for the URL_ID from the URL Excel file
    # So what we have done we are relate input .xlsx file and output .xlsx file on
    url = df_urls.loc[df_urls['URL_ID'] == str(url_id), 'URL'].values[0]
    
    # Calculate the polarity score
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)

    
    # Calculate Subjectivity Score
    subjectivity_score = (positive_score - negative_score) / (word_count + 0.000001)
        
    # ******************************** 	Analysis of Readability ***************************************************************************************************************
    
    def calculate_readability(text):
       
        #Calculate average sentence length
        average_sentence_length = word_count / sentence_count

        return average_sentence_length

    average_sentence_length = calculate_readability(text)    
 

    # Calculate the count of complex words
    complex_word_count = sum(len(word) > 2 for word in words)

    #  Calculate the percentage of complex words
    percentage_complex_words = (complex_word_count / word_count) * 100

    # Calculate the Fog Index
    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)

    #********************************* Average Number of Words Per Sentence***************************************************************************************

    # Calculate the average number of words per sentence
    average_words_per_sentence = word_count / sentence_count


    #********************************** Word Count ****************************************************************************************************************

    # Remove punctuation and convert text to lowercase
    cleaned_text = text.translate(str.maketrans('', '', string.punctuation))
    cleaned_text = cleaned_text.lower()

    # Defining a function as we don't want to get confused between word count of text and cleaned_text

    def word_counts(text):
       
        # Tokenize the text into words and getting the final word count
        words = nltk.word_tokenize(cleaned_text)
       
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words]

        # Count the remaining words
        word_count = len(words)

        return word_count

    # Final word count of our cleaned_text by calling out the function
    word_count_final = word_counts(cleaned_text)  

    #************************************  Syllable Count Per Word  **************************************************************************************************************
    # Here we have used a syllables library which directly gives us the syllables in a word, plus the condition mentioned in For loop for 'es' and 'ed'
    def count_syllables(word):
        return syllables.estimate(word)

    def count_syllables_per_word(words):
        syllable_count = 0                 # Starting with zero
        for word in words:
        # Exclude words ending with "es" or "ed"
            if word.endswith(("es", "ed")):
                continue                   # Don't count     
            syllable_count += count_syllables(word)
        return syllable_count
  
    
     # Count the number of syllables per word
    syllables_per_word = count_syllables_per_word(words)

    #************************************* Personal Pronouns ***********************************************************************************************************************
   
    # Calculate the count of personal pronouns
    pronouns_pattern = r"\b(I|we|my|ours|us)\b"                                                    # Pronouns we want to exclude
    pronoun_count = len(re.findall(pronouns_pattern, text, flags=re.IGNORECASE))                   # direct sum of pronouns found in text using regex/re

    #************************************** Average word Lenght *******************************************************************************************************************
    
    # Calculate the total number of characters in each word
    total_characters = sum(len(word) for word in text)

    # Calculate average word length
    average_word_length = total_characters / word_count

  
    # Creating a new .xlsx file with columns as given in out Outout structure file and assigning them values

    new_df = pd.DataFrame({
        'URL_ID': [url_id],
        'URL': [url], 
        'POSITIVE SCORE': [positive_score],
        'NEGATIVE SCORE':[negative_score],
        'POLARITY SCORE':[polarity_score],
        'SUBJECTIVITY SCORE':[subjectivity_score],
        'AVG SENTENCE LENGTH':[average_sentence_length],
        'PERCENTAGE OF COMPLEX WORDS':[percentage_complex_words],
        'FOG INDEX':[fog_index],
        'AVG NUMBER OF WORDS PER SENTENCE':[average_words_per_sentence],
        'COMPLEX WORD COUNT':[complex_word_count],
        'WORD COUNT':[word_count_final],
        'SYLLABLE PER WORD':[syllables_per_word],
        'PERSONAL PRONOUNS':[pronoun_count],
        'AVG WORD LENGTH':[average_word_length]
        
       })

    # Concatenate the new DataFrame with the output DataFrame
    df_output = pd.concat([df_output, new_df], ignore_index=True)

# Sort the DataFrame by URL_ID column in ascending order
df_output = df_output.sort_values(by='URL_ID', ascending=True)



#Specify the output file path where we want to save the output

output_file = r"C:\Users\91880\Desktop\Task\Output\Output.xlsx"


# Save the DataFrame to the Excel file
df_output.to_excel(output_file, index=False)

print("Results saved to", output_file)
    