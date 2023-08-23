# IMPORTS & LIBRARIES
from cProfile import label
from pandas import * # for reading excel sheet
import pdfplumber # for reading in PDF data
import re # for preprocessing data (replacing string sequences with specific tokens)
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score # for seeing how our classifier is doing
import os
os.system('clear') # clear the terminal

# GLOBAL VARIABLES
XLS_DICT = {}
ALL_DOCS = []
TRAIN_DOCS = []
TEST_DOCS = []
OVERALL_LABELS = ['UNKNOWN', 'CAT_0', 'CAT_1', 'CAT_2', 'CAT_2_1', 'CAT_3', 'CAT_3_1', 'CAT_4', 'CAT_4_1', 'CAT_5', 'CAT_5_1', 'CAT_6', 'CAT_7']

# CLASSES
class Document:
    def __init__(self, d_row, d_filename, d_title, d_author, d_year, d_text, d_labels, d_training = False) -> None:
        self.row = d_row # an integer
        self.filename = d_filename # a single string
        self.title = d_title # a single string
        self.author = d_author # a single string
        self.year = d_year # a single string
        self.text = d_text # a single string
        self.labels = d_labels # a list of strings
        self.x_matrix = []
        self.y_golds = self.set_golds_from_labels() # list of correct (1) and incorrect (0) labels; should be length 13, which is num of categories
        self.y_preds = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.is_training = d_training
    
    def set_golds_from_labels(self):
        gold_list = []
        for label in OVERALL_LABELS:
            if label in self.labels:
                gold_list.append(1)
            else:
                gold_list.append(0)
        return gold_list

    def set_single_pred_by_label(self, new_pred_label, new_pred_value):
        new_pred_index = -1

        # go through the y-preds (by index)
        for i in range(0, len(OVERALL_LABELS)):
            if new_pred_label == OVERALL_LABELS[i]:
                new_pred_index = i

        # if the new pred index is -1, that means it wasn't found
        if new_pred_index == -1:
            print(f"!! - ERROR: tried to set single pred by label, but could not find the label <{new_pred_label}>")
            return
        
        new_pred_list = []
        for i in range(0, len(OVERALL_LABELS)):
            if i == new_pred_index:
                new_pred_list.append(new_pred_value)
            else:
                new_pred_list.append(self.y_preds[i])

        self.y_preds = new_pred_list

    def printout_text_only(self):
        print(f"--------------------------------------")
        print(f"DOCUMENT TEXT (whole):\n{self.text}")
        print(f"--------------------------------------")

    def printout(self):
        print(f"--------------------------------------")
        print(f"ROW: {self.row}")
        print(f"TITLE: {self.title}")
        print(f"FILENAME: {self.filename}")
        print(f"AUTHOR: {self.author}")
        print(f"YEAR: {self.year}")
        split_text = self.text.split(".")
        print(f"TEXT (excerpt from beginning): {split_text[:10]}")
        print(f"X MATRIX: {self.x_matrix}")
        print(f"LABELS: {self.labels}")
        print(f"Y GOLDS: {self.y_golds}")
        print(f"Y PREDS: {self.y_preds}")
        print(f"IS TRAINING: {self.is_training}")
        print(f"--------------------------------------")
        print(f"\n")

# MAIN METHODS
def find_row_for_pdf(long_pdf_name) -> int:
    short_pdf_name = long_pdf_name[:-4]
    for row_num, pdf_name in XLS_DICT["PDF_NAME"].items():
        str_pdf_name = str(pdf_name)
        str_pdf_name = str_pdf_name.strip()
        # print(f"Trying to match short pdf name <{short_pdf_name}> with pdf name <{pdf_name}> (as str <{str_pdf_name}>)")
        if short_pdf_name == str_pdf_name:
            # print(f"!! - found matching pdf name for: {short_pdf_name}")
            return row_num
    print(f"ERROR >> no row was found matching short pdf name {short_pdf_name}: with pdf name <{pdf_name}> (as str <{str_pdf_name}>)")
    return None

def get_labels_by_row(row_num):
    label_list = []

    for label_title in OVERALL_LABELS:
        if XLS_DICT[label_title][row_num] == 1:
            label_list.append(label_title)

    if len(label_list) == 0:
        print(f"POTENTIAL ERROR - no labels found in exel sheet for row {row_num}")
    
    return label_list

def preprocess_text(document_text):
    text_copy = document_text
    # lowercase all text
    text_copy = text_copy.lower()
    # print(f"-> PREPROCESSING: after lowercase:\n{text_copy}")

    # replace references with CITATION: (Person 1900), (Person et al. 2000), Person (2011)
    citation_pattern = r"(\w+\s)?\(.*\d+.*\)"
    citation_repl = "CIT"
    text_copy = re.sub(citation_pattern, citation_repl, text_copy)
    # print(f"PREPROCESSING: after removing citations, len: {len(text_copy)}, num tokens: {len(text_copy.split(' '))}")
    # print(f"-> PREPROCESSING: after removing citations:\n{text_copy}")

    # remove all newlines
    newline_pattern = r"\n"
    newline_repl = " "
    text_copy = re.sub(newline_pattern, newline_repl, text_copy)
    # print(f"PREPROCESSING: after removing newlines, len: {len(text_copy)}, num tokens: {len(text_copy.split(' '))}")
    # print(f"-> PREPROCESSING: after removing newlines:\n{text_copy}")

    # remove formatting like "this content downloaded from"
    format_pattern = r"DFODJFWOIFJWEFOKWJEFOWHO" # TO DO
    format_repl = " "
    text_copy = re.sub(format_pattern, format_repl, text_copy)
    # print(f"PREPROCESSING: after removing formatting, len: {len(text_copy)}, num tokens: {len(text_copy.split(' '))}")
    # print(f"-> PREPROCESSING: after removing formatting:\n{text_copy}")

    # remove all extra white space
    whitespace_pattern = r"\s\s+"
    whitespace_repl = " "
    text_copy = re.sub(whitespace_pattern, whitespace_repl, text_copy)
    # print(f"PREPROCESSING: after removing extra whitespace, len: {len(text_copy)}, num tokens: {len(text_copy.split(' '))}")
    # print(f"-> PREPROCESSING: after removing extra whitespace:\n{text_copy}")

    # collapse all other numbers
    number_pattern = r"(\d+[\s\/]?)+"
    number_repl = "NUM "
    text_copy = re.sub(number_pattern, number_repl, text_copy)
    # print(f"PREPROCESSING: after collapsing numbers, len: {len(text_copy)}, num tokens: {len(text_copy.split(' '))}")
    # print(f"-> PREPROCESSING: after collapsing numbers:\n{text_copy}")

    # remove punctuation except periods
    punct_pattern = r"[-,;!?\“\"\”\'\’]"
    punct_repl = ""
    text_copy = re.sub(punct_pattern, punct_repl, text_copy)
    # print(f"PREPROCESSING: after removing punctuation, len: {len(text_copy)}, num tokens: {len(text_copy.split(' '))}")
    # print(f"-> PREPROCESSING: after removing punctuation:\n{text_copy}")
    return text_copy

# MAIN

# >> READ IN DATA
cwd_path = os.path.dirname(os.path.abspath(__file__))

# -- READ IN EXCEL DATA
print(f"\n** NOW READING IN EXCEL DATA **\n")
xlsx_folder_path = cwd_path + "/xlsx/"
xlsx_file_path = xlsx_folder_path + "american-speech-dataset.xlsx"
xlsx = ExcelFile(xlsx_file_path)
xlsx_df = xlsx.parse(xlsx.sheet_names[0])
xlsx_dict = xlsx_df.to_dict()
XLS_DICT = xlsx_dict
print(f"XLS_DICT keys are: {XLS_DICT.keys()}\n")
# for x, y in x_dict.items():
#     print(f"x: {x}")
#     print(f"y: {y}")

# a test to understand the dictionary structure
# result: x_dict["SOME_COLUMN"][row#]
# some_data = x_dict["PDF_NAME"][200]
# print(f">> some data: <<\n{some_data}\n")

# -- READ IN PDF DATA
print(f"\n** NOW READING IN PDF DATA **\n")
pdf_folder_path = cwd_path + "/pdfs/"

readin_doc_count = 0

# get every file in the PDF folder
for readin_filename in os.listdir(pdf_folder_path):
    readin_doc_count += 1

    pdf_file_path = pdf_folder_path + readin_filename
    with pdfplumber.open(pdf_file_path) as pdf:
        # print(f"{pdf.metadata}\n")
        overall_pdf_text = ""
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3, layout=False, x_density=7.25, y_density=13)
            overall_pdf_text += page_text

    # find the row with this pdf_name
    p_row_num = find_row_for_pdf(readin_filename)
    # print(f"Row for {filename} is {p_row_num}")

    # construct documents from all of this information
    p_row = p_row_num
    p_filename = readin_filename
    p_title = XLS_DICT["TITLE"][p_row_num]
    p_author = XLS_DICT["AUTHORS"][p_row_num]
    temp_year = str(XLS_DICT["YEAR"][p_row_num])
    p_year = temp_year[:-2] # to remove float decimal
    p_text = overall_pdf_text # TO DO: correct this to include all pages
    p_labels = get_labels_by_row(p_row_num) # need method for this
    readin_doc = Document(p_row, p_filename, p_title, p_author, p_year, p_text, p_labels, False)
    # new_doc.printout()

    # add this document to an overall list
    ALL_DOCS.append(readin_doc)

print(f"\n** Found {readin_doc_count} PDF files total **")

# >> PREPROCESS DATA
print(f"\n** NOW PREPROCESSING TEXT DATA **\n")
for document in ALL_DOCS:
    print(f"\n\n>> initiating preprocessing for: {document.filename}")
    # document.printout()
    # print(f">> original document text:\n{document.text}")
    document.text = preprocess_text(document.text)
    # print(f"\n>> document text after preprocessing:\n{document.text}\n\n")

# >> VECTORIZER
print(f"\n** NOW AT VECTORIZER - HANDLING TEXT TO FEATURES **\n")

# make the vectorizer (which will then convert our texts to a vocabulary of features (the n-grams))
vectorizer = TfidfVectorizer(
    preprocessor=None, # None >> we have already handled preprocessing on our own above
    ngram_range = (1, 2), # get n-grams of these lengths
    binary=False, # False >> we want counts for each token
    analyzer = "word", # Word >> we want word n-grams
    min_df = 2 # 2 >> we want to remove any terms which only occur once
    )

# >> STRATIFIED K-FOLD
print(f"\n** NOW SPLITTING DATA INTO TRAIN, TEST - USING STRATIFIED K-FOLD **\n")
all_texts_list = [doc.text for doc in ALL_DOCS]
vectorizer.fit(all_texts_list)
to_strat_X = vectorizer.transform(all_texts_list)
to_strat_Y = [1 if "CAT_1" in doc.labels else 0 for doc in ALL_DOCS]

skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(to_strat_X, to_strat_Y)
skf_split = skf.split(to_strat_X, to_strat_Y)

X_train_i = None
Y_train_i = None

X_test_i = None
Y_test_i = None
# for train_index, test_index
index_num = 1
for thing in skf_split:
    print(f"THING #{index_num}:\n{thing}")
    index_num = index_num + 1
    # print("TRAIN:", train_index, "TEST:", test_index)
    # X_train_s, X_test_s = to_strat_X[train_index], to_strat_X[test_index]
    # Y_train_s, Y_test_s = to_strat_Y[train_index], to_strat_Y[test_index]



# gather the texts for training and testing, and update their is_training field

# fit the vectorizer to the training documents (making columns for each feature)
train_texts = [doc.text for doc in TRAIN_DOCS]
vectorizer.fit(train_texts)

# use the vectorizer to transform the documents (creating a feature matrix)
X_TRAIN = vectorizer.transform(train_texts)
print(f"\n>> created X train matrix (after fitting then transforming training texts):\n{X_TRAIN}\n\n")

test_texts = [doc.text for doc in TEST_DOCS]
X_TEST = vectorizer.transform(test_texts)

# >> LABEL ENCODER
print(f"\n** NOW AT LABEL ENCODER - HANDLING TEXT TO LABELS **\n")

LE_CAT_1 = LabelEncoder()
CAT_1_train_golds = ["CAT_1" if "CAT_1" in doc.labels else "0_c1" for doc in TRAIN_DOCS]
LE_CAT_1.fit(CAT_1_train_golds)
print(f"LE CLASSES (CAT_1):\n{LE_CAT_1.classes_}")
print(f"CAT_1 training labels passed in:\n{CAT_1_train_golds}")
CAT_1_train_transformed = LE_CAT_1.transform(CAT_1_train_golds)
print(f"CAT_1 TRAIN transformed:\n{CAT_1_train_transformed}")

# >> TRAIN CLASSIFIER(S)
print(f"\n** NOW TRAINING CLASSIFIERS **\n")

CLF_CAT_1 = LogisticRegression(max_iter=1000)
CLF_CAT_1.fit(X_TRAIN, CAT_1_train_transformed)

# >> PREDICT WITH CLASSIFIER(S)
print(f"\n** NOW PREDICTING WITH CLASSIFIERS **\n")
print(f"\n>> X test matrix:\n{X_TEST}\n\n")
CAT_1_preds = CLF_CAT_1.predict(X_TEST)
CAT_1_preds_translated = LE_CAT_1.inverse_transform(CAT_1_preds)
print(f">> predictions for CAT_1 complete:\n{CAT_1_preds}\n{CAT_1_preds_translated}")
CAT_1_preds_proba = CLF_CAT_1.predict_proba(X_TEST)
CAT_1_preds_proba_formatted = [format(prob[1], '.4f') for prob in CAT_1_preds_proba]
print(f">> positive predictions for CAT_1 as probabilities:\n{CAT_1_preds_proba_formatted}")

# >> REVIEW PERFORMANCE
print(f"\n** NOW REVIEWING PERFORMANCE **\n")
CAT_1_test_golds = ["CAT_1" if "CAT_1" in doc.labels else "0_c1" for doc in TEST_DOCS]
CAT_1_test_transformed = LE_CAT_1.transform(CAT_1_test_golds)
CAT_1_test_translated = LE_CAT_1.inverse_transform(CAT_1_test_transformed)
print(f"CAT_1 TEST (golds) transformed:\n{CAT_1_test_transformed}\n{CAT_1_test_translated}\n")

# OVERALL SUMMARY
# update predictions for each label in each test document
for i in range(0, len(CAT_1_preds)):
    TEST_DOCS[i].set_single_pred_by_label("CAT_1", CAT_1_preds[i])

# show the overall labels

print(f"----------------------------------------------------")
print(f"OVERALL LABELS:\n{OVERALL_LABELS}")
print(f"----------------------------------------------------")

# print out a short summary of predictions for each test document
for i in range(0, len(TEST_DOCS)):
    doc = TEST_DOCS[i]
    print(f"- #{i+1}: {doc.title} ({doc.filename})\npreds: <{doc.y_preds}>\n")

# f1 = f1_score(CAT_1_test_transformed, CAT_1_preds)
# print(f"\n** CAT 1 - F1 score: {int(f1 * 100)}% ({f1})")
# accuracy = accuracy_score(CAT_1_test_transformed, CAT_1_preds)
# print(f"** CAT 1 - accuracy score: {int(accuracy * 100)}% ({accuracy})\n")

c_report = classification_report(CAT_1_test_transformed, CAT_1_preds)
print(f"\nCLASSIFICATION REPORT:\n{c_report}\n")