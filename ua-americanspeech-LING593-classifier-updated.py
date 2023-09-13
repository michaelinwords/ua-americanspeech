### AUTHOR - Ahn Michael
### GOAL - For internship in UA American Speech project, train a model to classify academic articles (PDFs)
###        into pre-defined categories, based on their texts, then output predictions to an XLSX
#region - IMPORTS & INITIALISATION
import os       # for navigating the file system
import re       # mainly for text preprocessing / normalisation
import PyPDF2       # for reading PDF files
import pandas as pd     # for handling excel files and storing them as dataframes
from sklearn.feature_extraction.text import TfidfVectorizer     #! for transforming text into feature vectors
from sklearn.model_selection import train_test_split    # for a simple train-test split of document data
from sklearn.model_selection import StratifiedKFold     # for stratified k-fold splitting (separating train and test data)
from sklearn.preprocessing import MultiLabelBinarizer       #x for encoding labels / converting list of present categories (["CAT-A", "CAT-D"]) to binary format ([1, 0, 0, 1])
from sklearn.linear_model import LogisticRegression     #! includes the logistic regression model
from sklearn.multiclass import OneVsRestClassifier      # for handling multi-label classification
from sklearn.metrics import classification_report, f1_score, accuracy_score     # for evaluating the model's performance
from joblib import dump, load      # for saving and loading model, vectorizer, and label binarizer (to avoid retraining every time we want to make predictions)
from termcolor import colored   # for printing colored text in terminal; 100% optional
import numpy as np
os.system('clear') # clear the terminal
#endregion - IMPORTS & INITIALISATION

#region - OPTIONS
SCRIPT_MODE = "TRAIN_TEST" # which mode the script runs in; TRAIN_TEST, FINAL_TRAIN, or PREDICT
# for TRAIN_TEST mode, need to have an XLSX with PDF metadata and the PDFs to train on; will split data
# for FINAL_TRAIN mode, need to have an XLSX with PDF metadata and the PDFs to train on; will use all PDFs/documents for training (no performance review)
# for PREDICT mode, need to have an XLSX with PDFs you want to predict the categories of, and those PDFs in a folder
TERMINAL_MAIN_COLOR = 'blue' # which color termcolor will use when outputting main script logs

TFIDF_PREPROCESSOR = None  # None >> We are already handling preprocessing elsewhere
TFIDF_NGRAM_RANGE = (1, 1) # Define ngram range, get n-grams of these lengths
TFIDF_BINARY = False # False >> We want counts for each token, not just presence
TFIDF_ANALYSER = 'word' # We want word n-grams
TFIDF_MIN_DF = 2 # 2 >> We want to remove any terms which only occur once

KFOLD_SPLITS = 5 # How many splits the Stratified KFOLD will use
NUM_TOP_FEATURES = 20 # How many top-weighted features to see for each class
#endregion - OPTIONS

#region - COLLECTIONS
ALL_CATEGORIES = ['UNKNOWN', 'CAT_0', 'CAT_1', 'CAT_2', 'CAT_2_1', 'CAT_3', 'CAT_3_1', 'CAT_4', 'CAT_4_1', 'CAT_5', 'CAT_5_1', 'CAT_6', 'CAT_7']
USING_CATEGORIES = ['CAT_1', 'CAT_2', 'CAT_2_1', 'CAT_3', 'CAT_4', 'CAT_5', 'CAT_5_1', 'CAT_6', 'CAT_7']
CATEGORIES_DETAILED_LABELS = ["AAL/AAVL (CAT_1)", "African Am. (CAT_2)", "African Diaspora (CAT_2_1)", "Mexican Am. & Latinx (CAT_3)", "Native Am. (CAT_4)", "Asian Am./Pacific Is. (CAT_5)", "Asian Diaspora (CAT_5_1)", "Women's Language (CAT_6)", "LGBTQ Speech (CAT_7)"]

CWD_PATH = "" # this is initialised at the beginning of MAIN

XLS_DF = None
XLS_ROWS = []

PDF_TEXTS_dict = {}
PDF_TEXTS_list = []
pdf_count = 0
pdfs_total_words = 0

ALL_DOCS = []
TRAIN_DOCS = []
TEST_DOCS = []

X_RAW = [] # List of all document PDF texts >> will go into tfidf-vectoriser to get X
Y_RAW = [] # List of category lists >> will go into multi-label binarizer to get Y

FEATURE_NAMES = None

X_ALL = []
Y_ALL = []

X_TRAIN = []
X_TEST = []

Y_TRAIN = []
Y_TEST = []

#endregion - COLLECTIONS

# VARIABLES
# starting with list of texts ()
# X-MATRIX
# Y-VECTOR/MATRIX
# list of PDF texts
# within document, all applicable labels (in list) for the document

#region - CLASSES
class Document:
    def __init__(self, n_pdfname, n_pdftext, n_title, n_authors, n_doi, n_year, n_month, n_volume, n_issue, n_category_list) -> None:
        self.pdfname = n_pdfname
        self.pdftext = n_pdftext
        self.title = n_title
        self.authors = n_authors
        self.doi = n_doi
        self.year = n_year
        self.month = n_month
        self.volume = n_volume
        self.issue = n_issue
        self.category_list = n_category_list # a list of strings; the actual categories/labels of the document, coming from the spreadsheet data (example: ["CAT-A", "CAT-D"])
        self.x_vector = [] # the vectorised representation of pdftext (output from TF-IDF vectorizer)
        self.y_golds = [] # list of correct (1) and incorrect (0) labels, in binary format, from spreadsheet data; should be length 13, which is the # of our categories; get from multilabelbinarizer
        self.y_preds = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] # unless wanting to validate training data, this will remain -1s/unset for a training document; it will be set with real predictions (1s and 0s) for test and prediction documents

    def to_string(self):
        doc_string = f"PRINT_OUT for document <{self.pdfname}>, titled <{self.title}> (authors: {self.authors})\
            \nDOI {self.doi} | YEAR {self.year} | MONTH {self.month} | VOLUME {self.volume} | ISSUE {self.issue} \
            \nCATEGORY LIST: {self.category_list} \
            \n\nTEXT: {self.pdftext[:200]} ..\n \
            \nX_VECTOR: {self.x_vector} \
            \nY_GOLDS: {self.y_golds} \
            \nY_PREDS: {self.y_preds}\n"
        return doc_string

#endregion - CLASSES

#region - OTHER FUNCTIONS
# Read in the XLSX file and load its data into global variables
def read_xlsx(file_name):
    global XLS_DF
    global XLS_ROWS
    XLS_FOLDER = "XLSX"

    XLS_DF = pd.read_excel(f"{CWD_PATH}/{XLS_FOLDER}/{file_name}")
    # print(f"XLS_DF before replacing NaN values:\n{XLS_DF}")
    # replace all NaN values in label columns with 0
    XLS_DF[USING_CATEGORIES] = XLS_DF[USING_CATEGORIES].fillna(0)
    print(f"XLS_DF:\n{XLS_DF}")
    XLS_ROWS = XLS_DF.to_dict(orient='records')
    num_xls_rows = len(XLS_ROWS)
    preview_xls_rows(XLS_ROWS, 2)
    print(f"\n>> found and read XLSX file ({file_name} - {num_xls_rows} rows)")

# Read in PDF files and load their data into global variables
def read_pdfs(folder_name):
    global pdf_count
    global pdfs_total_words
    global PDF_TEXTS_dict
    global PDF_TEXTS_list

    for filename in os.listdir(folder_name):
        pdf_count += 1
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_name, filename)
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ' '.join([page.extract_text() for page in reader.pages])
                text = preprocess_text(text)
                PDF_TEXTS_dict[filename] = text
                word_count = len(text.split())
                pdfs_total_words += word_count
                # print(f"\n> {filename} had ~{word_count} words")

    average_word_count = round(pdfs_total_words / pdf_count)
    print(f">> found {pdf_count} PDF files -- {pdfs_total_words} total words (avg: {average_word_count})\n")

    PDF_TEXTS_list = [v for k, v in PDF_TEXTS_dict.items()]
    print(f"> PDF TEXT PREVIEW (after preprocessing):\n{PDF_TEXTS_list[0][:2000]} ..\n")

# Show the first and last parts of the xls data
def preview_xls_rows(xls_rows, first_and_last_count):
    print(f"\n> XLS_ROWS PREVIEW:")
    preview_list = []

    preview_list = [row for row in xls_rows[:first_and_last_count]]
    preview_list.append("..")
    for row in xls_rows[-first_and_last_count:]:
        preview_list.append(row)
    for row in preview_list:
        print(f"\t >> {row}\n")

# Do preprocessing and normalisation on the text
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
    # format_pattern = r"DFODJFWOIFJWEFOKWJEFOWHO" # TO DO
    # format_repl = " "
    # text_copy = re.sub(format_pattern, format_repl, text_copy)
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

def get_present_category_list(single_xls_row):
    category_list = []
    for category in USING_CATEGORIES:
        if single_xls_row[category].iloc[0] == 1:
            category_list.append(category)
    
    # print(f">> For PDF <{single_xls_row['PDF_NAME'].iloc[0]}>, made category list:\t{category_list}\n")
    return category_list

def make_documents():
    global ALL_DOCS
    pdf_count = 0
    for pdf_name, pdf_text in PDF_TEXTS_dict.items():
        pdf_name = pdf_name.strip()
        pdf_count += 1
        # print(f">> Found PDF ({pdf_count}) {pdf_name} with following text:\n{pdf_text[:300]}\n")

        doc_xls_row = XLS_DF.loc[XLS_DF['PDF_NAME'].str.strip() == pdf_name[:-4]]
        if doc_xls_row.empty:
            print(f"⚠️ This xls_row is empty (likely means the PDF_NAME <{pdf_name[:-4]}> wasn't found):\n{doc_xls_row}")

        doc_title = doc_xls_row['TITLE'].iloc[0].strip()
        doc_authors = doc_xls_row['AUTHORS'].iloc[0].strip()
        doc_doi = doc_xls_row['DOI'].iloc[0].strip()
        doc_year = doc_xls_row['YEAR'].iloc[0]
        doc_month = doc_xls_row['MONTH'].iloc[0]
        doc_volume = doc_xls_row['VOLUME'].iloc[0]
        doc_issue = doc_xls_row['ISSUE'].iloc[0]
        doc_category_list = get_present_category_list(doc_xls_row) # construct this manually

        new_doc = Document(pdf_name, pdf_text, doc_title, doc_authors, doc_doi, doc_year, doc_month, doc_volume, doc_issue, doc_category_list)
        ALL_DOCS.append(new_doc)
        # print(f">> Made a document for PDF name <{pdf_name}> and added to ALL_DOCS list:\n{new_doc.to_string()}\n")

    print(f">> Completed making {len(ALL_DOCS)} document objects - two quick preview documents:\n\n{ALL_DOCS[0].to_string()}\n{ALL_DOCS[-1].to_string()}")

def sanitise_string(s):
    # Remove unpermitted characters from the string to allow Excel writing
    # Allow line feed (ASCII 10) and carriage return (ASCII 13)
    allowed_chars = {10, 13}
    
    # Filter out "illegal" characters
    sanitised_string = "".join([char for char in s if ord(char) >= 32 or ord(char) in allowed_chars])
    
    return sanitised_string

def create_xlsx_from_all_docs():
    # Extract data from each Document object in ALL_DOCS
    global ALL_DOCS
    rows_data = []
    for doc in ALL_DOCS:
        row_data = [
            sanitise_string(doc.pdfname),
            sanitise_string(doc.pdftext),
            sanitise_string(doc.title),
            sanitise_string(doc.authors),
            sanitise_string(doc.doi),
            sanitise_string(str(doc.year)),
            sanitise_string(str(doc.month)),
            sanitise_string(str(doc.volume)),
            sanitise_string(str(doc.issue)),
            ", ".join([sanitise_string(cat) for cat in doc.category_list]) 
        ]
        rows_data.append(row_data) 
    
    # Create a dataframe from the extracted data
    columns = ["PDF Name", "PDF Text", "Title", "Authors", "DOI", "Year", "Month", "Volume", "Issue", "Categories"]
    df = pd.DataFrame(rows_data, columns=columns)
    
    # Save the dataframe as an xlsx file
    output_path = "XLSX/all_docs.xlsx"
    df.to_excel(output_path, index=False)
    print(f">> Converted ALL_DOCS into XLSX, saved @ {output_path} ..\n")
    
    return output_path

# Vectorise the documents using TF-IDF
def tfidf_vectorise(docs, preprocessor=None, ngram_range=(1,2), binary=False, analyzer='word', min_df=2):

    # Create TF-IDF vectorizer
    vectoriser = TfidfVectorizer(preprocessor=preprocessor,
                                 ngram_range=ngram_range,
                                 binary=binary,
                                 analyzer=analyzer,
                                 min_df=min_df)

    return vectoriser.fit_transform(docs), vectoriser

# Save the trained model (or vectoriser) to a file
def save_model(model, filename):
    dump(model, filename)

# Load a trained model (or vectoriser) from a file
def load_model(filename):
    return load(filename)

# Predict categories for the given documents using the trained model
def predict_categories(model, vectorizer, docs):
    X = vectorizer.transform(docs)
    return model.predict(X)

# Update our XLSX file with the model's category predictions
def update_xlsx(xlsx_path, predictions):
    # TODO: implement this function
    pass
#endregion - OTHER FUNCTIONS

#region - MAIN

CWD_PATH = os.path.dirname(os.path.abspath(__file__))

#region - TRAIN MODE
print(colored(f"\n📄 ** SCRIPT START in TEST_TRAIN mode **", TERMINAL_MAIN_COLOR))
if SCRIPT_MODE.strip() == "TRAIN_TEST":

    # - READ IN DATA
    print(colored(f"\n📄 READING IN DATA", TERMINAL_MAIN_COLOR))

    # ---- READ IN EXCEL DATA
    # QUESTION: how should the excel data (the spreadsheet rows/columns) best be stored for the model training - dataframe, list of rows, actual dictionary?
    print(colored(f"\n📄 READING EXCEL DATA", TERMINAL_MAIN_COLOR))
    read_xlsx("american-speech-dataset-complete-rows-final-sep-13-2023.xlsx")

    # ---- READ IN PDF DATA
    print(colored(f"\n📄 READING PDF DATA", TERMINAL_MAIN_COLOR))
    read_pdfs("PDFs-train")

    # ---- CONSTRUCT DOCUMENT OBJECTS
    print(colored(f"\n📄 CONSTRUCTING DOCUMENT OBJECTS", TERMINAL_MAIN_COLOR))
    make_documents()
    create_xlsx_from_all_docs()

    # - MAKE VECTORISER (and vectorise training documents)
    print(colored(f"\n📄 MAKING and FIT-TRANSFORMING TF-IDF VECTORISER", TERMINAL_MAIN_COLOR))
    # - FIT-TRANSFORM VECTORISER (TF-IDF - vectorizer.fit_transform(training_documents))
    # fit: learns the vocabulary of the entire collection of training documents and calculates term frequencies and document frequencies
    # transform: uses the learned vocabulary and IDF values from fitting to convert each document into a vector
    # output of fit_transform: will be a matrix, where each row is a document from the collection and each column is a word/n-gram from the learned vocabulary

    # Customizable TF-IDF options

    # Fit the vectorizer on all texts
    X_RAW = [doc.pdftext for doc in ALL_DOCS] # all texts, in a list
    # print(f">> X_RAW is currently:\n{X_RAW[:10]} ..\n")
    print(f">> X_RAW has been made (not displaying all texts, as that would be too long)\n")
    X_ALL, vectoriser = tfidf_vectorise(X_RAW, TFIDF_PREPROCESSOR, TFIDF_NGRAM_RANGE, TFIDF_BINARY, TFIDF_ANALYSER, TFIDF_MIN_DF)
    print(f">> X_ALL is currently:\n{X_ALL[:10]} ..\n")
    FEATURE_NAMES = vectoriser.get_feature_names_out()
    print(f"FEATURE_NAMES preview: {FEATURE_NAMES[:100]} ..")

    # - ENCODE LABELS (MultiLabelBinarizer, into binary format
    print(colored(f"\n📄 ENCODING LABELS with MLB", TERMINAL_MAIN_COLOR))
    # MLB replaces my previous set_golds_from_labels function, previously in the document class
    Y_RAW = [doc.category_list for doc in ALL_DOCS] # list of category lists
    print(f">> Y_RAW is currently:\n{Y_RAW[:10]} ..\n")
    mlb = MultiLabelBinarizer(classes=USING_CATEGORIES)
    mlb.fit(USING_CATEGORIES)
    # ---- TRANSFORM LABELS for each document
    Y_ALL = mlb.transform(Y_RAW)
    print(f">> Y_ALL is currently:\n{Y_ALL[:10]} ..\n")

    # - SPLIT DATA (Stratified K-Fold or simple train-test-split, into training and testing sets)
    # NOTE: Implementation is using the common method of using a single primary category for stratification;
    # especially useful when the categories are imbalanced, to ensure training and test sets have similar distributions
    # of this primary category; does not ensure stratification across all categories, which can be more complex
    print(colored(f"\n📄 SPLITTING DATA (TRAIN-TEST SPLIT using SKF)", TERMINAL_MAIN_COLOR))
    
    # Convert multilabel data to single label
    single_label = ["".join(map(str, label)) for label in Y_ALL]

    # Initialize the StratifiedKFold instance with KFOLD_SPLITS splits
    skf = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)

    # Get the train and test indices for the first split
    train_indices, test_index = next(skf.split(X_ALL, single_label))

    # Extract the training and testing data using these indices
    X_TRAIN, X_TEST = X_ALL[train_indices], X_ALL[test_index]
    single_label_train, single_label_test = [single_label[i] for i in train_indices], [single_label[i] for i in test_index]

    # Convert single labels back to multilabel format
    Y_TRAIN = [[int(char) for char in label] for label in single_label_train]
    Y_TEST = [[int(char) for char in label] for label in single_label_test]
    Y_TRAIN, Y_TEST = np.array(Y_TRAIN), np.array(Y_TEST)

    #######
    # PREVIOUSLY, simple split: X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X_ALL, Y_ALL, test_size=0.2)

    print(f">> After splitting, there are {X_TRAIN.shape[0]} training documents/rows and {X_TEST.shape[0]} test documents/rows -- {X_ALL.shape[0]} documents/rows total")

    # skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=13)
    # X = [pdf for pdf in PDF_TEXTS_dict.keys()] # list of the pdf names
    # y = [record['CAT_1'] for record in XLS_ROWS]  # Using CAT_1for stratification
    # train_index, test_index = next(skf.split(X, y))
    # TRAIN_DOCS = [X[i] for i in train_index]
    # TEST_DOCS = [X[i] for i in test_index]

    # - TRAIN CLASSIFIER (fit the model on the training data: model.fit(X_train, y_train))
    print(colored(f"\n📄 TRAINING CLASSIFIER", TERMINAL_MAIN_COLOR))

    # Instantiate the classifier
    clf = OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver='liblinear'))  # 'liblinear' is a good solver choice for ovr
    # OneVsRestClassifier - is a fit for this task, as it trains a separate binary classifier
    # for each category, as I imagined early on; this does not work properly just setting as an 'ovr' parameter
    # Train the classifier
    print(f"X_TRAIN is:\n\n{X_TRAIN}\n\nY_TRAIN is:\n\n{Y_TRAIN}\n\n")
    clf.fit(X_TRAIN, Y_TRAIN)

    # - REVIEW PERFORMANCE (Classification Report)
    print(colored(f"\n📄 REVIEWING PERFORMANCE", TERMINAL_MAIN_COLOR))

    print(f"-- This training was on {X_TRAIN.shape[0]} documents (out of {X_ALL.shape[0]} total documents)\n")
    print(f"TFIDF settings - PREPROCESSOR {TFIDF_PREPROCESSOR}, NGRAM_RANGE {TFIDF_NGRAM_RANGE}, BINARY {TFIDF_BINARY}, ANALYSER {TFIDF_ANALYSER}, MIN_DF {TFIDF_MIN_DF}")
    print(f"Stratified K-Fold - KFOLD_SPLITS: {KFOLD_SPLITS}\n")

    for i, class_label in enumerate(clf.classes_):
        estimator = clf.estimators_[i]

        top_pos_indices = np.argsort(estimator.coef_[0])[-NUM_TOP_FEATURES:][::-1]
        top_pos_weights = estimator.coef_[0][top_pos_indices]
        print(f">> Top {NUM_TOP_FEATURES} positive features for class {class_label} ({CATEGORIES_DETAILED_LABELS[i]}):")
        pos_index_weight_list = []
        for index, weight in zip(top_pos_indices, top_pos_weights):
            pos_index_weight_list.append(f"{FEATURE_NAMES[index]} ({weight:.5f})")
        print(f"{pos_index_weight_list}\n")

        top_neg_indices = np.argsort(estimator.coef_[0])[:NUM_TOP_FEATURES]
        top_neg_weights = estimator.coef_[0][top_neg_indices]
        print(f">> Top {NUM_TOP_FEATURES} negative features for class {class_label} ({CATEGORIES_DETAILED_LABELS[i]}):")
        neg_index_weight_list = []
        for index, weight in zip(top_neg_indices, top_neg_weights):
            neg_index_weight_list.append(f"{FEATURE_NAMES[index]} ({weight:.5f})")
        print(f"{neg_index_weight_list}\n\n")

    subset_accuracy = clf.score(X_TEST, Y_TEST)
    print(f"SUBSET ACCURACY: {subset_accuracy * 100:.2f}%")

    # Vectorise the TEST_DOCS and evaluate the model
    Y_PRED = clf.predict(X_TEST)
    report_dict = classification_report(Y_TEST, Y_PRED, labels=range(len(USING_CATEGORIES)), target_names=CATEGORIES_DETAILED_LABELS, zero_division=1, output_dict=True)
    # Here, zero division defines what to return if the denominator is 0 (due to some outcomes - true positives, false positives, true negatives, false negatives - possibly having no instances)
    report_dict['per_label_accuracy'] = (Y_TEST == Y_PRED).mean()
    report_str = classification_report(Y_TEST, Y_PRED, labels=range(len(USING_CATEGORIES)), target_names=CATEGORIES_DETAILED_LABELS, zero_division=1) + f'\nper_label_accuracy: {report_dict["per_label_accuracy"]:.2f}\n'
    print(f"CLASSIFICATION REPORT:\n{report_str}")

    # - SAVE MODEL
    print(colored(f"\n📄 SAVING MODEL", TERMINAL_MAIN_COLOR))
    clf_save_name = "model.joblib"
    vectoriser_save_name = "vectoriser.joblib"
    save_model(clf, "model.joblib")
    save_model(vectoriser, "vectoriser.joblib")
    print(f">> Successfully saved classifier as {clf_save_name} and vectoriser as {vectoriser_save_name}")

    # - SCRIPT COMPLETE
    print(colored(f"\n📄 SCRIPT (train_test mode) COMPLETED SUCCESSFULLY ✅\n", TERMINAL_MAIN_COLOR))
#endregion - TRAIN MODE

elif SCRIPT_MODE.strip() == "FINAL_TRAIN":
    # - SCRIPT COMPLETE
    print(colored(f"\n📄 SCRIPT (final_train mode) COMPLETED SUCCESSFULLY ✅\n", TERMINAL_MAIN_COLOR))
    pass

#region - PREDICT MODE
elif SCRIPT_MODE.strip() == "PREDICT":
    print(colored(f"\n📄 ** SCRIPT START in PREDICT mode **\n", TERMINAL_MAIN_COLOR))

    # - LOAD MODEL & VECTORISER DATA
    print(colored(f"\n📄 LOADING MODEL & VECTORISER DATA", TERMINAL_MAIN_COLOR))
    clf = load_model("model.joblib")
    vectoriser = load_model("vectoriser.joblib")

    # - READ IN DATA
    print(colored(f"\n📄 READING IN DATA", TERMINAL_MAIN_COLOR))

    # ---- READ IN EXCEL DATA
    print(colored(f"\n📄 READING EXCEL DATA", TERMINAL_MAIN_COLOR))
    read_xlsx("american-speech-predictions.xlsx")

    # ---- READ IN PDF DATA
    print(colored(f"\n📄 READING PDF DATA", TERMINAL_MAIN_COLOR))
    read_pdfs("PDFs-predict")

    # - PREDICT XLSX CATEGORIES
    print(colored(f"\n📄 PREDICTING XLSX CATEGORIES", TERMINAL_MAIN_COLOR))
    predictions = predict_categories(clf, vectorizer, new_pdfs)

    # ---- UPDATE XLSX FILE
    print(colored(f"\n📄 UPDATING XLSX FILE WITH PREDICTIONS", TERMINAL_MAIN_COLOR))
    update_xlsx_with_predictions("sample_data.xlsx", predictions)

    # - SCRIPT COMPLETE
    print(colored(f"\n📄 SCRIPT (predict mode) COMPLETED SUCCESSFULLY ✅", TERMINAL_MAIN_COLOR))
#endregion - PREDICT MODE

else:
    print(f"SCRIPT_MODE set to <{SCRIPT_MODE}> does not match an existing script mode (use one of these: TRAIN_TEST, FINAL_TRAIN, PREDICT)")
#endregion - MAIN