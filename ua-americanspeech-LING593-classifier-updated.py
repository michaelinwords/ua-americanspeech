### AUTHOR - Ahn Michael
### GOAL - For internship in UA American Speech project, train a model to classify academic articles (PDFs)
###        into pre-defined categories, based on their texts, then output predictions to an XLSX

#region - OPTIONS
SCRIPT_MODE = "TRAIN" # which mode the script runs in; either TRAIN or PREDICT
# for TRAIN mode, need to have an XLSX with PDF metadata and the PDFs to train on
# for PREDICT mode, need to have an XLSX with PDFs you want to predict the categories of, and those PDFs in a folder
#endregion - OPTIONS

#region - IMPORTS & INITIALISATION
import os       # for navigating the file system
import re       # mainly for text preprocessing / normalisation
import PyPDF2       # for reading PDF files
import pandas as pd     # for handling excel files and storing them as dataframes
from sklearn.feature_extraction.text import TfidfVectorizer     # for transforming text into feature vectors
from sklearn.model_selection import StratifiedKFold     # for stratified k-fold splitting (separating train and test data)
from sklearn.preprocessing import MultiLabelBinarizer       # for encoding labels / converting list of present categories (["CAT-A", "CAT-D"]) to binary format ([1, 0, 0, 1])
from sklearn.linear_model import LogisticRegression     # includes the logistic regression model
from sklearn.multiclass import OneVsRestClassifier      # for handling multi-label classification
from sklearn.metrics import classification_report, f1_score, accuracy_score     # for evaluating the model's performance
from joblib import dump, load      # for saving and loading model, vectorizer, and label binarizer (to avoid retraining every time we want to make predictions)
os.system('clear') # clear the terminal
#endregion - IMPORTS & INITIALISATION

#region - COLLECTIONS
OVERALL_LABELS = ['UNKNOWN', 'CAT_0', 'CAT_1', 'CAT_2', 'CAT_2_1', 'CAT_3', 'CAT_3_1', 'CAT_4', 'CAT_4_1', 'CAT_5', 'CAT_5_1', 'CAT_6', 'CAT_7']

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
#endregion - COLLECTIONS

#region - CLASSES
class Document:
    def __init__(self, n_row, n_pdfname, n_pdftext, n_title, n_authors, n_doi, n_year, n_month, n_volume, n_issue, n_labels, n_test_or_train) -> None:
        self.row = n_row # should be an integer, specifically not a string
        self.pdfname = n_pdfname
        self.pdftext = n_pdftext
        self.title = n_title
        self.authors = n_authors
        self.doi = n_doi
        self.year = n_year
        self.month = n_month
        self.volume = n_volume
        self.issue = n_issue
        self.labels = n_labels # a list of strings; the actual categories/labels of the document, coming from the spreadsheet data (example: ["CAT-A", "CAT-D"])
        self.test_or_train = n_test_or_train # whether the document is for testing or training
        self.x_vector = [] # the vectorised representation of pdftext (output from TF-IDF vectorizer)
        self.y_golds = [] # list of correct (1) and incorrect (0) labels, in binary format, from spreadsheet data; should be length 13, which is the # of our categories; get from multilabelbinarizer
        self.y_preds = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] # unless wanting to validate training data, this will remain -1s/unset for a training document; it will be set with real predictions (1s and 0s) for test and prediction documents
#endregion - CLASSES

#region - OTHER FUNCTIONS
# Read in the XLSX file and load its data into global variables
def read_xlsx(file_name):
    global XLS_DF
    global XLS_ROWS
    XLS_FOLDER = "XLSX"

    XLS_DF = pd.read_excel(f"{CWD_PATH}/{XLS_FOLDER}/{file_name}")
    # replace all NaN values in label columns with 0
    XLS_DF = XLS_DF[OVERALL_LABELS].fillna(0)
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
def preprocess_text(some_text):
    preprocessed_text = ""
    preprocessed_text = some_text.lower()  # lowercase
    return preprocessed_text

# Vectorise the documents using TF-IDF
def tfidf_vectorise(docs, preprocessor=None, ngram_range=(1,1), binary=False, analyzer='word', min_df=1):
    vectoriser = TfidfVectorizer(
        preprocessor=preprocessor,
        ngram_range=ngram_range,
        binary=binary,
        analyzer=analyzer,
        min_df=min_df
    )
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
print(f"\n** SCRIPT START in TRAIN mode **")
if SCRIPT_MODE == "TRAIN":

    # - READ IN DATA
    print(f"\n READING IN DATA")

    # ---- READ IN EXCEL DATA
    # QUESTION: how should the excel data (the spreadsheet rows/columns) best be stored for the model training - dataframe, list of rows, actual dictionary?
    print(f"\nREADING EXCEL DATA")
    read_xlsx("american-speech-dataset-complete-rows.xlsx")

    # ---- READ IN PDF DATA
    print(f"\nREADING PDF DATA")
    read_pdfs("PDFs-train")

    # - SPLIT DATA (Stratified K-Fold, into training and testing sets)
    # NOTE: Implementation is using the common method of using a single primary category for stratification;
    # especially useful when the categories are imbalanced, to ensure training and test sets have similar distributions
    # of this primary category; does not ensure stratification across all categories, which can be more complex
    print(f"\nSPLITTING DATA with SKF")
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=13)
    X = [pdf for pdf in PDF_TEXTS_dict.keys()] # list of the pdf names
    y = [record['CAT_1'] for record in XLS_ROWS]  # Using CAT_1for stratification
    train_index, test_index = next(skf.split(X, y))
    TRAIN_DOCS = [X[i] for i in train_index]
    TEST_DOCS = [X[i] for i in test_index]

    # ---- CONSTRUCT DOCUMENT OBJECTS
    print(f"\nCONSTRUCTING DOCUMENT OBJECTS")

    # - MAKE VECTORISER (and vectorise training documents)
    print(f"\nMAKING and FIT-TRANSFORMING TF-IDF VECTORISER")
    # - FIT-TRANSFORM VECTORISER (TF-IDF - vectorizer.fit_transform(training_documents))
    # fit: learns the vocabulary of the entire collection of training documents and calculates term frequencies and document frequencies
    # transform: uses the learned vocabulary and IDF values from fitting to convert each document into a vector
    # output of fit_transform: will be a matrix, where each row is a document from the collection and each column is a word/n-gram from the learned vocabulary

    # Customizable TF-IDF options
    preprocessor = None  # None >> We are already handling preprocessing elsewhere
    ngram_range = (1, 2) # Define ngram range, get n-grams of these lengths
    binary = False # False >> We want counts for each token, not just presence
    analyzer = 'word' # We want word n-grams
    min_df = 2 # 2 >> We want to remove any terms which only occur once

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(preprocessor=preprocessor,
                                 ngram_range=ngram_range,
                                 binary=binary,
                                 analyzer=analyzer,
                                 min_df=min_df)

    # Fit the vectorizer on all texts
    X = vectorizer.fit_transform(ALL_TEXTS)


    # - ENCODE LABELS (MultiLabelBinarizer, into binary format
    print(f"\nENCODING LABELS with MLB")
    # MLB replaces my previous set_golds_from_labels function, previously in the document class
    mlb = MultiLabelBinarizer(classes=OVERALL_LABELS)
    mlb.fit(OVERALL_LABELS)
    # ---- TRANSFORM LABELS for each document
    #! for each train_doc in train_docs:
    # we pass .transform a list (even though it only has one item) because it expects a list, 
    # since it usually works on multiple documents at one time; it outputs a matrix (with only one row, 
    # since one document), and we get that first (only) document
    #! train_doc.y_golds = mlb.transform([document.labels])[0] 


    # - TRAIN CLASSIFIER (fit the model on the training data: model.fit(X_train, y_train))
    print(f"\nTRAINING CLASSIFIER")
    # OneVsRestClassifier - is a fit for this task, as it trains a separate binary classifier
    # for each category, as I imagined early on
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
    clf.fit(X_train, y_train)

    # - REVIEW PERFORMANCE (Classification Report)
    print(f"\nREVIEWING PERFORMANCE")
    # Vectorise the TEST_DOCS and evaluate the model
    test_texts = [doc.pdf_text for doc in ALL_DOCS if doc.pdf_name in TEST_DOCS]
    X_test = vectorizer.transform(test_texts)
    y_test = [[1 if cat in doc.categories else 0 for cat in categories] for doc in ALL_DOCS if doc.pdf_name in TEST_DOCS]
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=categories, zero_division=1))

    # - SAVE MODEL
    print(f"\nSAVING MODEL")
    save_model(clf, "model.joblib")
    save_model(vectorizer, "vectorizer.joblib")

    # - SCRIPT COMPLETE
    print(f"\nSCRIPT (train mode) COMPLETED SUCCESSFULLY ✅")
#endregion - TRAIN MODE

#region - PREDICT MODE
else:
    print(f"\n** SCRIPT START in PREDICT mode **\n")

    # - LOAD MODEL & VECTORISER DATA
    print(f"\nLOADING MODEL & VECTORISER DATA")
    clf = load_model("model.joblib")
    vectorizer = load_model("vectorizer.joblib")

    # - READ IN DATA
    print(f"\nREADING IN DATA")

    # ---- READ IN EXCEL DATA
    print(f"\nREADING EXCEL DATA")
    read_xlsx("american-speech-predictions.xlsx")

    # ---- READ IN PDF DATA
    print(f"\nREADING PDF DATA")
    read_pdfs("PDFs-predict")

    # - PREDICT XLSX CATEGORIES
    print(f"\nPREDICTING XLSX CATEGORIES")
    predictions = predict_categories(clf, vectorizer, new_pdfs)

    # ---- UPDATE XLSX FILE
    print(f"\nUPDATING XLSX FILE WITH PREDICTIONS")
    update_xlsx_with_predictions("sample_data.xlsx", predictions)

    # - SCRIPT COMPLETE
    print(f"\nSCRIPT (predict mode) COMPLETED SUCCESSFULLY ✅")
#endregion - PREDICTMODE
#endregion - MAIN