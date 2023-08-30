### AUTHOR - Ahn Michael
### GOAL - For internship in UA American Speech project, train a model to classify academic articles (PDFs)
###        into pre-defined categories, based on their texts, then output predictions to an XLSX

#region - OPTIONS
SCRIPT_MODE = "TRAIN" # which mode the script runs in; either TRAIN or PREDICT
# for TRAIN mode, need to have an XLSX with PDF metadata and the PDFs to train on
# for PREDICT mode, need to have an XLSX with PDFs you want to predict the categories of, and those PDFs in a folder
TERMINAL_MAIN_COLOR = 'blue' # which color termcolor will use when outputting main script logs
#endregion - OPTIONS

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
os.system('clear') # clear the terminal
#endregion - IMPORTS & INITIALISATION

#region - COLLECTIONS
OVERALL_CATEGORIES = ['UNKNOWN', 'CAT_0', 'CAT_1', 'CAT_2', 'CAT_2_1', 'CAT_3', 'CAT_3_1', 'CAT_4', 'CAT_4_1', 'CAT_5', 'CAT_5_1', 'CAT_6', 'CAT_7']

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
    XLS_DF[OVERALL_CATEGORIES] = XLS_DF[OVERALL_CATEGORIES].fillna(0)
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

def get_present_category_list(single_xls_row):
    category_list = []
    for category in OVERALL_CATEGORIES:
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

    print(f">> Completed making {len(ALL_DOCS)} document objects - two quick preview documents:\n\n{ALL_DOCS[0].to_string()}\n{ALL_DOCS[-1].to_string()}\n")

def simple_split_test_train():
    TRAIN_DOCS, TEST_DOCS = train_test_split(ALL_DOCS, test_size=0.2)
    print(f">> After splitting, there are {len(TRAIN_DOCS)} training documents and {len(TEST_DOCS)} test documents -- {len(ALL_DOCS)} documents total\n")

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
print(colored(f"\n📄 ** SCRIPT START in TRAIN mode **", TERMINAL_MAIN_COLOR))
if SCRIPT_MODE.strip() == "TRAIN":

    # - READ IN DATA
    print(colored(f"\n📄 READING IN DATA", TERMINAL_MAIN_COLOR))

    # ---- READ IN EXCEL DATA
    # QUESTION: how should the excel data (the spreadsheet rows/columns) best be stored for the model training - dataframe, list of rows, actual dictionary?
    print(colored(f"\n📄 READING EXCEL DATA", TERMINAL_MAIN_COLOR))
    read_xlsx("american-speech-dataset-complete-rows.xlsx")

    # ---- READ IN PDF DATA
    print(colored(f"\n📄 READING PDF DATA", TERMINAL_MAIN_COLOR))
    read_pdfs("PDFs-train")

    # ---- CONSTRUCT DOCUMENT OBJECTS
    print(colored(f"\n📄 CONSTRUCTING DOCUMENT OBJECTS", TERMINAL_MAIN_COLOR))
    make_documents()

    # - SPLIT DATA (Stratified K-Fold, into training and testing sets)
    # NOTE: Implementation is using the common method of using a single primary category for stratification;
    # especially useful when the categories are imbalanced, to ensure training and test sets have similar distributions
    # of this primary category; does not ensure stratification across all categories, which can be more complex
    print(colored(f"\n📄 SPLITTING DATA (TRAIN-TEST SPLIT)", TERMINAL_MAIN_COLOR))
    simple_split_test_train()
    exit()

    # skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=13)
    # X = [pdf for pdf in PDF_TEXTS_dict.keys()] # list of the pdf names
    # y = [record['CAT_1'] for record in XLS_ROWS]  # Using CAT_1for stratification
    # train_index, test_index = next(skf.split(X, y))
    # TRAIN_DOCS = [X[i] for i in train_index]
    # TEST_DOCS = [X[i] for i in test_index]

    # - MAKE VECTORISER (and vectorise training documents)
    print(colored(f"\n📄 MAKING and FIT-TRANSFORMING TF-IDF VECTORISER", TERMINAL_MAIN_COLOR))
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
    print(colored(f"\n📄 ENCODING LABELS with MLB", TERMINAL_MAIN_COLOR))
    # MLB replaces my previous set_golds_from_labels function, previously in the document class
    mlb = MultiLabelBinarizer(classes=OVERALL_CATEGORIES)
    mlb.fit(OVERALL_CATEGORIES)
    # ---- TRANSFORM LABELS for each document
    #! for each train_doc in train_docs:
    # we pass .transform a list (even though it only has one item) because it expects a list, 
    # since it usually works on multiple documents at one time; it outputs a matrix (with only one row, 
    # since one document), and we get that first (only) document
    #! train_doc.y_golds = mlb.transform([document.labels])[0] 


    # - TRAIN CLASSIFIER (fit the model on the training data: model.fit(X_train, y_train))
    print(colored(f"\n📄 TRAINING CLASSIFIER", TERMINAL_MAIN_COLOR))
    # OneVsRestClassifier - is a fit for this task, as it trains a separate binary classifier
    # for each category, as I imagined early on
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
    clf.fit(X_train, y_train)

    # - REVIEW PERFORMANCE (Classification Report)
    print(colored(f"\n📄 REVIEWING PERFORMANCE", TERMINAL_MAIN_COLOR))
    # Vectorise the TEST_DOCS and evaluate the model
    test_texts = [doc.pdf_text for doc in ALL_DOCS if doc.pdf_name in TEST_DOCS]
    X_test = vectorizer.transform(test_texts)
    y_test = [[1 if cat in doc.categories else 0 for cat in categories] for doc in ALL_DOCS if doc.pdf_name in TEST_DOCS]
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=categories, zero_division=1))

    # - SAVE MODEL
    print(colored(f"\n📄 SAVING MODEL", TERMINAL_MAIN_COLOR))
    save_model(clf, "model.joblib")
    save_model(vectorizer, "vectorizer.joblib")

    # - SCRIPT COMPLETE
    print(colored(f"\n📄 SCRIPT (train mode) COMPLETED SUCCESSFULLY ✅", TERMINAL_MAIN_COLOR))
#endregion - TRAIN MODE

#region - PREDICT MODE
else:
    print(colored(f"\n📄 ** SCRIPT START in PREDICT mode **\n", TERMINAL_MAIN_COLOR))

    # - LOAD MODEL & VECTORISER DATA
    print(colored(f"\n📄 LOADING MODEL & VECTORISER DATA", TERMINAL_MAIN_COLOR))
    clf = load_model("model.joblib")
    vectorizer = load_model("vectorizer.joblib")

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
#endregion - MAIN