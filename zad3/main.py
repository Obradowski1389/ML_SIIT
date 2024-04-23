import json, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


stop_words = ['a', 'avaj', 'ako', 'al', 'ali', 'arh', 'au', 'ah', 'aha', 'aj', 'bar', 'bi', 'bila', 'bili', 'bilo', 'bismo', 'biste', 'bih', 'bijasmo', 'bijaste', 'bijah', 'bijahu', 'bijaše', 'biće', 'blizu', 'broj', 'brr', 'bude', 'budimo', 'budite', 'budu', 'budući', 'bum', 'buć', 'vam', 'vama', 'vas', 'vaša', 'vaše', 'vašim', 'vašima', 'valjda', 'veoma', 'verovatno', 'već', 'većina', 'vi', 'video', 'više', 'vrlo', 'vrh', 'ga', 'gde', 'gic', 'god', 'gore', 'gđekoje', 'da', 'dakle', 'dana', 'danas', 'daj', 'dva', 'de', 'deder', 'delimice', 'delimično', 'dem', 'do', 'dobar', 'dobiti', 'dovečer', 'dokle', 'dole', 'donekle', 'dosad', 'doskoro', 'dotad', 'dotle', 'došao', 'doći', 'drugamo', 'drugde', 'drugi', 'e', 'evo', 'eno', 'eto', 'eh', 'ehe', 'ej', 'želela', 'želele', 'želeli', 'želelo', 'želeh', 'želeći', 'želi', 'za', 'zaista', 'zar', 'zatim', 'zato', 'zahvaliti', 'zašto', 'zbilja', 'zimus', 'znati', 'zum', 'i', 'ide', 'iz', 'izvan', 'izvoli', 'između', 'iznad', 'ikada', 'ikakav', 'ikakva', 'ikakve', 'ikakvi', 'ikakvim', 'ikakvima', 'ikakvih', 'ikakvo', 'ikakvog', 'ikakvoga', 'ikakvom', 'ikakvome', 'ikakvoj', 'ili', 'im', 'ima', 'imam', 'imao', 'ispod', 'ih', 'iju', 'ići', 'kad', 'kada', 'koga', 'kojekakav', 'kojima', 'koju', 'krišom', 'lani', 'li', 'mali', 'manji', 'me', 'mene', 'meni', 'mi', 'mimo', 'misli', 'mnogo', 'mogu', 'mora', 'morao', 'moj', 'moja', 'moje', 'moji', 'moju', 'moći', 'mu', 'na', 'nad', 'nakon', 'nam', 'nama', 'nas', 'naša', 'naše', 'našeg', 'naši', 'naći', 'ne', 'negde', 'neka', 'nekad', 'neke', 'nekog', 'neku', 'nema', 'nemam', 'neko', 'neće', 'nećemo', 'nećete', 'nećeš', 'neću', 'ni', 'nikada', 'nikoga', 'nikoje', 'nikoji', 'nikoju', 'nisam', 'nisi', 'niste', 'nisu', 'ništa', 'nijedan', 'no', 'o', 'ova', 'ovako', 'ovamo', 'ovaj', 'ovde', 'ove', 'ovim', 'ovima', 'ovo', 'ovoj', 'od', 'odmah', 'oko', 'okolo', 'on', 'onaj', 'one', 'onim', 'onima', 'onom', 'onoj', 'onu', 'osim', 'ostali', 'otišao', 'pa', 'pak', 'pitati', 'po', 'povodom', 'pod', 'podalje', 'poželjan', 'poželjna', 'poizdalje', 'poimence', 'ponekad', 'popreko', 'pored', 'posle', 'potaman', 'potrbuške', 'pouzdano', 'početak', 'pojedini', 'praviti', 'prvi', 'preko', 'prema', 'prije', 'put', 'pljus', 'radije', 's', 'sa', 'sav', 'sada', 'sam', 'samo', 'sasvim', 'sva', 'svaki', 'svi', 'svim', 'svog', 'svom', 'svoj', 'svoja', 'svoje', 'svoju', 'svu', 'svugde', 'se', 'sebe', 'sebi', 'si', 'smeti', 'smo', 'stvar', 'stvarno', 'ste', 'su', 'sutra', 'ta', 'taèno', 'tako', 'takođe', 
'tamo', 'tvoj', 'tvoja', 'tvoje', 'tvoji', 'tvoju', 'te', 'tebe', 'tebi', 'ti', 'tima', 'to', 'tome', 'toj', 'tu', 'u', 'uvek', 'uvijek', 'uz', 'uza', 'uzalud', 'uzduž', 'uzeti', 'umalo', 'unutra', 'upotrebiti', 'uprkos', 'učinio', 'učiniti', 'halo', 'hvala', 'hej', 'hm', 'hop', 'hoće', 'hoćemo', 'hoćete', 'hoćeš', 'hoću', 'htedoste', 'htedoh', 'htedoše', 'htela', 'htele', 'hteli', 'hteo', 'htejasmo', 'htejaste', 'htejahu', 'hura', 'često', 'čijem', 'čiji', 'čijim', 'čijima', 'šic', 'štagod', 
'što', 'štogod', 'ja', 'je', 'jedan', 'jedini', 'jedna', 'jedne', 'jedni', 'jedno', 'jednom', 'jer', 'jesam', 'jesi', 'jesmo', 'jesu', 'jim', 'joj', 'ju', 'juče', 'njegova', 'njegovo', 'njezin', 'njezina', 'njezino', 'njemu', 'njen', 'njim', 'njima', 'njihova', 'njihovo', 'njoj', 'nju', 'će', 'ćemo', 'ćete', 'ćeš', 'ću']    


def load_data(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    
    tokens = [word for word in tokens if word not in stop_words]
    
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text
    

def preprocess_and_split(train_data, test_data):
    train_texts = [preprocess_text(sample['strofa']) for sample in train_data]
    test_texts = [preprocess_text(sample['strofa']) for sample in test_data]
    
    tfidf_vectorizer = TfidfVectorizer()

    tfidf_train_matrix = tfidf_vectorizer.fit_transform(train_texts)
    tfidf_test_matrix = tfidf_vectorizer.transform(test_texts)

    print("Dimenzije TF-IDF matrice trening podataka:", tfidf_train_matrix.shape)
    print("Dimenzije TF-IDF matrice test podataka:", tfidf_test_matrix.shape)
    return tfidf_train_matrix, tfidf_test_matrix, [sample['zanr'] for sample in train_data], [sample['zanr'] for sample in test_data]
    
    # vectorizer = CountVectorizer()
    # X_train = vectorizer.fit_transform(train_texts)
    # X_test = vectorizer.transform(test_texts)
    # return X_train, X_test, [sample['zanr'] for sample in train_data], [sample['zanr'] for sample in test_data]


def main(train_path, test_path):
    train_data = load_data(train_path)
    test_data = load_data(test_path)

    X_train, X_test, y_train, y_test = preprocess_and_split(train_data, test_data)
    svm_classifier = SVC(kernel='linear', random_state=42)

    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)

    print(classification_report(y_test, y_pred))

    micro_f1 = f1_score(y_test, y_pred, average='micro')
    print("Mikro F1 mera:", micro_f1)
    

if __name__ == "__main__":
    main("train.json", "test.json")
