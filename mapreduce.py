def mapper(docID, docContent):
    words = docContent.split()
    for word in words:
        yield (word.lower(), docID)

def reducer(word, docIDs):
    uniqueDocIDs = list(set(docIDs))
    yield (word, uniqueDocIDs)

def inverted_index(documents):
    intermediate_data = {}

    for docID, content in documents.items():
        for word, doc in mapper(docID, content):
            if word not in intermediate_data:
                intermediate_data[word] = []
            intermediate_data[word].append(doc)

    inverted_index = {}
    for word, docs in intermediate_data.items():
        for word, unique_docs in reducer(word, docs):
            inverted_index[word] = unique_docs

    return inverted_index

documents = {
    "D1": "the cat sat on the mat",
    "D2": "the dog sat on the log"
}
print(inverted_index(documents))
