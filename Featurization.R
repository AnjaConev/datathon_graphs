library(text2vec)
library(data.table)
library(jsonlite)
library(magrittr)
library(doc2vec)
library(uwot)
library(dbscan)

graph <- fread("Training/training_graph.csv")

results <- fromJSON("Training/node_features_text.json") 
res_ids <- names(results)
txt_count <- lapply(results, length)
result <- lapply(results, paste0, collapse = " ") %>% data.table()
names(result) <- "text"
result[, doc_id := ids]
result[, nwords := txt_count]

prep_fun <- tolower
tok_fun <- word_tokenizer
train_tokens <- word_tokenizer(result$text)

it_train <- itoken(train_tokens,
                  ids = result$doc_id,
                  progressbar = TRUE)
vocab <- create_vocabulary(it_train)

vectorizer <- vocab_vectorizer(vocab)
t1 <- Sys.time()
dtm_train <- create_dtm(it_train, vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))

# By normalization we assume transformation of the rows of DTM so we adjust values measured on different scales to a notionally common scale.
# For the case when length of the documents vary we can apply L1 normalization. 
# It means we will transform rows in a way that sum of the row values will be equal to 1
dtm_train_l1_norm <- normalize(dtm_train, "l1")

# define tfidf model
tfidf <- TfIdf$new()

# fit model to train data and transform train data with fitted model
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)

dtm_asmatrix <- dtm_train_tfidf %>% as.matrix()
ids <- rownames(dtm_asmatrix)
dtm_asmatrix <- dtm_asmatrix %>% as.data.table()
dtm_asmatrix[, id := ids]
fwrite(dtm_asmatrix, "TF_IDF_features.csv")

## PARAGRAPH/DOC_EMBEDDINGS PART!!!

d2v <- paragraph2vec(result, type = "PV-DBOW", dim = 128, lr = 0.05, iter = 100, window = result[, nwords] %>% unlist %>% max, 
                     hs = TRUE, negative = 0, sample = 0.00001, min_count = 0, threads = 8)

embedding <- as.matrix(d2v, which = "docs") 
doc_vocab <- summary(d2v, which = "docs")
ids <- rownames(embedding)
embedding <- embedding %>% as.data.table()
embedding[, id := ids]
fwrite(embedding, "DOC2Vec_features.csv")
model  <- top2vec(d2v, 
                  control.dbscan = list(minPts = 50), 
                  control.umap = list(n_neighbors = 15L, n_components = 3), umap = umap, 
                  trace = TRUE)
print(model)

## GLOVE PART!!!

tcm = create_tcm(it_train, vectorizer, skip_grams_window = result[, nwords] %>% unlist %>% max)
glove = GlobalVectors$new(rank = 128, x_max = 10)

wv_main = glove$fit_transform(tcm, n_iter = 100, convergence_tol = 0.01, n_threads = 8)
wv_context = glove$components

word_vectors = wv_main + t(wv_context)
ids <- rownames(word_vectors)
word_vectors <- word_vectors %>% as.data.table()
word_vectors[, id := as.integer(ids)]

j <- 1
avg_emb_list <- lapply(results, function(entry){
  len <- length(entry)
  emb <- vector("numeric", 128) # Size of embedding
  for (i in 1:len) {
    index <- entry[i]
    #print(class(index))
    #print(class(word_vectors[, id]))
    emb <- emb + word_vectors[id == index]
  }
  emb <- emb/len
  message(j,"\r",appendLF=FALSE)
  flush.console()
  j <<- j + 1
  emb
})
avg_emb <- rbindlist(avg_emb_list)
avg_emb[, id := res_ids]
