#####################################
# Load libraries
# Load your vocabulary and training data
#####################################
library(text2vec)
library(glmnet)
myvocab <- scan(file = "myvocab.txt", what = character())

train = read.table("train.tsv",
                    stringsAsFactors = FALSE,
                    header = TRUE)

train$review <- gsub('<.*?>', ' ', train$review)

#####################################
# Train a binary classification model
#####################################
# Create matrix corresponding to vocab
it_train = itoken(train$review,
                    preprocessor = tolower,
                    tokenizer = word_tokenizer)
vectorizer = vocab_vectorizer(create_vocabulary(myvocab,
                                                  ngram = c(1L, 2L)))
dtm_train = create_dtm(it_train, vectorizer)

set.seed(7568)
fit = glmnet(x = dtm_train,
                y = train$sentiment,
                alpha = 0.05,
                family='binomial')

#####################################
# Load test data, and
# Compute prediction
#####################################
test <- read.table("test.tsv", stringsAsFactors = FALSE,
                header = TRUE)

test$review <- gsub('<.*?>', ' ', test$review)

# Create matrix corresponding to vocab
it_test = itoken(test$review,
                    preprocessor = tolower,
                    tokenizer = word_tokenizer)
vectorizer = vocab_vectorizer(create_vocabulary(myvocab,
                                            ngram = c(1L, 2L)))
dtm_test = create_dtm(it_test, vectorizer)

prob = predict(fit, newx=dtm_test, s=0.05)
output = data.frame(id=c(test$id),
                    prob=c(prob))

#####################################
# Store your prediction for test data in a data frame
# "output": col 1 is test$id
#           col 2 is the predicted probs
#####################################
write.table(output, file = "mysubmission.txt",
            row.names = FALSE, sep='\t')
