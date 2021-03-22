set.seed(5)
library(neuralnet)

dff_train <- read.csv(file = "Utrain.csv", header=T)
dff_test <- read.csv(file = "Utest.csv", header = T)

head(dff_train)
dim(dff_train)

dff_train <- dff_train[,-1]
dff_test <- dff_test[,-1]

df_train_clean <- data.frame(nrow=nrow(dff_train))
for(i in 1:ncol(dff_train)) {
    if(class(dff_train[,i])=='numeric' | class(dff_train[,i])=="integer") {
        df_train_clean <- cbind(df_train_clean, dff_train[,i])
    }
}

df_test_clean <- data.frame(nrow=nrow(dff_test))
for(i in 1:ncol(dff_test)) {
    if(class(dff_test[,i])=='numeric' | class(dff_test[,i])=="integer") {
        df_test_clean <- cbind(df_test_clean, dff_test[,i])
    }
}

df_test_clean <- df_test_clean[, -1]

df_train_clean <- df_train_clean[, -1]
colnames(df_train_clean) <- c(paste0("Feature", c(1:ncol(df_train_clean))))
df_test_clean <- df_test_clean[, -1]
colnames(df_test_clean) <- c(paste0("Feature", c(1:ncol(df_test_clean))))

head(df_train_clean, 3)
str(df_train_clean)

df_train_clean <- cbind(df_train_clean, attacks = dff_train$attack_cat)
df_test_clean <- cbind(df_test_clean, attacks = dff_test$attack_cat)


nn = neuralnet(df_train_clean$attacks ~ ., data=df_train_clean, hidden = c(10,10), linear.output=F)     

plot(nn, rep="best", file = "NeuralNet.png")
head(nn$result.matrix,10)

pred <- predict(nn, df_test_clean)
class(pred)

tab <- table(df_test_clean$attacks, apply(pred, 1, which.max))
tab

truth <- table(df_test_clean$attacks)

acc <- function() {
	if(identical(sum(tab[,1]),sum(truth))==TRUE) 
	{	print("We have 100% accuracy in prediction") }
	if(ncol(tab)>1) {
		c2 <- sum(tab[,2])
		c <- sum(tab)
		p<- paste("We have", (1-(c2/c))*100, "accuracy", sep = " ")
		print(p) }
	}

a<- acc()
pa <- paste("Our final accuracy is", a, sep=" ")

print(pa)