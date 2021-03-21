library(nnet)

setwd("D://R-4.0.2//ExcelWorks")
dff_train <- read.csv(file = "Utrain.csv", header=T)
dff_test <- read.csv(file = "Utest.csv", header = T)

head(dff_train)
dim(dff_train)
dff_train <- dff_train[,-1]
dff_test <- dff_test[,-1]

head(table(dff_train$source_ip))
head(table(dff_test$source_ip))

t1 <- tapply(dff_train$attack_cat, dff_train$attack_cat)
t2 <- tapply(dff_test$attack_cat, dff_test$attack_cat)

table(t1)
table(dff_train$attack_cat)
table(t2)
table(dff_test$attack_cat)

Source_train <- tapply(dff_train$source_ip, dff_train$source_ip)
Dest_train <- tapply(dff_train$dest_ip, dff_train$dest_ip)
Proto_train <- tapply(dff_train$proto, dff_train$proto)

Source_test <- tapply(dff_test$source_ip, dff_test$source_ip)
Dest_test <- tapply(dff_test$dest_ip, dff_test$dest_ip)
Proto_test <- tapply(dff_test$proto, dff_test$proto)

table(Source_train)
table(dff_train$source_ip)

table(Source_test)
table(dff_test$source_ip)

V1 <- vector(len = length(t1))
for(i in 1:length(V1)){
      if(t1[i]==1) 
      { V1[i]<- 1 }  
    }

V2 <- vector(len = length(t2))
for(i in 1:length(V2)){
      if(t2[i]==1) 
      { V2[i]<- 1 }  
    }

unique(V1)
head(V2, 25)
head(t2, 25)

train <- as.data.frame(cbind(Source_train, Dest_train, Proto_train, V1))
test <- as.data.frame(cbind(Source_test, Dest_test, Proto_test, V2))

colnames(test) <- c("Source_train", "Dest_train", "Proto_train", "V1")

dim(train)
dim(test)

summary(train)
summary(test)

MyNN <- nnet(V1 ~ Source_train + Dest_train + Proto_train, data=train, size=3, na.action=na.omit)

MyPred <- predict(MyNN, newdata=test, type = "raw")

table(MyPred, test$V1)

TS <- table(MyPred>0.5, test$V1)
TS

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
a<- accuracy(TS)
pa<- paste("The accuracy of our prediction using only the source, dest and protocol features is", a, sep=" ")

# print(pa)

dfs_train <- data.frame(nrow=nrow(dff_train))
for(i in 1:ncol(dff_train)) {
    if(class(dff_train[,i])=='numeric' | class(dff_train[,i])=="integer") {
        dfs_train <- cbind(dfs_train, dff_train[,i])
    }
}

dfs_test <- data.frame(nrow=nrow(dff_test))
for(i in 1:ncol(dff_test)) {
    if(class(dff_test[,i])=='numeric' | class(dff_test[,i])=="integer") {
        dfs_test <- cbind(dfs_test, dff_test[,i])
    }
}

head(dfs_train, 3)
head(dfs_test, 3)

dfs_test <- dfs_test[, -1]

dfs_train <- dfs_train[, -1]
colnames(dfs_train) <- c(paste0("Feature", c(1:ncol(dfs_train))))
dfs_test <- dfs_test[, -1]
colnames(dfs_test) <- c(paste0("Feature", c(1:ncol(dfs_test))))

dim(dfs_train)
dim(dfs_test)
names(dfs_train)
names(dfs_test)

train <- as.data.frame(cbind(train, dfs_train))
test <- as.data.frame(cbind(test, dfs_test))

head(train, 3)
head(test, 3)
dim(train)
dim(test)

q<- vector(len = 35)
for(i in 1:35) {
	LastNN <- nnet(V1 ~ ., data=train, size=i, na.action=na.omit, maxit= 80, MaxNWts = 3000)
	lp <- predict(LastNN, test)
	t<- table(lp>0.5, test$V1)
	q[i] <- accuracy(t) }


summary(q)
pq <- paste("The best overall accuracy we got is", max(q), sep = " ")


print(pq)