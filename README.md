# stat4630-final
machine learning analysis of banking clients to determine likelihood of default

        '''
        library(ROCR)
        setwd("Documents/STAT ML F19")
        credit_default <- read.csv("default of credit card clients.csv")
        attach(credit_default)

        hist(BILL_AMT1)
        hist(BILL_AMT2)
        hist(BILL_AMT3)
        hist(BILL_AMT4)
        hist(BILL_AMT5)
        hist(BILL_AMT6)


        hist(PAY_AMT1)
        hist(PAY_AMT2)
        hist(PAY_AMT3)
        hist(PAY_AMT4)
        hist(PAY_AMT5)
        hist(PAY_AMT6)


        hist(AGE)
        hist(LIMIT_BAL)

        data<-read.csv("wcgs.csv", header=TRUE)
        attach(data)
        hist(ncigs)

        ##create boxplots to see if predictors have different distributions based on presence of heart disease
        par(mfrow=c(3,2))


        boxplot(BILL_AMT1~default.payment.next.month, main="Bill_AMT1 against Default Next Payment")
        boxplot(BILL_AMT2~default.payment.next.month, main="Bill_AMT2 against Default Next Payment")
        boxplot(BILL_AMT3~default.payment.next.month, main="Bill_AMT3 against Default Next Payment")
        boxplot(BILL_AMT4~default.payment.next.month, main="Bill_AMT4 against Default Next Payment")
        boxplot(BILL_AMT5~default.payment.next.month, main="Bill_AMT5 against Default Next Payment")
        boxplot(BILL_AMT6~default.payment.next.month, main="Bill_AMT6 against Default Next Payment")


        par(mfrow=c(3,2))


        boxplot(PAY_AMT1~default.payment.next.month, main="PAY_AMT1 against Default Next Payment")
        boxplot(PAY_AMT2~default.payment.next.month, main="PAY_AMT2 against Default Next Payment")
        boxplot(PAY_AMT3~default.payment.next.month, main="PAY_AMT3 against Default Next Payment")
        boxplot(PAY_AMT4~default.payment.next.month, main="PAY_AMT4 against Default Next Payment")
        boxplot(PAY_AMT5~default.payment.next.month, main="PAY_AMT5 against Default Next Payment")
        boxplot(PAY_AMT6~default.payment.next.month, main="PAY_AMT6 against Default Next Payment")

        boxplot(AGE~default.payment.next.month, main="AGE against Default Next Payment")

        boxplot(LIMIT_BAL~default.payment.next.month, main="Limit Balance against Default Next Payment")
        ### DATA CLEANING 
        credit_default <- credit_default[which(credit_default$EDUCATION %in% c(1, 2, 3, 4)), ]
        credit_default <- credit_default[which(credit_default$MARRIAGE != 0), ]
        credit_default <- credit_default[, -1]

        ### Setting categorical variables as factors
        credit_default[, c(2:4, 6:11, 24)] <- lapply(credit_default[, c(2:4, 6:11, 24)], factor)
        names(credit_default)[15]="Y"
        str(credit_default)

        # selecting all quantitative variables
        credit_default <- credit_default[,c(1,5,12:ncol(credit_default))]


        ### LOGIT MODEL
        logit_mod_complex <- glm(Y ~ ., data = credit_default, family = binomial)

        # logit_mod <- glm(Y ~ AGE+LIMIT_BAL, data = credit_default, family = binomial)

        # logit_mod3 <- glm(Y ~ AGE+LIMIT_BAL+BILL_AMT6+BILL_AMT5+BILL_AMT4+PAY_AMT4+PAY_AMT6+PAY_AMT5, data = credit_default, family = binomial)


        # splitting our data into train & test set
        set.seed(pi)
        data <- credit_default
        sample.data<-sample.int(nrow(data), floor(.50*nrow(data)), replace = F)
        train<-data[sample.data, ]
        test<-data[-sample.data, ]


        # ROC curve for logistic 
        preds <- predict(logit_mod_complex, newdata = test, type = "response")
        rates <- prediction(preds, test$Y)
        roc_result <- performance(rates, measure = "tpr", x.measure = "fpr")
        plot(roc_result, main = "ROC Curve: Logistic Regression (AUROC = 0.66)")
        lines(x = c(0,1), y = c(0,1), col="red")

        performance(rates, "auc") # AUROC 


        ### LDA MODEL 
        lda.cdef <- lda(Y ~ ., data = train)
        # lda.cdef2 <- lda(Y ~ log(LIMIT_BAL)+log(AGE))

        # ROC curve LDA
        preds <- predict(lda.cdef2,test)$posterior[,2]
        rates <- prediction(preds, test$Y)
        roc_result <- performance(rates, measure = "tpr", x.measure = "fpr")
        plot(roc_result, main = "ROC Curve: LD Model (AUROC = 0.64)")
        lines(x = c(0,1), y = c(0,1), col="red")

        performance(rates, "auc") # AUROC 


        # Logistic: 5 and 10 fold CV test error
        set.seed(pi)
        cv.glm(credit_default, logit_mod_complex, K = 5)$delta
        # 0.1649347 0.1649301
        cv.glm(credit_default, logit_mod_complex, K = 10)$delta
        # 0.1649344 0.1649300


        # LDA: 5 and 10 fold CV test error
        cv.da <- function(object, newdata)
        {
          return(predict(object, newdata = newdata)$class)
        }
        errorest(Y ~ ., data = credit_default, model = lda,
                 estimator = "cv", est.para = control.errorest(k=5),
                 predict = cv.da)$err
        # 0.2231344
        errorest(Y ~ ., data = credit_default, model = lda,
                 estimator = "cv", est.para = control.errorest(k=10),
                 predict = cv.da)$err
        # 0.2231344


        ### LOGISTIC WITH CATEGORICAL 
        # setting up dataset with categorical variables
        # reran original data cleaning code
        credit_default <- credit_default[,c(1:5,12:ncol(credit_default))]
        names(credit_default)[ncol(credit_default)]="Y"

        logit_mod_wcat <- glm(Y ~ ., data = credit_default, family = binomial)
        # summary(logit_mod_wcat)


        # ROC 
        preds <- predict(logit_mod_wcat, newdata = test, type = "response")
        rates <- prediction(preds, test$Y)
        roc_result <- performance(rates, measure = "tpr", x.measure = "fpr")
        plot(roc_result, main = "ROC Curve: Enhanced Logistic (AUROC = 0.62)")
        lines(x = c(0,1), y = c(0,1), col="red")
        performance(rates, "auc")

        # CV 
        set.seed(pi)
        cv.glm(credit_default, logit_mod_wcat, K = 5)$delta
        # 0.1645125 0.1644965
        cv.glm(credit_default, logit_mod_wcat, K = 10)$delta
        # 0.1645197 0.1645098

        setwd("C:/Users/firem/Desktop/University of Virginia/fall19/STAT 4630/project")
        credit_default <- read.csv("credit.csv",header=TRUE, sep=",")

        credit_default <- credit_default[which(credit_default$EDUCATION %in% c(1, 2, 3, 4)), ]
        credit_default <- credit_default[which(credit_default$MARRIAGE != 0), ]
        credit_default <- credit_default[, -1]

        credit_default[, c(2:4, 6:11, 24)] <- lapply(credit_default[, c(2:4, 6:11, 24)], factor)
        str(credit_default)

        attach(credit_default)

        library(MASS)
        library(tree) ##to fit trees
        library(randomForest) ##for random forests (and bagging)
        library(gbm)

        #regression
        set.seed(1)
        sample.data<-sample.int(nrow(credit_default), floor(.50*nrow(credit_default)), replace = F)
        train<-credit_default[sample.data, ]
        test<-credit_default[-sample.data, ]

        tree.class.train<-tree(LIMIT_BAL~default.payment.next.month+SEX+EDUCATION+MARRIAGE+PAY_0+PAY_2+PAY_3+PAY_4+PAY_5+PAY_6+BILL_AMT1+BILL_AMT2+BILL_AMT3+BILL_AMT4+BILL_AMT5+BILL_AMT6+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6, data=train)
        plot(tree.class.train)
        text(tree.class.train, cex=0.75, pretty=0)

        summary(tree.class.train)

        # pruning

        set.seed(11)
        cv.credit<-cv.tree(tree.class.train, K=10)
        cv.credit

        plot(cv.credit$size, cv.credit$dev, type='b', ylab="Deviance", xlab="Size")

        trees.num<-cv.credit$size[which.min(cv.credit$dev)]
        print(trees.num)

        prune.credit<- prune.tree(tree.class.train, best = trees.num)
        plot(prune.credit)
        text(prune.credit, cex = 0.75, pretty = 0)

        yhat<-predict(prune.credit, newdata=test)
        credit.test<-test[,"LIMIT_BAL"]
        mse.tree<-mean((credit.test-yhat)^2)
        print(mse.tree)


        #classification 
        tree.class.train<-tree(default.payment.next.month~LIMIT_BAL+SEX+EDUCATION+MARRIAGE+PAY_0+PAY_2+PAY_3+PAY_4+PAY_5+PAY_6+BILL_AMT1+BILL_AMT2+BILL_AMT3+BILL_AMT4+BILL_AMT5+BILL_AMT6+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6)
        plot(tree.class.train)
        text(tree.class.train, cex=0.75, pretty=0)
        summary(tree.class.train)
        
        '''
