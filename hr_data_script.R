#Set Directory and load data
setwd("/home/hduser/Desktop/Data Science Project/")
#load the data into R
hr_data <- read.csv("338_cert_proj_datasets_v3.0.csv")
hr_data <- hr_data[,c(1,2,3,4,5,6,8,9,10,7)]
hr_data$left <- ifelse(hr_data$left>0,"Yes","No")
View(hr_data)

#Find the correlation values of the attributes of our data
#selecting columns with numeric values
hr_corr <- hr_data[,c(1,2,3,4,5,6,7)]
str(hr_corr)
cr <- cor(hr_corr)
library(corrplot)
corrplot(cr, type = "lower")
corrplot(cr, method = "number")
#average_montly_hours and number_project are highly correlated in the dataset (0.42)

#Divide dataset into left employees & non left employees
library("dplyr")
hr_left <- filter(hr_data,left=="Yes")
hr_non_left <- filter(hr_data,left=="No")

#Visualize the characteristics of the whole data and only the people who left, use plots
#and histograms
#Satisfaction Level
hist(hr_data$satisfaction_level)
hist(hr_left$satisfaction_level)

#Last Evaluation
hist(hr_data$last_evaluation)
hist(hr_left$last_evaluation)

#Number of Projects
hist(hr_data$number_project)
hist(hr_left$number_project)

#Average Monthly Hours
hist(hr_data$average_montly_hours)
hist(hr_left$average_montly_hours)

#Time Spend
hist(hr_data$time_spend_company)
hist(hr_left$time_spend_company)

#Work Accident
hist(hr_data$Work_accident)
hist(hr_left$Work_accident)

#Promotion Last 5 years
hist(hr_data$promotion_last_5years)
hist(hr_left$promotion_last_5years)

#Evaluate the values of each attributes for both left and non-left employees
#Salary - employees who left were not being paid much
plot(table(hr_left$salary))
plot(table(hr_non_left$salary))

#Satisfaction Level - many employees who left had satisfaction level very low
plot(table(hr_left$satisfaction_level))
plot(table(hr_non_left$satisfaction_level))

#Number Of Projects - maximum people who left had only two projects
plot(table(hr_left$number_project))
plot(table(hr_non_left$number_project))

#Last Evaluation - few employees who didnt leave have were given last evaluation near to 0  
plot(table(hr_left$last_evaluation))
plot(table(hr_non_left$last_evaluation))

#Average Monthly Hours - Employees who left had less average monthly hours
plot(table(hr_left$average_montly_hours))
plot(table(hr_non_left$average_montly_hours))

#Time Spent in Company - after employed for 6 years, employees have left, maximum(3 years)
plot(table(hr_left$time_spend_company))
plot(table(hr_non_left$time_spend_company))

#Work Accident - Employees who didnt leave have more work accidents
plot(table(hr_left$Work_accident))
plot(table(hr_non_left$Work_accident))

#Department - Employees from sales department are higher have more likely left the company. 
plot(table(hr_left$department))
plot(table(hr_non_left$department))

#Promotion in last 5 years - Most employees have not been promoted in last 5 years
plot(table(hr_left$promotion_last_5years))
plot(table(hr_non_left$promotion_last_5years))

#Analyse the department wise turnouts and find out the percentage of employees
#leaving from each department
library(data.table)
setDT(hr_left)[ , 100 * .N / nrow( hr_left ), by = department ]

#converting columns back to factors for creating models
hr_data$left = factor(hr_data$left)
hr_data$Work_accident = factor(hr_data$Work_accident)
hr_data$promotion_last_5years = factor(hr_data$promotion_last_5years)
str(hr_data)

#divide the data into Training and Testing datasets
set.seed(2)
id<-sample(2,nrow(hr_data),prob = c(0.7,0.3),replace = TRUE)
hr_data_train<-hr_data[id==1,]
hr_data_test<-hr_data[id==2,]

#Building Decision Tree
library(rpart)
colnames(hr_data)
hr_data_model <- rpart(left ~ .,data = hr_data_train)
hr_data_model_less <- rpart(left ~ satisfaction_level+last_evaluation+average_montly_hours+number_project+Work_accident+time_spend_company,data = hr_data_train)
hr_data_model

#We can plot it as
plot(hr_data_model,margin = 0.1)

#margin is used to adjust the size of the plot, For viewing labels
text(hr_data_model,use.n = TRUE,pretty = TRUE,cex =0.8)

#create subset and verify
temp<-hr_data_train[hr_data_train$satisfaction_level>0.5 & hr_data_train$number_project>2,]
table(temp$left)

#Prediction of test dataset
pred_hr<-predict(hr_data_model_less,newdata = hr_data_test,type = "class")
head(pred_hr)

#Now we need to compare it with actual values
table(pred_hr,hr_data_test$left)

#For creating confusion matrix we can use the following
library(caret)
confusionMatrix(table(pred_hr,hr_data_test$left))


#Random Forest
library(randomForest)
hr_forest<-randomForest(left ~ .,data = hr_data_train, ntree = 1000)
hr_forest_less <-randomForest(left ~ satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+department,data = hr_data_train, ntree = 10)
print(hr_forest)

#Prediction of Test set
pred1_hr<-predict(hr_forest,newdata = hr_data_test,type = "class")
pred2_hr<-predict(hr_forest_less,newdata = hr_data_test,type = "class")
pred1_hr
table(pred1_hr,hr_data_test$left)

#confusion matrix
library(caret)
confusionMatrix(table(pred1_hr,hr_data_test$left))
importance(hr_forest)

#NaiveBayes
library(e1071)
hr_naive<- naiveBayes(left ~ .,data = hr_data_train)
hr_naive_1 <- naiveBayes(left ~ satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+department,data = hr_data_train)
hr_naive

#Prediction of Test set
pred2_hr <- predict(hr_naive,newdata = hr_data_test,type = "class")
head(pred2_hr)
table(hr_data_test$left, pred2_hr)
pred1_hr <- predict(hr_naive_1,newdata = hr_data_test,type = "class")
head(pred1_hr)

#Confusion matrix
library(caret)
confusionMatrix(table(pred2_hr,hr_data_test$left))
confusionMatrix(table(pred1_hr,hr_data_test$left))


#SVM
svm_model <- svm(left ~ .,data = hr_data_train,kernel = "linear",cost = 0.1,scale = F)
svm_model_less <- svm(left ~ satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+department,data = hr_data_train,kernel = "linear",cost = 0.1,scale = F)
pred3_hr <- predict(svm_model,newdata = hr_data_test,type = "class")
table(hr_data_test$left, pred3_hr)

#Confusion matrix
library(caret)
confusionMatrix(table(pred3_hr,hr_data_test$left))


#Accuracy
#SVM - 78.72%
#Random Forest - 98.70%
#Naive Bayes - 85.25%
#Decision Tree - 96.87%
#Best Model according to the accuracy is Random Forest