import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#I got help from stackoverflow, some youtube videos and r/learnpython on reddit.

#I read where putting "(object)" next to the class is good practice going fwd because of inheritance issues 

#I used some machine learning to figure out what best pred was

class analysisData(object): 

    def __init__(self, filename):
       self.variables = []
       self.filename = filename
       
    def parseFile(self):
        self.dataset = pd.read_csv(self.filename)
        self.variables = self.dataset.columns 
        
dataParser = analysisData('./candy-data.csv') 
dataParser.parseFile()

class LinearAnalysis(object): 

    def __init__(self, targetY): 
    	self.bestX = None
    	self.targetY = targetY
    	self.fit = None

    def runSimpleAnalysis(self, dataParser): 
       
        dataset = dataParser.dataset

        best_pred = 0
        for column in dataParser.variables: 
            if column == self.targetY or column == 'competitorname':
                continue

            x_values = dataset[column].values.reshape(-1,1) 
            y_values = dataset[self.targetY].values

            regr = LinearRegression()
            regr.fit(x_values, y_values)
            preds = regr.predict(x_values) 
            score = r2_score(y_values, preds)

            if score > best_pred:
                best_pred = score
                self.bestX = column
        
        self.fit = best_pred
        print(self.bestX)
        print(self.fit)

linear_analysis = LinearAnalysis(targetY='sugarpercent')
linear_analysis.runSimpleAnalysis(dataParser)