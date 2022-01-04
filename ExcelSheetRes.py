import csv

from numpy import mod
import pandas as pd

modelAnswer = ['B']*25
StudentAnswers = []
with open('outputs') as f:
    for line in f:
        studentPaper = line.split(',')
        for question in studentPaper:
            grade = question.split(':')[1].strip()
            StudentAnswers.append(grade)
with open('./results.csv', 'a', newline='') as excel:
    # create the csv writer
    writer = csv.writer(excel)
    result = [] 
    for i in range(len(modelAnswer)):
        if modelAnswer[i] == StudentAnswers[i]:
            result.append('TRUE')
        else :
            result.append('WRONG')
    writer.writerow(result)
    read_file = pd.read_csv ('./results.csv')
    read_file.to_excel ('./res.xlsx', index = None, header=None)

"""
# Open an Excel workbook
workbook = xlsxwriter.Workbook('./results.xlsx')
# Set up a format
correctFormat = workbook.add_format(properties={'bold': True, 'font_color': 'white','bg_color':'green'})
WrongFormat =  workbook.add_format(properties={'bold': True, 'font_color': 'white','bg_color':'red'})

result = [x==y for x,y in zip(modelAnswer,StudentAnswers)]
# Create a sheet
worksheet = workbook.add_worksheet('results')
newRowLocation = worksheet.max_row +1


for row_num, value in enumerate(data):
    worksheet.write(row_num, 0, value)

row_num += 1
worksheet.write(row_num, 0, '=SUM(A1:A{})'.format(row_num))

workbook.close()


for r in range(len(result)):
    if result[r] :
            worksheet.write(newRowLocation, 1+r, result[r], correctFormat)
    else:
            worksheet.write(newRowLocation, 1+r, result[r], WrongFormat)
"""