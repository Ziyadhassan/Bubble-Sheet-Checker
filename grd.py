#!env python
import xlsxwriter
import json


def main():
  answer = json.load(open('outputs'))
  workbook = xlsxwriter.Workbook('answers.xlsx')
  worksheet = workbook.add_worksheet('student answers')
  for i, (k, v) in enumerate(answer.items()):
    worksheet.write(0, i, k)
    worksheet.write(1, i, v)
  workbook.close()


if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    print('Error: ', e)
    exit(1)
