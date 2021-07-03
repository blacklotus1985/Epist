import mysql.connector
from sqlalchemy import create_engine
import pandas as pd

mydb = mysql.connector.connect(
  host=" neo4j.epistolarita.simultech.it",
  user="root",
  password="simultech",
  port = 49153,
  database = 'performagroup'
)
mycursor = mydb.cursor()

mycursor.execute("Select * from documenti")
#curriculum = mycursor.execute("Select * from curriculum")


myresult = mycursor.fetchall()

for x in myresult:
    print(x[2])
