import datetime
import json
import eml_parser
import os
import flair
import sqlite3
import operator
import string
from sqlite3 import Error
from collections import defaultdict
import time
import math
from multiprocessing import Pool
from multiprocessing import set_start_method
from multiprocessing import get_context
from multiprocessing import freeze_support

def push_SQL(rating, keyword, date, label, conn):
    sql = '''INSERT INTO realData (subject, sentiment_score, keyword, date, P_N) VALUES (?,?,?,?,?)'''
    vals = ('Election', rating, keyword, date, label)

    cur = conn.cursor()
    cur.execute(sql, vals)
    conn.commit()

def define_topic(body):
    # Split the body and create a dictionary
    wordCounts = defaultdict(int)
    words = body.split(' ')

    # Go through body and count words related to topics
    for word in words:
        wordCounts[word] += 1

    # Add up counts
    trump_count = wordCounts['Trump'] + wordCounts['president'] + wordCounts['President'] + wordCounts['Pence'] + \
                  wordCounts['Mike'] + wordCounts['GOP'] + wordCounts['Republican'] + wordCounts['Republicans'] + \
                  wordCounts['Donald'] + wordCounts['Mitch'] + wordCounts['McConnell'] + wordCounts['right'] + \
                  wordCounts['Barrett']
    biden_count = wordCounts['Biden'] + wordCounts['Kamala'] + wordCounts['Vice President'] + wordCounts['Joe'] + \
                  wordCounts['Harris'] + wordCounts['DNC'] + wordCounts['Democrat'] + wordCounts['Democrats'] + \
                  wordCounts['Barack'] + wordCounts['Obama'] + wordCounts['Nancy'] + wordCounts['Pelosi'] + wordCounts[
                      'left'] + wordCounts['Schumer'] + wordCounts['Pete'] + wordCounts['Buttigieg']

    # Determine topic
    if trump_count > biden_count:
        keyword = 'Trump'
    elif biden_count > trump_count:
        keyword = 'Biden'
    else:
        keyword = 'Undefined'
    return keyword


def analyze_sentiment(flair_sentiment, raw_email):
    # Sentiment Analyzer
    parsed_eml = EML_Parsing(raw_email)
    body = json.dumps(parsed_eml["body"][0]['content'], default=json_serial).replace("\\r", ' ').replace("\\n", '')
    s = flair.data.Sentence(body)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    label = str(total_sentiment[0]).split(' ')[0]  # Label
    rating = float(str(total_sentiment[0]).split(' ')[1].strip('()'))

    # Get email date from parsed JSON object
    date = parsed_eml['header']['date']

    return [body, label, rating, date]


def EML_Parsing(raw_email):
    ep = eml_parser.EmlParser(include_raw_body=True, include_attachment_data=False, )
    parsed_eml = ep.decode_email_bytes(raw_email)
    return parsed_eml


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def json_serial(obj):
    if isinstance(obj, datetime.datetime):
        serial = obj.isoformat()
        return serial


def flair_set_up():
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    return flair_sentiment

'''Define function to run mutiple processors and pool the results together'''
def run_multiprocessing(func, i, n_processors):
    with get_context("spawn").Pool() as pool:
        return pool.map(func, i)


def main():
    # Set up Flair model
    flair_sentiment = flair_set_up()

    # Create database connection
    database = "proofpointDB.db"
    conn = create_connection(database)


    with open(filename, 'rb') as fhdl:
        raw_email = fhdl.read()

    # Extract body, sentiment, and parse EML
    holder = analyze_sentiment(flair_sentiment, raw_email)

    # Extract Label and body
    body = holder[0]
    label = holder[1]
    rating = holder[2]
    date = holder[3]

    # Extract topic
    keyword = define_topic(body)

    push_SQL(rating, keyword, date, label, conn)


if __name__ == "__main__":
    set_start_method("spawn")
    files = []
    # Collect directory
    with open('directory_path.config', 'rb') as config:
        directory = config.read().decode("utf-8")

    # Run through directory and extract EML files
    for filename in os.listdir(directory):
        if filename.endswith(".eml"):
            files.append(directory+'\\'+filename)

    run_multiprocessing(main, files, 2)


