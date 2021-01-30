import datetime
import json
import eml_parser
import os
import flair
import sqlite3
from sqlite3 import Error
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import zipfile


def zip_file(file):
    """
    Zips the file so that it is able to be emailed to us automaically is the client allows
    :param file:
    :return:
    """
    zipf = zipfile.ZipFile("snapshot.zip", "w", zipfile.ZIP_DEFLATED)
    zipf.write(file)

    zipf.close()


def send_snapshot():
    """
    Emails the zip file to us automatically if the client allows
    :return:
    """
    zip_file("proofpointDB.db")
    subject = "Snapshot of Database"
    body = "This is an email with attachment sent from Python"
    sender_email = "proofpointcse498@gmail.com"
    receiver_email = "proofpointcse498@gmail.com"
    password = "proof4point"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    filename = "snapshot.zip"  # In same directory as script

    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)


def grab_stats_election(curr):
    """
    Collect different stats from the database to create
    :param curr:
    :return:
    """
    # collects # of emails of positive or negative sentiment for each candidate
    sql = '''SELECT COUNT(sentiment_score) FROM realData WHERE keyword="Trump" AND P_N="POSITIVE"; '''
    curr.execute(sql)
    posTrump = curr.fetchone()[0]
    sql = '''SELECT COUNT(sentiment_score) FROM realData WHERE keyword="Trump" AND P_N="NEGATIVE"; '''
    curr.execute(sql)
    negTrump = curr.fetchone()[0]
    sql = '''SELECT COUNT(sentiment_score) FROM realData WHERE keyword="Biden" AND P_N="POSITIVE"; '''
    curr.execute(sql)
    posBiden = curr.fetchone()[0]
    sql = '''SELECT COUNT(sentiment_score) FROM realData WHERE keyword="Biden" AND P_N="NEGATIVE"; '''
    curr.execute(sql)
    negBiden = curr.fetchone()[0]

    # collects average sentiment of a positive/negative email for each candidate
    sql = '''SELECT AVG(sentiment_score) FROM realData WHERE keyword="Trump" AND P_N="POSITIVE"; '''
    curr.execute(sql)
    avgPosTrump = curr.fetchone()[0]
    sql = '''SELECT AVG(sentiment_score) FROM realData WHERE keyword="Trump" AND P_N="NEGATIVE"; '''
    curr.execute(sql)
    avgNegTrump = curr.fetchone()[0]
    sql = '''SELECT AVG(sentiment_score) FROM realData WHERE keyword="Biden" AND P_N="POSITIVE"; '''
    curr.execute(sql)
    avgPosBiden = curr.fetchone()[0]
    sql = '''SELECT AVG(sentiment_score) FROM realData WHERE keyword="Biden" AND P_N="NEGATIVE";'''
    curr.execute(sql)
    avgNegBiden = curr.fetchone()[0]

    # gives each candidate a score based on # of pos/neg emails weighted by average sentiment of that type of email
    trumpScore = ((posTrump * avgPosTrump) + (negBiden * avgNegBiden)) / (posTrump + negBiden + negTrump + posBiden)
    bidenScore = ((posBiden * avgPosBiden) + (negTrump * avgNegTrump)) / (posBiden + negTrump + negBiden + posTrump)

    return trumpScore, bidenScore


def make_stock_prediction():
    database = "stockDB.db"
    conn = sqlite3.connect(database)
    curr = conn.cursor()

    # These SQL commands pull the total number of positive/negative emails about
    # a specific stock on a specific date. The emails will be from the 2nd date
    # in the command
    sql = '''SELECT COUNT(sentiment_score) FROM realData WHERE keyword="FB" AND date < '2020-11-24' AND date > '2020-11-23' AND P_N="POSITIVE"; '''
    curr.execute(sql)
    stockPosEmails = curr.fetchone()[0]

    sql = '''SELECT COUNT(sentiment_score) FROM realData WHERE keyword="FB" AND date < '2020-11-24' AND date > '2020-11-23' AND P_N="NEGATIVE"; '''
    curr.execute(sql)
    stockNegEmails = curr.fetchone()[0]

    # These SQL commands calculate the average sentiment of a positive/negative
    # email on the selected day
    sql = '''SELECT AVG(sentiment_score) FROM realData WHERE keyword="FB" AND date < '2020-11-24' AND date > '2020-11-23' AND P_N="POSITIVE"; '''
    curr.execute(sql)
    posStockSentiment = curr.fetchone()[0]
    if posStockSentiment == None:
        posStockSentiment = 0

    sql = '''SELECT AVG(sentiment_score) FROM realData WHERE keyword="FB" AND date < '2020-11-24' AND date > '2020-11-23' AND P_N="NEGATIVE"; '''
    curr.execute(sql)
    negStockSentiment = curr.fetchone()[0]
    if negStockSentiment == None:
        negStockSentiment = 0

    # x = score assigned to a stock
    # y = percent change in stock price day after date

    x = np.array(
        [-0.96, -0.95, -0.96, -0.81, -0.02, 0.41, -0.40, -0.13, -0.17, 0.32, 0.59, 0.47, -0.22, -0.12, 0.12, 0.14,
         0.30])

    y = np.array(
        [2.00, -0.42, -3.08, -0.26, 1.78, -0.13, -2.84, 0.62, 1.16, 0.01, -0.61, 0.88, 0.08, 0.63, 2.40, -2.70, 3.16])

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # gives each candidate a score based on # of pos/neg emails weighted by average sentiment of that type of email
    stockScore = ((stockPosEmails * posStockSentiment) - (stockNegEmails * negStockSentiment)) / (
                stockPosEmails + stockNegEmails)

    print(stockScore)
    reg = LinearRegression().fit(x, y)
    print(reg.coef_)
    # This is the predicted percentage the stock price will change the day
    # after the emails are sent
    print(reg.predict(np.array([stockScore]).reshape(-1, 1)))


def make_prediction_election(conn):
    """
    Create a predictive model (regression) using the approval ratings of the candidates throughout the year in
    addition to the sentiment and volumes of the emails collected
    :param conn:
    :return:
    """
    curr = conn.cursor()

    holder = grab_stats_election(curr)
    trumpScore = holder[0]
    bidenScore = holder[1]

    # data set of pre election favorability v percent of popular vote
    x = np.array([51, 43, 56, 50, 56, 55, 52, 51, 50, 62, 46, 55, 43, 34])
    y = np.array([53.5, 46.5, 54.7, 45.3, 48.4, 47.9, 48.3, 50.7, 45.7, 52.9, 47.2, 51.1, 51.1, 48.9])

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    reg = LinearRegression().fit(x, y)

    trump_score = reg.predict(np.array([trumpScore * 100]).reshape(-1, 1))
    biden_score = reg.predict(np.array([bidenScore * 100]).reshape(-1, 1))

    sql = '''INSERT INTO predictions (subject, keyword, prediction) VALUES (?,?,?)'''
    vals = ("Election", "Trump", round(trump_score[0][0], 2))

    cur = conn.cursor()
    cur.execute(sql, vals)
    conn.commit()

    sql = '''INSERT INTO predictions (subject, keyword, prediction) VALUES (?,?,?)'''
    vals = ("Election", "Biden", round(biden_score[0][0], 2))

    cur = conn.cursor()
    cur.execute(sql, vals)

    conn.commit()


def push_SQL(topic, rating, keyword, date, label, url, attachment, conn):
    """
    Push SQL code into the database
    :param topic: Election, Stock
    :param rating: Sentiment Score
    :param keyword: Subject within topic
    :param date: Date the email was received
    :param label: Positive or Negative
    :param url: Binary if there is one
    :param attachment: Binary if there is one
    :param conn: Connection to database
    :return:
    """
    sql = '''INSERT INTO test (subject, sentiment_score, keyword, date, P_N, url, attachment) VALUES (?,?,?,?,?,?,?)'''
    vals = (topic, rating, keyword, date, label, url, attachment)

    cur = conn.cursor()
    cur.execute(sql, vals)
    conn.commit()


def define_topic(body):
    """
    Define the topic and the category the email is classified as
    :param body: The body of the email
    :return: topic and keyowrd
    """
    # Split the body and create a dictionary
    wordCounts = defaultdict(int)
    words = body.split(' ')
    for word in words:
        wordCounts[word] += 1

    Trump_count = wordCounts['Trump'] + wordCounts['president'] + wordCounts['President'] + wordCounts['Pence'] + \
                  wordCounts['Mike'] + wordCounts['GOP'] + wordCounts['Republican'] + wordCounts['Republicans'] + \
                  wordCounts['Donald'] + wordCounts['Mitch'] + wordCounts['McConnell'] + wordCounts['right'] + \
                  wordCounts['Barrett']
    Biden_count = wordCounts['Biden'] + wordCounts['Kamala'] + wordCounts['Vice'] + wordCounts['Joe'] + \
                  wordCounts['Harris'] + wordCounts['DNC'] + wordCounts['Democrat'] + wordCounts['Democrats'] + \
                  wordCounts['Barack'] + wordCounts['Obama'] + wordCounts['Nancy'] + wordCounts['Pelosi'] + wordCounts[
                      'left'] + wordCounts['Schumer'] + wordCounts['Pete'] + wordCounts['Buttigieg']
    msft_count = wordCounts['Microsoft'] + wordCounts['MSFT'] + wordCounts['$MSFT']
    aapl_count = wordCounts['Apple'] + wordCounts['AAPL'] + wordCounts['$AAPL']
    amzn_count = wordCounts['Amazon'] + wordCounts['AMZN'] + wordCounts['$AMZN']
    goog_count = wordCounts['Google'] + wordCounts['Alphabet'] + wordCounts['GOOG'] + wordCounts['$GOOG']
    googl_count = wordCounts['Google'] + wordCounts['Alphabet'] + wordCounts['GOOGL'] + wordCounts['$GOOGL']
    fb_count = wordCounts['Facebook'] + wordCounts['FB'] + wordCounts['$FB']
    brk_count = wordCounts['Berkshire'] + wordCounts['BRK.B'] + wordCounts['$BRK.B']
    jnj_count = wordCounts['Johnson'] + wordCounts['JNJ'] + wordCounts['$JNJ']
    v_count = wordCounts['Visa'] + wordCounts['V'] + wordCounts['$V']
    pg_count = wordCounts['Proctor'] + wordCounts['Gamble'] + wordCounts['PG'] + wordCounts['$PG']
    jpm_count = wordCounts['JPMorgan'] + wordCounts['JPM'] + wordCounts['$JPM']
    unh_count = wordCounts['UnitedHealth'] + wordCounts['UNH'] + wordCounts['$UNH']
    ma_count = wordCounts['Mastercard'] + wordCounts['MA'] + wordCounts['$MA']
    intc_count = wordCounts['Intel'] + wordCounts['INTC'] + wordCounts['$INTC']
    vz_count = wordCounts['Verizon'] + wordCounts['VZ'] + wordCounts['$VZ']
    hd_count = wordCounts['Home'] + wordCounts['Depot'] + wordCounts['HD'] + wordCounts['$HD']
    t_count = wordCounts['AT&T'] + wordCounts['T'] + wordCounts['$T']
    dow_count = wordCounts['Dow'] + wordCounts['DJI'] + wordCounts['$DJI']
    nasdaq_count = wordCounts['NASDAQ'] + wordCounts['IXIC'] + wordCounts['$IXIC']
    sp_count = wordCounts['S&P500'] + wordCounts['INX'] + wordCounts['$INX']
    tesla_count = wordCounts['TSLA'] + wordCounts['Tesla'] + wordCounts['$TSLA']

    topic_count = {'Trump': Trump_count, 'Biden': Biden_count, 'MSFT': msft_count, \
                   'AAPL': aapl_count, 'AMZN': amzn_count, 'GOOG': goog_count, \
                   'GOOGL': googl_count, 'FB': fb_count, 'BRK': brk_count, \
                   'JNJ': jnj_count, 'V': v_count, 'PG': pg_count, \
                   'JPM': jpm_count, 'UNH': unh_count, 'MA': ma_count, \
                   'INTC': intc_count, 'VZ': vz_count, 'HD': hd_count, \
                   'T': t_count, 'DOW': dow_count, 'NASDAQ': nasdaq_count, \
                   'SP': sp_count, 'TSLA' : tesla_count}

    topic_values = topic_count.values()

    # looks for the max value in topic_count and makes that the keyword
    # if all of the values are 0 the topic will be undefined and the
    # keyword is N/A
    keyword = max(topic_count, key=topic_count.get)

    if max(topic_values) == 0:
        topic = 'Undefinded'
        keyword = 'N/A'
    elif keyword == 'Trump' or keyword == 'Biden':
        topic = 'Election'
    else:
        topic = 'Stocks'
    return [topic, keyword]


def analyze_sentiment(flair_sentiment, raw_email):
    """
    Put the body of the email through the sentiment analyzer
    :param flair_sentiment: sentiment analyzer model
    :param raw_email: email data
    :return:
    """
    holder = EML_Parsing(raw_email)
    # Sentiment Analyzer
    parsed_eml = holder[0]
    url = holder[1]
    attachment = holder[2]
    body = json.dumps(parsed_eml["body"][0]['content'], default=json_serial).replace("\\r", ' ').replace("\\n", '')
    s = flair.data.Sentence(body)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    label = str(total_sentiment[0]).split(' ')[0]  # Label
    rating = float(str(total_sentiment[0]).split(' ')[1].strip('()'))

    # Get email date from parsed JSON object
    date = parsed_eml['header']['date']

    return [body, label, rating, date, url, attachment]


def EML_Parsing(raw_email):
    """
    Parse the EML data to extract the body of the email as well as attachments and url's contained in the emails
    :param raw_email: Raw email data
    :return: eml body, url(binary), attachments(binary)
    """
    ep = eml_parser.EmlParser(include_raw_body=True, include_attachment_data=False, )
    parsed_eml = ep.decode_email_bytes(raw_email)

    # checks if there is an attachment in the email by checking
    # if there is an attachment key in the parsed eml file
    if 'attachment' in parsed_eml.keys():
        attachment = True
    else:
        attachment = False

    # checks if there is a url specified in the body of the email, does not
    # include an image or something with a link attached to it
    try:
        url_list = parsed_eml['body'][1]['uri']
        url = True
    except KeyError:
        url = False
    except IndexError:
        url = False

    return [parsed_eml, url, attachment]


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
    """
    Create JSON object out of eml data
    :param obj:
    :return:
    """
    if isinstance(obj, datetime.datetime):
        serial = obj.isoformat()
        return serial


def flair_set_up():
    """
    Set up flair model
    :return: flair model
    """
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    return flair_sentiment


def main():
    # Set up Flair model
    flair_sentiment = flair_set_up()
    count = 0

    # Create database connection
    database = "proofpointDB.db"
    conn = create_connection(database)

    # Collect directory
    with open('directory_path.config', 'rb') as config:
        directory = config.read().decode("utf-8")

    # Run through directory and extract EML files
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(subdir, file))
            if file.endswith(".eml"):
                count += 1
                with open(file, 'rb') as fhdl:
                    raw_email = fhdl.read()

                # Extract body, sentiment, and parse EML
                holder = analyze_sentiment(flair_sentiment, raw_email)

                # Extract Label and body
                body = holder[0]
                label = holder[1]
                rating = holder[2]
                date = holder[3]
                url = holder[4]
                attachment = holder[5]

                # Extract topic
                holder = define_topic(body)
                topic = holder[0]
                keyword = define_topic(body)[1]

                push_SQL(topic, rating, keyword, date, label, url, attachment, conn)
        print(count)

        # Sends snapshot of data because apparently no one else will
        if count % 10000 == 0:
            send_snapshot()

    make_prediction_election(conn) # Make the prediction for the categories based on the data
    make_stock_prediction()


if __name__ == "__main__":
    main()
