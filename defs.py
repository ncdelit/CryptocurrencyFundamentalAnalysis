def RetrieveUsers(subreddit,Terms):
    users = []
    for submission in subreddit.submissions(1515373826, 1515978626):

    #if "whitepaper" in comment.body:
    #    print(comment.author)
    #    print(comment.body)
    #print(submission.title)

        while True:
            try:
                submission.comments.replace_more(limit=None)
                SubCom = submission.comments.list()
                for comment in SubCom:
                    #print(comment.body)
                    #print(comment.author)
                    if any(x in comment.body for x in Terms):
                        #print(comment.body)
                        print(comment.author)
                        #print(comment.body)
                        if not ("Gyazo_Bot" == comment.author) and not ("RemindMeBot" == comment.author):
                            users.append(comment.author)

            except Exception,e:
                print(str(e))
                break
            break
    users = list(set(users))

def GetComments(users,NumberOfComments,reddit):

    overallcommentcount = 0
    commentcount = 0
    usercount = 0
    trycount = 0

    w, h =  NumberOfComments, len(users);
    userswithcomments = [['' for x in range(w)] for y in range(h)]

    for i in range(len(users)):
        trycount = 0
        while True:
            try:
                print(users[i])
                commentcount = 0
                commentslist = reddit.redditor(users[i]).comments.new(limit = NumberOfComments)
                for comment in commentslist:

                    #comments.append(comment.body)      # add all comments by users who have mentioned "whitepaper" to the array 'comments'
                    userswithcomments[usercount][commentcount] = comment.body.encode('utf-8')
                    #print(userswithcomments[usercount][commentcount])
                    commentcount = commentcount +1
                    overallcommentcount = overallcommentcount +1
                    #print("commentcount:" + str(commentcount))

                usercount = usercount +1
                #print("usercount:" + str(usercount))
            except Exception,e:
                if trycount >= 5:
                    usercount = usercount +1
                    break
                print(str(e))
                trycount = trycount + 1
                continue
            break

    return userswithcomments

def ScoreUsers(users,userswithcomments,TopPercentage,Terms):

    w3, h3 = 1,len(users);

    freqWP = [[0 for x in range(w3)] for y in range(h3)]

    lengths = []

    for row in userswithcomments:
        lengths.append(len(row))

    maxl = max(list(lengths))


    # for row in range(len(userswithcomments)):
    #     print("user:" +str(users[row]))
    #     print("============")
    #     for column in range(len(userswithcomments[0])):
    #         print(userswithcomments[row][column])

    for row in range(len(userswithcomments)):
        for column in range(maxl):
            try:
                if("use case" in userswithcomments[row][column].lower()):
                    # print("====================")
                    # print("use case")
                    # print("====================")
                    #print(userswithcomments[row][column].lower())
                    freqWP[row][0] = freqWP[row][0] +1

                if("white paper" in userswithcomments[row][column].lower()):
                    # print("===================")
                    # print("white paper")
                    # print("===================")
                    # print(userswithcomments[row][column].lower())
                    freqWP[row][0] = freqWP[row][0] +1

                if("repo " in userswithcomments[row][column].lower()):
                    # print("==================")
                    # print("repo ")
                    # print("==================")
                    # print(userswithcomments[row][column].lower())

                    freqWP[row][0] = freqWP[row][0] +1

                for word in userswithcomments[row][column].split():
                    if any(x in word.lower() for x in Terms):
                        # print("================")
                        # print(word)
                        # print("================")
                        #print(userswithcomments[row][column].lower())
                        freqWP[row][0] = freqWP[row][0] +1

                # if "usecase" in str(userswithwords[row][column]):
                #     freqUC[row][0] = freqUC[row][0] +1
                #print("countcolumn:" +str(column))

            except Exception,e:
                print(str(e))

            #print("countrow:" +str(row))

    counter2 = 0
    # print("")
    # print("Indicator Counter")
    # for row in freqWP:
    #     print(str(users[counter2]) +":"+ str(row))
    #     counter2= counter2+1

    import operator


    TopUsers = int(round(TopPercentage*len(users)))

    usersSind = zip(*sorted(enumerate(freqWP), key=operator.itemgetter(1)))[0][-TopUsers:]
    # print(usersSind)


    usersS = (users[int(i)] for i in usersSind)

    return usersS,usersSind

def WriteToDatabase(users):
    forPickle = []

    for x in usersS:
        forPickle.append(str(x))

    import pickle

    db = {}
    db['Users'] = forPickle

    with open('database.db', 'rb') as fh: db = pickle.load(fh)

def CleanComments(NumberOfComments,userswithcomments,users,usersSind):
    w3, h3 = NumberOfComments,len(users);

    #Sophistication metrics:
    temp = [['' for x in range(w3)] for y in range(h3)]

    temp = list([userswithcomments[i] for i in usersSind])

    UserSubset = [[word for line in col for word in line.split()] for col in temp]

    # for row in UserSubset:
    #     print(row)

    Merged = [item for sublist in UserSubset for item in sublist]

    import re, string, timeit


    Merged = [''.join(c for c in s if c not in string.punctuation) for s in Merged]

    return Merged

def FrequencyTable(Merged,otherWords):
    from nltk.corpus import stopwords
    from collections import Counter

    filtered_words2 = [word for word in Merged if word not in stopwords.words('english') and word not in otherWords]

    SubsetCount = Counter(filtered_words2)
    return SubsetCount

def ProjectDetermination(SubsetCount,TopN):
    """
    PROJECT DETERMININATION

    This code finds all coins in the TopN most common words
    """

    import operator

    w3, h3 = 2,TopN;
    projects = [[0 for x in range(w3)] for y in range(h3)]

    TopWords = []

    import collections
    orderedDict = collections.OrderedDict()

    from collections import OrderedDict

    List0 = dict(sorted(SubsetCount.iteritems(), key=operator.itemgetter(1), reverse=True)[:TopN])

    List = [ [k,v] for k, v in List0.items()]

    import urllib, json
    for row in range(len(List)):
        if len(List[row][0].decode('utf8')) == 3 or len(List[row][0].decode('utf8')) == 4:
            url = "https://min-api.cryptocompare.com/data/price?fsym="+str(List[row][0])+"&tsyms=USD,EUR"
            response = urllib.urlopen(url)
            data = json.loads(response.read())
            if 'Error' not in str(data):
                projects[row][0] = List[row][0]
                projects[row][1] = List[row][1]

    def transpose(grid):
        return zip(*grid)

    def removeBlankRows(grid):
        return [list(row) for row in grid if any(row)]


    projects= removeBlankRows(transpose(removeBlankRows(transpose(projects))))
    return projects

def CommentsToSentences(userswithcomments,usersSind):
    UserSubset2 = [userswithcomments[i] for i in usersSind]

    commentsvector = [item for sublist in UserSubset2 for item in sublist]

    import nltk

    sentences = []

    for i in commentsvector:
        sentences.append(nltk.sent_tokenize(unicode(i, errors='replace')))

    return sentences

def ComputeSentimentTable(sentences,projects):
    """
    SENTIMENT ANALYSIS

    Here, we perform a sentiment analysis by filtering the initial userswithcomments array and then splitting the comments out into sentances.

    We then only take the sentances which contain a coin mention, and we apply the sentiment function using the VADER method to those sentances

    This method seems to be about 60 percent accurate or so.

    Note that at the moment, it seems to be the case that there will be more sentiment score sentances than actual coin mentions in the frequency table. This is due to the fact that the frquency counter doesn't tally when numbers or anything else is in front of the coin mention, whereas the sentiment scoring will take any sentance that has the coin anywhere in it.
    """
    import pandas as pd
    import numpy as np
    import nltk
    counter = 0

    sentences = [item for sublist in sentences for item in sublist]

    w3, h3 = 4,len(sentences);
    WordCoin = [[0 for x in range(w3)] for y in range(h3)]

    for sentence in sentences:
        for word in sentence.split():
            for i in range(len(projects)):
                if str(projects[i][0]) in str(word):
                    WordCoin[counter][0] = projects[i][0]
                    WordCoin[counter][1] = sentence
                    WordCoin[counter][3] = projects[i][1]
                    counter = counter +1


    from nltk.sentiment.vader import SentimentIntensityAnalyzer


    sid = SentimentIntensityAnalyzer()
    for i in range(len(WordCoin)):
     #print(WordCoin[i][1])
        WordCoin[i][2] = sid.polarity_scores(str(WordCoin[i][1]))['compound']
             #print('{0}: {1}, '.format(k, ss[k]))

    return WordCoin

def ComputeAnalysisTable(WordCoin,projects):
    import pandas as pd
    import numpy as np
    import nltk

    def nthcolumn(matrix,n):
        return [row[n] for row in matrix]

    # from operator import itemgetter
    #
    # def nthcolumn(n, matrix):
    #     nthvalue = itemgetter(n)
    #     return [nthvalue(row) for row in matrix]

    df = pd.DataFrame({'Coin':nthcolumn(WordCoin, 0),'Sentence':nthcolumn(WordCoin, 1),'Sentiment':nthcolumn(WordCoin, 2),'Total':nthcolumn(WordCoin, 3)})


    CoinSentimentSummary = df.pivot_table(index=['Coin'], values=['Sentiment'], aggfunc=np.sum)

    # print(CoinSentimentSummary)

    CoinSentimentSummary = CoinSentimentSummary.reset_index()

    #Drop the first row which seems to be a zero for some reason
    CoinSentimentSummary.drop = CoinSentimentSummary.drop(CoinSentimentSummary.index[0],inplace=True)

    CoinSentimentSummary = CoinSentimentSummary.reset_index()

    ForDF = []

    #Sort in order to get into the same order as the projects array
    for index1,row1 in CoinSentimentSummary.iterrows():
        for i in range(len(nthcolumn(projects, 1))):
            if row1['Coin'] == projects[i][0]:
                ForDF.append(projects[i][1])


    CoinSentimentSummary['Total'] = ForDF

    CoinSentimentSummary['Rel'] = CoinSentimentSummary['Sentiment']/CoinSentimentSummary['Total']

    Avg = sum(CoinSentimentSummary['Sentiment'])/sum(CoinSentimentSummary['Total'])

    StDev = CoinSentimentSummary['Rel'].std()

    CoinSentimentSummary['StDev'] = (CoinSentimentSummary['Rel'] - Avg)/StDev

    CoinSentimentSummary = CoinSentimentSummary.sort_values(by=['Rel'],ascending=False)

    print(CoinSentimentSummary)

def AssociatedWords(WordCoin,otherWords):
    import operator
    import collections
    orderedDict = collections.OrderedDict()


    from collections import OrderedDict
    sentencespluscoins = []
    coinschecker = ['']
    for row1 in range(len(WordCoin)):
        boo = 0
        sentencespluscoins = []
        del sentencespluscoins[:]

        for i in coinschecker:
            if WordCoin[row1][0] == i:
                boo = boo +1

        if boo == 0:
            for row2 in range(len(WordCoin)):
                if WordCoin[row1][0] == WordCoin[row2][0]:
                    for word in str(WordCoin[row2][1]).split():
                        sentencespluscoins.append(word)

        if boo == 0:
            from nltk.corpus import stopwords
            from collections import Counter
            filtered_words2 = [word for word in sentencespluscoins if word not in stopwords.words('english') and word not in otherWords]

            SubsetCount = Counter(filtered_words2)
            SubsetCount = dict(sorted(SubsetCount.iteritems(), key=operator.itemgetter(1), reverse=True)[:25])
            coinschecker.append(WordCoin[row1][0])
            print("")
            print(str(WordCoin[row1][0]))
            print("....")
            print(SubsetCount)
