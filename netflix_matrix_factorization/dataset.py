import os
import numpy as np
import pickle

def read_ratings(filename, ratings, ratings_test, u2uid, m2mid):
    file1 = open(filename, 'rt')

    parts = filename.split('/')
    movieid = int(parts[-1][3:-4])
    if movieid not in m2mid:
        m2mid[movieid] = len(m2mid)
    movieid = m2mid[movieid]

    userid = -1
    first = True
    while True:
        line = file1.readline()
        if not line:
            break
        if first:
            first = False
            continue
        parts = line.split(',')
        userid = int(parts[0])
        if userid not in u2uid:
            u2uid[userid] = len(u2uid)
        userid = u2uid[userid]
        rating = float(parts[1])
        if movieid in ratings_test and userid in ratings_test[movieid]:
            ratings_test[movieid][userid] = rating
        else:
            ratings.append([movieid, userid, rating])
    file1.close()
    
    return
    
def go_through_files(directory, ratings, ratings_test, u2uid, m2mid):
    info = os.listdir(directory)
    
    for i in info:
        tmp = directory + '/' + i
        if os.path.isdir(tmp):
            go_through_files(tmp, ratings, ratings_test, u2uid, m2mid)
        if os.path.isfile(tmp) and tmp.endswith('.txt'):
            read_ratings(tmp, ratings, ratings_test, u2uid, m2mid)

    return
    

def read_test_file(filename, ratingst, u2uid, m2mid):
    file1 = open(filename, 'rt')

    movieid = -1
    userid = -1
    while True:
        line = file1.readline()
        if not line:
            break
        line = line.replace('\n','')
        if line.endswith(':'):
            movieid = int(line[:-1])
            if movieid not in m2mid:
                m2mid[movieid] = len(m2mid)
            movieid = m2mid[movieid]
            if movieid not in ratingst:
                ratingst[movieid] = {}
            continue
             
        userid = int(line)            
        if userid not in u2uid:
            u2uid[userid] = len(u2uid)
        userid = u2uid[userid]
        if userid not in ratingst[movieid]:
            ratingst[movieid][userid] = -1
                
    file1.close()
    
    return

def read_data():
    ratings = []
    ratings_test = {}
    u2uid = {}
    m2mid = {}

    if os.path.isfile('train_m.npy'):
        print('Loading from preprocessed files')
        train_m = np.load('train_m.npy')
        train_u = np.load('train_u.npy')
        train_r = np.load('train_r.npy')

        test_m = np.load('test_m.npy')
        test_u = np.load('test_u.npy')
        test_r = np.load('test_r.npy')

        u2uid = pickle.load(open('u2uid.pickle', 'rb'))
        m2mid = pickle.load(open('m2mid.pickle', 'rb'))
        print('Loading done')
    else:
        read_test_file('probe.txt', ratings_test, u2uid, m2mid)
        go_through_files('./training_set', ratings, ratings_test, u2uid, m2mid)

        ratings_test2 = []
        for k1 in ratings_test:
            for k2 in ratings_test[k1]:
                ratings_test2.append([k1,k2,ratings_test[k1][k2]])
        
        pickle.dump(u2uid,open('u2uid.pickle', 'wb'))
        pickle.dump(m2mid,open('m2mid.pickle', 'wb'))

        train_m = np.zeros(len(ratings)).astype(np.int32)
        train_u = np.zeros(len(ratings)).astype(np.int32)
        train_r = np.zeros(len(ratings)).astype(np.int32)

        for i in range(len(ratings)):
            train_m[i] = ratings[i][0]
            train_u[i] = ratings[i][1]
            train_r[i] = ratings[i][2]

        test_m = np.zeros(len(ratings_test2)).astype(np.int32)
        test_u = np.zeros(len(ratings_test2)).astype(np.int32)
        test_r = np.zeros(len(ratings_test2)).astype(np.int32)

        for i in range(len(ratings_test2)):
            test_m[i] = ratings_test2[i][0]
            test_u[i] = ratings_test2[i][1]
            test_r[i] = ratings_test2[i][2]

        np.save('train_m.npy', train_m)
        np.save('train_u.npy', train_u)
        np.save('train_r.npy', train_r)

        np.save('test_m.npy', test_m)
        np.save('test_u.npy', test_u)
        np.save('test_r.npy', test_r)

    print('Number of users:', len(u2uid))
    print('Number of movies:', len(m2mid))
    print('Number of training ratings:', train_m.shape[0])
    print('Number of testing ratings:', test_m.shape[0])

    train_J = np.zeros(train_m.shape[0]).astype(np.float32)
    test_J = np.zeros(test_m.shape[0]).astype(np.float32)

    return train_m, train_u, train_r, train_J, test_m, test_u, test_r, test_J, u2uid, m2mid
