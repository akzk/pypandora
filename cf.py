"""
    Collaborative Filtering
"""
from math import sqrt
import sqlite3

class CF(object):
    def __init__(self, dbpath):
        """
            Connecting sqlite database for collaborative filtering.

            dbpath  : path of sqlite database. If it dose not exist, it will be build and create table.
        """
        self.conn = sqlite3.connect(dbpath)
        self.__create_tables()
    
    def commit(self):
        self.conn.commit()

    def add_rating(self, user, item, rating, is_commit=True):
        """
            Adding data

            user    : user origin id
            item    : item origin id
            rating  : user rating of item
        """
        assert isinstance(user, int)
        assert isinstance(item, int)
        assert isinstance(rating, float)

        user_id = self.__get_user_id(user)
        item_id = self.__get_item_id(item)

        if user_id is None:
            self.__add_user(user, is_commit)
            user_id = self.__get_user_id(user)
        if item_id is None:
            self.__add_item(item, is_commit)
            item_id = self.__get_item_id(item)
        
        self.__add_rating(user_id, item_id, rating, is_commit)

    def update_all_distance(self, metric='euclidean', is_commit=True):
        """
            Computing and updating distance between all users
        """
        user_ids = self.__get_all_user_ids()
        for i, _ in enumerate(user_ids):
            self.__update(user_ids[i], user_ids[i+1:], metric)
        if is_commit: self.conn.commit()
    
    def update_distance(self, user, metric='euclidean', is_commit=True):
        """
            Computing and updating distance between one user and other users

            user    : user origin id
        """
        user_id = self.__get_user_id(user)
        if user_id is None: return
        user_ids = self.__get_all_user_ids()
        self.__update(user_id, user_ids, metric)
        if is_commit: self.conn.commit()
    
    def get_distance(self, a, b):
        """
            Searching distance between user a and b

            a       : user a's origin id
            b       : user b's origin id
        """
        a_id = self.__get_user_id(a)
        b_id = self.__get_user_id(b)
        distance = self.__get_distance(a_id, b_id)
        return distance
    
    def get_closest_users(self, user, num=10):
        """
            Searching users closest to the user with given user_id(origin id)

            user    : user origin id
            num     : number of closest users 
        """
        user_id = self.__get_user_id(user)
        if user_id is None: return []
        return self.__get_closest_users(user_id, num)
    
    def __get_closest_users(self, user_id, num):
        """
            Searching users closest to the user with given user_id

            user_id : user id
            num     : number of closest users
        """
        sqls = [
            f"""
                select b_id, distance from distance where a_id = {user_id} 
                and distance != -1 order by distance limit {num}
            """,
            f"""
                select a_id, distance from distance where b_id = {user_id}
                and distance != -1 order by distance limit {num}
            """
        ]

        cursor = self.conn.cursor()
        result = []

        for sql in sqls:
            cursor.execute(sql)
            for row in cursor.fetchall():
                result.append({"user_id": row[0], "distance": row[1]})
            if len(result) > 0: break

        return result
    
    def __update(self, a_id, b_ids, metric='euclidean'):
        """
            Computing and updating distance between one user and other users

            a_id        : one user id
            b_ids       : other users' id
            metric      : 'euclidean' or 'pearson'
        """
        for b_id in b_ids:
            if a_id != b_id:
                distance = self.__compute_distance(a_id, b_id, metric)
                self.__add_distance(a_id, b_id, distance, False)
    
    def __get_all_user_ids(self):
        """
            Return a list of all user ids
        """
        cursor = self.conn.cursor()
        cursor.execute("select id from user")
        return [x[0] for x in cursor.fetchall()]

    def __get_distance(self, a_id, b_id):
        """
            Return distance between user a and b

            a_id        : user a's id
            b_id        : user b's id
        """
        cursor = self.conn.cursor()
        cursor.execute("select distance from distance where a_id = %d and b_id = %d" % (a_id, b_id))
        result = cursor.fetchone()
        if result is not None:
            return result[0]
    
    def __compute_distance(self, a_id, b_id, method):
        """
            Computing distance of user a and user b. If there is no item rated by both a and b, the distance will be -1.0
        """
        method = method.lower()
        assert method == 'euclidean' or method == 'pearson'

        inner = self.__get_rated_item_inner(a_id, b_id)
        if len(inner) == 0: return -1.0

        a_ratings = sorted(self.__get_item_rating_list(a_id, inner), key=lambda x:x[0])
        b_ratings = sorted(self.__get_item_rating_list(b_id, inner), key=lambda x:x[0])
        
        if method == 'euclidean':
            return self.__euclidean(a_ratings, b_ratings)
        elif method == 'pearson':
            return self.__pearson(a_ratings, b_ratings)
    
    def __euclidean(self, a_ratings, b_ratings):
        return sum([(a_ratings[i][1]-b_ratings[i][1])**2 for i, _ in enumerate(a_ratings)])
    
    def __pearson(self, a_ratings, b_ratings):
        n = len(a_ratings)
        sum1 = sum([x[1] for x in a_ratings])
        sum2 = sum([x[1] for x in b_ratings])
        sum1sq = sum([x[1]**2 for x in a_ratings])
        sum2sq = sum([x[1]**2 for x in b_ratings])
        p_sum = sum(a_ratings[i][1]*b_ratings[i][1] for i, _ in enumerate(a_ratings))
        num = p_sum-(sum1*sum2/n)
        den = sqrt((sum1sq-sum1**2/n)*(sum2sq-sum2**2/n))
        if den == 0: return 0.0
        return num/den
    
    def __get_rated_item_inner(self, a_id, b_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            select item_id from (select item_id, count(item_id) as num from rating \
            where user_id in (%d, %d) group by item_id) where num = 2
        """ % (a_id, b_id))
        return [x[0] for x in cursor.fetchall()]
    
    def __get_item_rating_list(self, user_id, item_ids):
        cursor = self.conn.cursor()
        cursor.execute("""
            select item_id, rating from rating where user_id = %d and item_id in (%s)
        """ % (user_id, ','.join([str(x) for x in item_ids])))
        return cursor.fetchall()
    
    def __add_item(self, origin_id, is_commit=True):
        cursor = self.conn.cursor()
        cursor.execute("insert into item(origin_id) values(%d)" % origin_id)
        if is_commit: self.conn.commit()
    
    def __add_user(self, origin_id, is_commit=True):
        cursor = self.conn.cursor()
        cursor.execute("insert into user(origin_id) values(%d)" % origin_id)
        if is_commit: self.conn.commit()
    
    def __add_rating(self, user_id, item_id, rating, is_commit=True):
        cursor = self.conn.cursor()
        cursor.execute("insert or replace into rating(user_id, item_id, rating) values(%d, %d, %f)" % (user_id, item_id, rating))
        if is_commit: self.conn.commit()
    
    def __add_distance(self, a_id, b_id, distance, is_commit=True):
        cursor = self.conn.cursor()
        cursor.execute("insert or replace into distance(a_id, b_id, distance) values(%d, %d, %f)" % (a_id, b_id, distance))
        if is_commit: self.conn.commit()
    
    def __get_user_id(self, origin_id):
        """
            Return user id by origin id
        """
        cursor = self.conn.cursor()
        cursor.execute("select id from user where origin_id = %d" % origin_id)
        result = cursor.fetchone()
        if result is not None:
            return result[0]
    
    def __get_item_id(self, origin_id):
        """
            Return item id by origin id
        """
        cursor = self.conn.cursor()
        cursor.execute("select id from item where origin_id = %d" % origin_id)
        result = cursor.fetchone()
        if result is not None:
            return result[0]

    def __get_rating(self, user_id, item_id):
        """
            Return rating by user id and item_id

            user    : user id
            item    : item id
        """
        cursor = self.conn.cursor()
        cursor.execute("select rating from rating where user_id = %d and item_id = %d" % (user_id, item_id))
        result = cursor.fetchone()
        if result is not None:
            return result[0]
    
    def __create_tables(self):
        """
            Creating tables
        """
        tables = [
            """
                CREATE TABLE IF NOT EXISTS user (
                    id          integer primary key     autoincrement not null,
                    origin_id   integer                 not null,
                    unique(origin_id)
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS item (
                    id          integer primary key     autoincrement not null,
                    origin_id   integer                 not null,
                    unique(origin_id)
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS rating (
                    id          integer primary key     autoincrement not null,
                    user_id     integer                 not null,
                    item_id     integer                 not null,
                    rating      real                    not null,
                    unique(user_id, item_id) on conflict replace
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS distance (
                    id          integer primary key     autoincrement not null,
                    a_id        integer                 not null,
                    b_id        integer                 not null,
                    distance    real                    not null,
                    unique(a_id, b_id) on conflict replace
                )
            """
        ]
        
        cursor = self.conn.cursor()
        for sql in tables:
            cursor.execute(sql)
        self.conn.commit()
