import requests
import pymongo
import datetime
import uuid
from bson.json_util import dumps
from random import sample as random_sample
import numpy as np
import threading, queue

BIASES = {"patribotics": -38, "Bipartisanism": -26, "fwdprogressives": -25, "HuffPost": -22, "MSNBC": -19,  "washingtonpost": -10, "CNN": -7, "propublica": -5,  "NPR": -5, "PBS": -5, "nytimes": -4, "ABC": 0, "business": 1, "CBSNews": 4, "Forbes": 5, "thehill": 10, "weeklystandard": 18, "TheTimesNUSA": 20, "amconmag": 27, "FoxNews": 27, "OANN": 28, "realDailyWire": 28, "BreitbartNews": 34, "newswarz": 38}


POLITICIANS = ["BarackObama", "realDonaldTrump", "HillaryClinton", "SpeakerPelosi"]
NETWORKS = list(BIASES.keys())
ORGANIZATIONS = ["TheDemocrats", "GOP"]
CELEBS_POLITICAL = ["seanhannity"]
CELEBS = []
ALL_TWITTERS = POLITICIANS + NETWORKS + ORGANIZATIONS + CELEBS_POLITICAL + CELEBS


def print_json(object):
    print(dumps(object, indent=3))


def thread_run(func, arg_list:list):
    def worker_func(q, f, out_list):
        while True:
            try: (i, f_args) = q.get()
            except queue.Empty: return
            out_list[i] = f(**f_args)
            q.task_done()

    assert type(arg_list) == list
    out_list = [0] * len(arg_list)

    q = queue.Queue()
    for i, args in enumerate(arg_list):
        q.put_nowait((i, args))

    for _ in range(4):
        threading.Thread(target=worker_func, daemon=True, args=(q, func, out_list)).start()
    q.join()
    return out_list


def parallel_worker_func(f, f_args, i, out_list):
        out_list[i] = f(**f_args)


def parallel_run(target_f, arg_list):
    from multiprocessing import Manager, Pool
    import copy
    import multiprocessing

    assert type(arg_list) == list

    manager = Manager()
    out_list = manager.list([0] * len(arg_list))

    pool_size = multiprocessing.cpu_count() * 2
    pool = Pool(max(1, pool_size))

    print("running multiprocess on {} workers".format(pool_size))
    results = [pool.apply_async(parallel_worker_func, (target_f, copy.deepcopy(args), i, out_list)) for i, args in enumerate(arg_list)]
    pool.close()
    pool.join()

    for r in results:
        if isinstance(r._value, BaseException):
            raise r._value

    return list(out_list)


def merge_lists(lists):
    return sum(lists, [])


class TwitterScraper:
    access_token = "AAAAAAAAAAAAAAAAAAAAADmpCwEAAAAArVTe5zTHz2ookNDQIlGImi9Fdqw%3D4GuHEFgH7ZQuSM3d4flg8eTTlafRVaTdUBrUTOJFdZMnnDQ6ji"
    client_key = "65bULXQXhB9DD9MWtiWmuj12Y"
    client_secret = "fpJFhiGd2iYQjauxCggUZYMY7mmmBRJzTJ7KQfk6pSIAYCoLSn"

    search_header = {'Authorization': 'Bearer {}'.format(access_token)}
    tweet_url = "https://api.twitter.com/1.1/tweets/search/30day/Test.json"
    user_url = "https://api.twitter.com/1.1/users/lookup.json"
    timeline_url = "https://api.twitter.com/1.1/statuses/user_timeline.json"
    followers_url = "https://api.twitter.com/1.1/followers/ids.json"

    @staticmethod
    def fetch_database():
        client = pymongo.MongoClient('mongodb://deltatrainer:legengerry@18.217.164.108:27017/admin?retryWrites=false',
                                     serverSelectionTimeoutMS=10)
        db = client["admin"]["17835"]
        return db

    @staticmethod
    def put_database_objects(objects: [dict]):
        """
        store objects in the 17835 bucket of the database
        :param objects:
        :return: None
        """
        assert type(objects) == list
        db = TwitterScraper.fetch_database()
        for obj in objects:
            existing_obj = db.find_one({"id": obj["id"]})
            if existing_obj is None:
                db.insert_one(obj)
            elif existing_obj["17835"]["type"] == "user":
                db.update_one({"id": obj["id"]}, {"$set": obj}, upsert=False)
            else:
                print("skipping database insertion of repeat tweet (id: {})".format(obj["id"]))

    @staticmethod
    def update_database_objects(objects: [dict]):
        """
        update objects in the database
        :param objects: ([dict]) objects to update
        :return None:
        """
        assert type(objects) == list
        db = TwitterScraper.fetch_database()
        for obj in objects:
            db.update_one({"id": obj["id"]}, {"$set": obj}, upsert=False)

    @staticmethod
    def fetch_database_objects(query: dict, fields: [str] = None):
        """
        fetch objects from the database matching given query
        :param query: (dict) the query
        :param fields: [str] optional, the fields in the objects to fetch (defaults to fetching all fields)
        :return objects: [dict] list of objects
        """
        fields = None if fields is None else {field: True for field in fields}
        db = TwitterScraper.fetch_database()
        objects = list(db.find(query, fields))
        print("fetched {} objects from database".format(len(objects)))
        return objects

    @staticmethod
    def delete_objects_in_database():
        """
        discresionary method to delete tweet objects
        :return None: None
        """
        pass
        # db = TwitterScraper.fetch_database()
        # db.delete_many({"lang": {"$exists": True, "$ne": "en"}})

    @staticmethod
    def count_tweets():
        """
        count and print the number of tweet objects in database
        :retun None: None
        """
        n_tweets = len(TwitterScraper.fetch_database_objects(query={"17835.type": "tweet"}, fields=["_id"]))
        print("N tweets in database: {}".format(n_tweets))

    @staticmethod
    def scrape_tweets():
        """
        general tweet scraper - scrape by content
        :return tweets: [dict] the tweet objects
        """
        search_params = {"query": 'Biden "vote"', "maxResults": "100",
                         "fromDate": "<202003021200>", "toDate": "<202003022400>"}
        search_id = uuid.uuid4().hex
        search_resp = requests.get(TwitterScraper.tweet_url, params=search_params,
                                   headers=TwitterScraper.search_header)
        tweets = search_resp.json()["statuses"]

        TwitterScraper.process_objects(tweets, search_params, search_id, object_type="tweet")
        TwitterScraper.put_database_objects(tweets)
        return tweets

    @staticmethod
    def process_objects(objects: [dict], query: dict, search_id: str, object_type: str):
        """
        add metadata to the 17835 attribute of the object and put it in the database
        modifies objects in place

        :param objects: ([dict])
        :param query: ([dict]) query used to fetch obejcts from twitter
        :param search_id: (str) the id of the query/search
        :param object_type: (str) the type of the object (user, tweet, ...)
        :return: None
        """
        for obj in objects:
            metadata = {"date_scraped": datetime.datetime.now(), "query": query, "search_id": search_id,
                        "type": object_type}
            obj["17835"] = metadata

    @staticmethod
    def update_tweet_follow_status(tweet_id: int, follows_screen_name: str):
        db = TwitterScraper.fetch_database()
        db.update_one({"id": tweet_id}, {"$set": {"17835.follows": follows_screen_name}})

    @staticmethod
    def fetch_user_tweets(screen_name):
        """
        fetch tweets from a specific user
        :param screen_name: (str) the twitter handle
        :return objects: ([dict]) the tweet objects
        """
        objects = TwitterScraper.fetch_database_objects(query={"user.screen_name": screen_name, "17835.type": "tweet"})
        return objects

    @staticmethod
    def fetch_all_tweets(group_by=None):
        """
        fetch all tweets in the database
        :param group_by: (str) the field to group the tweets by
            (if grouped, returns tweets as dict mapping field to list of tweets in that group)
        :return objects/objects_map: ([dict]/{group: [dict]) all tweet objects (grouped if group_by is not None)
        """
        objects = TwitterScraper.fetch_database_objects(query={"17835.type": "tweet"},
                                                        fields=["full_text", "screen_name", "17835.bias", "17835.follows"])
        if group_by is None:
            return objects
        else:
            if group_by == "follows":
                group_func = lambda obj: obj["17835"].get("follows", None)
            elif group_by == "screen_name":
                group_func = lambda obj: obj["screen_name"]
            else:
                raise ValueError("invalid group by: {}".format(group_by))

            object_map = {}
            for obj in objects:
                key = group_func(obj)
                if key is None:
                    continue
                else:
                    object_map[key] = object_map.get(key, []) + [obj]
            return object_map

    @staticmethod
    def scrape_user_timeline(screen_name: str=None, user_id: int=None):
        """
        scrape a user's n most recent tweets (max 200)
        can also scrape multiple batches of 200 in a row, requires extra dev
        saves them in the database
        :param screen_name: (str) twitter handle of user to scrape
        :param user_id: (int) the id of the twitter user (must be passed if screen name is not)
        :return tweets: [obj] the tweet objects
        """
        search_params = {"count": 200, "exclude_replies": True,
                         "include_rts": False, "tweet_mode": "extended", "lang": "en"}
        if screen_name is None:
            assert user_id is not None
            search_params["user_id"] = user_id
        else:
            assert user_id is None
            search_params["screen_name"] = screen_name

        search_id = uuid.uuid4().hex
        search_resp = requests.get(TwitterScraper.timeline_url, params=search_params,
                                   headers=TwitterScraper.search_header)
        tweets = search_resp.json()
        if "error" in tweets:
            # raise Exception(tweets["error"])
            print("Error: unable to scrape tweets from {}".format(screen_name if screen_name is not None else user_id))
            return []

        tweets = list(filter(lambda tweet: tweet["lang"] == "en", tweets))

        TwitterScraper.process_objects(tweets, search_params, search_id, object_type="tweet")
        TwitterScraper.put_database_objects(tweets)
        return tweets

    @staticmethod
    def scrape_users_timelines(screen_names: [str] = None, user_ids: [int] = None):
        """
        scrape the timelines of multiple users and return
        :param screen_names: ([str]) screen names of timelines to scrape
        :param user_ids: ([int]) user ids of timelines to scrape
        :return tweets: ([dict]) list of tweet objects
        """
        tweets = []
        if screen_names is None:
            assert user_ids is not None
            for user_id in user_ids:
                tweets += TwitterScraper.scrape_user_timeline(user_id=user_id)
        else:
            assert user_ids is None
            for screen_name in screen_names:
                tweets += TwitterScraper.scrape_user_timeline(screen_name=screen_name)
        return tweets

    @staticmethod
    def scrape_user_profiles(screen_names: [str]):
        """
        scrape the user profiles of the screen names specified then save them in the database
        :param screen_names: ([str]) a list of handles of the users to scrape
        :return profiles: [dict] the user profile objects
        """
        search_params = {"screen_name": screen_names}
        search_id = uuid.uuid4().hex
        search_resp = requests.get(TwitterScraper.user_url, params=search_params,
                                   headers=TwitterScraper.search_header)
        profiles = search_resp.json()

        TwitterScraper.process_objects(profiles, search_params, search_id, object_type="user")
        TwitterScraper.put_database_objects(profiles)
        return profiles

    @staticmethod
    def scrape_user_followers(screen_name: str):
        """
        get the user ids of accounts that follow screen_name
        :param screen_name: (str) screen name of account whos followers we want
        :return follower_ids: ([int]) user ids of followers of screen_name
        """
        user = TwitterScraper.scrape_user_profiles([screen_name])[0]

        search_params = {"screen_name": screen_name, "count": 5000}
        search_id = uuid.uuid4().hex
        search_resp = requests.get(TwitterScraper.followers_url, params=search_params,
                                   headers=TwitterScraper.search_header).json()

        if "errors" in search_resp:
            print("Unable to scrape followers of {} ({})".format(screen_name, search_resp["errors"][0]["message"]))
            return []
        else:
            follower_ids = search_resp["ids"]
            user["17835"]["follower_ids"] = follower_ids

            TwitterScraper.put_database_objects([user])
            return follower_ids

    @staticmethod
    def scrape_users_followers(screen_names: [str]):
        """
        get the user ids of accounts that follow any of screen_names
        :param screen_names: ([str]) screen names of accounts whos followers we want
        :return follower_ids: ([int]) user ids of followers of screen_names
        """
        follower_ids = []
        for screen_name in screen_names:
            follower_ids += TwitterScraper.scrape_user_followers(screen_name)
        return follower_ids

    @staticmethod
    def scrape_user_followers_timelines(screen_name: str, n_followers: int):
        """
        get the follower ids of user then get those followers' timelines
        :param screen_name: (str) screen name of the user whos followers we want
        :param n_followers: (int) number of followers to scrape
        :return tweets: ([dict]) the aggregated tweets of all followers of screen_name
        """
        # def scrape_user_followers_helper(follower_id):
        #     new_tweets = TwitterScraper.scrape_user_timeline(user_id=follower_id)
        #     for tweet in new_tweets:
        #         tweet["17835"]["follows"] = screen_name
        #         tweet["17835"]["bias"] = BIASES.get(screen_name, None)
        #     TwitterScraper.update_database_objects(new_tweets)
        #     print("scraped {} tweets from {}".format(len(new_tweets), follower_id))
        #     return new_tweets
        #
        # print("Scraping followers of {}".format(screen_name))
        # follower_ids_raw = TwitterScraper.scrape_user_followers(screen_name=screen_name)
        # follower_ids = random_sample(follower_ids_raw, min(n_followers, len(follower_ids_raw)))
        # arg_list = [{"follower_id": follower_id} for follower_id in follower_ids]
        # tweets = merge_lists(thread_run(scrape_user_followers_helper, arg_list=arg_list))
        # print("found {} tweets".format(len(tweets)))
        # return tweets

        print("Scraping followers of {}".format(screen_name))
        follower_ids = random_sample(TwitterScraper.scrape_user_followers(screen_name=screen_name), n_followers)
        tweets = []
        for follower_id in follower_ids:
            new_tweets = TwitterScraper.scrape_user_timeline(user_id=follower_id)
            for tweet in new_tweets:
                tweet["17835"]["follows"] = screen_name
                tweet["17835"]["bias"] = BIASES.get(screen_name, None)
            TwitterScraper.update_database_objects(new_tweets)
            tweets += new_tweets
        print("found {} tweets".format(len(tweets)))
        return tweets

    @staticmethod
    def scrape_users_followers_timelines(screen_names: [str], n_followers=500):
        """
        for each screen name in screen_names, get the follower ids of user then get those followers' timelines
        sorts the tweets by the screen_name whos follwoers they belong to
        :param screen_names: ([str]) screen names of the user whos followers we want
        :param n_followers: (int) number of followers to scrape
        :return tweets: ({str: [dict]}) maps screen_name to the aggregated tweets of all followers of screen_name
        """

        # arg_list = [{"screen_name": screen_name, "n_followers": n_followers} for screen_name in screen_names]
        # tweets = parallel_run(TwitterScraper.scrape_user_followers_timelines, arg_list=arg_list)
        # tweets = merge_lists(tweets)
        tweets = {}
        for screen_name in screen_names:
            tweets[screen_name] = TwitterScraper.scrape_user_followers_timelines(screen_name=screen_name, n_followers=n_followers)
        return tweets


if __name__ == '__main__':
    # TwitterScraper.count_tweets()
    tweets = TwitterScraper.scrape_users_followers_timelines(NETWORKS[3:], n_followers=100)
    # tweets = TwitterScraper.fetch_all_tweets()
    # objects = TwitterScraper.fetch_user_tweets("seanhannity")
    print()
